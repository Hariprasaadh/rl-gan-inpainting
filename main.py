import os
import sys
import argparse
from PIL import Image
import torch
import torchvision.transforms as T

from src.data.mask_generator import generate_fixed_eval_masks
from src.data.dataset import InpaintingDataset
from src.data.transforms import denormalize_image, feather_composite
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.training.pretrain_gan import train_gan_baseline
from src.training.train_rl_bandit import train_rl_bandit
from src.training.train_rl_ablations import train_ablation_c, train_ablation_d
from src.training.joint_finetune import train_joint_finetune
from src.evaluation.evaluate import evaluate_all
from src.rl.actions import ACTION_NAMES
from src.rl.state import StateBuilder


def cmd_generate_eval_masks(args: argparse.Namespace) -> None:
    """Generate fixed deterministic evaluation masks partitioned by severity."""
    output_dir = args.output_dir or "data/masks/fixed_eval_masks"
    print(f"Generating fixed evaluation masks into: {output_dir}")
    saved = generate_fixed_eval_masks(
        output_dir=output_dir,
        count_per_bucket=args.count,
        image_size=args.image_size,
        seed=args.seed,
    )
    total = sum(len(v) for v in saved.values())
    print(f"Successfully generated {total} fixed evaluation masks across 4 severity buckets.")


def cmd_train_gan(args: argparse.Namespace) -> None:
    """Train GAN baseline (Experiment A)."""
    train_gan_baseline(config_path=args.config, override_epochs=args.epochs)


def cmd_train_rl(args: argparse.Namespace) -> None:
    """Train PPO contextual bandit controller (Experiment B)."""
    train_rl_bandit(config_path=args.config, override_timesteps=args.timesteps)


def cmd_train_ablations(args: argparse.Namespace) -> None:
    """Train ablation models (Experiments C & D)."""
    if args.mode in ("c", "all"):
        train_ablation_c(config_path=args.config, timesteps=args.timesteps)
    if args.mode in ("d", "all"):
        train_ablation_d(config_path=args.config, timesteps=args.timesteps)


def cmd_joint_finetune(args: argparse.Namespace) -> None:
    """Run optional joint fine-tuning (Experiment E)."""
    train_joint_finetune(config_path=args.config, override_epochs=args.epochs)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate models on fixed test set and generate report."""
    dataset = InpaintingDataset(
        image_dir=args.data_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        is_train=False,
        synthetic_size=args.samples,
        fixed_seed=42,
    )
    generator = InpaintingGenerator(base_channels=32, strategy_dim=64, latent_dim=256)
    if args.gan_checkpoint and os.path.exists(args.gan_checkpoint):
        ckpt = torch.load(args.gan_checkpoint, map_location=args.device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"], strict=False)

    evaluate_all(
        dataset=dataset,
        generator=generator,
        rl_agent_path=args.rl_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.samples,
    )


def cmd_demo(args: argparse.Namespace) -> None:
    """Run interactive demonstration on sample image and save side-by-side comparison."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = InpaintingDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
        synthetic_size=5,
        is_train=False,
        fixed_seed=42,
    )
    sample = dataset[args.sample_idx]
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    masked_image = sample["masked_image"].unsqueeze(0).to(device)

    generator = InpaintingGenerator(base_channels=32, strategy_dim=64, latent_dim=256).to(device).eval()
    if args.gan_checkpoint and os.path.exists(args.gan_checkpoint):
        ckpt = torch.load(args.gan_checkpoint, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"], strict=False)

    with torch.no_grad():
        # Coarse pass
        coarse = generator.coarse_forward(masked_image, mask)
        coarse_comp = masked_image + coarse * mask

        # Fixed baseline
        gan_out = generator(masked_image, mask, strategy=None)
        gan_comp = feather_composite(image, gan_out["completed"], mask)

        # RL Adaptive
        chosen_action = 0
        if args.rl_checkpoint and os.path.exists(args.rl_checkpoint):
            from stable_baselines3 import PPO
            agent = PPO.load(args.rl_checkpoint, device=device)
            sb = StateBuilder(latent_dim=generator.encoder.latent_dim)
            latent = generator.extract_state_embedding(masked_image, mask)
            obs = sb.build_state(latent, sample["stats"], coarse_composite=coarse_comp, mask=mask)
            act, _ = agent.predict(obs, deterministic=True)
            chosen_action = int(act)

        rl_refined = generator.refine(coarse_comp, mask, strategy=chosen_action)
        rl_comp = feather_composite(image, rl_refined, mask)

    # Convert to PIL images for visualization grid
    img_gt = Image.fromarray(denormalize_image(image[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
    img_masked = Image.fromarray(denormalize_image(masked_image[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
    img_coarse = Image.fromarray(denormalize_image(coarse_comp[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
    img_baseline = Image.fromarray(denormalize_image(gan_comp[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())
    img_rl = Image.fromarray(denormalize_image(rl_comp[0], to_uint8=True).permute(1, 2, 0).cpu().numpy())

    # Build side-by-side grid
    w, h = img_gt.size
    grid = Image.new("RGB", (w * 5, h))
    grid.paste(img_gt, (0, 0))
    grid.paste(img_masked, (w, 0))
    grid.paste(img_coarse, (w * 2, 0))
    grid.paste(img_baseline, (w * 3, 0))
    grid.paste(img_rl, (w * 4, 0))

    out_path = os.path.join(args.output_dir, "demo_comparison.png")
    grid.save(out_path)
    print(f"Demo comparison saved to: {out_path}")
    print(f"RL Policy selected Action {chosen_action}: {ACTION_NAMES.get(chosen_action, 'Unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL-GAN: Adaptive Image Inpainting using Reinforcement Learning Guided GAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: generate-eval-masks
    p_masks = subparsers.add_parser("generate-eval-masks", help="Generate fixed deterministic eval masks")
    p_masks.add_argument("--output-dir", type=str, default="data/masks/fixed_eval_masks")
    p_masks.add_argument("--count", type=int, default=25, help="Masks per severity bucket")
    p_masks.add_argument("--image-size", type=int, default=256)
    p_masks.add_argument("--seed", type=int, default=42)

    # Command: train-gan
    p_gan = subparsers.add_parser("train-gan", help="Pretrain GAN Baseline (Experiment A)")
    p_gan.add_argument("--config", type=str, default="configs/pretrain_gan.yaml")
    p_gan.add_argument("--epochs", type=int, default=None)

    # Command: train-rl
    p_rl = subparsers.add_parser("train-rl", help="Train PPO Contextual Bandit (Experiment B)")
    p_rl.add_argument("--config", type=str, default="configs/rl_agent_bandit.yaml")
    p_rl.add_argument("--timesteps", type=int, default=None)

    # Command: train-ablations
    p_abl = subparsers.add_parser("train-ablations", help="Train Ablation Experiments (C & D)")
    p_abl.add_argument("--config", type=str, default="configs/rl_agent_bandit.yaml")
    p_abl.add_argument("--mode", type=str, default="all", choices=["c", "d", "all"])
    p_abl.add_argument("--timesteps", type=int, default=2000)

    # Command: joint-finetune
    p_jnt = subparsers.add_parser("joint-finetune", help="Run Joint Fine-Tuning (Experiment E)")
    p_jnt.add_argument("--config", type=str, default="configs/joint_finetune.yaml")
    p_jnt.add_argument("--epochs", type=int, default=None)

    # Command: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Benchmark evaluation across models and severity buckets")
    p_eval.add_argument("--data-dir", type=str, default="data/raw/sample_images")
    p_eval.add_argument("--mask-dir", type=str, default=None)
    p_eval.add_argument("--gan-checkpoint", type=str, default="checkpoints/gan_baseline/best_model.pt")
    p_eval.add_argument("--rl-checkpoint", type=str, default="checkpoints/rl_agent_bandit/ppo_bandit_final.zip")
    p_eval.add_argument("--output-dir", type=str, default="results")
    p_eval.add_argument("--image-size", type=int, default=128)
    p_eval.add_argument("--samples", type=int, default=30)
    p_eval.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Command: demo
    p_demo = subparsers.add_parser("demo", help="Generate side-by-side demo comparison")
    p_demo.add_argument("--image-dir", type=str, default="data/raw/sample_images")
    p_demo.add_argument("--gan-checkpoint", type=str, default="checkpoints/gan_baseline/best_model.pt")
    p_demo.add_argument("--rl-checkpoint", type=str, default="checkpoints/rl_agent_bandit/ppo_bandit_final.zip")
    p_demo.add_argument("--output-dir", type=str, default="results")
    p_demo.add_argument("--sample-idx", type=int, default=0)
    p_demo.add_argument("--image-size", type=int, default=128)
    p_demo.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.command == "generate-eval-masks":
        cmd_generate_eval_masks(args)
    elif args.command == "train-gan":
        cmd_train_gan(args)
    elif args.command == "train-rl":
        cmd_train_rl(args)
    elif args.command == "train-ablations":
        cmd_train_ablations(args)
    elif args.command == "joint-finetune":
        cmd_joint_finetune(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
