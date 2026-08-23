import os
import argparse
import json
from typing import Dict, Any, Optional
import numpy as np
import torch

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.data.dataset import InpaintingDataset
from src.models.generator import InpaintingGenerator
from src.rl.actions import ACTION_NAMES
from src.rl.state import StateBuilder
from src.evaluation.metrics import InpaintingMetricEvaluator
from src.evaluation.severity_eval import SeverityEvaluator
from src.baselines.lama_inference import LaMaInference


def evaluate_all(
    dataset: InpaintingDataset,
    generator: InpaintingGenerator,
    rl_agent_path: Optional[str] = None,
    output_dir: str = "results",
    device: str = "cpu",
    compute_lpips: bool = False,
    num_samples: int = 50,
) -> Dict[str, Any]:
    """Run full benchmark evaluation across models, severity buckets, and action distributions."""
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("evaluate", log_file=os.path.join(output_dir, "eval.log"))
    logger.info(f"Starting evaluation on {num_samples} test samples (device: {device})...")

    dev = torch.device(device)
    generator = generator.to(dev).eval()

    # Load RL agent if available
    rl_agent = None
    if rl_agent_path and os.path.exists(rl_agent_path):
        from stable_baselines3 import PPO
        rl_agent = PPO.load(rl_agent_path, device=dev)
        logger.info(f"Loaded RL Agent policy from: {rl_agent_path}")

    lama_model = LaMaInference(device=device)

    # Evaluators
    eval_gan = InpaintingMetricEvaluator(device=dev, compute_lpips=compute_lpips)
    eval_rl = InpaintingMetricEvaluator(device=dev, compute_lpips=compute_lpips)
    eval_lama = InpaintingMetricEvaluator(device=dev, compute_lpips=compute_lpips)

    state_builder = StateBuilder(
        latent_dim=generator.encoder.latent_dim,
        include_mask_stats=True,
        include_quality_proxy=True,
    )

    action_counts = {i: 0 for i in range(4)}
    n = min(len(dataset), num_samples)

    for i in range(n):
        item = dataset[i]
        image = item["image"].unsqueeze(0).to(dev)
        mask = item["mask"].unsqueeze(0).to(dev)
        masked_image = item["masked_image"].unsqueeze(0).to(dev)
        stats = item["stats"]

        with torch.no_grad():
            # 1. GAN Baseline (Fixed neutral)
            gan_out = generator(masked_image, mask, strategy=None)
            eval_gan.update(gan_out["completed"], image, mask)

            # 2. RL-GAN (Adaptive)
            if rl_agent is not None:
                latent_emb = generator.extract_state_embedding(masked_image, mask)
                coarse = generator.coarse_forward(masked_image, mask)
                coarse_comp = masked_image + coarse * mask
                obs = state_builder.build_state(
                    latent_embedding=latent_emb,
                    mask_stats=stats,
                    coarse_composite=coarse_comp,
                    mask=mask,
                )
                action, _ = rl_agent.predict(obs, deterministic=True)
                act_idx = int(action)
                action_counts[act_idx] += 1

                rl_refined = generator.refine(coarse_comp, mask, strategy=act_idx)
                rl_completed = masked_image + rl_refined * mask
            else:
                rl_completed = gan_out["completed"]

            eval_rl.update(rl_completed, image, mask)

            # 3. LaMa External Reference
            lama_completed = lama_model.inpaint(masked_image, mask)
            eval_lama.update(lama_completed, image, mask)

    # Severity analysis
    sev_evaluator = SeverityEvaluator(generator, rl_agent, device=device, compute_lpips=compute_lpips)
    sev_results = sev_evaluator.evaluate_dataset(dataset, num_samples=num_samples)
    sev_table_md = sev_evaluator.format_table(sev_results)

    # Format Summary Ablation Table
    sum_gan = eval_gan.summary()
    sum_rl = eval_rl.summary()
    sum_lama = eval_lama.summary()

    ablation_rows = [
        {"Model": "A: GAN Baseline (Fixed Strategy)", "PSNR": f"{sum_gan['psnr']['mean']:.2f} ± {sum_gan['psnr']['std']:.2f}", "SSIM": f"{sum_gan['ssim']['mean']:.3f}", "L1": f"{sum_gan['l1']['mean']:.4f}"},
        {"Model": "B: RL-GAN (Adaptive PPO)", "PSNR": f"{sum_rl['psnr']['mean']:.2f} ± {sum_rl['psnr']['std']:.2f}", "SSIM": f"{sum_rl['ssim']['mean']:.3f}", "L1": f"{sum_rl['l1']['mean']:.4f}"},
        {"Model": "LaMa (External Reference)", "PSNR": f"{sum_lama['psnr']['mean']:.2f} ± {sum_lama['psnr']['std']:.2f}", "SSIM": f"{sum_lama['ssim']['mean']:.3f}", "L1": f"{sum_lama['l1']['mean']:.4f}"},
    ]

    total_actions = max(1, sum(action_counts.values()))
    action_dist_lines = []
    for act_idx, count in action_counts.items():
        pct = (count / total_actions) * 100.0
        name = ACTION_NAMES.get(act_idx, f"Action {act_idx}")
        action_dist_lines.append(f"- **{name}**: {pct:.1f}% ({count}/{total_actions})")

    report_md = f"""# RL-GAN Inpainting Evaluation Report

## 1. Model Comparison & Ablation Table

| Setting / Model | PSNR (dB) [^] | SSIM [^] | L1 Loss [v] |
|---|:---:|:---:|:---:|
| **A: GAN Baseline (Fixed Strategy)** | {sum_gan['psnr']['mean']:.2f} | {sum_gan['ssim']['mean']:.3f} | {sum_gan['l1']['mean']:.4f} |
| **B: RL-GAN (Adaptive Controller)** | {sum_rl['psnr']['mean']:.2f} | {sum_rl['ssim']['mean']:.3f} | {sum_rl['l1']['mean']:.4f} |
| **LaMa Reference (Non-RL)** | {sum_lama['psnr']['mean']:.2f} | {sum_lama['ssim']['mean']:.3f} | {sum_lama['l1']['mean']:.4f} |

---

## 2. Learned Action Distribution
{chr(10).join(action_dist_lines)}

---

## 3. Mask Severity Analysis (Hypothesis Testing)
{sev_table_md}
"""

    with open(os.path.join(output_dir, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    json_data = {
        "gan_summary": sum_gan,
        "rl_summary": sum_rl,
        "lama_summary": sum_lama,
        "action_counts": action_counts,
        "severity_summary": sev_results["summary_table"],
    }
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    logger.info(f"\n{report_md}")
    from src.utils.logger import close_logger
    close_logger(logger)
    return json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL-GAN Inpainting")
    parser.add_argument("--samples", type=int, default=30, help="Number of test samples")
    parser.add_argument("--rl-checkpoint", type=str, default="checkpoints/rl_agent_bandit/ppo_bandit_final.zip")
    args = parser.parse_args()

    dataset = InpaintingDataset(image_dir=None, image_size=128, synthetic_size=args.samples, is_train=False, fixed_seed=42)
    generator = InpaintingGenerator(base_channels=32, strategy_dim=64, latent_dim=256)
    evaluate_all(dataset, generator, rl_agent_path=args.rl_checkpoint, num_samples=args.samples)
