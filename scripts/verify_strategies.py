"""Critical checkpoint before PPO: verify 4 strategies produce different outputs.

Pass condition: best strategy varies across images. If Global always wins -> fix losses.
"""

import os
import sys
import argparse
import json

# Ensure project root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from PIL import Image

from src.data.dataset import InpaintingDataset
from src.data.transforms import denormalize_image
from src.models.rl_inpainting_model import RLInpaintingModel
from src.rl.reward import compute_image_metrics


def verify_strategies(
    backbone_checkpoint: str = "checkpoints/pretrained/deepfillv2_places2.pth",
    adapters_checkpoint: str = "checkpoints/adapters/adapters_final.pt",
    data_dir: str = "data/raw/sample_images",
    image_size: int = 256,
    num_images: int = 5,
    output_dir: str = "results/strategy_verify",
    device: str = "cpu",
    state_only: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    model = RLInpaintingModel(cnum=48, strategy_dim=64, latent_dim=256).to(device_t).eval()

    if backbone_checkpoint and os.path.exists(backbone_checkpoint):
        print(f"Loading backbone from {backbone_checkpoint}")
        model.load_pretrained_backbone(backbone_checkpoint, strict=False)
    else:
        print(f"Backbone checkpoint not found at {backbone_checkpoint}, using init weights (variance will be synthetic).")

    if adapters_checkpoint and os.path.exists(adapters_checkpoint):
        print(f"Loading adapters from {adapters_checkpoint}")
        model.load_adapters(adapters_checkpoint)

    dataset = InpaintingDataset(
        image_dir=data_dir if os.path.exists(data_dir) else None,
        image_size=image_size,
        is_train=False,
        synthetic_size=num_images,
        fixed_seed=42,
    )

    # ---- State embedding variance check ----
    print("\n=== State Embedding Variance Check ===")
    embeddings = []
    for i in range(min(num_images, len(dataset))):
        item = dataset[i]
        mi = item["masked_image"].unsqueeze(0).to(device_t)
        mask = item["mask"].unsqueeze(0).to(device_t)
        emb = model.extract_state_embedding(mi, mask)  # (1,256)
        embeddings.append(emb.detach().cpu().numpy().flatten())
        print(f"Image {i} stats: {item['stats']}")

    embeddings = np.stack(embeddings, axis=0)  # (N,256)
    # Pairwise distances
    if len(embeddings) > 1:
        dists = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                d = np.linalg.norm(embeddings[i] - embeddings[j])
                dists.append(d)
        print(f"Mean pairwise embedding L2 distance: {np.mean(dists):.4f} (should be > 0.1)")
        print(f"Min distance: {np.min(dists):.4f}")
        if np.mean(dists) < 1e-5:
            print("FAIL: State embeddings are identical across masks -> encoder collapsed!")
        else:
            print("PASS: State embeddings vary across masks.")

    if state_only:
        print("State-only flag set, stopping before strategy output check.")
        return

    # ---- Strategy diversity check ----
    print("\n=== Strategy Output Diversity Check ===")
    best_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    results = []

    for i in range(min(num_images, len(dataset))):
        item = dataset[i]
        image = item["image"].unsqueeze(0).to(device_t)
        mask = item["mask"].unsqueeze(0).to(device_t)
        masked_image = item["masked_image"].unsqueeze(0).to(device_t)

        with torch.no_grad():
            coarse = model.coarse_forward(masked_image, mask)
            coarse_comp = masked_image + coarse * mask

            metrics_per_strategy = {}
            for strat in range(4):
                refined = model.refine(coarse_comp, mask, strategy=strat)
                completed = masked_image + refined * mask
                m = compute_image_metrics(completed, image, compute_lpips=False)
                metrics_per_strategy[strat] = m
                print(f"  Image {i} Strategy {strat} -> PSNR {m['psnr']:.2f} SSIM {m['ssim']:.3f}")

            # Determine best by PSNR
            best_strat = max(metrics_per_strategy, key=lambda k: metrics_per_strategy[k]["psnr"])
            best_counts[best_strat] += 1
            results.append({"image_idx": i, "metrics": metrics_per_strategy, "best": int(best_strat)})

            # Save visual grid for this image
            with torch.no_grad():
                outs = []
                outs.append(image)
                outs.append(masked_image)
                outs.append(masked_image + coarse * mask)
                for s in range(4):
                    r = model.refine(coarse_comp, mask, strategy=s)
                    comp = masked_image + r * mask
                    outs.append(comp)
                # Denormalize and create PIL grid
                pil_images = []
                for t in outs:
                    arr = denormalize_image(t[0], to_uint8=True).permute(1, 2, 0).cpu().numpy()
                    pil_images.append(Image.fromarray(arr))
                w, h = pil_images[0].size
                grid = Image.new("RGB", (w * len(pil_images), h))
                for idx, pim in enumerate(pil_images):
                    grid.paste(pim, (idx * w, 0))
                grid.save(os.path.join(output_dir, f"verify_image_{i:02d}.png"))

    print("\nBest strategy distribution:")
    for k, v in best_counts.items():
        print(f"  Strategy {k}: {v}/{num_images} images")

    # Save JSON
    with open(os.path.join(output_dir, "verify_results.json"), "w") as f:
        # Convert numpy floats to python floats for json
        def convert(o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            raise TypeError
        json.dump({"best_counts": best_counts, "results": results}, f, indent=2, default=convert)

    total = sum(best_counts.values())
    max_pct = max(best_counts.values()) / max(1, total)
    if max_pct > 0.8:
        print(f"\nFAIL condition: same strategy wins on {max_pct*100:.1f}% (>80%) of images -> adapters not differentiated!")
    else:
        print(f"\nPASS: Strategy wins distributed (max {max_pct*100:.1f}%). Adapters appear differentiated.")

    # Also check reward non-degeneracy (quick)
    print("\n=== Reward Non-Degeneracy Check (1 image, 4 actions) ===")
    from src.rl.reward import compute_reward
    item = dataset[0]
    image = item["image"].unsqueeze(0).to(device_t)
    mask = item["mask"].unsqueeze(0).to(device_t)
    masked_image = item["masked_image"].unsqueeze(0).to(device_t)
    with torch.no_grad():
        coarse = model.coarse_forward(masked_image, mask)
        coarse_comp = masked_image + coarse * mask
        coarse_metrics = compute_image_metrics(coarse_comp, image, compute_lpips=False)
        # Need dummy disc delta 0, use reward with zero
        for strat in range(4):
            refined = model.refine(coarse_comp, mask, strategy=strat)
            completed = masked_image + refined * mask
            refined_metrics = compute_image_metrics(completed, image, compute_lpips=False)
            rew, breakdown = compute_reward(coarse_metrics, refined_metrics, 0.0, strat)
            print(f"  Strat {strat} reward {rew:.4f} d_psnr {breakdown['d_psnr']:.4f} d_psnr_raw {breakdown['d_psnr_raw']:.4f}")

    print(f"\nVerification outputs saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify strategy diversity")
    parser.add_argument("--backbone-checkpoint", type=str, default="checkpoints/pretrained/deepfillv2_places2.pth")
    parser.add_argument("--adapters-checkpoint", type=str, default="checkpoints/adapters/adapters_final.pt")
    parser.add_argument("--data-dir", type=str, default="data/raw/sample_images")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/strategy_verify")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-only", action="store_true", help="Only check state embedding variance")
    args = parser.parse_args()
    verify_strategies(
        backbone_checkpoint=args.backbone_checkpoint,
        adapters_checkpoint=args.adapters_checkpoint,
        data_dir=args.data_dir,
        image_size=args.image_size,
        num_images=args.num_images,
        output_dir=args.output_dir,
        device=args.device,
        state_only=args.state_only,
    )
