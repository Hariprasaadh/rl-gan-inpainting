"""verify_strategies.py — Checkpoint 3: confirm adapter specialisation.

Run all 4 strategies on 5 (or N) test images, print PSNR/SSIM/LPIPS for each,
and save side-by-side grids to results/strategy_verify/.

PASS condition: the best strategy differs across images (no single strategy
wins >80% of the time).  If all images prefer the same strategy, the adapter
pretraining losses need adjustment.

Usage
-----
  python scripts/verify_strategies.py \
      --backbone  checkpoints/pretrained/deepfillv2_places2.pth \
      --adapters  checkpoints/adapters/adapters_final.pt \
      --data-dir  data/raw/val2017 \
      --n-images  8 \
      --output    results/strategy_verify

Optional:
  --state-only   Only print state embedding statistics (Checkpoint 2).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips as _lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

from src.models.rl_inpainting_model import RLInpaintingModel
from src.models.strategy_adapters import ADAPTER_NAMES
from src.data.dataset import InpaintingDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def to_np(t: torch.Tensor) -> np.ndarray:
    """(B, 3, H, W) tensor in [-1, 1] -> (H, W, 3) numpy in [0, 1]."""
    return ((t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0).clip(0, 1)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, lpips_net=None) -> Dict[str, float]:
    psnr = float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
    ssim = float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))

    lpips_val = float("nan")
    if lpips_net is not None:
        pred_t = torch.tensor(pred).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        gt_t   = torch.tensor(gt).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            lpips_val = float(lpips_net(pred_t, gt_t).mean().item())

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}


# ---------------------------------------------------------------------------
# State embedding check (Checkpoint 2)
# ---------------------------------------------------------------------------

def verify_state_embeddings(
    model: RLInpaintingModel,
    dataset: InpaintingDataset,
    n: int,
    device: torch.device,
) -> None:
    print("\n=== CHECKPOINT 2: State Embedding Variability ===")
    embeddings = []
    for i in range(n):
        sample  = dataset[i]
        masked  = sample["masked_image"].unsqueeze(0).to(device)
        mask    = sample["mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.extract_state_embedding(masked, mask)
        embeddings.append(emb.squeeze(0).cpu())

    stacked = torch.stack(embeddings)            # (n, 256)
    std_per_dim = stacked.std(dim=0)
    mean_std    = std_per_dim.mean().item()
    zero_dims   = (std_per_dim < 1e-4).sum().item()

    print(f"  Num images      : {n}")
    print(f"  Embedding dim   : {stacked.shape[1]}")
    print(f"  Mean std/dim    : {mean_std:.4f}   (>0.01 is good)")
    print(f"  Collapsed dims  : {zero_dims}/{stacked.shape[1]}  (0 is ideal)")

    if mean_std < 0.005:
        print("  [WARN] Embeddings are near-constant — encoder head may need more training.")
    else:
        print("  [PASS] Embeddings vary meaningfully across different masks.")


# ---------------------------------------------------------------------------
# Strategy verification (Checkpoint 3)
# ---------------------------------------------------------------------------

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (to_np(t) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_grid(images: List[Image.Image], labels: List[str], W: int, H: int) -> Image.Image:
    """Build a horizontal comparison grid with text labels."""
    n   = len(images)
    pad = 20
    grid = Image.new("RGB", (n * W, H + pad), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)
    for idx, (img, lbl) in enumerate(zip(images, labels)):
        grid.paste(img.resize((W, H)), (idx * W, pad))
        draw.text((idx * W + 4, 2), lbl, fill=(255, 220, 80))
    return grid


def verify_strategies(
    model: RLInpaintingModel,
    dataset: InpaintingDataset,
    n_images: int,
    device: torch.device,
    output_dir: str,
    lpips_net=None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    wins_per_action: Dict[int, int] = defaultdict(int)
    all_metrics: List[Dict] = []

    print(f"\n=== CHECKPOINT 3: Strategy Verification ({n_images} images) ===")
    print(f"{'Image':>6}  {'Best':>8}  {'G-PSNR':>7}  {'L-PSNR':>7}  {'B-PSNR':>7}  {'T-PSNR':>7}")

    for img_idx in range(n_images):
        sample  = dataset[img_idx]
        gt      = sample["image"].unsqueeze(0).to(device)
        mask    = sample["mask"].unsqueeze(0).to(device)
        masked  = sample["masked_image"].unsqueeze(0).to(device)
        gt_np   = to_np(gt)

        with torch.no_grad():
            # Coarse pass
            coarse_out, backbone_refined = model.backbone(masked, mask)
            coarse_comp = masked * (1 - mask) + coarse_out * mask

            row: Dict[str, object] = {}
            psnrs: Dict[int, float] = {}
            pil_images = [tensor_to_pil(gt), tensor_to_pil(masked), tensor_to_pil(coarse_comp)]
            labels     = ["GT", "Masked", "Coarse"]

            for action in range(4):
                strat_emb = model.strategy_embeddings(
                    torch.tensor([action], device=device)
                )
                refined = model.adapters[action](backbone_refined, mask, strat_emb)
                completed = masked * (1 - mask) + refined * mask
                pred_np   = to_np(completed)
                m = compute_metrics(pred_np, gt_np, lpips_net)
                psnrs[action]  = m["psnr"]
                row[action]    = m
                pil_images.append(tensor_to_pil(completed))
                labels.append(f"A{action}:{ADAPTER_NAMES[action]}\nPSNR={m['psnr']:.2f}")

        best_action = max(psnrs, key=lambda a: psnrs[a])
        wins_per_action[best_action] += 1
        all_metrics.append(row)

        psnr_strs = "  ".join(f"{psnrs[a]:7.2f}" for a in range(4))
        print(f"  {img_idx:4d}  {ADAPTER_NAMES[best_action]:>8}  {psnr_strs}")

        # Save comparison grid
        h, w = gt_np.shape[:2]
        grid  = make_grid(pil_images, labels, W=w, H=h)
        grid.save(os.path.join(output_dir, f"img_{img_idx:04d}.png"))

    # Summary
    total = sum(wins_per_action.values())
    print(f"\n{'='*50}")
    print("  Action Distribution (best strategy wins):")
    for a in range(4):
        cnt = wins_per_action.get(a, 0)
        pct = cnt / max(total, 1) * 100
        bar = "#" * int(pct / 5)
        print(f"    {ADAPTER_NAMES[a]:12s} (A{a}): {pct:5.1f}%  {bar}")

    dominant = max(wins_per_action, key=lambda a: wins_per_action.get(a, 0))
    dominant_pct = wins_per_action.get(dominant, 0) / max(total, 1) * 100

    print()
    if dominant_pct > 80:
        print(f"  [FAIL] {ADAPTER_NAMES[dominant]} wins {dominant_pct:.0f}% of images.")
        print("  Action: check adapter loss weights in pretrain_adapters.py")
    else:
        print(f"  [PASS] No single strategy dominates (max {dominant_pct:.0f}%).")
        print(f"  Grids saved to: {output_dir}/")

    # Save per-action average PSNR
    for a in range(4):
        avg_psnr = np.mean([m[a]["psnr"] for m in all_metrics if a in m])
        print(f"    {ADAPTER_NAMES[a]:12s}: avg PSNR = {avg_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify strategy adapter specialisation")
    parser.add_argument("--backbone",    type=str, default="checkpoints/pretrained/deepfillv2_places2.pth")
    parser.add_argument("--adapters",    type=str, default="checkpoints/adapters/adapters_final.pt")
    parser.add_argument("--data-dir",    type=str, default="data/raw/val2017")
    parser.add_argument("--n-images",    type=int, default=8)
    parser.add_argument("--image-size",  type=int, default=256)
    parser.add_argument("--output",      type=str, default="results/strategy_verify")
    parser.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--state-only",  action="store_true", help="Only run Checkpoint 2 (state embedding)")
    parser.add_argument("--no-lpips",    action="store_true", help="Skip LPIPS computation")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build model
    model = RLInpaintingModel(cnum=48, strategy_dim=64, latent_dim=256).to(device)
    model.load_pretrained_backbone(args.backbone)
    if os.path.exists(args.adapters):
        model.load_adapters(args.adapters)
    else:
        print(f"[WARN] No adapter checkpoint found at {args.adapters}. Using random weights.")
    model.eval()

    # Dataset
    dataset = InpaintingDataset(
        image_dir=args.data_dir,
        image_size=args.image_size,
        is_train=False,
        fixed_seed=42,
    )

    # Checkpoint 2: state embeddings
    verify_state_embeddings(model, dataset, n=min(args.n_images, len(dataset)), device=device)

    if args.state_only:
        return

    # Optional LPIPS
    lpips_net = None
    if _HAS_LPIPS and not args.no_lpips:
        lpips_net = _lpips_lib.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in lpips_net.parameters():
            p.requires_grad = False

    # Checkpoint 3: strategy verification
    verify_strategies(
        model     = model,
        dataset   = dataset,
        n_images  = min(args.n_images, len(dataset)),
        device    = device,
        output_dir= args.output,
        lpips_net = lpips_net,
    )


if __name__ == "__main__":
    main()
