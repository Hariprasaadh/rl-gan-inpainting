"""evaluate_baseline.py — Experiment A: DeepFill-v2 frozen backbone, no RL.

Evaluates the backbone output on the fixed 100-image eval set (created by
create_eval_set.py) and prints a table of PSNR/SSIM/LPIPS per severity bucket.

This is the Experiment A baseline — all subsequent experiments (B: PPO bandit,
C/D: ablations, E: joint fine-tuning) should improve over this.

Usage
-----
  python scripts/evaluate_baseline.py \
      --backbone   checkpoints/pretrained/deepfillv2_places2.pth \
      --eval-dir   data/masks/fixed_eval_masks \
      --output-dir results/baseline_A \
      --device     cuda
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms

try:
    import lpips as _lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

from src.models.deepfillv2 import DeepFillV2Generator, load_deepfillv2_checkpoint


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

to_tensor = transforms.ToTensor()


def load_eval_pair(
    base_dir: str,
    entry: Dict,
    image_size: int,
    device: torch.device,
):
    """Load an image-mask pair from eval manifest entry."""
    img_path  = os.path.join(base_dir, entry["image"])
    mask_path = os.path.join(base_dir, entry["mask"])

    img  = Image.open(img_path).convert("RGB").resize((image_size, image_size))
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size))

    img_t  = to_tensor(img).unsqueeze(0).to(device)    # (1, 3, H, W) in [0, 1]
    img_t  = img_t * 2 - 1                             # -> [-1, 1]
    mask_t = to_tensor(mask).unsqueeze(0).to(device)   # (1, 1, H, W) in [0, 1]
    mask_t = (mask_t > 0.5).float()                    # binarize
    masked_t = img_t * (1 - mask_t)                    # zero out hole

    return img_t, mask_t, masked_t


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lpips_net=None,
) -> Dict[str, float]:
    """Compute PSNR/SSIM/LPIPS. pred and gt are (1, 3, H, W) in [-1, 1]."""
    pred_np = ((pred.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    gt_np   = ((gt.squeeze(0).permute(1, 2, 0).cpu().numpy()   + 1) / 2).clip(0, 1)

    psnr = float(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
    ssim = float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0))

    lpips_val = float("nan")
    if lpips_net is not None:
        with torch.no_grad():
            lpips_val = float(lpips_net(pred, gt).mean().item())

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}


def save_comparison(
    img: torch.Tensor,
    masked: torch.Tensor,
    pred: torch.Tensor,
    save_path: str,
) -> None:
    """Save a 3-panel comparison image."""
    def to_pil(t: torch.Tensor) -> Image.Image:
        arr = ((t.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    panels = [to_pil(img), to_pil(masked), to_pil(pred)]
    labels = ["GT", "Masked", "DeepFill-v2"]
    W, H   = panels[0].size
    grid   = Image.new("RGB", (W * 3, H), color=(20, 20, 20))
    for i, (panel, _) in enumerate(zip(panels, labels)):
        grid.paste(panel, (i * W, 0))
    grid.save(save_path)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    backbone_path: str,
    eval_dir: str,
    output_dir: str,
    image_size: int = 256,
    device_str: str = "cuda",
    compute_lpips: bool = True,
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = DeepFillV2Generator(cnum=48).to(device).eval()
    load_deepfillv2_checkpoint(model, backbone_path)

    # LPIPS
    lpips_net = None
    if _HAS_LPIPS and compute_lpips:
        lpips_net = _lpips_lib.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in lpips_net.parameters():
            p.requires_grad = False

    # Load manifest
    manifest_path = os.path.join(eval_dir, "eval_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Eval manifest not found: {manifest_path}\n"
            "Run: python scripts/create_eval_set.py  first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Evaluate per bucket
    all_results: Dict[str, List[Dict]] = defaultdict(list)

    for bucket_name, entries in manifest.items():
        bucket_out = os.path.join(output_dir, bucket_name)
        os.makedirs(bucket_out, exist_ok=True)
        print(f"\n  Evaluating bucket: {bucket_name} ({len(entries)} images)")

        for idx, entry in enumerate(entries):
            img_t, mask_t, masked_t = load_eval_pair(eval_dir, entry, image_size, device)

            with torch.no_grad():
                _, refined = model(masked_t, mask_t)
            completed = masked_t * (1 - mask_t) + refined * mask_t

            metrics = compute_metrics(completed, img_t, lpips_net)
            metrics["missing_ratio"] = entry.get("missing_ratio", 0.0)
            all_results[bucket_name].append(metrics)

            save_comparison(
                img_t, masked_t, completed,
                os.path.join(bucket_out, f"{idx:04d}_comparison.png"),
            )

            print(
                f"    [{bucket_name}] {idx+1:2d}/{len(entries)}  "
                f"PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  "
                f"LPIPS={metrics['lpips']:.4f}",
                end="\r",
            )
        print()

    # Summary table
    print("\n" + "="*70)
    print("  EXPERIMENT A BASELINE: DeepFill-v2 (no RL)")
    print("="*70)
    print(f"  {'Bucket':12s}  {'N':>4}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'LPIPS':>8}")
    print("  " + "-"*62)

    overall_psnr, overall_ssim, overall_lpips = [], [], []
    summary: Dict = {}

    for bucket in ["small", "medium", "large", "xlarge"]:
        results = all_results.get(bucket, [])
        if not results:
            continue
        avg_psnr  = np.mean([r["psnr"]  for r in results])
        avg_ssim  = np.mean([r["ssim"]  for r in results])
        avg_lpips = np.nanmean([r["lpips"] for r in results])
        print(
            f"  {bucket:12s}  {len(results):>4}  "
            f"{avg_psnr:>10.3f}  {avg_ssim:>8.4f}  {avg_lpips:>8.4f}"
        )
        overall_psnr.extend([r["psnr"]  for r in results])
        overall_ssim.extend([r["ssim"]  for r in results])
        overall_lpips.extend([r["lpips"] for r in results if not np.isnan(r["lpips"])])
        summary[bucket] = {
            "psnr": float(avg_psnr), "ssim": float(avg_ssim), "lpips": float(avg_lpips),
            "n": len(results),
        }

    print("  " + "-"*62)
    print(
        f"  {'OVERALL':12s}  {len(overall_psnr):>4}  "
        f"{np.mean(overall_psnr):>10.3f}  {np.mean(overall_ssim):>8.4f}  "
        f"{np.nanmean(overall_lpips):>8.4f}"
    )
    print("="*70)

    summary["overall"] = {
        "psnr": float(np.mean(overall_psnr)),
        "ssim": float(np.mean(overall_ssim)),
        "lpips": float(np.nanmean(overall_lpips)),
        "n": len(overall_psnr),
    }

    results_path = os.path.join(output_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DeepFill-v2 baseline (Experiment A)")
    parser.add_argument("--backbone",    type=str, default="checkpoints/pretrained/deepfillv2_places2.pth")
    parser.add_argument("--eval-dir",   type=str, default="data/masks/fixed_eval_masks")
    parser.add_argument("--output-dir", type=str, default="results/baseline_A")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-lpips",   action="store_true")
    args = parser.parse_args()

    evaluate_baseline(
        backbone_path = args.backbone,
        eval_dir      = args.eval_dir,
        output_dir    = args.output_dir,
        image_size    = args.image_size,
        device_str    = args.device,
        compute_lpips = not args.no_lpips,
    )


if __name__ == "__main__":
    main()
