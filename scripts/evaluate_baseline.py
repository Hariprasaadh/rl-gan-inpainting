"""Evaluate DeepFill-v2 backbone (no adapter, neutral pass) on 100 fixed eval images.

Experiment A: frozen backbone, no RL. Produces PSNR/SSIM/LPIPS baseline.
"""

import os
import sys
import argparse
import json

# Ensure project root is on sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.data.dataset import InpaintingDataset
from src.models.rl_inpainting_model import RLInpaintingModel
from src.evaluation.metrics import InpaintingMetricEvaluator
from src.utils.logger import setup_logger, close_logger


def evaluate_baseline(
    backbone_checkpoint: str = "checkpoints/pretrained/deepfillv2_places2.pth",
    data_dir: str = "data/raw/sample_images",
    mask_dir: str = None,
    image_size: int = 256,
    num_samples: int = 100,
    device: str = "cpu",
    output_dir: str = "results",
    compute_lpips: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("evaluate_baseline", log_file=os.path.join(output_dir, "baseline_eval.log"))
    logger.info(f"Starting Experiment A baseline eval ({num_samples} samples) device={device}")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Dataset: use fixed eval masks if exists, else synthetic masks
    eval_mask_dir = mask_dir or "data/masks/fixed_eval_masks"
    if eval_mask_dir and os.path.exists(eval_mask_dir):
        # Check if subdirs exist (buckets)
        has_masks = any(os.path.exists(os.path.join(eval_mask_dir, sub)) for sub in os.listdir(eval_mask_dir))
        if not has_masks:
            eval_mask_dir = None
    else:
        eval_mask_dir = None

    if eval_mask_dir:
        logger.info(f"Using fixed eval masks from {eval_mask_dir}")
    else:
        logger.info("No fixed eval masks found, using synthetic masks with fixed seed")

    dataset = InpaintingDataset(
        image_dir=data_dir if data_dir and os.path.exists(data_dir) else None,
        mask_dir=eval_mask_dir,
        image_size=image_size,
        is_train=False,
        synthetic_size=num_samples,
        fixed_seed=42,
    )

    model = RLInpaintingModel(cnum=48, strategy_dim=64, latent_dim=256).to(dev).eval()
    if backbone_checkpoint and os.path.exists(backbone_checkpoint):
        logger.info(f"Loading backbone from {backbone_checkpoint}")
        model.load_pretrained_backbone(backbone_checkpoint)
    else:
        logger.warning(f"Backbone checkpoint not found at {backbone_checkpoint}, using init weights")

    evaluator = InpaintingMetricEvaluator(device=dev, compute_lpips=compute_lpips)

    n = min(len(dataset), num_samples)
    for i in range(n):
        item = dataset[i]
        image = item["image"].unsqueeze(0).to(dev)
        mask = item["mask"].unsqueeze(0).to(dev)
        masked_image = item["masked_image"].unsqueeze(0).to(dev)

        with torch.no_grad():
            # Neutral: backbone only, no adapter (strategy=None)
            coarse = model.coarse_forward(masked_image, mask)
            coarse_comp = masked_image + coarse * mask
            # Use backbone refine directly without adapter (neutral)
            refined_backbone = model.backbone.refine_forward(coarse_comp, mask)
            completed = masked_image + refined_backbone * mask
            evaluator.update(completed, image, mask)

        if (i + 1) % 20 == 0:
            logger.info(f"Evaluated {i+1}/{n}")

    summary = evaluator.summary()
    logger.info(f"Experiment A Summary: {json.dumps(summary, indent=2)}")

    # Save json and md
    with open(os.path.join(output_dir, "baseline_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "baseline_report.md"), "w") as f:
        f.write("# Experiment A: Frozen DeepFill-v2 Backbone Baseline (No RL)\n\n")
        f.write(f"Samples: {n}\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v['mean']:.4f} ± {v['std']:.4f} (n={v['count']})\n")

    logger.info(f"Results saved to {output_dir}")
    close_logger(logger)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline (Experiment A)")
    parser.add_argument("--backbone-checkpoint", type=str, default="checkpoints/pretrained/deepfillv2_places2.pth")
    parser.add_argument("--data-dir", type=str, default="data/raw/sample_images")
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--compute-lpips", action="store_true")
    args = parser.parse_args()
    evaluate_baseline(
        backbone_checkpoint=args.backbone_checkpoint,
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        image_size=args.image_size,
        num_samples=args.samples,
        device=args.device,
        output_dir=args.output_dir,
        compute_lpips=args.compute_lpips,
    )
