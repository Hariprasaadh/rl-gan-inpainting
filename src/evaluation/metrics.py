from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from src.rl.reward import get_lpips_model


class InpaintingMetricEvaluator:
    """Computes and tracks evaluation metrics (PSNR, SSIM, LPIPS, MSE) across a test set."""

    def __init__(self, device: torch.device = torch.device("cpu"), compute_lpips: bool = True):
        self.device = device
        self.compute_lpips = compute_lpips
        self.lpips_net = get_lpips_model(device) if compute_lpips else None

        self.records: Dict[str, List[float]] = {
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "l1": [],
            "l1_hole": [],
        }

    def reset(self) -> None:
        """Reset accumulated metric lists."""
        for k in self.records:
            self.records[k].clear()

    def update(
        self,
        predicted: torch.Tensor,
        ground_truth: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute metrics for a single prediction vs ground truth pair.

        Args:
            predicted: (1, 3, H, W) or (3, H, W) in [-1, 1]
            ground_truth: (1, 3, H, W) or (3, H, W) in [-1, 1]
            mask: (1, 1, H, W) or (1, H, W) in {0, 1}

        Returns:
            Dict of metric values for this item.
        """
        if predicted.dim() == 3:
            predicted = predicted.unsqueeze(0)
        if ground_truth.dim() == 3:
            ground_truth = ground_truth.unsqueeze(0)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(0)

        # Convert to numpy (H, W, 3) in [0, 1]
        pred_np = ((predicted.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)
        gt_np = ((ground_truth.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)

        psnr_val = float(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
        ssim_val = float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0))

        # L1 total
        l1_val = float(torch.mean(torch.abs(predicted - ground_truth)).item())

        # L1 in hole
        l1_hole_val = l1_val
        if mask is not None:
            mask_exp = mask.expand_as(predicted)
            hole_sum = torch.sum(mask_exp)
            if hole_sum > 0:
                l1_hole_val = float((torch.sum(torch.abs(predicted - ground_truth) * mask_exp) / hole_sum).item())

        # LPIPS
        lpips_val = 0.0
        if self.compute_lpips and self.lpips_net is not None:
            with torch.no_grad():
                d = self.lpips_net(predicted.to(self.device), ground_truth.to(self.device))
                lpips_val = float(d.mean().item())
        else:
            lpips_val = float(F.mse_loss(predicted, ground_truth).item())

        item_metrics = {
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "l1": l1_val,
            "l1_hole": l1_hole_val,
        }

        for k, v in item_metrics.items():
            self.records[k].append(v)

        return item_metrics

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return mean and standard deviation for each metric."""
        out = {}
        for k, v in self.records.items():
            if len(v) > 0:
                out[k] = {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "count": len(v),
                }
            else:
                out[k] = {"mean": 0.0, "std": 0.0, "count": 0}
        return out
