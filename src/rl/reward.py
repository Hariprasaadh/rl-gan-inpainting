from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from src.rl.actions import get_action_cost


# Global LPIPS model cache for reward calculation
_LPIPS_VGG: Optional[lpips.LPIPS] = None


def get_lpips_model(device: torch.device = torch.device("cpu")) -> Optional[lpips.LPIPS]:
    """Get cached LPIPS model."""
    global _LPIPS_VGG
    if _LPIPS_VGG is None:
        try:
            _LPIPS_VGG = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
            for p in _LPIPS_VGG.parameters():
                p.requires_grad = False
        except Exception:
            _LPIPS_VGG = None
    return _LPIPS_VGG


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute PSNR between two images in [0, 1] or [0, 255]."""
    return float(peak_signal_noise_ratio(gt, pred, data_range=1.0))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute SSIM between two images (H, W, C) in [0, 1]."""
    # Multichannel SSIM with channel_axis=2
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))


def normalize_psnr_delta(d_psnr: float) -> float:
    """Normalize PSNR change (e.g., +2 dB -> ~0.2) and clamp to stable range."""
    return float(np.clip(d_psnr / 10.0, -2.0, 2.0))


def compute_image_metrics(
    image_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    compute_lpips: bool = True,
) -> Dict[str, float]:
    """Compute quality metrics (PSNR, SSIM, LPIPS) between prediction and ground truth.

    Args:
        image_tensor: (1, 3, H, W) or (3, H, W) in [-1, 1]
        gt_tensor: (1, 3, H, W) or (3, H, W) in [-1, 1]
        compute_lpips: Whether to compute LPIPS distance
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if gt_tensor.dim() == 3:
        gt_tensor = gt_tensor.unsqueeze(0)

    # Convert to [0, 1] numpy (H, W, C) for skimage metrics
    pred_np = ((image_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)
    gt_np = ((gt_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0.0, 1.0)

    psnr_val = compute_psnr(pred_np, gt_np)
    ssim_val = compute_ssim(pred_np, gt_np)

    lpips_val = 0.0
    if compute_lpips:
        lpips_net = get_lpips_model(image_tensor.device)
        if lpips_net is not None:
            with torch.no_grad():
                dist = lpips_net(image_tensor, gt_tensor)
                lpips_val = float(dist.mean().item())
        else:
            # Fallback normalized MSE as perceptual proxy
            lpips_val = float(F.mse_loss(image_tensor, gt_tensor).item())

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
    }


def compute_reward(
    prev_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    disc_score_delta: float,
    action: int,
    weights: Tuple[float, float, float, float, float] = (1.0, 1.0, 0.5, 0.5, 0.3),
) -> Tuple[float, Dict[str, float]]:
    """Compute improvement-based RL reward R_t with action cost penalty.

    R_t = alpha * d_psnr + beta * d_ssim - gamma * d_lpips + delta * disc_delta - lambda * action_cost

    Args:
        prev_metrics: dict of coarse/previous-step metrics {'psnr', 'ssim', 'lpips'}
        new_metrics: dict of refined/new-step metrics {'psnr', 'ssim', 'lpips'}
        disc_score_delta: change in discriminator confidence D_refined - D_coarse
        action: chosen action integer
        weights: (alpha, beta, gamma, delta, lambda)

    Returns:
        total_reward: scalar float
        breakdown: dict of individual components for TensorBoard logging
    """
    alpha, beta, gamma, delta, lam = weights

    d_psnr_raw = new_metrics["psnr"] - prev_metrics["psnr"]
    d_psnr = normalize_psnr_delta(d_psnr_raw)
    d_ssim = float(new_metrics["ssim"] - prev_metrics["ssim"])
    d_lpips = float(new_metrics["lpips"] - prev_metrics["lpips"])  # lower is better
    cost = get_action_cost(action)

    reward = (
        alpha * d_psnr
        + beta * d_ssim
        - gamma * d_lpips
        + delta * disc_score_delta
        - lam * cost
    )

    # Clip reward for numerical stability during PPO training
    reward = float(np.clip(reward, -10.0, 10.0))

    breakdown = {
        "reward": reward,
        "d_psnr": d_psnr,
        "d_psnr_raw": d_psnr_raw,
        "d_ssim": d_ssim,
        "d_lpips": d_lpips,
        "disc_delta": disc_score_delta,
        "action_cost": cost,
        "new_psnr": new_metrics["psnr"],
        "new_ssim": new_metrics["ssim"],
        "new_lpips": new_metrics["lpips"],
    }

    return reward, breakdown
