from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F


def compute_border_discontinuity(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Compute edge gradient discontinuity along mask boundary as a quality proxy.

    Args:
        image: (3, H, W) or (1, 3, H, W) in [-1, 1]
        mask: (1, H, W) or (1, 1, H, W) in {0, 1}

    Returns:
        Scalar float proxy in [0.0, 1.0].
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)

    # Dilate mask slightly to find boundary transition zone
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=mask.device)
    dilated = F.conv2d(mask.float(), kernel, padding=1)
    boundary = ((dilated > 0) & (mask == 0)).float()

    if boundary.sum() == 0:
        return 0.0

    # Sobel filters for edge magnitude
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    gray = image.mean(dim=1, keepdim=True)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

    # Mean gradient along boundary
    boundary_grad = (grad_mag * boundary).sum() / (boundary.sum() + 1e-6)
    norm_val = float(torch.clamp(boundary_grad / 2.0, 0.0, 1.0).item())
    return norm_val


class StateBuilder:
    """Constructs state observation vectors for the RL agent."""

    def __init__(
        self,
        latent_dim: int = 256,
        include_mask_stats: bool = True,
        include_quality_proxy: bool = True,
    ):
        self.latent_dim = latent_dim
        self.include_mask_stats = include_mask_stats
        self.include_quality_proxy = include_quality_proxy

        stats_dim = 4 if include_mask_stats else 0
        proxy_dim = 1 if include_quality_proxy else 0
        self.state_dim = latent_dim + stats_dim + proxy_dim

    def build_state(
        self,
        latent_embedding: torch.Tensor,
        mask_stats: Dict[str, float],
        coarse_composite: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Build a flat numpy observation vector.

        Args:
            latent_embedding: (1, latent_dim) or (latent_dim,) tensor from generator encoder
            mask_stats: dict containing 'missing_ratio', 'bbox_ratio', 'num_regions', 'center_dist'
            coarse_composite: optional composite tensor for boundary proxy
            mask: optional mask tensor for boundary proxy

        Returns:
            np.ndarray of shape (state_dim,), dtype float32.
        """
        if isinstance(latent_embedding, torch.Tensor):
            latent_vec = latent_embedding.detach().cpu().flatten().numpy()
        else:
            latent_vec = np.asarray(latent_embedding, dtype=np.float32).flatten()

        parts = [latent_vec]

        if self.include_mask_stats:
            stats_vec = np.array([
                float(mask_stats.get("missing_ratio", 0.0)),
                float(mask_stats.get("bbox_ratio", 0.0)),
                float(mask_stats.get("num_regions", 0.0)),
                float(mask_stats.get("center_dist", 0.0)),
            ], dtype=np.float32)
            parts.append(stats_vec)

        if self.include_quality_proxy:
            if coarse_composite is not None and mask is not None:
                proxy_val = compute_border_discontinuity(coarse_composite, mask)
            else:
                proxy_val = 0.0
            parts.append(np.array([proxy_val], dtype=np.float32))

        state = np.concatenate(parts, axis=0).astype(np.float32)
        return state
