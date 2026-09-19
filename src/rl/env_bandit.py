from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn

from src.rl.actions import ACTION_NAMES, get_action_cost
from src.rl.state import StateBuilder
from src.rl.reward import compute_image_metrics, compute_reward
from src.models.discriminator import SNPatchGANDiscriminator
from src.data.dataset import InpaintingDataset

# Prefer new backbone; fallback to legacy generator for backward compat
try:
    from src.models.rl_inpainting_model import RLInpaintingModel as InpaintingGenerator
except ImportError:
    from src.models.generator import InpaintingGenerator  # type: ignore


class InpaintingBanditEnv(gym.Env):
    """Gymnasium Environment for Version 1 Contextual Bandit RL Controller.

    Episode Length: Exactly 1 step.
    Action Space: Discrete(4) -> [0: Global, 1: Local, 2: Boundary, 3: Texture].
    Reward: Improvement-based (dPSNR, dSSIM, -dLPIPS, dDisc - action cost).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        dataset: InpaintingDataset,
        generator: InpaintingGenerator,
        discriminator: SNPatchGANDiscriminator,
        device: str = "cpu",
        reward_weights: Tuple[float, float, float, float, float] = (1.0, 1.0, 0.5, 0.5, 0.3),
        compute_lpips: bool = False,
        include_mask_stats: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.generator = generator.eval()
        self.discriminator = discriminator.eval()
        self.device = torch.device(device)
        self.reward_weights = reward_weights
        self.compute_lpips = compute_lpips

        # Freeze model parameters in bandit env
        for p in self.generator.parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = False

        self.state_builder = StateBuilder(
            latent_dim=generator.encoder.latent_dim,
            include_mask_stats=include_mask_stats,
            include_quality_proxy=True,
        )

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.state_builder.state_dim,),
            dtype=np.float32,
        )

        # Episode state cache
        self.current_image: Optional[torch.Tensor] = None
        self.current_mask: Optional[torch.Tensor] = None
        self.current_masked: Optional[torch.Tensor] = None
        self.current_coarse: Optional[torch.Tensor] = None
        self.current_coarse_comp: Optional[torch.Tensor] = None
        self.current_stats: Dict[str, float] = {}
        self.coarse_metrics: Dict[str, float] = {}
        self.coarse_disc_score: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Sample a fresh item from dataset
        img, mask, masked_img, meta = self.dataset.sample()
        self.current_image = img.unsqueeze(0).to(self.device)
        self.current_mask = mask.unsqueeze(0).to(self.device)
        self.current_masked = masked_img.unsqueeze(0).to(self.device)
        self.current_stats = meta.get("stats", {})

        with torch.no_grad():
            self.current_coarse = self.generator.coarse_forward(self.current_masked, self.current_mask)
            self.current_coarse_comp = self.current_masked + self.current_coarse * self.current_mask

            # Compute coarse baseline metrics and discriminator score
            self.coarse_metrics = compute_image_metrics(
                self.current_coarse_comp,
                self.current_image,
                compute_lpips=self.compute_lpips,
            )
            disc_conf = self.discriminator.get_confidence(self.current_coarse_comp, self.current_mask)
            self.coarse_disc_score = float(disc_conf.item())

            # Extract state observation
            latent_emb = self.generator.extract_state_embedding(self.current_masked, self.current_mask)

        obs = self.state_builder.build_state(
            latent_embedding=latent_emb,
            mask_stats=self.current_stats,
            coarse_composite=self.current_coarse_comp,
            mask=self.current_mask,
        )

        info = {
            "coarse_psnr": self.coarse_metrics["psnr"],
            "coarse_ssim": self.coarse_metrics["ssim"],
            "coarse_lpips": self.coarse_metrics["lpips"],
            "coarse_disc_score": self.coarse_disc_score,
            "mask_stats": self.current_stats,
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_idx = int(action)

        with torch.no_grad():
            # Run refinement conditioned on chosen action strategy
            refined_img = self.generator.refine(
                self.current_coarse_comp,
                self.current_mask,
                strategy=action_idx,
            )
            completed_img = self.current_masked + refined_img * self.current_mask

            # Compute refined metrics and discriminator confidence
            refined_metrics = compute_image_metrics(
                completed_img,
                self.current_image,
                compute_lpips=self.compute_lpips,
            )
            refined_disc_conf = self.discriminator.get_confidence(completed_img, self.current_mask)
            refined_disc_score = float(refined_disc_conf.item())
            disc_delta = refined_disc_score - self.coarse_disc_score

        # Compute improvement-based reward
        reward, breakdown = compute_reward(
            prev_metrics=self.coarse_metrics,
            new_metrics=refined_metrics,
            disc_score_delta=disc_delta,
            action=action_idx,
            weights=self.reward_weights,
        )

        # Single-step episode ends immediately
        terminated = True
        truncated = False

        # Reset next observation
        next_obs, reset_info = self.reset()

        info = {
            **breakdown,
            "action": action_idx,
            "action_name": ACTION_NAMES.get(action_idx, "Unknown"),
            "completed_psnr": refined_metrics["psnr"],
            "completed_ssim": refined_metrics["ssim"],
            "completed_lpips": refined_metrics["lpips"],
            "completed_disc_score": refined_disc_score,
        }

        return next_obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.current_image is None:
            return None
        with torch.no_grad():
            img_np = ((self.current_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        return img_np
