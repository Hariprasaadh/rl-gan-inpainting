from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from src.rl.actions import ACTION_NAMES, ACTION_STOP, get_action_cost
from src.rl.state import StateBuilder
from src.rl.reward import compute_image_metrics, compute_reward
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.data.dataset import InpaintingDataset


class InpaintingMultiStepEnv(gym.Env):
    """Gymnasium Environment for Version 2 Multi-Step Iterative Refinement.

    Episode Length: Variable (1 to max_steps), terminated early if action == STOP.
    Action Space: Discrete(5) -> [0: Global, 1: Local, 2: Boundary, 3: Texture, 4: Stop].
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        dataset: InpaintingDataset,
        generator: InpaintingGenerator,
        discriminator: SNPatchGANDiscriminator,
        device: str = "cpu",
        max_steps: int = 4,
        reward_weights: Tuple[float, float, float, float, float] = (1.0, 1.0, 0.5, 0.5, 0.3),
        compute_lpips: bool = False,
        include_mask_stats: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.generator = generator.eval()
        self.discriminator = discriminator.eval()
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.reward_weights = reward_weights
        self.compute_lpips = compute_lpips

        # Freeze generator and discriminator parameters
        for p in self.generator.parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = False

        self.state_builder = StateBuilder(
            latent_dim=generator.encoder.latent_dim,
            include_mask_stats=include_mask_stats,
            include_quality_proxy=True,
        )

        # 5 actions (including STOP)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.state_builder.state_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_image: Optional[torch.Tensor] = None
        self.current_mask: Optional[torch.Tensor] = None
        self.current_masked: Optional[torch.Tensor] = None
        self.current_composite: Optional[torch.Tensor] = None
        self.current_stats: Dict[str, float] = {}
        self.current_metrics: Dict[str, float] = {}
        self.current_disc_score: float = 0.0

    def _get_obs(self) -> np.ndarray:
        with torch.no_grad():
            latent_emb = self.generator.extract_state_embedding(self.current_masked, self.current_mask)
        return self.state_builder.build_state(
            latent_embedding=latent_emb,
            mask_stats=self.current_stats,
            coarse_composite=self.current_composite,
            mask=self.current_mask,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        img, mask, masked_img, meta = self.dataset.sample()
        self.current_image = img.unsqueeze(0).to(self.device)
        self.current_mask = mask.unsqueeze(0).to(self.device)
        self.current_masked = masked_img.unsqueeze(0).to(self.device)
        self.current_stats = meta.get("stats", {})

        with torch.no_grad():
            coarse = self.generator.coarse_forward(self.current_masked, self.current_mask)
            self.current_composite = self.current_masked + coarse * self.current_mask

            self.current_metrics = compute_image_metrics(
                self.current_composite,
                self.current_image,
                compute_lpips=self.compute_lpips,
            )
            disc_conf = self.discriminator.get_confidence(self.current_composite, self.current_mask)
            self.current_disc_score = float(disc_conf.item())

        obs = self._get_obs()
        info = {
            "initial_psnr": self.current_metrics["psnr"],
            "initial_ssim": self.current_metrics["ssim"],
            "initial_lpips": self.current_metrics["lpips"],
            "mask_stats": self.current_stats,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_idx = int(action)
        self.current_step += 1

        # Check for STOP action
        if action_idx == ACTION_STOP:
            # STOP ends episode early with 0 cost and 0 additional improvement
            reward = 0.0
            terminated = True
            truncated = False
            obs = self._get_obs()
            info = {
                "action": ACTION_STOP,
                "action_name": "Stop",
                "reward": 0.0,
                "action_cost": 0.0,
                "step": self.current_step,
                "final_psnr": self.current_metrics["psnr"],
                "final_ssim": self.current_metrics["ssim"],
            }
            return obs, reward, terminated, truncated, info

        # Apply refinement action
        with torch.no_grad():
            refined = self.generator.refine(
                self.current_composite,
                self.current_mask,
                strategy=action_idx,
            )
            next_composite = self.current_masked + refined * self.current_mask

            next_metrics = compute_image_metrics(
                next_composite,
                self.current_image,
                compute_lpips=self.compute_lpips,
            )
            next_disc_conf = self.discriminator.get_confidence(next_composite, self.current_mask)
            next_disc_score = float(next_disc_conf.item())
            disc_delta = next_disc_score - self.current_disc_score

        reward, breakdown = compute_reward(
            prev_metrics=self.current_metrics,
            new_metrics=next_metrics,
            disc_score_delta=disc_delta,
            action=action_idx,
            weights=self.reward_weights,
        )

        # Update current state for multi-step trajectory
        self.current_composite = next_composite
        self.current_metrics = next_metrics
        self.current_disc_score = next_disc_score

        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs()

        info = {
            **breakdown,
            "action": action_idx,
            "action_name": ACTION_NAMES.get(action_idx, "Unknown"),
            "step": self.current_step,
            "current_psnr": next_metrics["psnr"],
            "current_ssim": next_metrics["ssim"],
        }

        return obs, reward, terminated, truncated, info
