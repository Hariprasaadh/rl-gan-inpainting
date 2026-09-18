"""RLInpaintingModel — unified wrapper for the DeepFill backbone + adapters.

This class replaces InpaintingGenerator as the single model object used by
env_bandit.py, train_rl_bandit.py, evaluate.py, and main.py.

Public interface (identical to InpaintingGenerator where it overlaps):
    model.coarse_forward(masked_img, mask)         -> (B, 3, H, W)
    model.extract_state_embedding(masked_img, mask)-> (B, latent_dim)
    model.refine(coarse_comp, mask, strategy=int)  -> (B, 3, H, W)
    model.forward(masked_img, mask, strategy=None) -> dict

Trainable parameters:
    - encoder_head  (DeepFillEncoder, ~150k params)
    - adapters[0:4] (~50k params each)
    - strategy_embeddings (~256 params)

Frozen parameters:
    - backbone (DeepFillV2Generator, ~4.5M params)

Usage
-----
model = RLInpaintingModel(cnum=48, strategy_dim=64, latent_dim=256)
model.load_pretrained_backbone("checkpoints/pretrained/deepfillv2_places2.pth")
model.load_adapters("checkpoints/adapters/adapters_final.pt")   # after pretraining
model.eval()
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn

from src.models.deepfillv2 import DeepFillV2Generator, load_deepfillv2_checkpoint
from src.models.encoder_deepfill import DeepFillEncoder
from src.models.strategy_adapters import build_adapters, ADAPTER_NAMES


class RLInpaintingModel(nn.Module):
    """Frozen DeepFill-v2 backbone + 4 trainable strategy adapters.

    Args:
        cnum:         DeepFill base channel count (48 for nipponjo weights).
        strategy_dim: Dimension of the per-action strategy embedding vector.
        latent_dim:   Dimension of the RL state observation vector (256).
        num_strategies: Number of discrete actions (4).
    """

    def __init__(
        self,
        cnum: int = 48,
        strategy_dim: int = 64,
        latent_dim: int = 256,
        num_strategies: int = 4,
    ):
        super().__init__()
        self.cnum          = cnum
        self.strategy_dim  = strategy_dim
        self.latent_dim    = latent_dim
        self.num_strategies = num_strategies

        # --- Frozen backbone ---
        self.backbone = DeepFillV2Generator(cnum=cnum)

        # --- Trainable components ---
        self.encoder_head = DeepFillEncoder(cnum=cnum, latent_dim=latent_dim)
        self.adapters     = build_adapters(strategy_dim=strategy_dim)
        self.strategy_embeddings = nn.Embedding(num_strategies, strategy_dim)

        # Neutral strategy embedding (for no-strategy baseline)
        self.register_buffer("neutral_embedding", torch.zeros(1, strategy_dim))

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def load_pretrained_backbone(self, path: str, strict: bool = True) -> None:
        """Load nipponjo DeepFill-v2 weights and freeze the backbone.

        Args:
            path:   Path to states_pt_places2.pth (or states_tf_places2.pth).
            strict: Raise if key mismatch found (recommended True).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DeepFill-v2 checkpoint not found at: {path}\n"
                "Download states_pt_places2.pth from:\n"
                "  https://github.com/nipponjo/deepfillv2-pytorch (Google Drive link in README)\n"
                "and place it at: checkpoints/pretrained/deepfillv2_places2.pth"
            )
        load_deepfillv2_checkpoint(self.backbone, path, strict=strict)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters in-place."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[RLInpaintingModel] Backbone frozen.")

    def load_adapters(self, path: str) -> None:
        """Load pretrained adapter weights.

        The checkpoint should be saved with save_adapters().
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Adapter checkpoint not found at: {path}\n"
                "Run: python main.py pretrain-adapters  first."
            )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.adapters.load_state_dict(ckpt["adapters"])
        self.encoder_head.load_state_dict(ckpt["encoder_head"])
        if "strategy_embeddings" in ckpt:
            self.strategy_embeddings.load_state_dict(ckpt["strategy_embeddings"])
        print(f"[RLInpaintingModel] Adapters + encoder_head loaded from: {path}")

    def save_adapters(self, path: str) -> None:
        """Save trainable components (adapters + encoder head) to a checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "adapters":            self.adapters.state_dict(),
                "encoder_head":        self.encoder_head.state_dict(),
                "strategy_embeddings": self.strategy_embeddings.state_dict(),
            },
            path,
        )
        print(f"[RLInpaintingModel] Saved adapters to: {path}")

    def trainable_parameters(self):
        """Return only the trainable (non-backbone) parameters."""
        return (
            list(self.encoder_head.parameters())
            + list(self.adapters.parameters())
            + list(self.strategy_embeddings.parameters())
        )

    # ------------------------------------------------------------------
    # Strategy embedding helpers
    # ------------------------------------------------------------------

    def get_strategy_embedding(
        self,
        strategy: Optional[Union[int, torch.Tensor]],
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Decode action index into strategy embedding tensor (B, strategy_dim)."""
        if device is None:
            device = next(self.strategy_embeddings.parameters()).device

        if strategy is None:
            return self.neutral_embedding.expand(batch_size, -1).to(device)

        if isinstance(strategy, int):
            idx = torch.tensor([strategy] * batch_size, dtype=torch.long, device=device)
            return self.strategy_embeddings(idx)

        if isinstance(strategy, torch.Tensor):
            strategy = strategy.to(device)
            if strategy.dtype in (torch.int32, torch.int64, torch.long):
                if strategy.dim() == 0:
                    strategy = strategy.unsqueeze(0).expand(batch_size)
                return self.strategy_embeddings(strategy)
            elif strategy.shape[-1] == self.strategy_dim:
                return strategy.float()
            elif strategy.shape[-1] == self.num_strategies:
                # One-hot
                return torch.matmul(strategy.float(), self.strategy_embeddings.weight)

        return self.neutral_embedding.expand(batch_size, -1).to(device)

    # ------------------------------------------------------------------
    # Core interface methods (identical signatures to InpaintingGenerator)
    # ------------------------------------------------------------------

    def extract_state_embedding(
        self, masked_image: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract 256-d state vector for the RL controller.

        Reuses bottleneck features from backbone (no extra forward pass).

        Args:
            masked_image: (B, 3, H, W) in [-1, 1], zeros inside hole
            mask:         (B, 1, H, W) in {0, 1}, 1 = hole

        Returns:
            (B, latent_dim) float32 tensor
        """
        with torch.no_grad():
            bottleneck = self.backbone.get_bottleneck_features(masked_image, mask)
        return self.encoder_head(bottleneck)

    def coarse_forward(
        self, masked_image: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Run backbone coarse pass only.

        Args:
            masked_image: (B, 3, H, W) in [-1, 1]
            mask:         (B, 1, H, W)

        Returns:
            coarse_out: (B, 3, H, W) in [-1, 1]
        """
        coarse_out, _ = self.backbone(masked_image, mask)
        return coarse_out

    def refine(
        self,
        coarse_composite: torch.Tensor,
        mask: torch.Tensor,
        strategy: Optional[Union[int, torch.Tensor]] = None,
        original_masked_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run backbone Stage-2 then apply the chosen strategy adapter.

        Args:
            coarse_composite:     (B, 3, H, W) — coarse + original composite
            mask:                 (B, 1, H, W)
            strategy:             int action index (0-3) or tensor
            original_masked_image: unused, kept for API compatibility

        Returns:
            refined: (B, 3, H, W) in [-1, 1]
        """
        b = coarse_composite.shape[0]
        device = coarse_composite.device

        # Backbone Stage-2 refinement (frozen)
        with torch.no_grad():
            ones = torch.ones_like(mask)
            x_in2 = torch.cat([coarse_composite, mask, ones], dim=1)
            refined_backbone = self.backbone.refine_net(x_in2)

        # Strategy adapter (trainable)
        strat_emb = self.get_strategy_embedding(strategy, batch_size=b, device=device)

        if isinstance(strategy, int):
            adapter = self.adapters[strategy]
        elif isinstance(strategy, torch.Tensor) and strategy.numel() == 1:
            adapter = self.adapters[int(strategy.item())]
        else:
            # Fallback: global adapter
            adapter = self.adapters[0]

        return adapter(refined_backbone, mask, strat_emb)

    def forward(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        strategy: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Complete forward pass (coarse + refine + adapter).

        Returns:
            Dict with keys:
              - coarse:           (B, 3, H, W) backbone Stage-1 output
              - coarse_composite: masked_image * (1-mask) + coarse * mask
              - refined:          (B, 3, H, W) adapter output
              - completed:        masked_image * (1-mask) + refined * mask
              - state_embedding:  (B, latent_dim)
        """
        b = masked_image.shape[0]
        device = masked_image.device

        # --- Backbone (frozen) ---
        with torch.no_grad():
            coarse_out, backbone_refined = self.backbone(masked_image, mask)
            bottleneck = self.backbone.get_bottleneck_features(masked_image, mask)

        coarse_composite = masked_image * (1 - mask) + coarse_out * mask

        # --- State embedding (encoder_head is trainable) ---
        state_embedding = self.encoder_head(bottleneck)

        # --- Strategy adapter (trainable) ---
        strat_emb = self.get_strategy_embedding(strategy, batch_size=b, device=device)

        if strategy is None:
            # No adapter: use backbone output directly
            refined = backbone_refined
        elif isinstance(strategy, int):
            refined = self.adapters[strategy](backbone_refined, mask, strat_emb)
        elif isinstance(strategy, torch.Tensor) and strategy.numel() == 1:
            refined = self.adapters[int(strategy.item())](backbone_refined, mask, strat_emb)
        else:
            refined = self.adapters[0](backbone_refined, mask, strat_emb)

        completed = masked_image * (1 - mask) + refined * mask

        return {
            "coarse":           coarse_out,
            "coarse_composite": coarse_composite,
            "refined":          refined,
            "completed":        completed,
            "state_embedding":  state_embedding,
        }

    # ------------------------------------------------------------------
    # Convenience properties for backward compatibility
    # ------------------------------------------------------------------

    @property
    def encoder(self):
        """Compatibility shim: expose encoder_head as .encoder for StateBuilder."""
        return self.encoder_head
