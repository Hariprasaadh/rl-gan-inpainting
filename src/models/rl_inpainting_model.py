"""Unified wrapper: frozen DeepFill-v2 backbone + trainable adapters + state embedding.

This replaces InpaintingGenerator for the RL environment.
Interface preserved: coarse_forward, refine, extract_state_embedding, forward
so env_bandit.py needs minimal change.
"""

from typing import Optional, Union, Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.deepfillv2 import DeepFillV2Generator, load_deepfillv2_checkpoint
from src.models.encoder import DeepFillEncoder
from src.models.strategy_adapters import GlobalAdapter, LocalAdapter, BoundaryAdapter, TextureAdapter


class RLInpaintingModel(nn.Module):
    """Glues DeepFill backbone + DeepFillEncoder + 4 strategy adapters."""

    def __init__(
        self,
        cnum: int = 48,
        cnum_in: int = 5,
        strategy_dim: int = 64,
        latent_dim: int = 256,
        num_strategies: int = 4,
    ):
        super().__init__()
        self.cnum = cnum
        self.cnum_in = cnum_in
        self.strategy_dim = strategy_dim
        self.latent_dim = latent_dim
        self.num_strategies = num_strategies

        self.backbone = DeepFillV2Generator(cnum=cnum, cnum_in=cnum_in)
        self.encoder_head = DeepFillEncoder(cnum=cnum, latent_dim=latent_dim)

        # Keep encoder attribute for backward compatibility (StateBuilder uses generator.encoder.latent_dim)
        self.encoder = self.encoder_head
        # Ensure .latent_dim accessible via both names
        self.encoder.latent_dim = latent_dim

        self.adapters = nn.ModuleList([
            GlobalAdapter(strategy_dim=strategy_dim),
            LocalAdapter(strategy_dim=strategy_dim),
            BoundaryAdapter(strategy_dim=strategy_dim),
            TextureAdapter(strategy_dim=strategy_dim),
        ])

        self.strategy_embeddings = nn.Embedding(num_strategies, strategy_dim)
        self.register_buffer("neutral_embedding", torch.zeros(1, strategy_dim))

        self.base_channels = cnum  # keep compat

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pretrained_backbone(self, path: str, strict: bool = False) -> Dict:
        """Load and freeze DeepFill backbone."""
        info = load_deepfillv2_checkpoint(self.backbone, path, strict=strict, verbose=True)
        # Freeze backbone params
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        return info

    def load_adapters(self, path: str) -> bool:
        """Load adapter weights if checkpoint exists."""
        if not path or not os.path.exists(path):
            print(f"[RLInpaintingModel] Adapters checkpoint not found at {path}, using init weights.")
            return False
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Handle dict wrapping
        if isinstance(ckpt, dict) and "adapters" in ckpt:
            state = ckpt["adapters"]
        elif isinstance(ckpt, dict) and "adapter_state_dict" in ckpt:
            state = ckpt["adapter_state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            # Check if ckpt is a flat state_dict with adapter keys
            state = ckpt
            # If keys contain generator/encoder, filter
            if any("backbone" in k for k in ckpt.keys()):
                # Might be full model checkpoint
                filtered = {}
                for k, v in ckpt.items():
                    if "adapter" in k.lower() or "strategy_embedding" in k:
                        nk = k.replace("module.", "")
                        filtered[nk] = v
                if len(filtered) > 0:
                    state = filtered

        # Try to load adapters ModuleList
        try:
            # If state has keys like "0.conv1.weight" (from ModuleList)
            if any(k.startswith("0.") for k in state.keys()):
                self.adapters.load_state_dict(state, strict=False)
            elif any("adapters." in k for k in state.keys()):
                # Remap adapters.N.xxx -> N.xxx
                remapped = {k.replace("adapters.", ""): v for k, v in state.items() if "adapters." in k}
                self.adapters.load_state_dict(remapped, strict=False)
            else:
                self.adapters.load_state_dict(state, strict=False)

            # Load strategy embeddings if present
            if "strategy_embeddings.weight" in ckpt:
                self.strategy_embeddings.load_state_dict(
                    {"weight": ckpt["strategy_embeddings.weight"]}, strict=False
                )
            elif "strategy_embeddings" in state:
                pass  # handled

            print(f"[RLInpaintingModel] Loaded adapters from {path}")
            return True
        except Exception as e:
            print(f"[RLInpaintingModel] Failed to load adapters: {e}")
            # Try alternative: load full checkpoint non-strict
            try:
                self.load_state_dict(ckpt, strict=False)
                print(f"[RLInpaintingModel] Loaded full checkpoint with strict=False fallback")
                return True
            except Exception as e2:
                print(f"[RLInpaintingModel] Fallback also failed: {e2}")
                return False

    def freeze_adapters(self):
        for p in self.adapters.parameters():
            p.requires_grad = False
        for p in self.strategy_embeddings.parameters():
            p.requires_grad = False

    def unfreeze_adapters(self):
        for p in self.adapters.parameters():
            p.requires_grad = True
        for p in self.strategy_embeddings.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_strategy_embedding(
        self,
        strategy: Optional[Union[int, torch.Tensor]] = None,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        if strategy is None:
            return self.neutral_embedding.expand(batch_size, -1).to(device)
        if isinstance(strategy, int):
            strategy = torch.tensor([strategy] * batch_size, dtype=torch.long, device=device)
        elif isinstance(strategy, torch.Tensor) and strategy.dtype in (torch.int64, torch.int32, torch.long):
            if strategy.dim() == 0:
                strategy = strategy.unsqueeze(0).expand(batch_size)
            strategy = strategy.to(device)
        if isinstance(strategy, torch.Tensor) and strategy.dim() == 1 and strategy.dtype == torch.long:
            return self.strategy_embeddings(strategy)
        elif isinstance(strategy, torch.Tensor) and strategy.shape[-1] == self.strategy_dim:
            return strategy.to(device)
        elif isinstance(strategy, torch.Tensor) and strategy.shape[-1] == self.num_strategies:
            return torch.matmul(strategy.to(device).float(), self.strategy_embeddings.weight)
        else:
            return self.neutral_embedding.expand(batch_size, -1).to(device)

    def extract_state_embedding(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Extract 256-d state vector from backbone bottleneck (cost-free)."""
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.encoder_head.parameters())):
            bottleneck = self.backbone.get_bottleneck_feature(masked_image, mask)
            emb = self.encoder_head(bottleneck)
        return emb

    def coarse_forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.backbone.coarse_forward(masked_image, mask)

    def refine(
        self,
        coarse_composite: torch.Tensor,
        mask: torch.Tensor,
        strategy: Optional[Union[int, torch.Tensor]] = None,
        original_masked_image: Optional[torch.Tensor] = None,
        strategy_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run refinement: backbone Stage2 + adapter[strategy].

        Accepts either strategy index or direct embedding.
        coarse_composite: (B,3,H,W) composite of masked_image + coarse*mask (or just coarse_composite)
        """
        b = coarse_composite.shape[0]
        device = coarse_composite.device

        # Backbone refine (frozen)
        with torch.set_grad_enabled(False):
            refined_backbone = self.backbone.refine_forward(coarse_composite, mask)

        if strategy is None and strategy_embedding is None:
            # Neutral: no adapter, return backbone output directly
            return refined_backbone

        if strategy_embedding is None:
            strat_emb = self.get_strategy_embedding(strategy, batch_size=b, device=device)
            strat_idx = int(strategy) if isinstance(strategy, int) else int(strategy[0]) if isinstance(strategy, torch.Tensor) and strategy.dim() == 1 else 0
            if isinstance(strategy, torch.Tensor) and strategy.numel() == 1:
                strat_idx = int(strategy.item())
            # Clamp idx
            if isinstance(strategy, int):
                strat_idx = strategy
            elif isinstance(strategy, torch.Tensor) and strategy.dtype == torch.long and strategy.dim() == 1:
                strat_idx = int(strategy[0].item())
            else:
                strat_idx = 0
                # If strategy is already embedding, strat_idx not used strictly
                if isinstance(strategy, int):
                    strat_idx = strategy
        else:
            strat_emb = strategy_embedding
            strat_idx = 0

        # Resolve idx safely
        if isinstance(strategy, int):
            idx = max(0, min(strategy, 3))
        elif isinstance(strategy, torch.Tensor) and strategy.dtype == torch.long:
            try:
                idx = int(strategy.flatten()[0].item())
                idx = max(0, min(idx, 3))
            except Exception:
                idx = 0
        else:
            idx = 0

        adapter = self.adapters[idx]
        # Adapter expects (B,3,H,W) image and mask
        adapted = adapter(refined_backbone, mask, strat_emb)
        return adapted

    def forward(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        strategy: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Complete forward: coarse -> backbone refine -> adapter."""
        coarse = self.backbone.coarse_forward(masked_image, mask)
        coarse_composite = masked_image + coarse * mask
        refined = self.refine(coarse_composite, mask, strategy=strategy)
        completed = masked_image + refined * mask
        state_embedding = self.extract_state_embedding(masked_image, mask)
        return {
            "coarse": coarse,
            "coarse_composite": coarse_composite,
            "refined": refined,
            "completed": completed,
            "state_embedding": state_embedding,
        }
