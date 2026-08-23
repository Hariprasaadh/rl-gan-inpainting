from typing import Optional, Union, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import GatedConv2d, InpaintingEncoder


class StrategyModulation(nn.Module):
    """Modulates feature maps according to the RL agent's chosen strategy vector.

    Applies Feature-wise Linear Modulation (FiLM):
        out = (1 + gamma(strategy)) * x + beta(strategy)
    """

    def __init__(self, channels: int, strategy_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(strategy_dim, channels * 2),
            nn.ELU(inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, strategy_embedding: torch.Tensor) -> torch.Tensor:
        # strategy_embedding: (B, strategy_dim)
        params = self.mlp(strategy_embedding)  # (B, channels * 2)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class CoarseNetwork(nn.Module):
    """Coarse inpainting sub-network (DeepFill v2 style).

    Produces initial rough completion from masked input.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.conv1 = GatedConv2d(in_channels, base_channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = GatedConv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = GatedConv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Dilated Bottleneck
        self.dilated1 = GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=2, dilation=2)
        self.dilated2 = GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=4, dilation=4)
        self.dilated3 = GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=8, dilation=8)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv1 = GatedConv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv2 = GatedConv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked_image, mask], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.dilated1(x)
        x = self.dilated2(x)
        x = self.dilated3(x)

        x = self.up1(x)
        x = self.deconv1(x)
        x = self.up2(x)
        x = self.deconv2(x)
        x = self.out_conv(x)
        return torch.tanh(x)


class RefinementNetwork(nn.Module):
    """Refinement sub-network conditioned on RL strategy vector.

    Applies adaptive contextual transformations based on the chosen strategy:
      - Action 0: Global completion
      - Action 1: Local refinement
      - Action 2: Boundary refinement
      - Action 3: Texture refinement
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32, strategy_dim: int = 64):
        super().__init__()
        self.strategy_dim = strategy_dim

        # Encoder
        self.conv1 = GatedConv2d(in_channels, base_channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = GatedConv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = GatedConv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Strategy modulation layers
        self.mod1 = StrategyModulation(base_channels * 4, strategy_dim)
        self.mod2 = StrategyModulation(base_channels * 4, strategy_dim)

        # Dilated Bottleneck
        self.dilated1 = GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=2, dilation=2)
        self.dilated2 = GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=4, dilation=4)
        self.dilated3 = GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=8, dilation=8)
        self.dilated4 = GatedConv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=16, dilation=16)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv1 = GatedConv2d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv2 = GatedConv2d(base_channels * 2, base_channels, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(
        self,
        coarse_composite: torch.Tensor,
        mask: torch.Tensor,
        strategy_embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([coarse_composite, mask], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.mod1(x, strategy_embedding)
        x = self.dilated1(x)
        x = self.dilated2(x)
        x = self.dilated3(x)
        x = self.dilated4(x)
        x = self.mod2(x, strategy_embedding)

        x = self.up1(x)
        x = self.deconv1(x)
        x = self.up2(x)
        x = self.deconv2(x)
        x = self.out_conv(x)
        return torch.tanh(x)


class InpaintingGenerator(nn.Module):
    """Complete RL-Guided Inpainting Generator (DeepFill v2 style).

    Consists of:
      - Shared Feature Encoder (state extraction for RL)
      - Coarse Inpainting Network
      - RL Strategy Vector Embedder (4 discrete strategies)
      - Refinement Network conditioned on RL strategy
    """

    def __init__(
        self,
        base_channels: int = 32,
        strategy_dim: int = 64,
        num_strategies: int = 4,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.strategy_dim = strategy_dim
        self.num_strategies = num_strategies

        # Strategy Embedding table (0: Global, 1: Local, 2: Boundary, 3: Texture)
        self.strategy_embeddings = nn.Embedding(num_strategies, strategy_dim)
        # Neutral strategy embedding for default baseline mode
        self.register_buffer("neutral_embedding", torch.zeros(1, strategy_dim))

        self.encoder = InpaintingEncoder(in_channels=4, base_channels=base_channels * 2, latent_dim=latent_dim)
        self.coarse_net = CoarseNetwork(in_channels=4, base_channels=base_channels)
        self.refine_net = RefinementNetwork(
            in_channels=4,
            base_channels=base_channels,
            strategy_dim=strategy_dim,
        )

    def get_strategy_embedding(
        self,
        strategy: Optional[Union[int, torch.Tensor]] = None,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Decode integer action or strategy vector into embedding tensor (B, strategy_dim)."""
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
            # One-hot representation
            return torch.matmul(strategy.to(device).float(), self.strategy_embeddings.weight)
        else:
            return self.neutral_embedding.expand(batch_size, -1).to(device)

    def extract_state_embedding(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Extract pooled 256-dim feature representation for the RL controller."""
        _, state_embedding = self.encoder(masked_image, mask)
        return state_embedding

    def coarse_forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run coarse inpainting pass."""
        return self.coarse_net(masked_image, mask)

    def refine(
        self,
        coarse_image: torch.Tensor,
        mask: torch.Tensor,
        strategy: Optional[Union[int, torch.Tensor]] = None,
        original_masked_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run refinement pass conditioned on strategy vector."""
        b, _, _, _ = coarse_image.shape
        strat_emb = self.get_strategy_embedding(strategy, batch_size=b, device=coarse_image.device)

        if original_masked_image is not None:
            coarse_composite = original_masked_image + coarse_image * mask
        else:
            coarse_composite = coarse_image

        return self.refine_net(coarse_composite, mask, strat_emb)

    def forward(
        self,
        masked_image: torch.Tensor,
        mask: torch.Tensor,
        strategy: Optional[Union[int, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Complete forward pass.

        Returns:
            Dict containing:
              - coarse: coarse network output (B, 3, H, W)
              - coarse_composite: masked_image + coarse * mask
              - refined: refinement network output (B, 3, H, W)
              - completed: masked_image + refined * mask (final composite image)
              - state_embedding: (B, latent_dim)
        """
        b = masked_image.shape[0]
        coarse = self.coarse_net(masked_image, mask)
        coarse_composite = masked_image + coarse * mask

        strat_emb = self.get_strategy_embedding(strategy, batch_size=b, device=masked_image.device)
        refined = self.refine_net(coarse_composite, mask, strat_emb)
        completed = masked_image + refined * mask

        _, state_embedding = self.encoder(masked_image, mask)

        return {
            "coarse": coarse,
            "coarse_composite": coarse_composite,
            "refined": refined,
            "completed": completed,
            "state_embedding": state_embedding,
        }
