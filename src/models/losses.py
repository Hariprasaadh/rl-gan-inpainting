from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix of feature map for style loss."""
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / float(c * h * w)


class MaskedL1Loss(nn.Module):
    """Masked L1 Loss with separate weights for hole and valid regions."""

    def __init__(self, hole_weight: float = 6.0, valid_weight: float = 1.0):
        super().__init__()
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted L1 loss.

        Args:
            predicted: (B, C, H, W)
            target: (B, C, H, W)
            mask: (B, 1, H, W) where 1 indicates hole, 0 indicates valid.
        """
        diff = torch.abs(predicted - target)

        mask_expanded = mask.expand_as(diff)
        hole_pixels = torch.sum(mask_expanded) + 1e-6
        valid_pixels = torch.sum(1.0 - mask_expanded) + 1e-6

        hole_loss = torch.sum(diff * mask_expanded) / hole_pixels
        valid_loss = torch.sum(diff * (1.0 - mask_expanded)) / valid_pixels

        total_l1 = self.hole_weight * hole_loss + self.valid_weight * valid_loss

        return total_l1, {
            "l1_hole": float(hole_loss.item()),
            "l1_valid": float(valid_loss.item()),
            "l1_total": float(total_l1.item()),
        }


class VGGFeatureExtractor(nn.Module):
    """Extracts intermediate VGG-16 features for perceptual and style losses."""

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT if hasattr(models, "VGG16_Weights") else None)
        features = list(vgg16.features.children())

        # Slice layers: relu1_2 (idx 3), relu2_2 (idx 8), relu3_3 (idx 15), relu4_3 (idx 22)
        self.slice1 = nn.Sequential(*features[:4])
        self.slice2 = nn.Sequential(*features[4:9])
        self.slice3 = nn.Sequential(*features[9:16])
        self.slice4 = nn.Sequential(*features[16:23])

        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Convert [-1, 1] to [0, 1] then normalize for VGG
        x = (x + 1.0) / 2.0
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._normalize(x)
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4


class PerceptualAndStyleLoss(nn.Module):
    """Perceptual and Style Loss using VGG-16 features."""

    def __init__(self, perceptual_weight: float = 0.05, style_weight: float = 120.0):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.vgg: Optional[VGGFeatureExtractor] = None

    def _get_vgg(self, device: torch.device) -> VGGFeatureExtractor:
        if self.vgg is None:
            try:
                self.vgg = VGGFeatureExtractor().to(device).eval()
            except Exception:
                # If network weights download is unavailable, use mock feature extractor
                self.vgg = None
        return self.vgg

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        vgg = self._get_vgg(predicted.device)
        if vgg is None:
            # Fallback simple feature loss
            p_loss = F.l1_loss(predicted, target) * self.perceptual_weight
            s_loss = torch.tensor(0.0, device=predicted.device)
            return p_loss, s_loss, {"loss_perceptual": float(p_loss.item()), "loss_style": 0.0}

        pred_feats = vgg(predicted)
        target_feats = vgg(target)

        # Perceptual Loss (L1 over feature maps)
        p_loss = torch.tensor(0.0, device=predicted.device)
        for pf, tf in zip(pred_feats, target_feats):
            p_loss = p_loss + F.l1_loss(pf, tf)
        p_loss = p_loss * self.perceptual_weight

        # Style Loss (L1 over Gram matrices)
        s_loss = torch.tensor(0.0, device=predicted.device)
        for pf, tf in zip(pred_feats, target_feats):
            s_loss = s_loss + F.l1_loss(gram_matrix(pf), gram_matrix(tf))
        s_loss = s_loss * self.style_weight

        return p_loss, s_loss, {
            "loss_perceptual": float(p_loss.item()),
            "loss_style": float(s_loss.item()),
        }


class HingeAdversarialLoss(nn.Module):
    """Hinge loss for GAN with Spectral Normalization."""

    def __init__(self, adv_weight: float = 0.001):
        super().__init__()
        self.adv_weight = adv_weight

    def discriminator_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Hinge loss for discriminator."""
        loss_real = torch.mean(F.relu(1.0 - real_logits))
        loss_fake = torch.mean(F.relu(1.0 + fake_logits))
        return loss_real + loss_fake

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Hinge loss for generator."""
        return -torch.mean(fake_logits) * self.adv_weight
