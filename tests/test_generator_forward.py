import pytest
import torch
import torch.nn as nn

from src.models.encoder import InpaintingEncoder
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.models.losses import MaskedL1Loss, PerceptualAndStyleLoss, HingeAdversarialLoss


def test_encoder_forward():
    encoder = InpaintingEncoder(in_channels=4, base_channels=16, latent_dim=128)
    masked_image = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 1.0

    features, state = encoder(masked_image, mask)
    assert len(features) == 3
    assert features[0].shape == (2, 16, 32, 32)
    assert features[1].shape == (2, 32, 16, 16)
    assert features[2].shape == (2, 64, 8, 8)
    assert state.shape == (2, 128)


def test_generator_forward_and_strategies():
    generator = InpaintingGenerator(base_channels=16, strategy_dim=32, num_strategies=4, latent_dim=128)
    masked_image = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 15:45, 15:45] = 1.0

    # 1. Forward with default neutral strategy
    out = generator(masked_image, mask)
    assert out["coarse"].shape == (2, 3, 64, 64)
    assert out["refined"].shape == (2, 3, 64, 64)
    assert out["completed"].shape == (2, 3, 64, 64)
    assert out["state_embedding"].shape == (2, 128)

    # 2. Forward with discrete strategy 0 (Global) vs strategy 3 (Texture)
    out_strat0 = generator(masked_image, mask, strategy=0)
    out_strat3 = generator(masked_image, mask, strategy=3)

    # Output tensors should differ due to strategy modulation
    diff = torch.abs(out_strat0["refined"] - out_strat3["refined"]).sum()
    assert diff > 0.0


def test_discriminator_forward_and_confidence():
    disc = SNPatchGANDiscriminator(in_channels=4, base_channels=16)
    image = torch.randn(2, 3, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)

    logits = disc(image, mask)
    assert logits.shape[0] == 2
    assert logits.shape[1] == 1

    conf = disc.get_confidence(image, mask)
    assert conf.shape == (2,)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_losses_and_step():
    generator = InpaintingGenerator(base_channels=16, strategy_dim=32, num_strategies=4, latent_dim=128)
    discriminator = SNPatchGANDiscriminator(in_channels=4, base_channels=16)

    l1_fn = MaskedL1Loss(hole_weight=6.0, valid_weight=1.0)
    adv_fn = HingeAdversarialLoss(adv_weight=0.01)

    images = torch.randn(2, 3, 64, 64)
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 10:30, 10:30] = 1.0
    masked_images = images * (1.0 - masks)

    # Generator forward
    out = generator(masked_images, masks, strategy=1)
    fake_completed = out["completed"]

    # Discriminator step
    real_logits = discriminator(images, masks)
    fake_logits = discriminator(fake_completed.detach(), masks)
    loss_d = adv_fn.discriminator_loss(real_logits, fake_logits)
    assert not torch.isnan(loss_d)

    # Generator step
    loss_l1, _ = l1_fn(fake_completed, images, masks)
    fake_logits_g = discriminator(fake_completed, masks)
    loss_adv_g = adv_fn.generator_loss(fake_logits_g)
    loss_g = loss_l1 + loss_adv_g
    assert not torch.isnan(loss_g)
