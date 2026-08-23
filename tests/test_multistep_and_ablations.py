import pytest
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.data.dataset import InpaintingDataset
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.rl.actions import ACTION_STOP
from src.rl.env_multistep import InpaintingMultiStepEnv
from src.rl.state import StateBuilder


def get_mock_multistep_env():
    dataset = InpaintingDataset(image_dir=None, image_size=64, synthetic_size=10, is_train=True)
    generator = InpaintingGenerator(base_channels=16, strategy_dim=32, num_strategies=4, latent_dim=128)
    discriminator = SNPatchGANDiscriminator(in_channels=4, base_channels=16)
    env = InpaintingMultiStepEnv(
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        device="cpu",
        max_steps=3,
        compute_lpips=False,
    )
    return env


def test_multistep_env_conformance():
    env = get_mock_multistep_env()
    check_env(env)


def test_multistep_stop_action():
    env = get_mock_multistep_env()
    obs, info = env.reset(seed=42)
    assert env.current_step == 0

    # Execute STOP action immediately
    next_obs, reward, terminated, truncated, step_info = env.step(ACTION_STOP)
    assert terminated is True
    assert reward == 0.0
    assert step_info["action"] == ACTION_STOP


def test_multistep_max_steps_termination():
    env = get_mock_multistep_env()
    obs, _ = env.reset(seed=42)

    # Step 1: Action 0 (not terminated)
    _, _, term1, _, _ = env.step(0)
    assert term1 is False

    # Step 2: Action 1 (not terminated)
    _, _, term2, _, _ = env.step(1)
    assert term2 is False

    # Step 3: Action 2 (reaches max_steps=3 -> terminated)
    _, _, term3, _, _ = env.step(2)
    assert term3 is True


def test_ablation_state_builder_dims():
    sb_full = StateBuilder(latent_dim=128, include_mask_stats=True, include_quality_proxy=True)
    assert sb_full.state_dim == 128 + 4 + 1

    sb_no_stats = StateBuilder(latent_dim=128, include_mask_stats=False, include_quality_proxy=True)
    assert sb_no_stats.state_dim == 128 + 0 + 1
