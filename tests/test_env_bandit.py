import pytest
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from src.data.dataset import InpaintingDataset
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.rl.env_bandit import InpaintingBanditEnv


def get_mock_env():
    dataset = InpaintingDataset(image_dir=None, image_size=64, synthetic_size=10, is_train=True)
    generator = InpaintingGenerator(base_channels=16, strategy_dim=32, num_strategies=4, latent_dim=128)
    discriminator = SNPatchGANDiscriminator(in_channels=4, base_channels=16)
    env = InpaintingBanditEnv(
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        device="cpu",
        compute_lpips=False,
    )
    return env


def test_env_conformance():
    env = get_mock_env()
    # Check Gymnasium env conformance with stable_baselines3 check_env
    check_env(env)


def test_env_reset_and_step():
    env = get_mock_env()
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    assert "coarse_psnr" in info

    # Step action 2 (Boundary)
    next_obs, reward, terminated, truncated, step_info = env.step(2)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert "action" in step_info
    assert step_info["action"] == 2
    assert "d_psnr" in step_info
    assert "action_cost" in step_info


def test_ppo_bandit_learn_smoke():
    env = get_mock_env()
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=16,
        batch_size=8,
        n_epochs=2,
        gamma=0.0,
        verbose=0,
    )
    # Smoke test: learn for 32 timesteps without error
    model.learn(total_timesteps=32)
    assert model.num_timesteps >= 32
