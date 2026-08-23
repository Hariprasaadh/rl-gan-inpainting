import pytest
import torch
import numpy as np

from src.rl.actions import get_action_cost, decode_action, ACTION_GLOBAL, ACTION_TEXTURE
from src.rl.reward import compute_reward, normalize_psnr_delta, compute_image_metrics


def test_action_costs_and_decoding():
    assert get_action_cost(ACTION_GLOBAL) == 0.00
    assert get_action_cost(ACTION_TEXTURE) == 0.20
    one_hot = decode_action(ACTION_TEXTURE, num_actions=4)
    assert one_hot.shape == (4,)
    assert one_hot[ACTION_TEXTURE] == 1.0
    assert one_hot.sum() == 1.0


def test_normalize_psnr_delta():
    assert normalize_psnr_delta(0.0) == 0.0
    assert abs(normalize_psnr_delta(2.0) - 0.2) < 1e-4
    assert normalize_psnr_delta(100.0) == 2.0  # Clamped to 2.0
    assert normalize_psnr_delta(-100.0) == -2.0  # Clamped to -2.0


def test_compute_reward_breakdown():
    prev = {"psnr": 20.0, "ssim": 0.70, "lpips": 0.30}
    new_good = {"psnr": 22.0, "ssim": 0.75, "lpips": 0.25}  # Improved
    new_bad = {"psnr": 18.0, "ssim": 0.65, "lpips": 0.35}   # Degraded

    r_good, info_good = compute_reward(prev, new_good, disc_score_delta=0.1, action=ACTION_GLOBAL)
    r_bad, info_bad = compute_reward(prev, new_bad, disc_score_delta=-0.1, action=ACTION_TEXTURE)

    # Positive improvements should yield higher reward than degradations
    assert r_good > r_bad
    assert "d_psnr" in info_good
    assert "d_ssim" in info_good
    assert "d_lpips" in info_good
    assert "disc_delta" in info_good
    assert "action_cost" in info_good
    assert info_good["action_cost"] == 0.0
    assert info_bad["action_cost"] == 0.20


def test_compute_image_metrics():
    img1 = torch.ones(1, 3, 32, 32)
    img2 = torch.ones(1, 3, 32, 32)
    metrics = compute_image_metrics(img1, img2, compute_lpips=False)
    assert "psnr" in metrics
    assert "ssim" in metrics
    assert metrics["ssim"] > 0.99
