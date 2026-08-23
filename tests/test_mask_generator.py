import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from src.data.mask_generator import (
    generate_irregular_mask,
    generate_targeted_severity_mask,
    generate_fixed_eval_masks,
    compute_mask_stats,
    get_severity_bucket,
    DynamicMaskGenerator,
)


def test_generate_irregular_mask_shape_and_values():
    h, w = 128, 128
    mask = generate_irregular_mask(h, w, max_strokes=5, max_len=40, max_width=15)
    assert mask.shape == (h, w)
    assert mask.dtype == np.uint8
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 1})
    assert mask.sum() > 0, "Mask should contain at least some masked pixels"


def test_compute_mask_stats():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 1  # 40x40 = 1600 pixels, missing_ratio = 0.16
    stats = compute_mask_stats(mask)
    assert "missing_ratio" in stats
    assert "bbox_ratio" in stats
    assert "num_regions" in stats
    assert "center_dist" in stats
    assert abs(stats["missing_ratio"] - 0.16) < 1e-4
    assert stats["bbox_ratio"] == 0.16
    assert get_severity_bucket(stats["missing_ratio"]) == "10-20%"


def test_mask_generator_reproducibility():
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)
    mask1 = generate_irregular_mask(64, 64, rng=rng1)
    mask2 = generate_irregular_mask(64, 64, rng=rng2)
    np.testing.assert_array_equal(mask1, mask2)


def test_targeted_severity_mask():
    for bucket in ["10-20%", "20-40%", "40-60%", "60%+"]:
        mask = generate_targeted_severity_mask(64, 64, bucket, rng=np.random.default_rng(42))
        assert mask.shape == (64, 64)
        assert mask.sum() > 0


def test_fixed_eval_masks_generation():
    temp_dir = tempfile.mkdtemp()
    try:
        saved = generate_fixed_eval_masks(temp_dir, count_per_bucket=2, image_size=64, seed=42)
        assert len(saved) == 4
        for bucket, files in saved.items():
            assert len(files) == 2
            for f in files:
                assert os.path.exists(f)
    finally:
        shutil.rmtree(temp_dir)


def test_dynamic_mask_generator():
    dmg = DynamicMaskGenerator(image_size=64, seed=10)
    mask_tensor, stats = dmg()
    assert isinstance(mask_tensor, torch.Tensor)
    assert mask_tensor.shape == (1, 64, 64)
    assert mask_tensor.dtype == torch.float32
    assert "missing_ratio" in stats
