import pytest
import torch
from src.data.dataset import InpaintingDataset


def test_inpainting_dataset_synthetic():
    dataset = InpaintingDataset(
        image_dir=None,
        image_size=64,
        synthetic_size=10,
        is_train=True,
    )
    assert len(dataset) == 10
    item = dataset[0]

    assert "image" in item
    assert "mask" in item
    assert "masked_image" in item
    assert "stats" in item

    # Check tensor shapes
    assert item["image"].shape == (3, 64, 64)
    assert item["mask"].shape == (1, 64, 64)
    assert item["masked_image"].shape == (3, 64, 64)

    # Check value ranges
    assert item["image"].min() >= -1.0
    assert item["image"].max() <= 1.0
    assert set(torch.unique(item["mask"]).tolist()).issubset({0.0, 1.0})

    # Check masked image consistency: where mask == 1, masked_image == 0
    mask_bool = (item["mask"] == 1.0).expand_as(item["masked_image"])
    assert torch.all(item["masked_image"][mask_bool] == 0.0)


def test_inpainting_dataset_sample():
    dataset = InpaintingDataset(image_dir=None, image_size=64, synthetic_size=5, is_train=False, fixed_seed=42)
    img, mask, masked_img, meta = dataset.sample()
    assert img.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert masked_img.shape == (3, 64, 64)
    assert "stats" in meta
