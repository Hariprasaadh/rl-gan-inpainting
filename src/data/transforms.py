from typing import Tuple, Union
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np


class ImageTransform:
    """Image transformation pipeline for RL-GAN inpainting.

    Normalizes images to [-1, 1] range and resizes to target resolution.
    """

    def __init__(self, image_size: Union[int, Tuple[int, int]] = 256, is_train: bool = True):
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        transforms_list = [
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
        ]
        if is_train:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))

        transforms_list.extend([
            T.ToTensor(),  # [0, 1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

        self.transform = T.Compose(transforms_list)

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.transform(img)


def denormalize_image(tensor: torch.Tensor, to_uint8: bool = False) -> torch.Tensor:
    """Convert tensor from [-1, 1] back to [0, 1] (or [0, 255] uint8).

    Args:
        tensor: (B, C, H, W) or (C, H, W) in [-1, 1]
        to_uint8: If True, returns uint8 tensor in [0, 255]

    Returns:
        Denormalized tensor.
    """
    img = (tensor + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
    if to_uint8:
        img = (img * 255.0).round().to(torch.uint8)
    return img
