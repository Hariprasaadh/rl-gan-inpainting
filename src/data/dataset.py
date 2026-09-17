import os
import glob
from typing import Optional, Tuple, Dict, Any, List, Union
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import ImageTransform
from src.data.mask_generator import (
    generate_irregular_mask,
    compute_mask_stats,
    generate_targeted_severity_mask,
)


class InpaintingDataset(Dataset):
    """Dataset for training and evaluating RL-GAN inpainting.

    Handles:
      - Real image loading (PNG/JPG/JPEG) from Places2, CelebA-HQ, etc.
      - Synthetic procedural image generation for offline testing/CPU development
      - Dynamic irregular masks during training
      - Fixed seeded / loaded masks for validation and testing
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        image_size: int = 256,
        split: str = "train",
        is_train: bool = True,
        synthetic_size: int = 100,
        fixed_seed: Optional[int] = None,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.split = split
        self.is_train = is_train
        self.synthetic_size = synthetic_size
        self.fixed_seed = fixed_seed
        self.transform = ImageTransform(image_size=image_size, is_train=is_train)

        # Collect image filepaths
        self.image_paths: List[str] = []
        if image_dir and os.path.exists(image_dir):
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.avif"]:
                self.image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
            self.image_paths.sort()

        # Check if we should use synthetic generator
        self.use_synthetic = len(self.image_paths) == 0

        # Collect fixed mask files if mask_dir provided
        self.mask_paths: List[str] = []
        if mask_dir and os.path.exists(mask_dir):
            for ext in ["*.png", "*.jpg"]:
                self.mask_paths.extend(glob.glob(os.path.join(mask_dir, "**", ext), recursive=True))
            self.mask_paths.sort()

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.synthetic_size
        return len(self.image_paths)

    def _generate_synthetic_image(self, index: int) -> Image.Image:
        """Procedurally generate colorful geometric scenes for testing without downloads."""
        rng = np.random.default_rng(index + (self.fixed_seed or 0))
        img = Image.new("RGB", (self.image_size, self.image_size), color=(
            int(rng.integers(50, 230)),
            int(rng.integers(50, 230)),
            int(rng.integers(50, 230)),
        ))
        draw = ImageDraw.Draw(img)

        # Draw random shapes (rectangles, ellipses, polygons)
        for _ in range(rng.integers(4, 12)):
            x1, y1 = int(rng.integers(0, self.image_size)), int(rng.integers(0, self.image_size))
            x2, y2 = int(rng.integers(0, self.image_size)), int(rng.integers(0, self.image_size))
            color = (int(rng.integers(0, 255)), int(rng.integers(0, 255)), int(rng.integers(0, 255)))
            shape_type = rng.integers(0, 3)
            if shape_type == 0:
                draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=color)
            elif shape_type == 1:
                draw.ellipse([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=color)
            else:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=int(rng.integers(2, 10)))

        return img

    def _load_image(self, index: int) -> Tuple[torch.Tensor, str]:
        if self.use_synthetic:
            pil_img = self._generate_synthetic_image(index)
            path = f"synthetic_{index:05d}"
        else:
            path = self.image_paths[index % len(self.image_paths)]
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception:
                pil_img = self._generate_synthetic_image(index)

        tensor = self.transform(pil_img)
        return tensor, path

    def _get_mask(self, index: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        # If fixed mask directory is given, load mask from disk
        if len(self.mask_paths) > 0:
            mask_path = self.mask_paths[index % len(self.mask_paths)]
            mask_pil = Image.open(mask_path).convert("L")
            if mask_pil.size != (self.image_size, self.image_size):
                mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = (np.array(mask_pil) > 128).astype(np.uint8)
        else:
            # Deterministic RNG for eval, dynamic for train
            rng = None
            if not self.is_train and self.fixed_seed is not None:
                rng = np.random.default_rng(self.fixed_seed + index)
            mask_np = generate_irregular_mask(
                self.image_size,
                self.image_size,
                max_strokes=4,
                max_len=max(20, int(self.image_size * 0.35)),
                max_width=max(8, int(self.image_size * 0.12)),
                rng=rng,
            )

        stats = compute_mask_stats(mask_np)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)  # (1, H, W)
        return mask_tensor, stats

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image, path = self._load_image(index)
        mask, stats = self._get_mask(index)

        # masked_image: valid pixels kept, missing pixels zeroed out
        masked_image = image * (1.0 - mask)

        return {
            "image": image,              # (3, H, W) in [-1, 1]
            "mask": mask,                # (1, H, W) in {0, 1}
            "masked_image": masked_image,  # (3, H, W) in [-1, 1]
            "stats": stats,
            "path": path,
        }

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Convenience method to sample a single item (used by RL Gym Environment)."""
        idx = int(np.random.randint(0, len(self)))
        item = self[idx]
        return (
            item["image"],
            item["mask"],
            item["masked_image"],
            {"stats": item["stats"], "path": item["path"]},
        )
