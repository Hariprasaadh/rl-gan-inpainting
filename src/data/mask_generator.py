import os
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
import torch


def compute_mask_stats(mask: np.ndarray) -> Dict[str, float]:
    """Compute geometric and coverage statistics of a mask.

    Args:
        mask: 2D uint8 numpy array where 1 indicates masked (missing) pixels and 0 indicates valid.

    Returns:
        Dict with:
          - missing_ratio: float in [0.0, 1.0]
          - bbox_ratio: bounding box area / total area
          - num_regions: normalized count of disconnected masked components
          - center_dist: distance of mask centroid from image center normalized by half diagonal
    """
    h, w = mask.shape[:2]
    total_pixels = float(h * w)
    masked_pixels = float(np.sum(mask > 0))
    missing_ratio = masked_pixels / total_pixels if total_pixels > 0 else 0.0

    if masked_pixels == 0:
        return {
            "missing_ratio": 0.0,
            "bbox_ratio": 0.0,
            "num_regions": 0.0,
            "center_dist": 0.0,
        }

    # Bounding box
    y_indices, x_indices = np.where(mask > 0)
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    bbox_area = float((ymax - ymin + 1) * (xmax - xmin + 1))
    bbox_ratio = bbox_area / total_pixels

    # Number of connected components
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
    # num_labels includes background (label 0)
    num_components = max(0, num_labels - 1)
    norm_num_components = float(np.clip(num_components / 10.0, 0.0, 1.0))

    # Centroid distance
    cy = float(np.mean(y_indices))
    cx = float(np.mean(x_indices))
    center_y, center_x = h / 2.0, w / 2.0
    max_dist = np.sqrt(center_y**2 + center_x**2)
    dist = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
    center_dist = float(dist / max_dist) if max_dist > 0 else 0.0

    return {
        "missing_ratio": float(missing_ratio),
        "bbox_ratio": float(bbox_ratio),
        "num_regions": float(norm_num_components),
        "center_dist": float(center_dist),
    }


def get_severity_bucket(missing_ratio: float) -> str:
    """Return bucket name for missing-area ratio."""
    if missing_ratio < 0.20:
        return "10-20%"
    elif missing_ratio < 0.40:
        return "20-40%"
    elif missing_ratio < 0.60:
        return "40-60%"
    else:
        return "60%+"


def generate_irregular_mask(
    h: int,
    w: int,
    max_strokes: int = 8,
    max_len: int = 80,
    max_width: int = 30,
    min_strokes: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate irregular stroke mask on the fly.

    Args:
        h: Height of the mask.
        w: Width of the mask.
        max_strokes: Maximum number of brush strokes.
        max_len: Maximum length of each stroke segment.
        max_width: Maximum stroke line width.
        min_strokes: Minimum number of strokes.
        rng: Optional seeded numpy random generator.

    Returns:
        np.ndarray of shape (h, w) with values in {0, 1} (uint8), where 1 = hole (missing).
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros((h, w), dtype=np.uint8)
    num_strokes = rng.integers(min_strokes, max(min_strokes + 1, max_strokes + 1))

    for _ in range(num_strokes):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        num_sub_strokes = rng.integers(3, 16)
        for _ in range(num_sub_strokes):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.integers(10, max(11, max_len))
            width = rng.integers(5, max(6, max_width))
            x2 = int(np.clip(x + length * np.cos(angle), 0, w - 1))
            y2 = int(np.clip(y + length * np.sin(angle), 0, h - 1))
            cv2.line(mask, (x, y), (x2, y2), 1, int(width))
            x, y = x2, y2

    # Occasionally add a circle or rectangle for diversity
    if rng.random() > 0.6:
        cx, cy = int(rng.integers(w // 4, max(w // 4 + 1, 3 * w // 4))), int(rng.integers(h // 4, max(h // 4 + 1, 3 * h // 4)))
        max_r = max(4, min(h, w) // 4)
        min_r = max(2, min(3, max_r - 1))
        radius = int(rng.integers(min_r, max_r + 1))
        cv2.circle(mask, (cx, cy), radius, 1, -1)

    # Ensure mask has at least some missing pixels
    if mask.sum() == 0:
        cv2.circle(mask, (w // 2, h // 2), max(2, min(h, w) // 6), 1, -1)

    return mask


def generate_targeted_severity_mask(
    h: int,
    w: int,
    target_bucket: str,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 50,
) -> np.ndarray:
    """Generate an irregular mask that falls within a specific severity bucket."""
    if rng is None:
        rng = np.random.default_rng()

    bucket_ranges = {
        "10-20%": (0.10, 0.20),
        "20-40%": (0.20, 0.40),
        "40-60%": (0.40, 0.60),
        "60%+": (0.60, 0.85),
    }
    low, high = bucket_ranges.get(target_bucket, (0.20, 0.40))

    best_mask = None
    best_diff = float("inf")
    target_mid = (low + high) / 2.0

    # Adjust generator parameters according to target severity
    if target_bucket == "10-20%":
        params = {"max_strokes": 4, "max_len": 50, "max_width": 15, "min_strokes": 1}
    elif target_bucket == "20-40%":
        params = {"max_strokes": 8, "max_len": 80, "max_width": 25, "min_strokes": 3}
    elif target_bucket == "40-60%":
        params = {"max_strokes": 14, "max_len": 100, "max_width": 35, "min_strokes": 6}
    else:  # 60%+
        params = {"max_strokes": 20, "max_len": 130, "max_width": 45, "min_strokes": 10}

    for _ in range(max_tries):
        mask = generate_irregular_mask(h, w, rng=rng, **params)
        ratio = mask.sum() / float(h * w)
        if low <= ratio <= high:
            return mask
        diff = abs(ratio - target_mid)
        if diff < best_diff:
            best_diff = diff
            best_mask = mask

    return best_mask if best_mask is not None else generate_irregular_mask(h, w, rng=rng)


def generate_fixed_eval_masks(
    output_dir: str,
    count_per_bucket: int = 25,
    image_size: int = 256,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate and save deterministic fixed evaluation masks for val/test."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    buckets = ["10-20%", "20-40%", "40-60%", "60%+"]
    saved_files: Dict[str, List[str]] = {b: [] for b in buckets}

    for bucket in buckets:
        bucket_dir = os.path.join(output_dir, bucket.replace("%", "pct").replace("+", "plus"))
        os.makedirs(bucket_dir, exist_ok=True)
        for i in range(count_per_bucket):
            mask = generate_targeted_severity_mask(image_size, image_size, bucket, rng=rng)
            mask_img = (mask * 255).astype(np.uint8)
            filepath = os.path.join(bucket_dir, f"mask_{i:04d}.png")
            cv2.imwrite(filepath, mask_img)
            saved_files[bucket].append(filepath)

    return saved_files


class DynamicMaskGenerator:
    """Callable mask generator for dynamic batch loading."""

    def __init__(
        self,
        image_size: int = 256,
        max_strokes: int = 8,
        max_len: int = 80,
        max_width: int = 30,
        seed: Optional[int] = None,
    ):
        self.image_size = image_size
        self.max_strokes = max_strokes
        self.max_len = max_len
        self.max_width = max_width
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def __call__(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        mask_np = generate_irregular_mask(
            self.image_size,
            self.image_size,
            max_strokes=self.max_strokes,
            max_len=self.max_len,
            max_width=self.max_width,
            rng=self.rng,
        )
        stats = compute_mask_stats(mask_np)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)  # (1, H, W)
        return mask_tensor, stats
