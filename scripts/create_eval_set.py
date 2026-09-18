"""create_eval_set.py — Build 100 fixed evaluation image-mask pairs.

Selects 25 images from each of 4 mask-severity buckets from COCO val2017
and saves them as a deterministic eval set for all experiments.

Severity buckets (by missing-pixel ratio):
  small   : 5–15%
  medium  : 15–30%
  large   : 30–50%
  xlarge  : 50–70%

Output structure:
  data/masks/fixed_eval_masks/
    small/
      0000_image.png
      0000_mask.png
      ...
    medium/ ...
    large/ ...
    xlarge/ ...
    eval_manifest.json

Usage
-----
  python scripts/create_eval_set.py \
      --data-dir  data/raw/val2017 \
      --output    data/masks/fixed_eval_masks \
      --per-bucket 25 \
      --image-size 256 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.data.mask_generator import MaskGenerator


BUCKETS = {
    "small":  (0.05, 0.15),
    "medium": (0.15, 0.30),
    "large":  (0.30, 0.50),
    "xlarge": (0.50, 0.70),
}


def make_mask(mask_gen: MaskGenerator, target_ratio: float, H: int, W: int) -> np.ndarray:
    """Generate a mask with missing_ratio close to target_ratio."""
    for _ in range(20):
        mask = mask_gen.generate(H, W)
        ratio = mask.mean()
        if abs(ratio - target_ratio) < 0.08:
            return mask
    return mask


def build_eval_set(
    data_dir: str,
    output_dir: str,
    per_bucket: int = 25,
    image_size: int = 256,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Collect image paths
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    if len(all_images) < per_bucket * len(BUCKETS):
        print(f"[WARN] Only {len(all_images)} images found in {data_dir}, "
              f"need {per_bucket * len(BUCKETS)}. Repeating some.")

    random.shuffle(all_images)

    to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    mask_gen = MaskGenerator(
        image_size=image_size,
        min_ratio=0.05,
        max_ratio=0.70,
    )

    manifest = {}
    img_cursor = 0

    for bucket_name, (lo, hi) in BUCKETS.items():
        bucket_dir = os.path.join(output_dir, bucket_name)
        os.makedirs(bucket_dir, exist_ok=True)
        bucket_entries = []
        target_ratio = (lo + hi) / 2.0

        count = 0
        while count < per_bucket:
            img_path = all_images[img_cursor % len(all_images)]
            img_cursor += 1

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_t = to_tensor(img)  # (3, H, W) in [0, 1]

            # Generate mask hitting the bucket range
            mask_np = make_mask(mask_gen, target_ratio, image_size, image_size)
            ratio   = float(mask_np.mean())
            if not (lo - 0.05 <= ratio <= hi + 0.05):
                continue

            # Save image
            img_arr  = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_pil  = Image.fromarray(img_arr)
            mask_arr = (mask_np * 255).astype(np.uint8)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr.squeeze(-1)
            mask_pil = Image.fromarray(mask_arr, mode="L")

            img_fname  = f"{count:04d}_image.png"
            mask_fname = f"{count:04d}_mask.png"
            img_pil.save(os.path.join(bucket_dir, img_fname))
            mask_pil.save(os.path.join(bucket_dir, mask_fname))

            bucket_entries.append({
                "image": os.path.join(bucket_name, img_fname),
                "mask":  os.path.join(bucket_name, mask_fname),
                "missing_ratio": ratio,
                "source_image":  img_path,
            })
            count += 1
            print(f"  [{bucket_name}] {count}/{per_bucket}  ratio={ratio:.3f}", end="\r")

        print(f"  [{bucket_name}] Done — {count} pairs saved.")
        manifest[bucket_name] = bucket_entries

    manifest_path = os.path.join(output_dir, "eval_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = per_bucket * len(BUCKETS)
    print(f"\nEval set created: {total} pairs in {output_dir}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed 100-image evaluation set")
    parser.add_argument("--data-dir",   type=str,  default="data/raw/val2017")
    parser.add_argument("--output",     type=str,  default="data/masks/fixed_eval_masks")
    parser.add_argument("--per-bucket", type=int,  default=25)
    parser.add_argument("--image-size", type=int,  default=256)
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    build_eval_set(
        data_dir   = args.data_dir,
        output_dir = args.output,
        per_bucket = args.per_bucket,
        image_size = args.image_size,
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()
