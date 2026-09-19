"""Create 100 fixed evaluation pairs (25 per severity bucket) from COCO val2017 or synthetic.

Must be done once and reused for all experiments.
Saves masks to data/masks/fixed_eval_masks/
If real images available, also copies a fixed list of image paths to data/masks/fixed_eval_manifest.json
"""

import os
import argparse
import json
import glob
from pathlib import Path

from src.data.mask_generator import generate_fixed_eval_masks


def create_eval_set(
    image_dir: str = "data/raw/val2017",
    mask_output_dir: str = "data/masks/fixed_eval_masks",
    count_per_bucket: int = 25,
    image_size: int = 256,
    seed: int = 42,
):
    print(f"Generating {count_per_bucket*4} fixed eval masks ({count_per_bucket} per bucket) -> {mask_output_dir}")
    saved = generate_fixed_eval_masks(
        output_dir=mask_output_dir,
        count_per_bucket=count_per_bucket,
        image_size=image_size,
        seed=seed,
    )
    total = sum(len(v) for v in saved.values())
    print(f"Generated {total} masks across buckets: {list(saved.keys())}")

    # Try to create image manifest if image_dir exists
    manifest_path = os.path.join(os.path.dirname(mask_output_dir), "fixed_eval_manifest.json")
    if image_dir and os.path.exists(image_dir):
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
        image_paths.sort()
        if len(image_paths) > 0:
            # Take first 100 deterministic
            selected = image_paths[:100]
            # Map mask files in deterministic order
            all_masks = []
            for bucket in ["10-20%", "20-40%", "40-60%", "60%+"]:
                all_masks.extend(sorted(saved.get(bucket, [])))
            pairs = []
            for i, img in enumerate(selected):
                mask = all_masks[i % len(all_masks)] if all_masks else None
                pairs.append({"image": img, "mask": mask, "index": i})
            with open(manifest_path, "w") as f:
                json.dump({"pairs": pairs, "total": len(pairs)}, f, indent=2)
            print(f"Manifest saved to {manifest_path} with {len(pairs)} pairs")
        else:
            print(f"No images found in {image_dir}, masks only.")
    else:
        print(f"Image dir {image_dir} not found, created masks only (synthetic eval will be used).")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create fixed eval set")
    parser.add_argument("--image-dir", type=str, default="data/raw/val2017")
    parser.add_argument("--mask-output-dir", type=str, default="data/masks/fixed_eval_masks")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_eval_set(
        image_dir=args.image_dir,
        mask_output_dir=args.mask_output_dir,
        count_per_bucket=args.count,
        image_size=args.image_size,
        seed=args.seed,
    )
