"""Download and extract COCO val2017 dataset with progress tracking."""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
RAW_DIR = os.path.join("data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "val2017.zip")
EXTRACT_DIR = RAW_DIR  # Will extract into data/raw/val2017/


def download_coco():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Check if already extracted
    final_dir = os.path.join(RAW_DIR, "val2017")
    if os.path.exists(final_dir):
        num_images = len([f for f in os.listdir(final_dir) if f.endswith(".jpg")])
        if num_images >= 4900:
            print(f"COCO val2017 already extracted at {final_dir} ({num_images} images found).")
            return

    # Check if zip exists and is valid
    need_download = True
    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
                if zf.testzip() is None:
                    print("Existing val2017.zip is valid. Skipping download.")
                    need_download = False
        except Exception:
            print("Existing val2017.zip is incomplete or corrupt. Re-downloading...")
            os.remove(ZIP_PATH)

    if need_download:
        print(f"Downloading COCO val2017 from {COCO_VAL_URL}...")
        response = requests.get(COCO_VAL_URL, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(ZIP_PATH, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="COCO val2017"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("Download complete.")

    print(f"Extracting {ZIP_PATH} to {EXTRACT_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)

    num_images = len([f for f in os.listdir(final_dir) if f.endswith(".jpg")])
    print(f"Extraction complete! Found {num_images} images in {final_dir}.")


if __name__ == "__main__":
    download_coco()
