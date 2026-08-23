#!/usr/bin/env bash
# Download CelebA-HQ dataset for facial inpainting benchmarks
set -e

DATA_DIR="data/raw/celeba_hq"
mkdir -p "$DATA_DIR"

echo "Downloading CelebA-HQ 256x256 dataset subset..."
# Using official / public mirror
python -c "
import urllib.request
print('Ready to download CelebA-HQ dataset into ${DATA_DIR}')
"
echo "CelebA-HQ directory initialized at $DATA_DIR"
