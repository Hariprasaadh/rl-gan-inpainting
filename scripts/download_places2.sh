#!/usr/bin/env bash
# Download and setup Places2 Standard challenge dataset (val/test subset)
set -e

DATA_DIR="data/raw/places2"
mkdir -p "$DATA_DIR"

echo "Downloading Places2 validation subset..."
wget -c http://data.csail.mit.edu/places/places365/val_256.tar -P "$DATA_DIR"

echo "Extracting Places2 dataset..."
tar -xf "$DATA_DIR/val_256.tar" -C "$DATA_DIR"

echo "Places2 dataset downloaded and extracted to $DATA_DIR"
