#!/usr/bin/env bash
# Download pretrained LaMa inpainting weights for baseline comparisons
set -e

CKPT_DIR="checkpoints/lama"
mkdir -p "$CKPT_DIR"

echo "Downloading LaMa (Big-LaMa) checkpoint..."
curl -L -o "$CKPT_DIR/big-lama.pt" "https://github.com/advimman/lama/releases/download/v1.0.0/big-lama.pt" || echo "Note: Use local baseline model if download is blocked."

echo "LaMa baseline weights ready in $CKPT_DIR"
