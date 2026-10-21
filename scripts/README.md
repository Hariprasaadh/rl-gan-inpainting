# RL-GAN Utility & Evaluation Scripts

This directory contains standalone Python scripts for data preparation, checkpoint acquisition, validation, and benchmark evaluation.

---

## 📜 Script Index

| Script | Purpose | Key Inputs | Primary Outputs |
|---|---|---|---|
| [`download_coco.py`](download_coco.py) | Downloads & unpacks COCO val2017 dataset (5,000 images, ~815 MB) | Automated HTTP download | `data/raw/val2017/*.jpg` |
| [`download_gan.py`](download_gan.py) | Downloads pretrained DeepFill-v2 Places2 backbone weights via gdown | Google Drive File ID | `checkpoints/pretrained/deepfillv2_places2.pth` |
| [`create_eval_set.py`](create_eval_set.py) | Generates 100 fixed evaluation masks partitioned across 4 severity buckets | Target image dir, seed | `data/masks/fixed_eval_masks/`, `data/masks/fixed_eval_manifest.json` |
| [`verify_strategies.py`](verify_strategies.py) | Critical pre-PPO safety checkpoint: verifies state variance and adapter diversity | Backbone & adapter weights | `results/strategy_verify/verify_results.json`, visual grids |
| [`evaluate_baseline.py`](evaluate_baseline.py) | Evaluates frozen DeepFill-v2 backbone (Experiment A, no adapters, no RL) | Backbone weights, test masks | `results/baseline_test/baseline_metrics.json`, `baseline_report.md` |
| [`run_multi_image_test.py`](run_multi_image_test.py) | Executes the 4-stage honest ablation test across representative scenes | All models, sample images | `results/demo_image_*.png`, `results/multi_image_evaluation_report.md` |

---

## 🛠 Detailed Usage & Workflows

### 1. `download_coco.py`

Downloads and extracts the COCO 2017 Validation Set:
```bash
python scripts/download_coco.py
```
- Requires the `requests` package.
- Automatically checks if `data/raw/val2017/` already contains $\ge 4,900$ images.
- Validates zip integrity before extraction.
- Displays a real-time progress bar.

### 2. `download_gan.py`

Downloads the official DeepFill-v2 Places2 pretrained checkpoint from Google Drive:
```bash
python scripts/download_gan.py
```
- Requires the `gdown` package (not part of `requirements.txt`; `pip install gdown`).
- Target checkpoint: `checkpoints/pretrained/deepfillv2_places2.pth`.
- Validates the downloaded file with `torch.load()` to confirm integrity.

### 3. `create_eval_set.py`

Generates 100 reproducible evaluation masks balanced across 4 severity tiers:
```bash
python main.py create-eval-set --image-dir data/raw/val2017 --output-dir data/masks/fixed_eval_masks --count 25 --image-size 256 --seed 42
# or directly (note the different flag name):
python scripts/create_eval_set.py --image-dir data/raw/val2017 --mask-output-dir data/masks/fixed_eval_masks --count 25 --image-size 256 --seed 42
```
- Creates 25 masks per bucket: `10-20pct`, `20-40pct`, `40-60pct`, `60pctplus`.
- Generates `data/masks/fixed_eval_manifest.json` pairing the first 100 images (sorted) with masks, if the image directory exists.
- The `main.py` entry point sets up the import path automatically.

### 4. `verify_strategies.py`

Performs an essential validation checkpoint prior to training the PPO agent:
```bash
python scripts/verify_strategies.py \
  --backbone-checkpoint checkpoints/pretrained/deepfillv2_places2.pth \
  --adapters-checkpoint checkpoints/adapters/adapters_final.pt \
  --data-dir data/raw/sample_images \
  --num-images 5 \
  --output-dir results/strategy_verify
```
**Validation Criteria:**
- **State Embedding Diversity**: prints the mean pairwise L2 distance between state embeddings across images (target $> 0.10$).
- **Strategy Output Diversity**: the gate passes when no single adapter has the best PSNR on more than $80\%$ of the test images. Per-strategy PSNR/SSIM, the best-strategy counts, and a four-action reward sanity check are written to `results/strategy_verify/verify_results.json` and the console.
- Add `--state-only` to run only the embedding check.

### 5. `evaluate_baseline.py`

Evaluates the neutral DeepFill-v2 backbone without RL controller:
```bash
python scripts/evaluate_baseline.py \
  --backbone-checkpoint checkpoints/pretrained/deepfillv2_places2.pth \
  --data-dir data/raw/sample_images \
  --mask-dir data/masks/fixed_eval_masks \
  --samples 100 \
  --output-dir results/baseline_test
```
Computes mean and standard deviation for PSNR, SSIM, LPIPS (add `--compute-lpips` for the VGG-based LPIPS), L1, and hole-interior L1 over `min(dataset size, --samples)` images. Outputs: `baseline_metrics.json` and `baseline_report.md` in `--output-dir`.

### 6. `run_multi_image_test.py`

Executes the complete 4-stage ablation test:
```bash
python scripts/run_multi_image_test.py
```
- Compares:
  - **Stage A**: DeepFill-v2 Backbone Only
  - **Stage B**: Fixed Adapter 0 (Global Completion)
  - **Stage C**: PPO-selected adapter (pure feed-forward inference)
  - **Stage D**: PPO-selected adapter + 400-step test-time refinement (AdamW over all trainable parameters, with the ground-truth image in the loss; an oracle upper bound)
- Uses three hand-designed masks (`build_controlled_mask`) on sample-image indices 1, 2 and 5.
- Uses `checkpoints/pretrained/deepfillv2_places2.pth`, `checkpoints/adapters/adapters_final.pt` and `checkpoints/rl_agent_bandit/ppo_bandit_final.zip`.
- Generates labeled dark-banner side-by-side comparison grids (`results/demo_image_1_alpine_lake.png`, etc.) and a quantitative markdown summary report.
