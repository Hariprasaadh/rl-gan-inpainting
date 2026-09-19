# RL-GAN Multi-Image Ablation Evaluation Report

> **Architecture**: Places2-pretrained DeepFillV2 backbone + COCO-val2017-trained adapters + PPO-trained RL controller.

## Ablation Stage Key

| Stage | Description | Scientifically Valid? |
|:---:|---|:---:|
| **A** | DeepFillV2 backbone only (no adapters, no RL) | Yes |
| **B** | Fixed adapter 0 (Global) — no RL selection | Yes |
| **C** | PPO-trained adapter selection — pure inference | Yes |
| **D** | Stage C + 400-step GT-supervised AdamW refinement | Uses ground truth |

## Quantitative Ablation Table

| # | Scene | RL Action | A: Backbone | B: Fixed Adapter | C: PPO-only | D: PPO+Refine | Delta C vs A | Delta D vs A |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Alpine Lake & Snowy Mountains | Action 2: Boundary Refinement | 34.45 dB | 35.21 dB | **32.25 dB** | 37.23 dB | **-2.21 dB** | +2.78 dB |
| **2** | Lush Alpine Meadow & Distant Peaks | Action 0: Global Completion | 33.72 dB | 34.54 dB | **34.54 dB** | 39.65 dB | **+0.82 dB** | +5.94 dB |
| **3** | Sunset Mountain Horizon & Dusk Sky | Action 2: Boundary Refinement | 40.58 dB | 40.00 dB | **36.80 dB** | 43.19 dB | **-3.78 dB** | +2.61 dB |

---

## Detailed Scene Analysis

### Test 1: Alpine Lake & Snowy Mountains
- **Scene**: Turquoise mountain lake with crystal water reflections and forested ridges.
- **Source**: `beautiful-lake-mountains_395237-44.avif`
- **PPO**: Trained PPO agent
- **Selected Action**: Action 2: Boundary Refinement

| Stage | PSNR | SSIM | vs Backbone |
|---|:---:|:---:|:---:|
| A: Backbone only | 34.45 dB | 0.9768 | baseline |
| B: Fixed adapter | 35.21 dB | 0.9784 | +0.75 dB |
| C: PPO-only (pure inference) | 32.25 dB | 0.9730 | **-2.21 dB** |
| D: PPO+Refinement (GT-supervised) | 37.23 dB | 0.9828 | +2.78 dB |

- **Output**: `demo_image_1_alpine_lake.png`

### Test 2: Lush Alpine Meadow & Distant Peaks
- **Scene**: Vibrant green grassy slopes, scattered fir trees, and fluffy summer clouds.
- **Source**: `beautiful-natural-image-1844362_1280.jpg`
- **PPO**: Trained PPO agent
- **Selected Action**: Action 0: Global Completion

| Stage | PSNR | SSIM | vs Backbone |
|---|:---:|:---:|:---:|
| A: Backbone only | 33.72 dB | 0.9703 | baseline |
| B: Fixed adapter | 34.54 dB | 0.9734 | +0.82 dB |
| C: PPO-only (pure inference) | 34.54 dB | 0.9734 | **+0.82 dB** |
| D: PPO+Refinement (GT-supervised) | 39.65 dB | 0.9872 | +5.94 dB |

- **Output**: `demo_image_2_lush_meadow.png`

### Test 3: Sunset Mountain Horizon & Dusk Sky
- **Scene**: Jagged snow-covered crests under pink/purple twilight horizon gradients.
- **Source**: `photo-masthead-375-BoK_p8LG.webp`
- **PPO**: Trained PPO agent
- **Selected Action**: Action 2: Boundary Refinement

| Stage | PSNR | SSIM | vs Backbone |
|---|:---:|:---:|:---:|
| A: Backbone only | 40.58 dB | 0.9908 | baseline |
| B: Fixed adapter | 40.00 dB | 0.9900 | -0.58 dB |
| C: PPO-only (pure inference) | 36.80 dB | 0.9834 | **-3.78 dB** |
| D: PPO+Refinement (GT-supervised) | 43.19 dB | 0.9946 | +2.61 dB |

- **Output**: `demo_image_3_sunset_peaks.png`


---

## Methodology Notes

- **Stage C (PPO-only)** is the correct measurement of the RL contribution.
  Single forward pass with PPO-selected adapter — no optimization, no ground truth used.

- **Stage D (PPO + Refinement)** applies 400 AdamW steps with ground-truth in the loss.
  This is test-time supervised fine-tuning — not a valid generalization metric.

- **Backbone**: DeepFillV2 pretrained on Places2 (frozen during adapter and RL training).
- **Adapters**: Trained 3 epochs on COCO val2017 (5,000 images, synthetic masks).
- **PPO**: Trained on COCO val2017 via stable-baselines3 PPO.
