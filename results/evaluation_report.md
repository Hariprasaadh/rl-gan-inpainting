# RL-GAN Inpainting Evaluation Report

## 1. Model Comparison & Ablation Table

| Setting / Model | PSNR (dB) [^] | SSIM [^] | L1 Loss [v] |
|---|:---:|:---:|:---:|
| **A: GAN Baseline (Fixed Strategy)** | 25.67 | 0.896 | 0.0520 |
| **B: RL-GAN (Adaptive Controller)** | 25.48 | 0.894 | 0.0542 |
| **LaMa Reference (Non-RL)** | 19.73 | 0.848 | 0.0630 |

---

## 2. Learned Action Distribution
- **Global Completion**: 0.0% (0/6)
- **Local Refinement**: 100.0% (6/6)
- **Boundary Refinement**: 0.0% (0/6)
- **Texture Refinement**: 0.0% (0/6)

---

## 3. Mask Severity Analysis (Hypothesis Testing)
| Missing Area | Samples | GAN Baseline (PSNR) | RL-GAN (PSNR) | Delta PSNR | GAN (SSIM) | RL-GAN (SSIM) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10-20%   |       5 |  27.12 dB |  26.94 dB | -0.17 dB | 0.907 | 0.907 |
| 20-40%   |       1 |  18.42 dB |  18.13 dB | -0.29 dB | 0.836 | 0.833 |
| 40-60%   |       0 |   0.00 dB |   0.00 dB | +0.00 dB | 0.000 | 0.000 |
| 60%+     |       0 |   0.00 dB |   0.00 dB | +0.00 dB | 0.000 | 0.000 |
