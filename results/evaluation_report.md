# RL-GAN Inpainting Evaluation Report

## 1. Model Comparison & Ablation Table

| Setting / Model | PSNR (dB) [^] | SSIM [^] | L1 Loss [v] |
|---|:---:|:---:|:---:|
| **A: GAN Baseline (Fixed Strategy)** | 15.33 | 0.683 | 0.2488 |
| **B: RL-GAN (Adaptive Controller)** | 15.33 | 0.683 | 0.2488 |
| **LaMa Reference (Non-RL)** | 13.96 | 0.432 | 0.2873 |

---

## 2. Learned Action Distribution
- **Global Completion**: 0.0% (0/1)
- **Local Refinement**: 0.0% (0/1)
- **Boundary Refinement**: 0.0% (0/1)
- **Texture Refinement**: 0.0% (0/1)

---

## 3. Mask Severity Analysis (Hypothesis Testing)
| Missing Area | Samples | GAN Baseline (PSNR) | RL-GAN (PSNR) | Delta PSNR | GAN (SSIM) | RL-GAN (SSIM) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10-20%   |       0 |   0.00 dB |   0.00 dB | +0.00 dB | 0.000 | 0.000 |
| 20-40%   |       2 |  19.49 dB |  19.49 dB | +0.00 dB | 0.828 | 0.828 |
| 40-60%   |       2 |  16.02 dB |  16.02 dB | +0.00 dB | 0.695 | 0.695 |
| 60%+     |       6 |  13.71 dB |  13.71 dB | +0.00 dB | 0.631 | 0.631 |
