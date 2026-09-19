# RL-GAN: Adaptive Image Inpainting using Reinforcement Learning Guided GAN

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2BCUDA124-EE4C2C.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Core Research Question:** *Can a Reinforcement Learning (PPO) controller dynamically choose optimal restoration strategies (Global, Local, Boundary, Texture) for a generative inpainter based on missing hole geometry and contextual difficulty?*

---

## 🏗 System Architecture

```text
                                Damaged Image + Hole Mask
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               │                                                         │
   [DeepFillV2 Frozen Backbone]                               [State Observation Vector]
               │                                                         │
    Coarse & Refined Features                                 - Bottleneck Latent Features (256d)
               │                                              - Mask Statistics (Area, Aspect Ratio)
               │                                              - Coarse Reconstruction Error Metrics
               │                                                         │
               │                                              [PPO Contextual Bandit Policy]
               │                                                         │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                  PPO Selects Strategy
                                            │
                ┌───────────────────┬───────┴───────┬───────────────────┐
                ▼                   ▼               ▼                   ▼
            Action 0            Action 1        Action 2            Action 3
        Global Adapter      Local Adapter   Boundary Adapter    Texture Adapter
       (Context + Perc)    (Hole L1 + SSIM) (Sobel Edge Loss)   (2D FFT Freq)
                │                   │               │                   │
                └───────────────────┼───────────────┴───────────────────┘
                                    │
                                    ▼
                      Final Reconstructed Inpainting
```

### Key Components:
1. **Backbone Generator**: Pretrained **DeepFillv2** with Gated Convolutions (Places2 weights, 84 parameters, frozen).
2. **Strategy Adapters**: 4 lightweight convolution heads, each trained with specialized loss objectives:
   - **Action 0 (`Global Completion`)**: Balanced hole/valid $L_1$ + Perceptual VGG loss.
   - **Action 1 (`Local Refinement`)**: $6.0\times$ hole-focused $L_1$ + SSIM loss.
   - **Action 2 (`Boundary Refinement`)**: Sobel edge-gradient boundary ring loss.
   - **Action 3 (`Texture Refinement`)**: 2D Fast Fourier Transform (FFT) frequency magnitude loss.
3. **PPO Contextual Bandit**: Stable-Baselines3 PPO agent operating in 1-step episodes ($\gamma = 0.0$), selecting the optimal adapter to maximize:
   $$\text{Reward} = \Delta \text{PSNR} + \Delta \text{SSIM} - 0.5 \cdot \Delta \text{LPIPS} + 0.3 \cdot \Delta \text{Discriminator} - \text{ActionCost}$$

---

## 📦 Dataset & Checkpoints Download

### 1. Download COCO Dataset (Val2017)
The adapters and PPO agent are trained on real-world scenes from the **COCO 2017 Validation Set** (5,000 images):

```bash
# Automated download & extraction:
python scripts/download_coco.py
```

*Manual alternative:*
1. Download `val2017.zip` from [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip) (815 MB).
2. Extract the archive into `data/raw/val2017/`.
3. The folder should contain 5,000 `.jpg` files (`data/raw/val2017/*.jpg`).

### 2. Download Pretrained DeepFillV2 Backbone
The base inpainting generator uses pretrained Places2 weights:

```bash
# Automated download:
python scripts/download_gan.py
```
*Destination path:* `checkpoints/pretrained/deepfillv2_places2.pth`

---

## ⚡ Environment Setup

Using [`uv`](https://github.com/astral-sh/uv) (recommended):
```bash
# Clone the repository
git clone https://github.com/Hariprasaadh/rl-gan-inpainting.git
cd rl-gan-inpainting

# Install all dependencies with CUDA 12.4 support
uv sync
```

Alternatively using standard Python venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Training Guide

### Step 1: Pretrain the 4 Strategy Adapters on COCO
Trains the 4 adapters with differentiated loss functions while keeping the DeepFillV2 backbone frozen:
```bash
python main.py pretrain-adapters --config configs/pretrain_adapters.yaml
```
- **Configuration**: [configs/pretrain_adapters.yaml](configs/pretrain_adapters.yaml)
- **Output checkpoint**: `checkpoints/adapters/adapters_final.pt`

### Step 2: Verify Strategy Diversity
Checks that the 4 adapters produce distinct outputs and verifies state encoder variance before running RL:
```bash
python main.py verify-strategies --data-dir data/raw/val2017 --num-images 5 --output-dir results/strategy_verify
```

### Step 3: Train PPO Contextual Bandit Policy
Trains the PPO policy to evaluate hole attributes and dynamically select the optimal adapter:
```bash
python main.py train-rl --config configs/rl_agent_bandit.yaml --timesteps 10000
```
- **Configuration**: [configs/rl_agent_bandit.yaml](configs/rl_agent_bandit.yaml)
- **Output checkpoint**: `checkpoints/rl_agent_bandit/ppo_bandit_final.zip`
- **TensorBoard logs**: View training curves with `tensorboard --logdir logs/rl_agent_bandit`

---

## 📊 Benchmark Evaluation & Ablation

Run the 4-stage honest ablation test across diverse real-world images:
```bash
python scripts/run_multi_image_test.py
```

### 4-Stage Ablation Framework:
| Stage | Model Configuration | Method | Scientifically Valid? |
|:---:|---|---|:---:|
| **Stage A** | DeepFillV2 Backbone Only | Pretrained baseline (0 adapters) | ✅ Yes |
| **Stage B** | Fixed Adapter Baseline | Default Action 0 (Global) | ✅ Yes |
| **Stage C** | **RL-GAN Adaptive Inference** | **PPO dynamically selects adapter** | ✅ **Yes (Core Test)** |
| **Stage D** | PPO + Test-Time Refinement | 400-step AdamW fine-tuning | ⚠️ Oracle Benchmark (uses GT) |

Detailed metrics and side-by-side grids are generated in `results/multi_image_evaluation_report.md` and `results/demo_image_*.png`.

---

## 🌐 Interactive Web Application

Launch the local web studio with an interactive drawing canvas, real-time telemetry, and a before/after split slider:

```bash
python app.py
```
Open **`http://localhost:7860`** in your browser.

- **Source & Mask Studio**: Draw custom brush masks or test on curated preset photos.
- **Inference Mode**: Defaults to **Instant Forward Pass (15ms)** using GPU acceleration.
- **5-Stage Pipeline View**: Inspect Ground Truth, Masked Input, Coarse Pass, Baseline GAN, and RL-GAN Adaptive output.

---

## 📁 Repository Structure

```text
rl-gan-inpainting/
├── app.py                         # Interactive Web Application server
├── main.py                        # CLI entrypoint for training & evaluation
├── pyproject.toml                 # Project metadata and uv dependencies
├── configs/
│   ├── pretrain_adapters.yaml     # Stage 2: Adapter pretraining configuration
│   ├── rl_agent_bandit.yaml       # Stage 4: PPO Contextual Bandit configuration
│   └── pretrain_gan.yaml          # Baseline GAN configuration
├── data/
│   ├── raw/
│   │   ├── sample_images/         # Curated preset demo images
│   │   └── val2017/               # (Git-ignored) 5,000 COCO images
│   └── masks/fixed_eval_masks/    # Deterministic test masks partitioned by severity
├── checkpoints/                   # (Git-ignored) Model weights
│   ├── pretrained/                # deepfillv2_places2.pth
│   ├── adapters/                  # adapters_final.pt
│   └── rl_agent_bandit/           # ppo_bandit_final.zip
├── src/
│   ├── data/                      # Dataset loader, transforms, irregular mask generator
│   ├── models/                    # DeepFillV2, Strategy Adapters, Losses, Encoder
│   ├── rl/                        # Bandit Gym Environment, StateBuilder, Rewards
│   └── training/                  # Adapter trainer, PPO bandit trainer
├── scripts/
│   ├── download_coco.py           # Automated COCO val2017 downloader
│   ├── download_gan.py            # DeepFillV2 checkpoint downloader
│   ├── run_multi_image_test.py    # 4-stage quantitative evaluation script
│   └── verify_strategies.py       # Strategy diversity verification
└── web/
    ├── index.html                 # Web studio interface
    ├── style.css                  # Dark-mode styling
    └── app.js                     # Canvas drawing & API integration
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
