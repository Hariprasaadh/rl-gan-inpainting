# RL-GAN: Adaptive Image Inpainting using Reinforcement Learning Guided GAN


> **Core Research Question:** *Can a Reinforcement Learning (PPO) controller dynamically choose optimal restoration strategies (Global, Local, Boundary, Texture) for a generative inpainter based on missing hole geometry and contextual difficulty, and does that adaptive control help more as the missing region gets harder to reconstruct?*

> [!TIP]
> 🎓 **Preparing to present or defend this project?** See the complete [Faculty Presentation & Defense Guide](FACULTY_PRESENTATION_GUIDE.md) for speaking scripts, slide outlines, mathematical derivations, live demo instructions, and tough Q&A answers. The written project report (BCSE432E) is in [`docs/`](docs/) as `RL_GAN_Inpainting_Report.pdf` / `.docx`.

> [!NOTE]
> 📖 **How it works:** step-by-step explanations of the model, training, evaluation and web studio are in [`docs/`](docs/README.md) ([Getting Started](docs/GETTING_STARTED.md) · [Architecture](docs/ARCHITECTURE.md) · [Training & RL](docs/TRAINING_AND_RL.md) · [Evaluation](docs/EVALUATION.md) · [Web Studio](docs/WEB_STUDIO.md)).

---

## 📑 Table of Contents

- [Overview & Key Innovations](#-overview--key-innovations)
- [System Architecture](#-system-architecture)
- [Mathematical Formulation](#-mathematical-formulation)
  - [State Space (261-d)](#1-state-space-s_t-in-mathbbr261)
  - [Action Space & Cost Regularization](#2-action-space-mathcala-and-action-costs)
  - [Multi-Objective Improvement Reward](#3-multi-objective-improvement-reward)
  - [Specialized Strategy Adapter Objectives](#4-specialized-strategy-adapter-objectives)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Data & Checkpoint Download](#-data--checkpoint-download)
- [End-to-End Workflow & Training Guide](#-end-to-end-workflow--training-guide)
  - [Phase 1: Dataset & Evaluation Set Generation](#phase-1-dataset--evaluation-set-generation)
  - [Phase 2: Strategy Adapter Pretraining](#phase-2-strategy-adapter-pretraining)
  - [Phase 3: Strategy Diversity & State Variance Verification](#phase-3-strategy-diversity--state-variance-verification)
  - [Phase 4: PPO Contextual Bandit Policy Training](#phase-4-ppo-contextual-bandit-policy-training)
  - [Phase 5: Benchmark Evaluation & Ablation Analysis](#phase-5-benchmark-evaluation--ablation-analysis)
- [Benchmark Results & Empirical Analysis](#-benchmark-results--empirical-analysis)
- [Interactive Web Studio](#-interactive-web-studio)
- [CLI Reference](#-cli-reference)
- [Configuration Guide](#-configuration-guide)
- [License](#-license)

---

## 💡 Overview & Key Innovations

Standard deep image inpainting models apply a single static generative policy to every missing region regardless of whether the defect is a thin scratch, an irregular boundary seam, a flat color patch, or a complex textured hole.

**RL-GAN** introduces an adaptive reinforcement learning control loop on top of a generative inpainting backbone:
1. **Cost-Free Bottleneck State Extraction**: Instead of running an auxiliary vision backbone, the state encoder taps directly into the DeepFill-v2 coarse bottleneck features (96 channels = 2·`cnum`) and pools them into a 256-d latent embedding, combined with geometric mask descriptors and boundary gradient discontinuity (261-d total observation vector).
2. **Specialized Strategy Adapters**: 4 lightweight residual convolution heads (operating on the RGB output of the backbone's refine stage) modulated via FiLM (Feature-wise Linear Modulation), pretrained with differentiated objectives targeting global context, hole reconstruction, boundary gradient continuity, and high-frequency (FFT) fidelity.
3. **PPO Contextual Bandit Controller**: A Stable-Baselines3 PPO agent trained on COCO val2017 images to select an adapter head per mask, trading off a quality-improvement reward against a hand-set action cost.
4. **Honest 4-Stage Ablation Framework**: Clear scientific decoupling between frozen backbone baselines (Stage A), fixed adapter baselines (Stage B), pure feed-forward RL inference (Stage C), and test-time ground-truth-supervised optimization (Stage D).

---

## 🏗 System Architecture

```text
                                 Damaged Image [3, H, W] + Hole Mask [1, H, W]
                                                       │
                 ┌─────────────────────────────────────┴─────────────────────────────────────┐
                 │                                                                           │
   [DeepFillV2 Frozen Backbone]                                                [State Observation Builder]
   - Gated Convolutions                                                         - Backbone Bottleneck Latent (256d)
   - Contextual Attention                                                       - Mask Statistics: Area, BBox, (4d)
   - Coarse Pass + Refine Pass                                                    Num Regions, Center Distance
                 │                                                              - Boundary Gradient Discontinuity (1d)
                 │                                                                           │
                 │ Coarse Composite (x_coarse)                                               ▼
                 │ + Refined Features (x_refine)                                  State Vector s_t ∈ ℝ²⁶¹
                 │                                                                           │
                 │                                                             [PPO Contextual Bandit Policy]
                 │                                                             (Actor-Critic Network, γ = 0.0)
                 │                                                                           │
                 └──────────────────────────────┬────────────────────────────────────────────┘
                                                │
                                      PPO Selects Action a_t
                                                │
                 ┌──────────────────────┬───────┴──────────────┬──────────────────────┐
                 ▼                      ▼                      ▼                      ▼
             Action 0               Action 1               Action 2               Action 3
          Global Adapter         Local Adapter          Boundary Adapter       Texture Adapter
          (Context + Perc)      (Hole L1 + SSIM)       (Sobel Boundary Ring)   (2D FFT Magnitude)
          Cost: 0.00             Cost: 0.10             Cost: 0.15             Cost: 0.20
                 │                      │                      │                      │
                 └──────────────────────┴───────┬──────────────┴──────────────────────┘
                                                │
                                                ▼
                                   Blended Final Composite
                            I_final = (1 - M) ⊙ I_gt + M ⊙ I_adapted
```

### Flow Diagram

```mermaid
graph TD
    A["Damaged Image + Hole Mask"] --> B["DeepFill-v2 Stage 1 (Coarse)"]
    A --> C["StateBuilder (261-d Vector)"]
    B --> D["DeepFill-v2 Stage 2 (Refine)"]
    B --> C
    C --> E["PPO Policy Network"]
    E -->|Selects Strategy| F{Action a_t}
    F -->|Action 0| G["Global Adapter (Context + Perceptual)"]
    F -->|Action 1| H["Local Adapter (6x Hole L1 + SSIM)"]
    F -->|Action 2| I["Boundary Adapter (Sobel Ring Loss)"]
    F -->|Action 3| J["Texture Adapter (2D FFT Magnitude)"]
    D --> G
    D --> H
    D --> I
    D --> J
    G --> K["Feathered Final Composition"]
    H --> K
    I --> K
    J --> K
```

---

## 📐 Mathematical Formulation

### 1. State Space $s_t \in \mathbb{R}^{261}$

The state representation $s_t = [z_t, m_t, g_t]$ encapsulates semantic context, hole geometry, and boundary quality:

1. **Latent Bottleneck Embedding** $z_t \in \mathbb{R}^{256}$:
   Extracted directly from the frozen DeepFill-v2 coarse bottleneck ($2 \times c_{\text{num}} = 96$ channels, output of the last dilated layer) via adaptive average pooling and a lightweight projection head:
   $$z_t = \text{EncoderHead}(\text{BottleneckFeature}(I_{\text{masked}}, M))$$
2. **Mask Geometry Statistics** $m_t \in \mathbb{R}^{4}$:
   - **Missing Ratio**: $\frac{\sum M}{H \times W}$ (fraction of pixels damaged)
   - **Bounding Box Ratio**: $\frac{\text{Area}(\text{bbox}(M))}{H \times W}$
   - **Number of Disjoint Regions**: Connected components via 8-connectivity, divided by 10 and clipped to $[0, 1]$
   - **Centroid Distance**: Normalized distance from mask center of mass to image center
3. **Boundary Discontinuity Proxy** $g_t \in \mathbb{R}^{1}$:
   Sobel edge gradient magnitude on a one-pixel ring just outside the hole, divided by 2 and clipped to $[0, 1]$:
   $$g_t = \text{clip}\left(\frac{1}{|\partial M|} \sum_{(x,y) \in \partial M} \|\nabla I_{\text{coarse}}(x,y)\|_2 \Big/ 2,\; 0,\; 1\right)$$

### 2. Action Space $\mathcal{A}$ and Action Costs

The action space consists of 4 discrete restoration strategies with associated computational cost penalties $C(a_t)$:

| Action | Strategy Name | FiLM Modulation | Primary Focus | Computation Cost $C(a_t)$ |
|:---:|---|---|---|:---:|
| `0` | **Global Completion** | Contextual FiLM | Broad scene balance, semantic consistency | `0.00` (Free baseline) |
| `1` | **Local Refinement** | Hole-weighted FiLM | Hole-interior fidelity, structural coherence | `0.10` |
| `2` | **Boundary Refinement** | Boundary-ring FiLM | Seam suppression, gradient continuity | `0.15` |
| `3` | **Texture Refinement** | Laplacian HF FiLM | High-frequency detail, stochastic patterns | `0.20` |

### 3. Multi-Objective Improvement Reward

The agent operates in a 1-step contextual bandit environment ($\gamma = 0.0$). The reward signal $R_t$ rewards quality gains relative to the coarse stage while penalizing heavy computations:

$$R_t = \alpha \cdot \Delta\widetilde{\text{PSNR}} + \beta \cdot \Delta\widetilde{\text{SSIM}} - \gamma \cdot \Delta\widetilde{\text{LPIPS}} + \delta \cdot \Delta\text{Disc} - \lambda \cdot C(a_t)$$

where deltas are normalized to avoid PSNR domination:
- $\Delta\widetilde{\text{PSNR}} = \text{clip}\left(\frac{\text{PSNR}_{\text{refined}} - \text{PSNR}_{\text{coarse}}}{10.0}, -2.0, 2.0\right)$
- $\Delta\widetilde{\text{SSIM}} = \text{clip}\left(\frac{\text{SSIM}_{\text{refined}} - \text{SSIM}_{\text{coarse}}}{0.10}, -2.0, 2.0\right)$
- $\Delta\widetilde{\text{LPIPS}} = \text{clip}\left(\frac{\text{LPIPS}_{\text{refined}} - \text{LPIPS}_{\text{coarse}}}{0.10}, -2.0, 2.0\right)$
- $\Delta\text{Disc} = D_{\text{patch}}(I_{\text{refined}}, M) - D_{\text{patch}}(I_{\text{coarse}}, M)$ (Spectral-Normalized PatchGAN confidence)
- Total reward is clipped to $[-10, 10]$.
- Training weights (`configs/rl_agent_bandit.yaml`): $\alpha=1.0, \beta=1.0, \gamma=0.5, \delta=0.3, \lambda=0.1$

### 4. Specialized Strategy Adapter Objectives

During Phase 2 pretraining, the DeepFill-v2 backbone is frozen and the 4 adapter heads are trained with differentiated losses (as implemented in `src/training/pretrain_adapters.py`):

$$\mathcal{L}_{\text{Global}} = \mathcal{L}_{1}^{\text{hole}} + \mathcal{L}_{1}^{\text{valid}} + \mathcal{L}_{\text{VGG}}$$

$$\mathcal{L}_{\text{Local}} = 6.0 \cdot \mathcal{L}_{1}^{\text{hole}} + 0.2 \cdot \mathcal{L}_{1}^{\text{valid}} + 0.5 \cdot \mathcal{L}_{\text{SSIM-proxy}}^{\text{hole}}$$

$$\mathcal{L}_{\text{Boundary}} = \left(\mathcal{L}_{1}^{\text{hole}} + \mathcal{L}_{1}^{\text{valid}}\right) + 2.0 \cdot \mathcal{L}_{\text{Sobel}}(\partial M)$$

$$\mathcal{L}_{\text{Texture}} = \left(\mathcal{L}_{1}^{\text{hole}} + \mathcal{L}_{1}^{\text{valid}}\right) + 0.1 \cdot \|\mathcal{F}(\hat{I}) - \mathcal{F}(I_{\text{gt}})\|_{1} + \mathcal{L}_{\text{VGG}}$$

Implementation details: $\mathcal{L}_{\text{VGG}}$ is computed by `PerceptualAndStyleLoss(perceptual_weight=0.05)`; $\mathcal{L}_{\text{SSIM-proxy}}$ is an L1 on 11×11 box-filtered means inside the hole; $\mathcal{L}_{\text{Sobel}}$ weights the Sobel-gradient L1 by 2.0 on the boundary ring and 0.2 elsewhere; the FFT term is computed over the whole image. Optimiser: Adam, lr $10^{-4}$, $\beta=(0.5, 0.999)$, 5 epochs, batch size 4.

---

## 📁 Repository Structure

```text
rl-gan-inpainting/
├── app.py                             # Interactive Web Studio server (port 7860)
├── main.py                            # Master CLI entrypoint (12 commands)
├── pyproject.toml                     # uv / pip build configuration & dependencies
├── requirements.txt                   # Flat pip dependency requirements
├── CLAUDE.md                          # Code-base guide for Claude Code (architecture, traps)
├── FACULTY_PRESENTATION_GUIDE.md      # Defense / presentation guide (checked against the code)
│
├── docs/                              # Documentation and project report
│   ├── README.md                      # Documentation index
│   ├── GETTING_STARTED.md             # Setup and pipeline commands
│   ├── ARCHITECTURE.md                # Model structure and data flow
│   ├── TRAINING_AND_RL.md             # Adapter pretraining, bandit environment, reward, PPO
│   ├── EVALUATION.md                  # Four-stage evaluation, metrics, results
│   ├── WEB_STUDIO.md                  # Web studio UI and HTTP API
│   ├── images/                        # Diagrams and figures used by the docs
│   ├── RL_GAN_Inpainting_Report.pdf   # Final report (PDF)
│   └── RL_GAN_Inpainting_Report.docx  # Final report (Word)
│
├── configs/                           # Experiment & training configuration files
│   ├── README.md                      # Detailed config documentation & schema guide
│   ├── pretrain_adapters.yaml         # Adapter pretraining config (COCO val2017)
│   ├── rl_agent_bandit.yaml           # PPO Contextual Bandit training config
│   ├── rl_agent_multistep.yaml        # Multi-step RL with STOP action config
│   ├── pretrain_gan.yaml              # Legacy GAN pretraining baseline
│   └── joint_finetune.yaml            # Joint fine-tuning configuration
│
├── data/                              # Datasets and deterministic evaluation masks
│   ├── raw/
│   │   ├── sample_images/             # 6 demo photographs (used by the web studio and ablation)
│   │   └── val2017/                   # COCO 2017 validation images (5,000; download with scripts/download_coco.py)
│   └── masks/
│       ├── fixed_eval_masks/          # 100 deterministic masks (25 per severity bucket)
│       │   ├── 10-20pct/              # Severity 1: 10% - 20% missing area
│       │   ├── 20-40pct/              # Severity 2: 20% - 40% missing area
│       │   ├── 40-60pct/              # Severity 3: 40% - 60% missing area
│       │   └── 60pctplus/             # Severity 4: >60% missing area
│       └── fixed_eval_manifest.json   # Deterministic evaluation pair manifest
│
├── checkpoints/                       # Model weights (git-ignored / downloaded)
│   ├── pretrained/
│   │   └── deepfillv2_places2.pth     # Places2-pretrained DeepFill-v2 backbone
│   ├── adapters/
│   │   ├── adapters_final.pt          # Pretrained 4-strategy adapter weights (+ strategy embeddings, encoder head)
│   │   └── adapters_epoch_*.pt        # Per-epoch adapter checkpoints (epochs 1-5)
│   └── rl_agent_bandit/
│       └── ppo_bandit_final.zip       # Trained PPO agent policy weights
│
├── src/                               # Source code modules
│   ├── models/
│   │   ├── deepfillv2.py              # Official DeepFill-v2 GatedConv backbone wrapper
│   │   ├── networks_tf.py             # TF-compatible DeepFill architecture definitions
│   │   ├── strategy_adapters.py       # 4 lightweight FiLM-modulated strategy adapters
│   │   ├── rl_inpainting_model.py     # Unified model: Backbone + Encoder + Adapters
│   │   ├── discriminator.py           # SNPatchGAN discriminator with realism score
│   │   ├── encoder.py                 # DeepFill bottleneck state encoder head
│   │   ├── generator.py               # Earlier small from-scratch generator
│   │   └── losses.py                  # Masked L1, Perceptual, Style, Adversarial losses
│   ├── rl/
│   │   ├── env_bandit.py              # Gymnasium Contextual Bandit environment (1-step)
│   │   ├── env_multistep.py           # Multi-step environment with STOP action
│   │   ├── state.py                   # StateBuilder (261-dim state observation vector)
│   │   ├── actions.py                 # Action constants, names, and computation costs
│   │   └── reward.py                  # Normalized multi-objective reward calculation
│   ├── training/
│   │   ├── pretrain_adapters.py       # Pretrains 4 adapters with differentiated losses
│   │   ├── train_rl_bandit.py         # PPO bandit trainer using Stable-Baselines3
│   │   ├── train_rl_ablations.py      # Ablation experiments C & D (earlier small generator)
│   │   ├── joint_finetune.py          # Joint fine-tuning (earlier small generator)
│   │   ├── pretrain_gan.py            # From-scratch GAN pretraining (earlier small generator)
│   │   └── trainer_utils.py           # Checkpointing, seed, config, device utils
│   ├── data/
│   │   ├── dataset.py                 # InpaintingDataset supporting real & synthetic pairs
│   │   ├── mask_generator.py          # Irregular stroke & shape mask generation
│   │   └── transforms.py              # Normalization, denormalization, feather blending
│   ├── evaluation/
│   │   ├── metrics.py                 # PSNR, SSIM, LPIPS streaming metric evaluator
│   │   ├── severity_eval.py           # Severity-bucketed evaluation pipeline
│   │   └── evaluate.py                # Comparative evaluation across models
│   ├── baselines/
│   │   └── lama_inference.py          # LaMa-style ResNet-FFC reference network
│   └── utils/
│       ├── logger.py                  # Console & TensorBoard metric loggers
│       └── seed.py                    # Deterministic reproducibility seeds
│
├── scripts/                           # Standalone utility & evaluation scripts
│   ├── README.md                      # Guide to all automation scripts
│   ├── download_coco.py               # Automated COCO val2017 downloader & unpacker
│   ├── download_gan.py                # Pretrained DeepFill-v2 Google Drive downloader
│   ├── create_eval_set.py             # Generates 100 balanced test masks & manifest
│   ├── evaluate_baseline.py           # Runs Experiment A baseline (frozen backbone)
│   ├── verify_strategies.py           # Pre-PPO checkpoint: state & strategy diversity
│   └── run_multi_image_test.py        # Runs 4-stage quantitative ablation benchmark
│
├── results/                           # Evaluation outputs, metrics, and visual grids
│   ├── multi_image_evaluation_report.md  # Detailed 4-stage ablation report
│   ├── evaluation_report.md              # Baseline vs RL-GAN severity breakdown
│   ├── baseline_test/                    # Experiment A baseline evaluation results
│   │   ├── baseline_metrics.json         # Raw metrics JSON
│   │   └── baseline_report.md            # Markdown summary
│   ├── strategy_verify/                  # Strategy diversity verification outputs
│   │   ├── verify_results.json           # Verification metrics per strategy
│   │   └── verify_image_*.png            # Verification comparison grids
│   └── demo_image_*.png                  # Labeled multi-stage visual comparison panels
│
└── web/                               # Interactive Web Studio frontend
    ├── index.html                     # HTML5 canvas studio layout
    ├── style.css                      # Cybernetic dark-mode interface styling
    └── app.js                         # Canvas brush drawing, slider, and API client
```

---

## ⚡ Installation & Environment Setup

### Method 1: Using `uv` (Recommended - Ultra Fast)

```bash
# Clone the repository
git clone https://github.com/Hariprasaadh/rl-gan-inpainting.git
cd rl-gan-inpainting

# Install all dependencies with PyTorch CUDA 12.4
uv sync
```

### Method 2: Standard Python Virtual Environment

Requires Python 3.10 or newer (the project was developed on Python 3.12 with PyTorch 2.6 + CUDA 12.4).

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies:
pip install -r requirements.txt
```

### Verification

Verify your PyTorch and CUDA installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 📦 Data & Checkpoint Download

### 1. Download Pretrained DeepFill-v2 Places2 Backbone

Automated download via Google Drive:
```bash
python scripts/download_gan.py
```
*Destination path:* `checkpoints/pretrained/deepfillv2_places2.pth`

*Manual Alternative:*
Download `states_pt_places2.pth` from [nipponjo/deepfillv2-pytorch](https://github.com/nipponjo/deepfillv2-pytorch) and save it to `checkpoints/pretrained/deepfillv2_places2.pth`.

### 2. Download COCO val2017 Dataset

Automated download and extraction (5,000 real-world scenes, ~815 MB):
```bash
python scripts/download_coco.py
```
*Destination path:* `data/raw/val2017/*.jpg`

*Manual Alternative:*
Download `val2017.zip` from [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip) and extract it into `data/raw/val2017/`.

---

## 🚀 End-to-End Workflow & Training Guide

### Phase 1: Dataset & Evaluation Set Generation

Create 100 fixed, reproducible evaluation masks partitioned evenly into 4 difficulty buckets (25 masks each for 10-20%, 20-40%, 40-60%, and 60%+):

```bash
python main.py create-eval-set --image-dir data/raw/val2017 --output-dir data/masks/fixed_eval_masks --count 25 --image-size 256
```

### Phase 2: Strategy Adapter Pretraining

Pretrain the 4 specialized adapter heads while keeping the DeepFill-v2 backbone frozen:

```bash
python main.py pretrain-adapters --config configs/pretrain_adapters.yaml
```

- **Objective**: Each adapter learns to specialize under its assigned loss function.
- **Checkpoint Output**: `checkpoints/adapters/adapters_final.pt`
- **TensorBoard Logs**: `tensorboard --logdir logs/adapters`

### Phase 3: Strategy Diversity & State Variance Verification

Run the pre-PPO safety checkpoint to guarantee that state embeddings vary across scenes and adapters produce genuinely distinct outputs:

```bash
python main.py verify-strategies --data-dir data/raw/sample_images --num-images 5 --output-dir results/strategy_verify
```

- **Check 1 (state variance)**: the script prints the mean pairwise L2 distance between state embeddings across images (target $> 0.10$).
- **Check 2 (strategy diversity)**: the gate passes when no single strategy has the best PSNR on more than 80% of test scenes (adapters are differentiated).

### Phase 4: PPO Contextual Bandit Policy Training

Train the Stable-Baselines3 PPO agent to observe the 261-d state vector and pick the optimal adapter:

```bash
python main.py train-rl --config configs/rl_agent_bandit.yaml --timesteps 10000
```

- **Key Hyperparameters**: $\gamma = 0.0$ (1-step contextual bandit), entropy coefficient $= 0.05$, learning rate $= 3 \times 10^{-4}$. The config default is 5,000 timesteps; `--timesteps 10000` overrides it.
- **Checkpoint Output**: `checkpoints/rl_agent_bandit/ppo_bandit_final.zip`
- **TensorBoard Logs**: `tensorboard --logdir logs/rl_agent_bandit`

### Phase 5: Benchmark Evaluation & Ablation Analysis

#### 1. Baseline Backbone Evaluation (Experiment A)
```bash
python main.py eval-baseline --data-dir data/raw/sample_images --mask-dir data/masks/fixed_eval_masks --samples 100 --output-dir results/baseline_test
```

#### 2. Full 4-Stage Honest Ablation Test
```bash
python scripts/run_multi_image_test.py
```
This produces side-by-side visual panels (`results/demo_image_*.png`) and generates `results/multi_image_evaluation_report.md`. It runs three test scenes with hand-designed masks.

---

## 📊 Benchmark Results & Empirical Analysis

### 4-Stage Ablation Framework

| Stage | Model Configuration | Description | Valid Generalization Test? |
|:---:|---|---|:---:|
| **Stage A** | **DeepFillV2 Backbone Only** | Pretrained Places2 weights without adapters or RL | ✅ Yes |
| **Stage B** | **Fixed Adapter Baseline** | Default static selection: Action 0 (Global Completion) | ✅ Yes |
| **Stage C** | **RL-GAN Adaptive Inference** | **PPO dynamically chooses adapter based on state** | ✅ **Yes (Core Research Test)** |
| **Stage D** | PPO + test-time refinement | 400 AdamW steps using the ground-truth image in the loss | Oracle upper bound (uses ground truth) |

### Quantitative Comparison Across Representative Scenes

| Scene Name | Primary Challenge | RL Selected Action | A: Backbone | B: Fixed Adapter | C: PPO-Selected (Ours) | D: PPO + Refine |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Alpine Lake & Snowy Mountains** | Water reflections, forest texture | **Action 2: Boundary** | 34.45 dB / 0.9768 | 35.21 dB / 0.9784 | **32.25 dB / 0.9730** | 37.23 dB / 0.9828 |
| **Lush Alpine Meadow & Peaks** | Large grass gradient, clouds | **Action 0: Global** | 33.72 dB / 0.9703 | 34.54 dB / 0.9734 | **34.54 dB / 0.9734** (+0.82 dB) | 39.65 dB / 0.9872 (+5.94 dB) |
| **Sunset Mountain Horizon** | Sharp sky gradient, rock ridge | **Action 2: Boundary** | 40.58 dB / 0.9908 | 40.00 dB / 0.9900 | **36.80 dB / 0.9834** | 43.19 dB / 0.9946 (+2.61 dB) |

Change relative to the backbone (Stage A): B = +0.76 / +0.82 / −0.58 dB; C = **−2.21 / +0.82 / −3.78 dB**; D = +2.78 / +5.94 / +2.61 dB (lake / meadow / sunset).

### Key Empirical Takeaways:
1. **Adapters refine the backbone**: the fixed Global adapter (Stage B) improves on the backbone by about +0.8 dB on two of the three scenes.
2. **Per-scene strategy choice**: the policy chose Global on the meadow scene (so Stage C = Stage B) and Boundary on the lake and sunset scenes. In the five-image strategy verification the Local adapter had the best PSNR on 3 of 5 images and Global on 2 of 5.
3. **Scientifically honest comparison**: Stage C is a single feed-forward pass without ground truth; Stage D uses the ground-truth image and shows the capacity of the adapters (37–43 dB).
4. **Severity analysis**: `SeverityEvaluator` and the 100 fixed masks (25 per bucket) provide the framework for studying how adaptive control behaves as the hole grows.

---

## 🌐 Interactive Web Studio

RL-GAN includes a high-performance interactive web application with a responsive drawing studio, real-time telemetry, and visual comparison sliders:

```bash
python app.py
```
Open **`http://localhost:7860`** in your browser.

![Web Studio](results/demo_image_2_lush_meadow.png)

### Features:
- **Interactive Mask Canvas**: Draw custom irregular brush strokes with adjustable diameter or clear with one click.
- **Preset Image Showcase**: Test on the sample photographs in `data/raw/sample_images` (or upload your own); a random defect is pre-drawn on load.
- **Dual Processing Modes**:
  - ⚡ **Instant Forward Pass**: one GPU forward pass using the PPO-selected adapter. Measured latency on an RTX 3050 was about 235 ms per request (including an extra backbone-only baseline pass and PSNR/SSIM computation).
  - 🎯 **Ultra-Precision**: 60 AdamW steps on the chosen adapter with an L1 loss against the original image, which shows how closely an adapter can fit a given image.
- **Strategy choice**: the PPO agent chooses the adapter. The `/api/inpaint` endpoint also accepts an optional `action` (0–3) field to force a strategy.
- **Interactive Split Slider**: Drag between before and after inpainting.
- **5-Stage Pipeline Inspector**: View Ground Truth, Masked Input, Coarse Stage, Backbone baseline, and RL-GAN output side-by-side.
- **Live Telemetry Cards**: Chosen strategy, PSNR/SSIM versus the backbone baseline, damaged-area percentage and latency.

---

## 💻 CLI Reference

The master CLI `main.py` provides 12 subcommands:

```bash
python main.py <command> [options]
```

| Subcommand | Description | Key Arguments |
|---|---|---|
| `create-eval-set` | Generate 100 fixed balanced eval pairs & manifest | `--image-dir`, `--output-dir`, `--count` (25), `--image-size` (256), `--seed` (42) |
| `generate-eval-masks` | Generate fixed deterministic masks only | `--output-dir`, `--count`, `--image-size`, `--seed` |
| `download-deepfill` | Print DeepFill-v2 download instructions | None |
| `pretrain-adapters` | Pretrain 4 strategy adapters with specialized losses | `--config` (`configs/pretrain_adapters.yaml`) |
| `verify-strategies` | Verify strategy diversity & state variance before PPO | `--backbone-checkpoint`, `--adapters-checkpoint`, `--data-dir`, `--num-images` (5), `--state-only` |
| `train-rl` | Train PPO Contextual Bandit controller | `--config` (`configs/rl_agent_bandit.yaml`), `--timesteps` |
| `eval-baseline` | Evaluate DeepFill-v2 backbone baseline (no RL) | `--backbone-checkpoint`, `--data-dir`, `--mask-dir`, `--samples` (100), `--compute-lpips` |
| `evaluate` | Evaluation across models and severity buckets | `--data-dir`, `--mask-dir`, `--gan-checkpoint`, `--rl-checkpoint`, `--samples` (30) |
| `demo` | Run demo on a sample image and save visual grid | `--image-dir`, `--gan-checkpoint`, `--rl-checkpoint`, `--output-dir`, `--sample-idx` (0) |
| `train-gan` | Pretrain the earlier small baseline GAN from scratch | `--config` (`configs/pretrain_gan.yaml`), `--epochs` |
| `train-ablations` | Ablation experiments C & D on the earlier small generator | `--config`, `--mode` (`c`, `d`, `all`), `--timesteps` (2000) |
| `joint-finetune` | Joint fine-tuning on the earlier small generator | `--config` (`configs/joint_finetune.yaml`), `--epochs` |

---

## ⚙️ Configuration Guide

Experiment hyperparameter files are stored in `configs/` (see [configs/README.md](configs/README.md) for full documentation):

- **[configs/pretrain_adapters.yaml](configs/pretrain_adapters.yaml)**: Adapter pretraining configuration on COCO val2017 ($5$ epochs, Adam $\text{lr}=10^{-4}$).
- **[configs/rl_agent_bandit.yaml](configs/rl_agent_bandit.yaml)**: Stable-Baselines3 PPO Contextual Bandit configuration ($\gamma=0.0$, entropy coefficient $0.05$, $261$-d state space, $4$ discrete actions).
- **[configs/rl_agent_multistep.yaml](configs/rl_agent_multistep.yaml)**: Multi-step RL configuration with an explicit STOP action (earlier small generator).
- **[configs/pretrain_gan.yaml](configs/pretrain_gan.yaml)**: Standalone GAN baseline pretraining configuration (earlier small generator).
- **[configs/joint_finetune.yaml](configs/joint_finetune.yaml)**: Joint fine-tuning parameters (earlier small generator).

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).
