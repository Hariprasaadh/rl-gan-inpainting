# Experiment Configurations (`configs/`)

This directory contains YAML configuration files controlling each stage of training, adapter fine-tuning, and reinforcement learning.

---

## 📑 Configuration Files Overview

| Configuration File | Stage / Purpose | Target Model | Key Hyperparameters |
|---|---|---|---|
| [`pretrain_adapters.yaml`](pretrain_adapters.yaml) | Pretrains 4 strategy adapters | Adapter Heads (`Global`, `Local`, `Boundary`, `Texture`) | `epochs: 5`, `lr: 0.0001`, `batch_size: 4` |
| [`rl_agent_bandit.yaml`](rl_agent_bandit.yaml) | Trains PPO contextual bandit | Stable-Baselines3 PPO Controller | `gamma: 0.0`, `ent_coef: 0.05`, `lr: 0.0003`, `total_timesteps: 5000` (the README's 10000 comes from `--timesteps 10000`) |
| [`rl_agent_multistep.yaml`](rl_agent_multistep.yaml) | Multi-step RL with STOP action (earlier small generator) | Multi-step PPO Agent | `max_steps: 4`, `gamma: 0.95`, `ent_coef: 0.02` |
| [`pretrain_gan.yaml`](pretrain_gan.yaml) | Standalone baseline GAN pretraining (earlier small generator) | Small from-scratch `InpaintingGenerator` | `epochs: 100`, `lr_g: 0.0002`, `lr_d: 0.0002` |
| [`joint_finetune.yaml`](joint_finetune.yaml) | Joint fine-tuning (earlier small generator) | Small generator + discriminator + PPO | `epochs: 3`, `lr_g: 0.00005`, `lr_d: 0.00005` |

> `pretrain_adapters.yaml` and `rl_agent_bandit.yaml` drive the DeepFill-v2 pipeline described in the project report and in [`docs/`](../docs/README.md). The other three configure the earlier small-generator experiments.

---

## ⚙️ Detailed Configuration Schemas

### 1. `pretrain_adapters.yaml`
Used by: `python main.py pretrain-adapters --config configs/pretrain_adapters.yaml`

```yaml
data:
  image_dir: "data/raw/val2017"      # Training image directory
  image_size: 256                    # Input resolution (H, W)
  batch_size: 4                      # Training batch size
  num_workers: 2                     # DataLoader worker threads
  synthetic_size: 200                # Synthetic mask batch count fallback

model:
  cnum: 48                           # Base channel multiplier (DeepFill-v2 standard)
  cnum_in: 5                         # 5-channel input: [masked_img (3), ones (1), mask (1)]
  strategy_dim: 64                   # Dimensionality of FiLM modulation vector
  latent_dim: 256                    # Latent state embedding dimension
  backbone_checkpoint: "checkpoints/pretrained/deepfillv2_places2.pth"

training:
  epochs: 5                          # Training epochs
  lr: 0.0001                         # Adam learning rate for adapter parameters
  device: "cuda"                     # "cuda" or "cpu"
  save_dir: "checkpoints/adapters"   # Output directory for adapter checkpoints
  log_dir: "logs/adapters"           # TensorBoard log path
  save_every: 1                      # Checkpoint frequency (epochs)
```

---

### 2. `rl_agent_bandit.yaml`
Used by: `python main.py train-rl --config configs/rl_agent_bandit.yaml`

```yaml
data:
  image_dir: "data/raw/val2017"
  mask_dir: null                     # Dynamic irregular mask generation
  image_size: 256
  synthetic_size: 200                # Number of procedural images when image_dir is not provided

model:
  cnum: 48
  cnum_in: 5
  strategy_dim: 64
  latent_dim: 256
  base_channels: 32                  # Used by the earlier small-generator path
  backbone_checkpoint: "checkpoints/pretrained/deepfillv2_places2.pth"
  adapters_checkpoint: "checkpoints/adapters/adapters_final.pt"
  checkpoint_gan: "checkpoints/gan_baseline/best_model.pt"  # optional fallback for the earlier small generator

rl:
  total_timesteps: 5000              # Total environment interaction steps (CLI --timesteps overrides)
  n_steps: 128                       # Steps collected per PPO rollout
  batch_size: 32                     # Minibatch size for PPO gradient updates
  n_epochs: 4                        # Optimization epochs per rollout
  learning_rate: 0.0003              # PPO policy and value network learning rate
  gamma: 0.0                         # 0.0 for Contextual Bandit (single-step episodes)
  ent_coef: 0.05                     # Entropy coefficient (encourages action exploration)
  vf_coef: 0.5                       # Value function loss coefficient
  max_grad_norm: 0.5                 # Gradient clipping threshold
  reward_weights: [1.0, 1.0, 0.5, 0.3, 0.1] # [alpha dPSNR, beta dSSIM, gamma dLPIPS (subtracted), delta dDisc, lambda ActionCost (subtracted)]
  compute_lpips: true                # Whether to compute perceptual loss in reward
  device: "cuda"
  seed: 42
  save_dir: "checkpoints/rl_agent_bandit"
  log_dir: "logs/rl_agent_bandit"
```

---

Notes: training builds the model with seed 42, loads the backbone and adapters, freezes the adapters, and reads the reward weights from the YAML value above.

---

### 3. `rl_agent_multistep.yaml`
Used for multi-step refinement episodes (`InpaintingMultiStepEnv`, built on the earlier small generator) where the agent can perform up to $N$ refinement steps before executing Action 4 (`STOP`).

```yaml
data:
  image_size: 128
  synthetic_size: 200

model:
  base_channels: 32                  # Earlier small generator
  strategy_dim: 64
  latent_dim: 256
  checkpoint_gan: "checkpoints/gan_baseline/best_model.pt"

rl:
  max_steps: 4                       # Maximum refinement iterations per episode
  total_timesteps: 5000
  gamma: 0.95                        # Discount factor for multi-step returns
  ent_coef: 0.02                     # Exploration over 5 actions
  reward_weights: [1.0, 1.0, 0.5, 0.5, 0.3]
  compute_lpips: false
  device: "cpu"
  save_dir: "checkpoints/rl_agent_multistep"
```
There is no `step_penalty` key; Action 4 (`STOP`) ends the episode with zero reward and zero cost.

---

### 4. `pretrain_gan.yaml` and `joint_finetune.yaml`
Configure the earlier small from-scratch generator (`base_channels: 32`, 128×128 images). `pretrain_gan.yaml` sets 100 epochs, Adam `lr_g = lr_d = 2e-4`, loss weights (`hole_weight: 6.0`, `perceptual_weight: 0.05`, `style_weight: 120.0`, `adv_weight: 0.001`). `joint_finetune.yaml` sets 3 epochs, `lr_g = lr_d = 5e-5`, `rl_steps_per_epoch: 512` and a divergence guard `max_divergence_loss: 25.0`.
