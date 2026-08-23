# Adaptive Image Inpainting using Reinforcement Learning Guided GAN (RL-GAN)
## Technical Implementation Plan — v2 (revised per design review)

This document is a complete, code-oriented implementation plan. Hand it to an AI coding IDE (Claude Code, Cursor, Copilot Workspace, etc.) one section at a time. Changes from v1 are marked **[REVISED]**.

**Core research question (keep this in view at every stage):** *Can an RL controller choose the appropriate restoration strategy for a GAN generator, and does that adaptive control help more as the missing region gets harder to reconstruct?* — not "did we build the best inpainting model," and not "can RL optimize several action dimensions at once."

---

## 1. Project Goal (one line)

Train an RL agent that observes a masked image and decides *how* a frozen/near-frozen GAN generator should complete it (which strategy to apply), optimizing a reward built from quality improvement (ΔPSNR, ΔSSIM, −ΔLPIPS, ΔDiscriminator confidence) minus an action cost — instead of applying one fixed strategy to every mask.

---

## 2. Repository Structure

```
rl-gan-inpainting/
├── configs/
│   ├── pretrain_gan.yaml
│   ├── rl_agent_bandit.yaml     # Version 1: 1-step contextual bandit
│   ├── rl_agent_multistep.yaml  # Version 2: multi-step with STOP
│   └── joint_finetune.yaml      # optional / advanced, see Section 7
├── data/
│   ├── raw/                     # downloaded Places2 / CelebA-HQ
│   ├── processed/                # resized, split train/val/test
│   └── masks/
│       └── fixed_eval_masks/     # deterministic masks, val/test only — see Section 4.3
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── mask_generator.py    # dynamic (train) vs fixed (eval) masks
│   │   └── transforms.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── generator.py         # gated-conv / DeepFill-style generator
│   │   ├── discriminator.py     # SN-PatchGAN
│   │   └── losses.py            # L1, perceptual, style, adversarial
│   ├── rl/
│   │   ├── env_bandit.py        # Version 1: single-step env, 5 actions, no STOP
│   │   ├── env_multistep.py     # Version 2: multi-step env with STOP
│   │   ├── state.py
│   │   ├── actions.py           # 5-action discrete space (Section 6.1)
│   │   └── reward.py            # PSNR/SSIM/LPIPS/Disc + action cost
│   ├── training/
│   │   ├── pretrain_gan.py                # Experiment A
│   │   ├── train_rl_bandit.py             # Experiment B (frozen GAN + PPO)
│   │   ├── train_rl_ablations.py          # Experiments C, D
│   │   ├── joint_finetune.py              # Experiment E — optional/advanced
│   │   └── trainer_utils.py
│   ├── baselines/
│   │   └── lama_inference.py    # wraps pretrained LaMa for comparison
│   ├── evaluation/
│   │   ├── metrics.py           # PSNR, SSIM, FID, LPIPS
│   │   ├── severity_eval.py     # mask-severity-bucketed evaluation (Section 9)
│   │   └── evaluate.py
│   └── utils/
│       ├── logger.py
│       └── seed.py
├── tests/
│   ├── test_mask_generator.py
│   ├── test_dataset.py
│   ├── test_generator_forward.py
│   ├── test_env_bandit.py
│   └── test_reward.py
├── notebooks/
│   └── colab_quickstart.ipynb
├── scripts/
│   ├── download_places2.sh
│   ├── download_celeba_hq.sh
│   └── download_lama_checkpoint.sh
├── requirements.txt
├── README.md
└── main.py
```

**[REVISED — new rule]** A `tests/` directory is now mandatory, one smoke test per milestone (see Section 10, Gate Rule). Do not let the AI IDE generate the next milestone's files until the current milestone's test passes.

---

## 3. Environment Setup

`requirements.txt`:
```
torch>=2.1
torchvision>=0.16
stable-baselines3>=2.2
gymnasium>=0.29
opencv-python
numpy
scikit-image
lpips
pytorch-fid
tensorboard
pyyaml
tqdm
pytest
```

GPU strongly recommended. On Colab: `Runtime > Change runtime type > GPU`.

---

## 4. Data Pipeline

### 4.1 Datasets
- **Places2** (primary) — scene diversity.
- **CelebA-HQ** (alternative) — structured faces, faster convergence checks.
- **[NEW]** Reserve a small fixed **test** split (separate from val) that is *never* touched until final evaluation — needed for the severity-bucketed comparison in Section 9 to be trustworthy.

### 4.2 Preprocessing
- Resize to 256×256 (128×128 for fast dev iterations).
- Normalize to `[-1, 1]`.
- Splits: train / val / test (e.g. 85 / 10 / 5).

### 4.3 Mask Generation — **[REVISED]**
Previous plan cached ~5,000 masks and reused them for everything. That's now split:

- **Training masks: generated dynamically, on the fly, every batch.** Never cache/reuse for training — this maximizes mask diversity and avoids the model overfitting to a fixed mask pool.
- **Validation/test masks: a fixed, deterministic, seeded set**, generated once and saved to `data/masks/fixed_eval_masks/`. Every model variant (baseline GAN, RL-GAN, LaMa) is evaluated against the *exact same* fixed masks, at the *exact same* severity buckets (Section 9). This is what makes your ablation table and severity comparison scientifically valid — without it you can't tell whether a score difference came from the model or from random mask luck.

```python
def generate_irregular_mask(h, w, max_strokes=6, max_len=60, max_width=20, rng=None):
    rng = rng or np.random.default_rng()
    mask = np.zeros((h, w), np.uint8)
    for _ in range(rng.integers(1, max_strokes)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        for _ in range(rng.integers(4, 18)):
            angle = rng.uniform(0, 2 * np.pi)
            length = rng.integers(10, max_len)
            width = rng.integers(5, max_width)
            x2 = np.clip(int(x + length * np.cos(angle)), 0, w - 1)
            y2 = np.clip(int(y + length * np.sin(angle)), 0, h - 1)
            cv2.line(mask, (x, y), (x2, y2), 1, int(width))
            x, y = x2, y2
    return mask
```
Pass a seeded `rng` when generating the fixed eval set (`np.random.default_rng(42)`); pass an unseeded/global `rng` during training.

**[NEW]** Also record each generated mask's **missing-area percentage** — you'll need it to bucket into the severity table in Section 9 (10–20% / 20–40% / 40–60% / 60%+).

### 4.4 `InpaintingDataset`
Unchanged from v1: returns `image`, `mask`, `masked_image = image * (1 - mask)`.

---

## 5. Model Architecture

*(Unchanged from v1 — reviewer explicitly endorsed this and told us not to complicate it further. Do not swap in a diffusion model, transformer, or anything beyond GAN + RL.)*

### 5.1 Encoder
Shared CNN backbone. Input: `masked_image` (3ch) ⊕ `mask` (1ch) = 4 channels. Output: multi-scale features, used both by generator and by the RL agent's state.

### 5.2 GAN Generator
Gated-convolution encoder-decoder (DeepFill v2 style): coarse network → refinement network, conditioned on the RL agent's chosen `strategy_vector` (see 6.3), `tanh` output.

### 5.3 Discriminator
SN-PatchGAN — spectral-normalized, patch-level real/fake scores; also the source of the `ΔDiscriminator` reward term.

### 5.4 Losses
L1 (masked region weighted 6× vs 1×), perceptual (VGG / `lpips`), style (Gram matrix, optional), adversarial (hinge + spectral norm).

---

## 6. RL Formulation — **[REVISED — the biggest change]**

### 6.1 Action Space — simplified to 5 discrete actions

Previous plan: `MultiDiscrete([2,3,3])` = 18 actions. **Replaced** with a single `Discrete(5)`:

| Action | Meaning |
|---|---|
| 0 | Global completion |
| 1 | Local refinement |
| 2 | Boundary refinement |
| 3 | Texture refinement |
| 4 | Stop *(Version 2 only — see 6.2)* |

Rationale: the research question is "can RL choose the right strategy," not "can RL jointly optimize 3 independent dimensions." A 5-way choice is far easier to explain, debug, and report an action-distribution histogram for (Section 9). Only expand back toward a richer/`MultiDiscrete` action space in a later, optional iteration once the 5-action version clearly works.

### 6.2 Two Versions — resolve the "1-step vs STOP" inconsistency

**Version 1 — Contextual Bandit (build this first, this is your debugging version):**
```
State → PPO → Strategy (actions 0–3, no STOP) → Generator → Reward → episode ends
```
- Episode length = exactly 1 step.
- Action space for this version: `Discrete(4)` (drop action 4/Stop entirely — it has no meaning in a 1-step episode).
- `gamma = 0.0` in PPO (no future steps to discount).

**Version 2 — Multi-Step Refinement (build only after Version 1 works):**
```
State₀ → Action₁ → Generator refinement → Reward₁ → State₁ → Action₂ → ... → Stop
```
- Episode length variable, capped at e.g. `max_steps=4`.
- Action space: `Discrete(5)` (Stop now meaningful — it ends the episode early if further refinement isn't worth the cost).
- `gamma ≈ 0.95–0.99` (there are now real future steps to value).
- This is the stronger research story ("progressive adaptive refinement") but is strictly a **stretch goal** — do not start here.

### 6.3 State
Encoder feature vector (pooled) ⊕ mask statistics: missing-area %, bounding-box size, number of disconnected mask regions ⊕ current coarse-output quality proxy. Unchanged from v1.

### 6.4 Reward — **[REVISED — LPIPS and action cost added]**

Old reward mixed raw PSNR/SSIM/discriminator/artifact-penalty. New reward is improvement-based and includes an explicit efficiency term:

```
R_t = α·ΔPSNR + β·ΔSSIM − γ·ΔLPIPS + δ·ΔD − λ·C(a_t)
```
- `ΔPSNR`, `ΔSSIM` — improvement over the coarse/previous-step output (not absolute value — this rewards *the agent's contribution*, not just "the generator is generally good").
- `ΔLPIPS` — **[NEW]** perceptual distance improvement (you already listed LPIPS as an eval metric in v1 but never put it in the reward — now it's in both places, consistently).
- `ΔD` — change in discriminator confidence.
- `C(a_t)` — **[NEW]** action cost: e.g. `0` for global completion, small positive cost for each refinement pass, larger cost for more expensive strategies. This gives the RL objective a genuine efficiency dimension: *improve quality while minimizing unnecessary computation* — a materially stronger research narrative than quality alone, and it's what makes "Stop" meaningful once you reach Version 2.

```python
def compute_reward(prev_metrics, new_metrics, disc_score_delta, action_cost,
                    weights=(1.0, 1.0, 0.5, 0.5, 0.3)):
    alpha, beta, gamma, delta, lam = weights
    d_psnr = normalize_psnr(new_metrics["psnr"] - prev_metrics["psnr"])
    d_ssim = new_metrics["ssim"] - prev_metrics["ssim"]
    d_lpips = new_metrics["lpips"] - prev_metrics["lpips"]     # lower LPIPS is better
    r = (alpha * d_psnr + beta * d_ssim - gamma * d_lpips
         + delta * disc_score_delta - lam * action_cost)
    return r, {"d_psnr": d_psnr, "d_ssim": d_ssim, "d_lpips": d_lpips,
               "disc_delta": disc_score_delta, "cost": action_cost}
```
Still true from v1: normalize/clip each term before weighting (PSNR, SSIM, LPIPS are on very different numeric scales), and log every raw component to TensorBoard separately so you can catch the agent gaming one term.

### 6.5 Environment (`env_bandit.py`) — Version 1
```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class InpaintingBanditEnv(gym.Env):
    """Version 1: single-step contextual bandit. No STOP action."""
    def __init__(self, dataset, generator, discriminator, device="cuda"):
        super().__init__()
        self.dataset = dataset
        self.generator = generator.eval()          # frozen — Experiment B
        self.discriminator = discriminator.eval()
        self.device = device
        self.action_space = spaces.Discrete(4)      # global / local / boundary / texture
        self.observation_space = spaces.Box(low=-10, high=10, shape=(256 + 4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.image, self.mask, self.masked_image = self.dataset.sample()
        return self._build_state(self.masked_image, self.mask), {}

    def step(self, action):
        strategy_vec = decode_action(action)                 # from actions.py
        with torch.no_grad():
            coarse = self.generator.coarse_forward(self.masked_image, self.mask)
            completed = self.generator.refine(coarse, self.mask, strategy_vec)
        reward, info = compute_reward_from_images(coarse, completed, self.image,
                                                    self.discriminator, action)
        obs, _ = self.reset()
        return obs, reward, True, False, info   # terminated=True: single-step episode

    def _build_state(self, masked_image, mask):
        ...  # encoder forward -> pooled vector, concat mask stats
```
`env_multistep.py` (Version 2) is the same skeleton with `terminated` driven by `action == STOP or step_count >= max_steps`, and `State_{t+1}` built from the *actual current partial completion*, not a fresh reset sample.

### 6.6 Algorithm — PPO (confirmed correct, unchanged)
`Discrete(4)`/`Discrete(5)` action spaces are natively supported by PPO in Stable-Baselines3; DDPG requires continuous actions and does not apply here. This resolves the "PPO or DDPG" question from the original proposal in favor of **PPO**.
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = InpaintingBanditEnv(train_dataset, generator, discriminator)
check_env(env)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_bandit",
            n_steps=2048, batch_size=64, learning_rate=3e-4, gamma=0.0)  # gamma=0 for Version 1
model.learn(total_timesteps=200_000)
model.save("checkpoints/rl_agent_bandit")
```

---

## 7. Training Procedure — **[REVISED — joint fine-tuning is now optional]**

### Experiment A — GAN Baseline (`pretrain_gan.py`)
Train generator + discriminator normally, fixed neutral strategy for every mask. This is your **baseline**, not just a warm-start — report its metrics standalone.

### Experiment B — Frozen GAN + PPO Controller (`train_rl_bandit.py`)
Freeze generator and discriminator completely. Train the Version-1 bandit PPO agent from Section 6.5. **This is the core, required result of the project.** If Experiment B shows a meaningful improvement over Experiment A, the project already succeeds as a valid RL research contribution — everything past this point is optional strengthening, not a dependency.

### Experiments C & D — Ablations (`train_rl_ablations.py`)
- **C: GAN + PPO + mask features** — does adding richer mask statistics to the state improve results over B?
- **D: GAN + PPO + perceptual reward** — does adding the `ΔLPIPS` term (vs. B without it) improve results?

### Experiment E — Joint Fine-Tuning (`joint_finetune.py`) — **optional / advanced, not required**
Only attempt after A–D are complete and documented. Unfreeze the generator (keep discriminator frozen or lightly fine-tuned); alternate short generator-update steps with short PPO bursts. **If this destabilizes (discriminator loss collapsing, reward variance exploding), stop and fall back to Experiment B as your headline result** — do not spend remaining project time trying to rescue Experiment E at the cost of the rest of the report.

---

## 8. Baselines — **[NEW]**

Add a modern non-RL baseline for context:

| | Model | Purpose |
|---|---|---|
| Baseline 1 | GAN (Experiment A) | Your own fixed-strategy baseline |
| Baseline 2 | **LaMa** (pretrained, inference-only via `baselines/lama_inference.py`) | Established strong inpainting model, for external context |
| Proposed | RL-GAN (Experiment B, +C/D/E) | Your contribution |

You do **not** need to beat LaMa. The point is to show your RL-GAN is in a reasonable ballpark of a modern baseline, and to answer the actual research question — *does RL improve adaptive control of a GAN* — not "did we build the world's best inpainting model."

---

## 9. Evaluation — **[REVISED — severity buckets + action distribution added]**

### 9.1 Standard metrics (unchanged, computed over the full image vs. ground truth, consistently)
PSNR, SSIM, FID, LPIPS.

### 9.2 Ablation table (expanded from v1)
| Setting | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|---|---|---|---|---|
| A: GAN baseline | | | | |
| B: GAN + PPO | | | | |
| C: GAN + PPO + mask features | | | | |
| D: GAN + PPO + perceptual reward | | | | |
| E: GAN + PPO + joint fine-tune *(optional)* | | | | |
| LaMa (reference, not directly compared) | | | | |

This directly answers *"which component actually improved performance"* — the kind of question faculty are likely to ask.

### 9.3 Action distribution — **[NEW]**
Report, after training, how often the agent chose each strategy:
```
Average RL actions per image: X
Global      31%
Local       28%
Boundary    22%
Texture     12%
(Stop        7%   — Version 2 only)
```
This is important specifically *because* this is an RL project — faculty will want to see the policy actually learned something structured, not collapsed to one action.

### 9.4 Evaluation by mask severity — **[NEW, most important addition]**
Bucket the **fixed eval mask set** (Section 4.3) by missing-area percentage and compare GAN baseline vs. RL-GAN in each bucket:

| Missing area | GAN (PSNR) | RL-GAN (PSNR) |
|---|---|---|
| 10–20% | | |
| 20–40% | | |
| 40–60% | | |
| 60%+ | | |

**Hypothesis to test:** RL-GAN's advantage over the plain GAN *grows* as mask severity increases. If the data supports this, your headline result becomes:

> "The adaptive controller provides the greatest benefit under difficult, irregular missing-region conditions" — a substantially stronger and more specific claim than "RL-GAN has higher average PSNR."

---

## 10. Suggested Build Order — **[REVISED — Gate Rule added]**

1. **M1 — Data**: `mask_generator.py` (dynamic + fixed-eval variants), `dataset.py`. Smoke test: visualize a grid of `(image, mask, masked_image)`.
2. **M2 — GAN only (Experiment A)**: `generator.py`, `discriminator.py`, `losses.py`. Train on a small subset until visibly plausible completions. Smoke test: forward pass shape check + one training step doesn't NaN.
3. **M3 — Gym env, dummy reward**: `env_bandit.py` with a random/dummy reward. Run `check_env()`, confirm PPO trains without crashing. Smoke test: `check_env()` passes, `model.learn(total_timesteps=1000)` completes.
4. **M4 — Real reward (Experiment B)**: wire in the real PSNR/SSIM/LPIPS/discriminator/cost reward from Section 6.4. Retrain PPO. Inspect action-distribution logs — **this is your required, core result.**
5. **M5 — Ablations (Experiments C, D)**: mask-feature state ablation, perceptual-reward ablation.
6. **M6 — Evaluation**: ablation table, action distribution, mask-severity table (Section 9), LaMa baseline comparison.
7. **M7 — Joint fine-tuning (Experiment E)** — *optional*, attempt only after M6 is complete and documented.
8. **M8 — Scale up**: full dataset / 256×256 / longer runs, regular checkpointing.

### Gate Rule — **[NEW, add this verbatim to your AI IDE's instructions]**
> Never proceed to the next milestone if the current milestone's smoke test in `tests/` is failing or missing. If a milestone's test fails, fix that milestone before generating any files for the next one. Do not generate the full multi-file structure for a later milestone speculatively — build one milestone, verify it, then move on.

This prevents the common failure mode of an AI IDE generating 20–30 files across several milestones and only then revealing that an early piece (e.g. mask generation, or the Gym env's observation shape) was broken the whole time.

---

## 11. Project Hierarchy (for your report/slides)

```
                        RL-GAN
                          │
              ┌───────────┴───────────┐
              │                       │
            GAN                     PPO
              │                       │
       Generator                Controller
       Discriminator            State / Actions(5) / Reward
       Losses                   (Bandit → optional multi-step)
              │                       │
              └───────────┬───────────┘
                          │
                    Evaluation
                          │
      ┌─────────┬─────────┼─────────┬───────────┐
      │         │         │         │           │
   PSNR       SSIM      LPIPS      FID     Action Distribution
      │         │         │         │           │
      └─────────┴─────────┴─────────┴───────────┘
                          │
           Ablations (A–E)  +  Mask-Severity Analysis  +  LaMa reference
```

---

## 12. Summary of Changes from v1 (design-review checklist)

- [x] Action space simplified: `MultiDiscrete([2,3,3])` (18 actions) → `Discrete(4/5)` (Section 6.1).
- [x] Resolved 1-step-vs-STOP inconsistency: two explicit versions — bandit (no Stop) then optional multi-step (Stop meaningful) (Section 6.2).
- [x] Reward now improvement-based and includes `ΔLPIPS` and an explicit action cost `C(a_t)` (Section 6.4).
- [x] Joint GAN/RL fine-tuning reclassified as optional Experiment E, not a dependency for project success (Section 7).
- [x] LaMa added as an external reference baseline (Section 8).
- [x] Training masks generated dynamically; validation/test masks fixed and deterministic (Section 4.3).
- [x] Ablation set expanded to A–E with a table answering "which component helped" (Section 9.2).
- [x] Action-distribution reporting added (Section 9.3).
- [x] Mask-severity-bucketed evaluation added — the strongest single result the project can produce (Section 9.4).
- [x] Gate Rule added: no milestone may start until the previous milestone's smoke test passes (Section 10).

---

## 13. What to Hand Your AI IDE First

Paste **Sections 2–6** verbatim as the first prompt ("scaffold this repo structure and implement the files described, Version 1 bandit environment only"). Then proceed milestone by milestone through Section 10, respecting the Gate Rule — do not ask for Experiment E (joint fine-tuning) until Experiments A–D and the Section 9 evaluation are complete and working.
