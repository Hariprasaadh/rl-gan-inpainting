# The Whole Project Explained

This document explains the RL-GAN adaptive inpainting project from two angles:

* **Part A, high level**: what the project is, why it exists and how the pieces fit, in plain language.
* **Part B, in detail**: every component, formula, training procedure, script and output, with the files that implement them.

If you only have five minutes read Part A. If you need to explain or modify the code, read Part B as well. For slide-by-slide talking points see [PRESENTATION_README.md](PRESENTATION_README.md).

---

# Part A. High-level view

## A1. The problem

**Image inpainting** means filling in missing or damaged regions of a photograph so that the result looks natural. The damaged region is described by a **mask**. Typical uses are removing objects, repairing scratches and restoring old photos.

Modern inpainting networks (GANs) do this well, but they apply **one fixed behaviour to every image**. Real damage varies:

| Damage | What the repair needs |
|---|---|
| Thin scratches | A smooth, invisible seam along the hole border |
| Large missing areas | The overall structure of the scene rebuilt |
| Textured regions (grass, fabric, bricks) | Fine, high-frequency detail |

A single compromise behaviour is rarely the best for each of these.

## A2. The idea

Keep one strong pretrained inpainting network, add several small **specialists**, and let a **reinforcement-learning agent choose the right specialist for each image**.

```text
damaged image + mask
        │
        ▼
 frozen DeepFill-v2 (pretrained on Places2)
        │  coarse fill
        ├────────► 261-number "state" ───► PPO agent ──► picks 1 of 4
        ▼                                                    │
 refined fill ──► [ Global | Local | Boundary | Texture ] ◄──┘
                              │
                              ▼
                      completed image
```

| Component | Role | Trained? |
|---|---|---|
| **Backbone** (DeepFill-v2, Places2) | Does the heavy inpainting | No, frozen |
| **Four adapters** (Global, Local, Boundary, Texture) | Small residual modules, each specialised for one kind of repair | Yes, in stage 1 |
| **PPO agent** | Looks at the damaged image and picks one adapter | Yes, in stage 2 |

## A3. Why reinforcement learning?

Choosing an adapter is a **decision** whose quality is only known after acting: you try an adapter and measure how good the result is. There is no label saying "the correct adapter for this image is number 2". RL learns the choice from a **reward**, the measured quality improvement.

The problem needs only **one decision per image** and the outcome is known immediately, so it is modelled as a **contextual bandit**: one step per episode, discount factor γ = 0, with PPO as the learning algorithm.

## A4. Key numbers

| Item | Value |
|---|---|
| Image size | 256 × 256 |
| Backbone | DeepFill-v2, Places2 pretrained, frozen |
| Adapters | 4 (about 1.5 MB in total) |
| RL state | 261 numbers (256 latent + 4 mask statistics + 1 boundary gradient) |
| Actions | 4 (0 Global, 1 Local, 2 Boundary, 3 Texture) |
| RL algorithm | PPO (Stable-Baselines3), 1-step episodes, γ = 0 |
| Reward weights (α, β, γ, δ, λ) | (1.0, 1.0, 0.5, 0.3, 0.1) |
| Interactive speed | about 235 ms per request on an RTX 3050 (warm) |

## A5. What you can do with it

* **Train**: pretrain the adapters, then train the PPO agent (`python main.py ...`).
* **Evaluate**: a four-stage ablation (A backbone, B fixed adapter, C PPO-selected, D PPO + per-image refinement) and a strategy-comparison tool.
* **Demo**: a browser app (`python app.py`) where you paint a mask and watch the agent choose a strategy and repair the image.

## A6. Results at a glance

PSNR in dB on three hand-masked scenes (higher is better):

| Scene | PPO chose | A: backbone | B: fixed Global | C: PPO-selected | D: PPO + refine |
|---|---|:---:|:---:|:---:|:---:|
| Alpine lake | Boundary | 34.45 | 35.21 | 32.25 | 37.23 |
| Lush meadow | Global | 33.72 | 34.54 | 34.54 | 39.65 |
| Sunset peaks | Boundary | 40.58 | 40.00 | 36.80 | 43.19 |

* The Global adapter adds about **+0.8 dB** over the backbone on the lake and meadow scenes.
* On the meadow the agent selected Global by itself, giving **+0.82 dB** without any ground truth.
* When the adapters are fitted to a specific image (stage D) PSNR reaches 37 to 43 dB, up to **+5.9 dB** above the backbone, with SSIM up to **0.9946**.
* Across five verification images the best adapter changes from image to image (Global wins two, Local three), so a fixed strategy is not best everywhere.

---

# Part B. In detail

## B1. Repository map

```text
rl-gan-inpainting/
├── main.py                 CLI entry point (all pipeline commands)
├── app.py                  Web studio server (http.server, port 7860)
├── web/                    index.html, style.css, app.js (front end)
├── configs/                YAML configs (pretrain_adapters, rl_agent_bandit, ...)
├── scripts/                download_gan/coco, create_eval_set, verify_strategies,
│                           evaluate_baseline, run_multi_image_test
├── src/
│   ├── models/             deepfillv2.py, networks_tf.py, encoder.py,
│   │                       strategy_adapters.py, rl_inpainting_model.py,
│   │                       discriminator.py, losses.py
│   ├── rl/                 state.py, actions.py, reward.py,
│   │                       env_bandit.py, env_multistep.py
│   ├── training/           pretrain_adapters.py, train_rl_bandit.py, trainer_utils.py
│   ├── data/               dataset.py, mask_generator.py, transforms.py
│   ├── evaluation/         metrics.py, severity_eval.py, evaluate.py
│   └── utils/              seed.py, logger.py
├── checkpoints/            pretrained/, adapters/, rl_agent_bandit/
├── data/                   raw/val2017, raw/sample_images, masks/fixed_eval_masks
├── results/                reports, demo panels, verification grids
└── docs/                   explanations, images, report, presentation
```

The files in `src/models/generator.py`, `pretrain_gan.py`, `train_rl_ablations.py`, `joint_finetune.py`, `env_multistep.py` and `src/baselines/lama_inference.py` belong to a separate, earlier line of work built around a small from-scratch generator; the pipeline described here does not use them.

## B2. Conventions

* All image tensors are in **[-1, 1]**.
* **Masks: 1 = hole, 0 = known.** `masked_image` has the hole zeroed.
* The completed image is `completed = masked_image + result * mask`: known pixels are never changed.
* Seeds: `src/utils/seed.py` (`seed_everything`) seeds Python, NumPy and PyTorch; training scripts use 42.
* Configs are read with `trainer_utils.load_config`; checkpoints are saved with `save_checkpoint`.

## B3. Data and masks

* **Images**: COCO val2017 (5,000 photos) resized to 256 × 256 (`src/data/dataset.py`). A folder of sample photographs (`data/raw/sample_images`) is used for demos and verification.
* **Training masks** are generated on the fly by `src/data/mask_generator.py`: random irregular **strokes** (1 to 4 strokes of random width and length, occasionally a circle).
* `compute_mask_stats` summarises a mask into four numbers used by the RL state (see B5).
* **Fixed evaluation set**: `python main.py create-eval-set` writes **100 masks, 25 in each missing-area bucket** (10 to 20%, 20 to 40%, 40 to 60%, 60% and above) plus a manifest that pairs them with the first 100 images. Masks come from a seeded generator (seed 42), so every experiment sees the same damage.
* `src/data/transforms.py` provides feathered blending used by the web app to soften the seam.

## B4. The frozen backbone (DeepFill-v2)

`src/models/deepfillv2.py` wraps a PyTorch port of the TensorFlow DeepFill-v2 network (`networks_tf.py`) loaded with Places2 weights (`checkpoints/pretrained/deepfillv2_places2.pth`).

* **Input**: 5 channels: masked RGB, a plane of ones and the mask.
* **Stage 1, coarse**: gated convolutions `ELU(W_f * x) ⊙ sigmoid(W_g * x)` with dilation rates up to 16 give a rough fill. The bottleneck layer `conv10_atrous` has 96 channels (2 × cnum).
* **Stage 2, refine**: a convolutional branch plus a **contextual attention** branch (which copies and blends patches from the known region) are concatenated and decoded into a sharper RGB output.
* **Frozen**: `RLInpaintingModel.load_pretrained_backbone` sets `requires_grad = False` and keeps the module in eval mode. Every gain therefore comes from the adapters and the controller.

Exposed methods: `coarse_forward`, `refine_forward`, `get_bottleneck_feature`.

## B5. The state (261 numbers)

Implemented in `src/rl/state.py` (`StateBuilder`) and `src/models/encoder.py` (`DeepFillEncoder`).

| Part | Size | Computation |
|---|:---:|---|
| Latent *z* | 256 | `conv10_atrous` features (96 ch) → global average pool → `Linear → LayerNorm → ELU → Linear` |
| Missing ratio | 1 | hole pixels / all pixels |
| Bounding-box ratio | 1 | hole bounding-box area / all pixels |
| Region count | 1 | connected components ÷ 10, clipped to [0, 1] |
| Centroid distance | 1 | hole centre of mass to image centre, normalised by the half-diagonal |
| Boundary gradient | 1 | mean Sobel magnitude of the coarse composite on a 1-px ring just outside the hole, ÷ 2, clipped |

The latent comes from the coarse pass that must be run anyway, so the state adds no extra backbone computation. The coarse composite used here is `masked + coarse × mask`.

## B6. The four adapters

Defined in `src/models/strategy_adapters.py`. All four share one skeleton and operate on the **3-channel RGB output of the refine stage**:

```text
Conv(3→64) → InstanceNorm → ELU → FiLM → Conv(64→64) → InstanceNorm → ELU → Conv(64→3)
out = tanh( x + 0.5 · tanh(Δ) )
```

* **FiLM** modulates the 64 hidden channels: `(1 + γ(e)) ⊙ h + β(e)`, where `e` is a learned **64-dimensional strategy embedding** and `γ, β` come from a small MLP. This lets the same layers behave differently per strategy.
* The output formula bounds the correction, so the adapter refines instead of replacing the backbone result.

| Action | Adapter | Specific behaviour | Cost |
|:---:|---|---|:---:|
| 0 | **Global** | Plain pass, no mask weighting | 0.00 |
| 1 | **Local** | Multiplies the input by `1 + 0.5 · mask` to emphasise the hole | 0.10 |
| 2 | **Boundary** | Builds a dilate-minus-erode ring (5×5) around the hole and blends the adapted output more strongly there | 0.15 |
| 3 | **Texture** | Adds a Laplacian high-pass term inside the hole plus a small high-frequency branch | 0.20 |

Costs are hand-set (`src/rl/actions.py`; `ACTION_STOP = 4` exists for the multi-step environment). The complete adapter set including embeddings is about 1.5 MB.

## B7. The unified model

`src/models/rl_inpainting_model.py` (`RLInpaintingModel`) is the single class every path uses:

| Method | Purpose |
|---|---|
| `coarse_forward(masked, mask)` | Stage 1 coarse prediction |
| `extract_state_embedding(masked, mask)` | The 256-d latent |
| `refine(coarse_composite, mask, strategy)` | Stage 2 + adapter; `strategy=None` gives the backbone-only result (the baseline "Stage A"); 0 to 3 selects an adapter |
| `forward(masked, mask, strategy)` | Complete pass returning coarse, refined, completed and the latent |
| `load_pretrained_backbone`, `load_adapters` | Checkpoint loading (adapters file also holds the strategy embeddings) |
| `freeze_adapters`, `unfreeze_adapters` | Toggle adapter training |

## B8. Stage 1: adapter pretraining

`python main.py pretrain-adapters --config configs/pretrain_adapters.yaml` (`src/training/pretrain_adapters.py`).

For every batch the frozen backbone computes the coarse composite and refined image. **Each adapter** receives the refined image, is composited into the hole, and is trained with **its own loss**; one Adam optimiser updates all adapters, the strategy embeddings and the encoder head (backbone frozen).

| Adapter | Loss |
|---|---|
| Global | L1(hole) + L1(valid) + VGG-16 perceptual loss (weight 0.05, `PerceptualAndStyleLoss`) |
| Local | 6·L1(hole) + 0.2·L1(valid) + 0.5·SSIM-style term (L1 of 11×11 box-filtered means inside the hole) |
| Boundary | L1(hole) + L1(valid) + 2·Sobel-gradient L1, weighted 2.0 on the hole-boundary ring and 0.2 elsewhere |
| Texture | L1(hole) + L1(valid) + 0.1·L1 between FFT magnitudes + VGG perceptual loss |

* Optimiser: Adam, lr 1e-4, betas (0.5, 0.999), batch size 4, 5 epochs.
* Output: `checkpoints/adapters/adapters_epoch_XXX.pt` and `adapters_final.pt`.
* Different losses pull each adapter toward a different behaviour: hole fidelity (Local), seam continuity (Boundary), detail (Texture), balance (Global).

### Strategy verification gate

`python main.py verify-strategies --data-dir data/raw/sample_images --num-images 5 ...` applies the four adapters to the same coarse composites, prints per-strategy PSNR/SSIM and checks that **no single strategy wins more than 80%** of the images. It also prints pairwise state-embedding distances and a four-action reward sanity check and writes comparison grids to `results/strategy_verify/`.

Recorded result on five images (PSNR dB):

| Image | Global | Local | Boundary | Texture | Best |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 30.28 | 30.18 | 28.97 | 29.82 | Global |
| 1 | 26.72 | 27.04 | 26.09 | 25.98 | Local |
| 2 | 25.94 | 26.23 | 24.16 | 24.90 | Local |
| 3 | 23.12 | 22.89 | 21.95 | 22.68 | Global |
| 4 | 30.35 | 30.46 | 28.04 | 28.42 | Local |

## B9. Stage 2: the RL formulation

### The environment

`src/rl/env_bandit.py` (`InpaintingBanditEnv`, a Gymnasium environment):

* `reset()`: sample an image and stroke mask, run the coarse pass, record coarse metrics and the discriminator score, build the 261-d state.
* `step(action)`: apply the chosen adapter, composite into the hole, measure PSNR, SSIM, LPIPS (when `compute_lpips` is true) and the discriminator confidence, compute the reward, and draw the next sample.
* Episodes are **one step long**. Observation: 261-D vector. Action: `Discrete(4)`.

### The reward

`src/rl/reward.py`:

```text
R = α·ΔPSNR~ + β·ΔSSIM~ − γ·ΔLPIPS~ + δ·ΔD − λ·C(a)
```

| Term | Definition |
|---|---|
| ΔPSNR~ | clip((PSNR_adapter − PSNR_coarse) / 10, ±2) |
| ΔSSIM~ | clip((SSIM_adapter − SSIM_coarse) / 0.1, ±2) |
| ΔLPIPS~ | clip((LPIPS_adapter − LPIPS_coarse) / 0.1, ±2); lower is better, hence subtracted |
| ΔD | change in the mean patch confidence of a spectral-normalised PatchGAN discriminator (`SNPatchGANDiscriminator`) |
| C(a) | action cost 0 / 0.10 / 0.15 / 0.20 |

Weights `(α, β, γ, δ, λ) = (1.0, 1.0, 0.5, 0.3, 0.1)` come from `configs/rl_agent_bandit.yaml`; the total is clipped to ±10. Each delta is measured against the **coarse composite**. Normalising each delta stops PSNR from dominating the other terms.

### Why a bandit

A multi-step formulation with a STOP action exists (`env_multistep.py`), but the best adapter is determined by the damage itself and the outcome is observable immediately, so a one-step bandit with **γ = 0** needs no credit assignment and gives low-variance updates.

## B10. PPO training

`python main.py train-rl --config configs/rl_agent_bandit.yaml [--timesteps N]` (`src/training/train_rl_bandit.py`).

The script builds the model (seed 42), loads the backbone and adapters, **freezes the adapters**, creates the discriminator for the reward's ΔD term and trains Stable-Baselines3 `PPO("MlpPolicy", ...)`:

| Setting | Value |
|---|---|
| Discount γ | 0.0 |
| Rollout length `n_steps` | 128 |
| Mini-batch / epochs | 32 / 4 |
| Learning rate | 3e-4 |
| Entropy coefficient | 0.05 |
| Value-loss coefficient / max grad-norm | 0.5 / 0.5 |
| Timesteps | `total_timesteps` in the config (5000), override with `--timesteps` |
| Seed | 42 |

**How PPO learns here.** MlpPolicy has an **actor** (probabilities over the four adapters) and a **critic** (V(s), the expected reward). Each iteration collects 128 one-step episodes, computes the advantage `A = r − V(s)` (γ = 0), then does 4 epochs of mini-batch updates minimising `−L_clip + 0.5·L_V − 0.05·H(π)`. The clipped objective `min(ρA, clip(ρ, 1−ε, 1+ε)A)` (ε = 0.2, the SB3 default) stops any update from moving the policy too far; the entropy bonus keeps all four adapters in play during exploration.

A callback logs the action distribution and every reward component to TensorBoard (`logs/rl_agent_bandit/`). The policy is saved as `checkpoints/rl_agent_bandit/ppo_bandit_final.zip`.

**At inference** `PPO.predict(state, deterministic=True)` returns the action; the app, ablation script and evaluators all build the same 261-d state and then call `refine(coarse_composite, mask, strategy=action)`.

## B11. Evaluation

### Metrics (`src/evaluation/metrics.py`)

PSNR and SSIM on the composited full image in [0, 1]; LPIPS (deep-feature distance); L1 and hole L1.

### The four-stage ablation (`scripts/run_multi_image_test.py`)

| Stage | Configuration | Uses ground truth? |
|:---:|---|:---:|
| **A** | Backbone only (`strategy=None`) | No |
| **B** | Fixed Global adapter (action 0) | No |
| **C** | PPO picks the adapter; one forward pass | No |
| **D** | C plus **400 AdamW steps** on the test image (lr 1.5e-3, cosine decay, hole-weighted L1 and an 8× gradient-matching loss against the target) | Yes |

Stages A to C are feed-forward measurements. Stage D shows how far the adapters can be pushed when optimised directly against a target image. The script runs three scenes (sample images 1, 2 and 5) with hand-designed masks (`build_controlled_mask`) and writes a seven-panel comparison per scene (ground truth, masked, coarse, A, B, C, D) and `results/multi_image_evaluation_report.md`.

### Other evaluators

* `python main.py eval-baseline`: backbone-only metrics on a set of images and the fixed masks (`results/baseline_test/`). The recorded 5-sample run gave PSNR 21.23 ± 1.82 dB, SSIM 0.864 ± 0.029, LPIPS 0.033 ± 0.014.
* `src/evaluation/severity_eval.py` (`SeverityEvaluator`): groups test items by missing ratio (10 to 20, 20 to 40, 40 to 60, 60%+) and reports baseline versus RL-guided PSNR/SSIM and action counts per bucket.

## B12. The web studio

`python app.py [port]` serves `http://localhost:7860`. It loads the models at import time, then serves `web/` plus:

| Endpoint | Purpose |
|---|---|
| `GET /api/presets` | List of sample images |
| `GET /api/image/<name>` | A sample image resized to 256×256 |
| `POST /api/inpaint` | Run inpainting: base64 image + mask, `mode` = `fast` or `precision`, optional `action` |

Request flow: decode and threshold the mask (luminance above 30 means hole) → coarse pass → 261-d state → PPO action (request `action` overrides it; with no PPO file the server uses action 1) → backbone-only baseline pass and adapter pass → feathered blend (`feather_composite`) → PSNR/SSIM of both against the original → JSON with images, action name, metrics and `elapsed_ms`.

* **Fast mode**: one forward pass, about 235 ms warm on the RTX 3050 (the figure includes the baseline pass and metric computation).
* **Precision mode**: after the decision, 60 AdamW steps (lr 3e-4) on the chosen adapter against the uploaded image, then the original weights are restored for the next request.
* Front end (`web/app.js`): canvas brush (6 to 48 px), random mask, upload, before/after split slider, 5-stage pipeline tab, decision card and telemetry cards.

## B13. Commands, in order

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/download_gan.py     # backbone -> checkpoints/pretrained/
python scripts/download_coco.py    # data/raw/val2017
python main.py create-eval-set --image-dir data/raw/val2017 --output-dir data/masks/fixed_eval_masks --count 25 --image-size 256
python main.py pretrain-adapters --config configs/pretrain_adapters.yaml
python main.py verify-strategies --data-dir data/raw/sample_images --num-images 5 --output-dir results/strategy_verify
python main.py train-rl --config configs/rl_agent_bandit.yaml --timesteps 10000
python main.py eval-baseline --data-dir data/raw/sample_images --mask-dir data/masks/fixed_eval_masks --samples 100
python scripts/run_multi_image_test.py
python app.py
```

Environment: Python 3.12, PyTorch 2.6 with CUDA 12.4 (developed on an RTX 3050), Windows/PowerShell. More in [GETTING_STARTED.md](GETTING_STARTED.md).

## B14. How the pieces connect (one image, end to end)

1. `app.py` or an evaluator receives `(image, mask)` and forms `masked = image · (1 − mask)`.
2. `RLInpaintingModel.coarse_forward` gives the coarse fill; the coarse composite is `masked + coarse · mask`.
3. `StateBuilder` builds the 261-d state from the 96-channel bottleneck (via `DeepFillEncoder`), `compute_mask_stats` and the Sobel boundary value.
4. PPO maps the state to an action (0 to 3).
5. `RLInpaintingModel.refine(coarse_composite, mask, strategy=action)` runs Stage 2 and the selected adapter.
6. `completed = masked + refined · mask` is returned; metrics are computed against the ground truth when it is available.

During training the same path runs inside `InpaintingBanditEnv.step`, which adds the reward computation.

## B15. Design choices and their reasons

| Choice | Reason |
|---|---|
| Frozen pretrained backbone | Keeps a strong prior; only about 1.5 MB of adapters plus a small policy are trained |
| Adapters on the RGB output | Small, fast to train, and easy to swap |
| Different loss per adapter | Produces distinct behaviours the agent can choose between |
| State from the coarse pass | Free (the pass is needed anyway) and informative about the hole |
| One-step bandit, γ = 0 | Outcome is immediate; no credit assignment; stable PPO updates |
| Normalised, clipped reward terms | Prevents one metric from dominating and keeps updates stable |
| Action costs | Makes the agent weigh benefit against extra specialisation |
| Entropy coefficient 0.05 | Keeps exploring all four adapters during training |
| Fixed masks and seed 42 | Reproducible comparisons across experiments |
| Single `RLInpaintingModel` | Training, evaluation and the web app share exactly the same inference path |

## B16. Further reading

* [ARCHITECTURE.md](ARCHITECTURE.md): modules and shapes
* [TRAINING_AND_RL.md](TRAINING_AND_RL.md): training and PPO
* [EVALUATION.md](EVALUATION.md): metrics and ablation
* [WEB_STUDIO.md](WEB_STUDIO.md): the web app and API
* [PRESENTATION_README.md](PRESENTATION_README.md): slide-by-slide guide with RL vocabulary
* [RL_GAN_Inpainting_Report.pdf](RL_GAN_Inpainting_Report.pdf): the full project report
* `../configs/README.md` and `../scripts/README.md`: every config key and script
