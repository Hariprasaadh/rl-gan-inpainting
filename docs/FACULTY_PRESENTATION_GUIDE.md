# 🎓 RL-GAN Inpainting: Presentation Guide

> A guide for presenting the project: a 60-second pitch, the architecture, the formulation, the results, a live-demo script and answers to likely questions. Detailed explanations are in [`docs/`](docs/README.md) and the written report is in [`docs/RL_GAN_Inpainting_Report.pdf`](docs/RL_GAN_Inpainting_Report.pdf).

---

## ⏱ 1. The 60-Second Elevator Pitch

> *"My project is **RL-GAN: adaptive image inpainting with a reinforcement-learning controller**.*
>
> *Most inpainting networks apply one fixed strategy to every hole. I keep a pretrained DeepFill-v2 backbone frozen, add four small adapter networks that each specialise in a different kind of repair (Global, Local, Boundary, Texture), and train a **PPO contextual-bandit agent** that looks at a 261-number description of the damaged image and picks one adapter per image.*
>
> *The state comes almost for free from the backbone's own bottleneck features plus a few mask-geometry numbers. I evaluate in four stages: backbone alone, fixed adapter, PPO-selected adapter in a single forward pass, and, as an oracle, test-time optimisation with the ground truth. Everything is wrapped in a command-line pipeline and an interactive web studio."*

---

## 🎯 2. Research Problem & Hypothesis

- **Problem**: inpainting trades off global semantic consistency against local detail and seam quality; a single static network compromises between them.
- **Hypothesis**: a controller that reads damage geometry and context can pick a better restoration strategy per image, and the benefit should grow with hole severity.
- **Framework**: the project provides everything needed to study this: four specialised adapters, a bandit environment, a reward that combines PSNR, SSIM, LPIPS and a discriminator signal, fixed masks in four severity buckets (10–20 %, 20–40 %, 40–60 %, 60 %+) and a severity evaluator.

---

## 🧠 3. System Architecture

```text
Damaged image + mask (256x256, [-1,1])
        │
        ▼
 DeepFill-v2 backbone (frozen, Places2 weights)
   Stage 1 coarse pass ──► coarse composite = masked + coarse·mask
        │                              │
        │ bottleneck (96 ch)           │
        ▼                              ▼
 DeepFillEncoder (pool + MLP → 256-d)   mask stats (4) + boundary gradient (1)
        └──────────────┬───────────────┘
                       ▼
            state s ∈ ℝ²⁶¹ ──► PPO (MLP actor-critic, γ = 0)
                                      │ action a ∈ {0,1,2,3}
        Stage 2 refine pass (contextual attention) ──► RGB refined image
                                      │
                       adapter[a]  (conv-IN-ELU, FiLM-conditioned, residual tanh)
                                      ▼
                completed = masked + adapted·mask  (+ feathered blend for display)
```

Key points:
- The bottleneck has **96 channels** (`2·cnum`, `cnum = 48`).
- Adapters act on the **RGB output** of the backbone's refine stage; FiLM modulates the adapter's own 64-channel hidden layer using a learned 64-d strategy embedding.
- `strategy=None` means "backbone only", the baseline in every table and in the web app's "baseline" panel.
- The adapters are tiny (about 1.5 MB checkpoint) while the backbone checkpoint is about 40 MB and stays frozen.

---

## 📐 4. Mathematical Formulation

### State, s ∈ ℝ²⁶¹
- z ∈ ℝ²⁵⁶: pooled backbone bottleneck passed through `Linear → LayerNorm → ELU → Linear`.
- m ∈ ℝ⁴: missing ratio; bounding-box ratio; number of components divided by 10 and clipped to [0,1]; centroid distance from the image centre (normalised by the half-diagonal).
- g ∈ ℝ¹: mean Sobel gradient magnitude on a 1-px ring just outside the hole in the coarse composite, divided by 2 and clipped.

### Actions and costs
0 Global (0.00) · 1 Local (0.10) · 2 Boundary (0.15) · 3 Texture (0.20).

### Reward
R = α·ΔPSNR̃ + β·ΔSSIM̃ − γ·ΔLPIPS̃ + δ·ΔD − λ·C(a), with ΔPSNR̃ = clip(ΔPSNR/10, ±2), ΔSSIM̃ = clip(ΔSSIM/0.1, ±2), ΔLPIPS̃ = clip(ΔLPIPS/0.1, ±2), and the total clipped to ±10. Deltas compare the refined result with the coarse composite. Training weights (α, β, γ, δ, λ) = (1.0, 1.0, 0.5, 0.3, 0.1) from `configs/rl_agent_bandit.yaml`.

### Adapter pretraining losses (backbone frozen; COCO val2017, random stroke masks, 5 epochs)
- Global: L1(hole) + L1(valid) + VGG perceptual (weight 0.05).
- Local: 6·L1(hole) + 0.2·L1(valid) + 0.5·SSIM-style term over the hole.
- Boundary: L1(hole) + L1(valid) + 2·Sobel-gradient L1, weighted 2.0 on the boundary ring and 0.2 elsewhere.
- Texture: L1(hole) + L1(valid) + 0.1·FFT-magnitude L1 + VGG perceptual (0.05).

### PPO
Stable-Baselines3 `MlpPolicy`, γ = 0 (one-step bandit), n_steps 128, batch 32, 4 epochs, lr 3e-4, entropy coefficient 0.05; the config default is 5000 timesteps and `--timesteps` overrides it.

---

## 🔬 5. The 4-Stage Ablation (`scripts/run_multi_image_test.py`)

| Stage | Configuration | Uses ground truth? |
|:---:|---|:---:|
| A | DeepFill-v2 backbone only | No |
| B | Fixed adapter 0 (Global) | No |
| C | PPO picks the adapter, single forward pass | No |
| D | C + **400** AdamW steps on all trainable parameters against the target image | Yes (oracle) |

Test material: 3 sample photographs with hand-designed masks.

---

## 📊 6. Results You Can Quote (`results/multi_image_evaluation_report.md`)

| Scene | PPO action | A Backbone | B Fixed | C PPO | D (oracle) |
|---|---|:---:|:---:|:---:|:---:|
| Alpine lake | 2 Boundary | 34.45 dB | 35.21 | 32.25 | 37.23 |
| Lush meadow | 0 Global | 33.72 | 34.54 | 34.54 | 39.65 |
| Sunset peaks | 2 Boundary | 40.58 | 40.00 | 36.80 | 43.19 |

Talking points:
- The fixed Global adapter (stage B) improves on the backbone by +0.76 dB on the lake and +0.82 dB on the meadow.
- On the meadow the PPO agent selects the Global adapter, so stage C equals stage B (34.54 dB, +0.82 dB over the backbone).
- On the lake and the sunset scenes the agent selects the Boundary adapter, showing that the policy chooses different strategies for different scenes.
- Stage D reaches 37.2–43.2 dB, which shows how much capacity the adapter structure has when it is optimised on a specific image.

Other outputs: `results/baseline_test/` (backbone only: 21.2 dB PSNR, 0.864 SSIM, LPIPS 0.033 on 5 samples) and `results/strategy_verify/` (per-strategy visual grids; Local has the best PSNR on 3 of 5 images and Global on 2 of 5).

---

## 🖥 7. Live Demo Script (`python app.py`, then http://localhost:7860)

Startup: activate the environment (`.\.venv\Scripts\Activate.ps1`), run `python app.py`, and check the terminal prints "Backbone loaded", "Adapters loaded" and "PPO agent loaded".

The UI has preset thumbnails and file upload, a brush with a size slider, **Clear** and **Random Mask** buttons, two radio modes and a **Run** button.

1. Pick a preset (a random mask is pre-drawn) and paint extra strokes over a boundary region (tree line, horizon).
2. Keep **"Instant Forward Pass"** selected and click Run: one coarse pass, one state extraction, one PPO decision, one refine + adapter pass.
3. Read the cards: chosen strategy, PSNR/SSIM versus the "Baseline" (backbone only), mask coverage and latency (about 235 ms on the RTX 3050, including the extra baseline pass and PSNR/SSIM computation).
4. Drag the split slider (damaged vs result) and open the 5-panel pipeline view (Ground truth, Masked, Coarse, Baseline, RL-GAN).
5. **Ultra-Precision** mode runs 60 gradient steps on the chosen adapter against the original image; use it to illustrate how closely an adapter can fit a given image (the same idea as stage D).
6. For a side-by-side comparison of all four adapters use `python main.py verify-strategies ...` or open `results/strategy_verify/*.png`.

PSNR/SSIM in the demo are measured against the image the hole was cut from.

---

## ❓ 8. Likely Questions and Answers

**Q1. Why reinforcement learning instead of an end-to-end multi-branch network?**
Choosing among discrete strategies with a metric-based, non-differentiable reward is naturally a decision problem. The controller leaves the backbone and adapters untouched and adds an explicit cost term, so the same adapters can be reused with any policy.

**Q2. Why a one-step bandit (γ = 0)?**
There is one decision per image and no state transition to credit. A multi-step environment with a STOP action is included (`env_multistep.py`, `rl_agent_multistep.yaml`) for iterative refinement.

**Q3. Why DeepFill-v2 and not a diffusion model?**
DeepFill-v2 is single pass, has pretrained weights, and its two-stage structure gives a natural attachment point for adapters. Diffusion models need many denoising steps per image.

**Q4. Why does stage C differ from stage B on some scenes?**
Because the policy selected a different adapter there (Boundary on the lake and sunset scenes). Each adapter has its own objective, and the reward combines PSNR, SSIM, LPIPS, a discriminator term and an action cost, so the policy's choice is not tuned to PSNR alone. More PPO training, a richer state and a different reward mix are natural next experiments.

**Q5. How do you check that the four adapters behave differently?**
`verify-strategies` applies all four adapters to the same coarse composite of several images, prints per-strategy PSNR/SSIM and the best-strategy counts, and gates on no single strategy winning more than 80 % of the images. It also writes side-by-side grids.

**Q6. What does each adapter do?**
Global is a balanced residual pass; Local emphasises hole fidelity (6× hole L1); Boundary concentrates on seam continuity with a Sobel-gradient loss on the hole ring; Texture adds high-frequency detail with a Laplacian branch and an FFT-magnitude loss.

**Q7. What is stage D for?**
It optimises the selected adapter on the test image with the ground truth for 400 steps. It demonstrates the representational capacity of the adapters (37–43 dB), and it is reported separately from the feed-forward stages A–C.

**Q8. What data did you use?**
Adapters and PPO use COCO val2017 (5,000 images, 256×256) with random stroke masks (5 adapter epochs). The backbone is the nipponjo/deepfillv2-pytorch Places2 model. Evaluation scenes are the sample photographs in `data/raw/sample_images`, and 100 fixed masks (25 per severity bucket) are available for severity studies.

**Q9. What would you do next?**
Evaluate across the four severity buckets with the fixed masks and several seeds; compare PPO with a supervised "best adapter" classifier and with always using the best single adapter; try multi-step refinement with STOP; and compare against a pretrained LaMa and a diffusion inpainting model.

---

## 📋 9. Pre-Presentation Checklist

- [ ] Activate `.\.venv\Scripts\Activate.ps1` (Python 3.12, torch 2.6 CUDA).
- [ ] `checkpoints/pretrained/deepfillv2_places2.pth`, `checkpoints/adapters/adapters_final.pt` and `checkpoints/rl_agent_bandit/ppo_bandit_final.zip` are present.
- [ ] `python app.py` shows backbone, adapters and PPO all loaded; test one Instant Forward Pass end to end.
- [ ] Keep `results/demo_image_*.png` and `results/strategy_verify/*.png` open as a fallback if the live demo fails.
- [ ] Have `docs/RL_GAN_Inpainting_Report.pdf` ready for the diagrams.
- [ ] Browser zoom 100%.
