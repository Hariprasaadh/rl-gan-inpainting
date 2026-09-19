"""
Multi-image evaluation with honest 4-stage ablation:
  Stage A: DeepFillV2 backbone only (no adapters, no RL)
  Stage B: Best fixed adapter (adapter 0 — Global) — no RL selection
  Stage C: PPO-selected adapter — NO test-time optimization
  Stage D: PPO-selected adapter + 400-step AdamW refinement (labelled)

Stages A–C are the scientifically valid comparisons.
Stage D is shown separately to isolate the optimization contribution.
"""
import os
import sys
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Use RLInpaintingModel (backbone + adapters)
try:
    from src.models.rl_inpainting_model import RLInpaintingModel
    _USE_RL_MODEL = True
except ImportError:
    from src.models.generator import InpaintingGenerator as RLInpaintingModel  # fallback
    _USE_RL_MODEL = False

from src.data.dataset import InpaintingDataset
from src.data.transforms import denormalize_image, feather_composite
from src.models.losses import MaskedL1Loss
from src.rl.reward import compute_image_metrics
from src.rl.actions import ACTION_NAMES
from src.rl.state import StateBuilder
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_header_labels(image_grid: Image.Image, titles: list) -> Image.Image:
    """Adds a clean dark banner with panel titles above the comparison grid."""
    w, h = image_grid.size
    panel_w = w // len(titles)
    header_h = 36

    new_img = Image.new("RGB", (w, h + header_h), color=(16, 20, 28))
    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, title in enumerate(titles):
        x_center = i * panel_w + panel_w // 2
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (x_center - tw // 2, (header_h - th) // 2),
            title,
            fill=(240, 244, 252),
            font=font,
        )
        if i > 0:
            draw.line(
                [(i * panel_w, 0), (i * panel_w, h + header_h)],
                fill=(42, 50, 68),
                width=2,
            )

    new_img.paste(image_grid, (0, header_h))
    return new_img


def build_controlled_mask(test_idx: int, device: torch.device) -> torch.Tensor:
    """Generates an organic, edge-safe defect mask tailored for each test image."""
    mask_np = np.zeros((256, 256), dtype=np.uint8)

    if test_idx == 1:
        cv2.line(mask_np, (75, 95), (115, 155), 1, 15)
        cv2.line(mask_np, (115, 155), (160, 120), 1, 15)
        cv2.circle(mask_np, (115, 155), 9, 1, -1)
    elif test_idx == 2:
        cv2.line(mask_np, (90, 80), (130, 130), 1, 15)
        cv2.line(mask_np, (130, 130), (160, 160), 1, 15)
        cv2.line(mask_np, (160, 160), (180, 200), 1, 15)
        cv2.circle(mask_np, (130, 130), 10, 1, -1)
    else:
        cv2.line(mask_np, (100, 110), (145, 155), 1, 15)
        cv2.line(mask_np, (145, 155), (185, 125), 1, 15)
        cv2.circle(mask_np, (145, 155), 9, 1, -1)

    return torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)


def build_generator(device: torch.device) -> nn.Module:
    """Construct and load RLInpaintingModel with backbone + adapters."""
    if _USE_RL_MODEL:
        model = RLInpaintingModel(cnum=48, cnum_in=5, strategy_dim=64, latent_dim=256).to(device)
        backbone_ckpt = "checkpoints/pretrained/deepfillv2_places2.pth"
        adapters_ckpt = "checkpoints/adapters/adapters_final.pt"

        if os.path.exists(backbone_ckpt):
            try:
                model.load_pretrained_backbone(backbone_ckpt, strict=False)
                print(f"  [Model] DeepFillV2 backbone loaded from {backbone_ckpt}", flush=True)
            except Exception as e:
                print(f"  [Model] WARNING: backbone load failed: {e}", flush=True)
        else:
            print(f"  [Model] WARNING: backbone checkpoint not found at {backbone_ckpt}", flush=True)

        if os.path.exists(adapters_ckpt):
            model.load_adapters(adapters_ckpt)
            print(f"  [Model] Adapters loaded from {adapters_ckpt}", flush=True)
        else:
            print(f"  [Model] WARNING: adapters checkpoint not found at {adapters_ckpt}", flush=True)
    else:
        # Legacy fallback
        from src.models.generator import InpaintingGenerator
        model = InpaintingGenerator(base_channels=32, strategy_dim=64, latent_dim=256).to(device)
        gan_ckpt = "checkpoints/gan_baseline/best_model.pt"
        if os.path.exists(gan_ckpt):
            ckpt = torch.load(gan_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["generator_state_dict"], strict=False)
            print(f"  [Model] Legacy GAN loaded from {gan_ckpt}", flush=True)

    return model


def run_refinement_optimization(
    generator: nn.Module,
    masked: torch.Tensor,
    mask: torch.Tensor,
    image_gt: torch.Tensor,
    strategy: int,
    steps: int = 400,
) -> None:
    """
    Test-time AdamW refinement using ground-truth supervision.
    NOTE: This uses the ground-truth image in the loss — valid for
    demonstrating convergence potential, but NOT a fair generalization test.
    Label results from this function as 'PPO + Refinement (GT-supervised)'.
    """
    l1_fn = MaskedL1Loss(hole_weight=35.0, valid_weight=1.0)
    generator.train()
    opt = torch.optim.AdamW(generator.parameters(), lr=0.0015, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=0.00002)

    for step in range(steps):
        opt.zero_grad()
        out = generator(masked, mask, strategy=strategy)
        loss_c, _ = l1_fn(out["coarse"], image_gt, mask)
        loss_r, _ = l1_fn(out["refined"], image_gt, mask)

        pred = out["refined"]
        p_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        g_dx = torch.abs(image_gt[:, :, :, 1:] - image_gt[:, :, :, :-1])
        m_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        loss_gx = torch.sum(torch.abs(p_dx - g_dx) * m_x) / (torch.sum(m_x) * 3 + 1e-6)

        p_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        g_dy = torch.abs(image_gt[:, :, 1:, :] - image_gt[:, :, :-1, :])
        m_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        loss_gy = torch.sum(torch.abs(p_dy - g_dy) * m_y) / (torch.sum(m_y) * 3 + 1e-6)

        total_loss = loss_c + loss_r + 8.0 * (loss_gx + loss_gy)
        total_loss.backward()
        opt.step()
        sched.step()

        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{steps} | Loss: {total_loss.item():.4f}", flush=True)

    generator.eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multi_test():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})",
        flush=True,
    )

    dataset = InpaintingDataset(
        image_dir="data/raw/sample_images",
        image_size=256,
        is_train=False,
        fixed_seed=42,
    )
    print(f"Total dataset images found: {len(dataset.image_paths)}", flush=True)

    test_specs = [
        {
            "index": 1,
            "slug": "alpine_lake",
            "title": "Alpine Lake & Snowy Mountains",
            "description": "Turquoise mountain lake with crystal water reflections and forested ridges.",
        },
        {
            "index": 2,
            "slug": "lush_meadow",
            "title": "Lush Alpine Meadow & Distant Peaks",
            "description": "Vibrant green grassy slopes, scattered fir trees, and fluffy summer clouds.",
        },
        {
            "index": 5,
            "slug": "sunset_peaks",
            "title": "Sunset Mountain Horizon & Dusk Sky",
            "description": "Jagged snow-covered crests under pink/purple twilight horizon gradients.",
        },
    ]

    print("\nSelected 3 Diverse Real-World Test Targets:", flush=True)
    for t in test_specs:
        fname = os.path.basename(dataset.image_paths[t["index"]])
        print(f"  Target: [Index {t['index']}] {fname} -> {t['title']}", flush=True)

    # ------------------------------------------------------------------
    # Load PPO agent
    # ------------------------------------------------------------------
    rl_agent = None
    rl_ckpt_path = "checkpoints/rl_agent_bandit/ppo_bandit_final.zip"
    ppo_trained = False
    if os.path.exists(rl_ckpt_path):
        rl_agent = PPO.load(rl_ckpt_path, device=device)
        ppo_trained = True
        print(f"\n[PPO] Loaded trained PPO agent from: {rl_ckpt_path}", flush=True)
    else:
        print(
            "\n[PPO] WARNING: No PPO checkpoint found at checkpoints/rl_agent_bandit/ppo_bandit_final.zip",
            flush=True,
        )
        print("[PPO] Stage C will use default action=0 (Global) as untrained fallback.", flush=True)

    results_summary = []

    for test_idx, spec in enumerate(test_specs, 1):
        sample_idx = spec["index"]
        img_filename = os.path.basename(dataset.image_paths[sample_idx])
        print("\n" + "=" * 78, flush=True)
        print(f"TEST {test_idx}/3: [Index {sample_idx}] {img_filename} ({spec['title']})", flush=True)
        print("=" * 78, flush=True)

        sample = dataset[sample_idx]
        image = sample["image"].unsqueeze(0).to(device)
        mask = build_controlled_mask(test_idx, device)
        masked = image * (1.0 - mask)

        # Build fresh model for inference stages A/B/C
        print("  [Model] Loading RLInpaintingModel...", flush=True)
        generator = build_generator(device)
        generator.eval()

        # ------------------------------------------------------------------
        # STAGE A: Backbone only — no adapters, no RL
        # strategy=None -> backbone refine_forward only, no adapter applied
        # ------------------------------------------------------------------
        print("\n  [Stage A] Backbone-only inference (DeepFillV2, no adapters)...", flush=True)
        with torch.no_grad():
            out_a = generator(masked, mask, strategy=None)
            comp_a = feather_composite(image, out_a["completed"], mask, calibrate_color=False)
            m_a = compute_image_metrics(comp_a, image, compute_lpips=False)
        print(f"  [Stage A] PSNR: {m_a['psnr']:.2f} dB | SSIM: {m_a['ssim']:.4f}", flush=True)

        # ------------------------------------------------------------------
        # STAGE B: Best fixed adapter (adapter 0 — Global) — no RL selection
        # ------------------------------------------------------------------
        print("\n  [Stage B] Fixed adapter 0 (Global) — no RL...", flush=True)
        with torch.no_grad():
            out_b = generator(masked, mask, strategy=0)
            comp_b = feather_composite(image, out_b["completed"], mask, calibrate_color=False)
            m_b = compute_image_metrics(comp_b, image, compute_lpips=False)
        print(f"  [Stage B] PSNR: {m_b['psnr']:.2f} dB | SSIM: {m_b['ssim']:.4f}", flush=True)

        # ------------------------------------------------------------------
        # STAGE C: PPO-selected adapter — pure RL inference, NO optimization
        # This is the scientifically valid RL contribution measurement
        # ------------------------------------------------------------------
        chosen_action = 0  # default if no PPO
        if rl_agent is not None:
            with torch.no_grad():
                coarse_init = generator.coarse_forward(masked, mask)
                coarse_init_comp = masked + coarse_init * mask
                sb = StateBuilder(latent_dim=generator.encoder.latent_dim)
                latent = generator.extract_state_embedding(masked, mask)
                obs = sb.build_state(
                    latent,
                    sample["stats"],
                    coarse_composite=coarse_init_comp,
                    mask=mask,
                )
                act, _ = rl_agent.predict(obs, deterministic=True)
                chosen_action = int(act)

        action_name = ACTION_NAMES.get(chosen_action, f"Strategy {chosen_action}")
        rl_source = "PPO-trained policy" if ppo_trained else "default fallback (PPO not trained)"
        print(f"\n  [Stage C] PPO action: {chosen_action} ({action_name}) — {rl_source}", flush=True)

        with torch.no_grad():
            out_c = generator(masked, mask, strategy=chosen_action)
            comp_c = feather_composite(image, out_c["completed"], mask, calibrate_color=False)
            m_c = compute_image_metrics(comp_c, image, compute_lpips=False)
        print(f"  [Stage C] PSNR: {m_c['psnr']:.2f} dB | SSIM: {m_c['ssim']:.4f}", flush=True)

        # ------------------------------------------------------------------
        # STAGE D: PPO-selected adapter + 400-step GT-supervised refinement
        # Labelled honestly — uses ground truth in loss
        # ------------------------------------------------------------------
        print(
            f"\n  [Stage D] PPO ({action_name}) + 400-step AdamW refinement (GT-supervised)...",
            flush=True,
        )
        # Re-build generator fresh so Stage C model weights are not contaminated
        generator_d = build_generator(device)
        run_refinement_optimization(generator_d, masked, mask, image, chosen_action, steps=400)

        with torch.no_grad():
            coarse_d = generator_d.coarse_forward(masked, mask)
            coarse_comp_d = masked + coarse_d * mask
            refined_d = generator_d.refine(coarse_comp_d, mask, strategy=chosen_action)
            comp_d = feather_composite(image, refined_d, mask, calibrate_color=False)
            m_d = compute_image_metrics(comp_d, image, compute_lpips=False)
        print(f"  [Stage D] PSNR: {m_d['psnr']:.2f} dB | SSIM: {m_d['ssim']:.4f}", flush=True)

        # ------------------------------------------------------------------
        # Build 7-panel comparison grid:
        # GT | Masked | Coarse | StageA | StageB | StageC | StageD
        # ------------------------------------------------------------------
        with torch.no_grad():
            coarse_vis = generator.coarse_forward(masked, mask)
            coarse_comp_vis = masked + coarse_vis * mask

        def to_pil(t: torch.Tensor) -> Image.Image:
            return Image.fromarray(
                denormalize_image(t[0], to_uint8=True).permute(1, 2, 0).cpu().numpy()
            )

        panels = [
            to_pil(image),
            to_pil(masked),
            to_pil(coarse_comp_vis),
            to_pil(comp_a),
            to_pil(comp_b),
            to_pil(comp_c),
            to_pil(comp_d),
        ]
        titles = [
            "1. Ground Truth",
            "2. Masked",
            "3. Coarse",
            f"A. Backbone ({m_a['psnr']:.1f}dB)",
            f"B. Fixed Adapter ({m_b['psnr']:.1f}dB)",
            f"C. PPO-only ({m_c['psnr']:.1f}dB)",
            f"D. PPO+Refine ({m_d['psnr']:.1f}dB)",
        ]

        w, h = panels[0].size
        grid = Image.new("RGB", (w * len(panels), h))
        for i, p in enumerate(panels):
            grid.paste(p, (i * w, 0))

        titled_grid = add_header_labels(grid, titles)
        out_path = f"results/demo_image_{test_idx}_{spec['slug']}.png"
        titled_grid.save(out_path)
        print(f"  [Output] Saved -> {out_path}", flush=True)

        results_summary.append({
            "test_idx": test_idx,
            "title": spec["title"],
            "description": spec["description"],
            "filename": img_filename,
            "out_path": out_path,
            "action": f"Action {chosen_action}: {action_name}",
            "ppo_trained": ppo_trained,
            # Stage metrics
            "psnr_a": m_a["psnr"], "ssim_a": m_a["ssim"],
            "psnr_b": m_b["psnr"], "ssim_b": m_b["ssim"],
            "psnr_c": m_c["psnr"], "ssim_c": m_c["ssim"],
            "psnr_d": m_d["psnr"], "ssim_d": m_d["ssim"],
            # Deltas vs backbone
            "delta_b": m_b["psnr"] - m_a["psnr"],
            "delta_c": m_c["psnr"] - m_a["psnr"],
            "delta_d": m_d["psnr"] - m_a["psnr"],
        })

    # ------------------------------------------------------------------
    # Write Markdown Report
    # ------------------------------------------------------------------
    report_file = "results/multi_image_evaluation_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# RL-GAN Multi-Image Ablation Evaluation Report\n\n")
        f.write(
            "> **Architecture**: Places2-pretrained DeepFillV2 backbone + COCO-val2017-trained adapters"
            f" + {'PPO-trained' if ppo_trained else 'untrained (default)'} RL controller.\n\n"
        )

        if not ppo_trained:
            f.write(
                "> WARNING: PPO checkpoint not found. Stage C uses action=0 as default fallback."
                " Run `python main.py train-rl` to generate the PPO agent.\n\n"
            )

        f.write("## Ablation Stage Key\n\n")
        f.write("| Stage | Description | Scientifically Valid? |\n")
        f.write("|:---:|---|:---:|\n")
        f.write("| **A** | DeepFillV2 backbone only (no adapters, no RL) | Yes |\n")
        f.write("| **B** | Fixed adapter 0 (Global) — no RL selection | Yes |\n")
        f.write(f"| **C** | {'PPO-trained' if ppo_trained else 'Default (no PPO)'} adapter selection — pure inference | Yes |\n")
        f.write("| **D** | Stage C + 400-step GT-supervised AdamW refinement | Uses ground truth |\n\n")

        f.write("## Quantitative Ablation Table\n\n")
        f.write(
            "| # | Scene | RL Action | "
            "A: Backbone | B: Fixed Adapter | C: PPO-only | D: PPO+Refine | "
            "Delta C vs A | Delta D vs A |\n"
        )
        f.write("|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results_summary:
            f.write(
                f"| **{r['test_idx']}** | {r['title']} | {r['action']} | "
                f"{r['psnr_a']:.2f} dB | {r['psnr_b']:.2f} dB | **{r['psnr_c']:.2f} dB** | "
                f"{r['psnr_d']:.2f} dB | **{r['delta_c']:+.2f} dB** | {r['delta_d']:+.2f} dB |\n"
            )

        f.write("\n---\n\n## Detailed Scene Analysis\n\n")
        for r in results_summary:
            f.write(f"### Test {r['test_idx']}: {r['title']}\n")
            f.write(f"- **Scene**: {r['description']}\n")
            f.write(f"- **Source**: `{r['filename']}`\n")
            f.write(f"- **PPO**: {'Trained PPO agent' if r['ppo_trained'] else 'Not trained — used default action=0'}\n")
            f.write(f"- **Selected Action**: {r['action']}\n\n")
            f.write("| Stage | PSNR | SSIM | vs Backbone |\n")
            f.write("|---|:---:|:---:|:---:|\n")
            f.write(f"| A: Backbone only | {r['psnr_a']:.2f} dB | {r['ssim_a']:.4f} | baseline |\n")
            f.write(f"| B: Fixed adapter | {r['psnr_b']:.2f} dB | {r['ssim_b']:.4f} | {r['delta_b']:+.2f} dB |\n")
            f.write(f"| C: PPO-only (pure inference) | {r['psnr_c']:.2f} dB | {r['ssim_c']:.4f} | **{r['delta_c']:+.2f} dB** |\n")
            f.write(f"| D: PPO+Refinement (GT-supervised) | {r['psnr_d']:.2f} dB | {r['ssim_d']:.4f} | {r['delta_d']:+.2f} dB |\n")
            f.write(f"\n- **Output**: `{os.path.basename(r['out_path'])}`\n\n")

        f.write("\n---\n\n## Methodology Notes\n\n")
        f.write(
            "- **Stage C (PPO-only)** is the correct measurement of the RL contribution.\n"
            "  Single forward pass with PPO-selected adapter — no optimization, no ground truth used.\n\n"
            "- **Stage D (PPO + Refinement)** applies 400 AdamW steps with ground-truth in the loss.\n"
            "  This is test-time supervised fine-tuning — not a valid generalization metric.\n\n"
            "- **Backbone**: DeepFillV2 pretrained on Places2 (frozen during adapter and RL training).\n"
            "- **Adapters**: Trained 3 epochs on COCO val2017 (5,000 images, synthetic masks).\n"
            f"- **PPO**: {'Trained on COCO val2017 via stable-baselines3 PPO.' if ppo_trained else 'Not yet trained. Run `python main.py train-rl`.'}\n"
        )

    print("\n" + "=" * 78, flush=True)
    print(f"All 3 tests completed! Report saved to {report_file}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    run_multi_test()
