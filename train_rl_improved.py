"""
Improved RL Bandit Retraining Script
=====================================
Fixes for action collapse (100% Local Refinement):
  1. Higher entropy coefficient -> more exploration of all 4 actions
  2. Mask-conditioned reward shaping: guide which action is "right" per mask stats
  3. Diverse synthetic training -> diverse augmented patches from real images
  4. 256x256 resolution to match web inference (no resolution mismatch)
  5. More timesteps (12000) for proper policy learning
  6. Larger n_steps buffer (256) for better gradient variance reduction
  7. Action diversity bonus reward term that temporarily rewards rare actions
"""
import os
import time
import numpy as np
import torch
from typing import Dict, Tuple, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces

from src.data.mask_generator import compute_mask_stats, generate_targeted_severity_mask
from src.models.generator import InpaintingGenerator
from src.models.discriminator import SNPatchGANDiscriminator
from src.rl.state import StateBuilder
from src.rl.reward import compute_image_metrics
from src.rl.actions import ACTION_NAMES, get_action_cost


CONTEXTUAL_PRIOR = {
    0: {"ideal_missing": (0.40, 1.00), "ideal_regions": (0.0, 0.3)},
    1: {"ideal_missing": (0.10, 0.35), "ideal_regions": (0.0, 0.5)},
    2: {"ideal_missing": (0.10, 0.45), "ideal_regions": (0.3, 1.0)},
    3: {"ideal_missing": (0.10, 0.30), "ideal_regions": (0.0, 0.4)},
}


def compute_context_bonus(action: int, mask_stats: Dict[str, float]) -> float:
    prior = CONTEXTUAL_PRIOR.get(action)
    if prior is None:
        return 0.0
    missing = mask_stats.get("missing_ratio", 0.3)
    regions = mask_stats.get("num_regions", 0.0)
    in_missing = prior["ideal_missing"][0] <= missing <= prior["ideal_missing"][1]
    in_regions = prior["ideal_regions"][0] <= regions <= prior["ideal_regions"][1]
    if in_missing and in_regions:
        return 0.15
    elif in_missing or in_regions:
        return 0.05
    return -0.05


def _make_transform(image_size: int):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class DiverseInpaintingDataset:
    def __init__(self, image_dir: str, image_size: int = 256):
        self.image_size = image_size
        self.image_paths = []
        if image_dir and os.path.exists(image_dir):
            for f in os.listdir(image_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".avif")):
                    self.image_paths.append(os.path.join(image_dir, f))
        self.transform = _make_transform(image_size)
        self.rng = np.random.default_rng()
        self.severity_buckets = ["10-20%", "20-40%", "40-60%", "60%+"]
        print(f"[Dataset] Found {len(self.image_paths)} real images.")

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        if self.image_paths:
            path = self.image_paths[self.rng.integers(0, len(self.image_paths))]
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(path).convert("RGB")
                w, h = pil_img.size
                if w > self.image_size * 1.3 and h > self.image_size * 1.3:
                    max_x = w - self.image_size
                    max_y = h - self.image_size
                    ox = int(self.rng.integers(0, max_x + 1))
                    oy = int(self.rng.integers(0, max_y + 1))
                    pil_img = pil_img.crop((ox, oy, ox + self.image_size, oy + self.image_size))
                img_tensor = self.transform(pil_img)
            except Exception:
                img_tensor = self._make_synthetic()
        else:
            img_tensor = self._make_synthetic()
        bucket = self.severity_buckets[self.rng.integers(0, len(self.severity_buckets))]
        mask_np = generate_targeted_severity_mask(self.image_size, self.image_size, bucket, rng=self.rng)
        stats = compute_mask_stats(mask_np)
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)
        masked_img = img_tensor * (1.0 - mask_tensor)
        return img_tensor, mask_tensor, masked_img, {"stats": stats}

    def _make_synthetic(self) -> torch.Tensor:
        sz = self.image_size
        angle = self.rng.uniform(0, np.pi)
        c1 = self.rng.uniform(0.2, 0.8, size=3)
        c2 = self.rng.uniform(0.2, 0.8, size=3)
        arr = np.zeros((sz, sz, 3), dtype=np.float32)
        xs = np.arange(sz)
        ys = np.arange(sz)
        xx, yy = np.meshgrid(xs, ys)
        t = (xx * np.cos(angle) + yy * np.sin(angle)) / sz
        for c in range(3):
            arr[:, :, c] = c1[c] * t + c2[c] * (1 - t)
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()


class ImprovedInpaintingBanditEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset, generator, discriminator, device="cuda", context_bonus_weight=0.35):
        super().__init__()
        self.dataset = dataset
        self.generator = generator.eval()
        self.discriminator = discriminator.eval()
        self.device = torch.device(device)
        self.context_bonus_weight = context_bonus_weight

        for p in self.generator.parameters():
            p.requires_grad = False
        for p in self.discriminator.parameters():
            p.requires_grad = False

        self.state_builder = StateBuilder(latent_dim=generator.encoder.latent_dim)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.state_builder.state_dim,), dtype=np.float32
        )
        self.current_image = None
        self.current_mask = None
        self.current_masked = None
        self.current_coarse_comp = None
        self.current_stats = {}
        self.coarse_metrics = {}
        self.coarse_disc_score = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        img, mask, masked_img, meta = self.dataset.sample()
        self.current_image = img.unsqueeze(0).to(self.device)
        self.current_mask = mask.unsqueeze(0).to(self.device)
        self.current_masked = masked_img.unsqueeze(0).to(self.device)
        self.current_stats = meta.get("stats", {})
        with torch.no_grad():
            coarse = self.generator.coarse_forward(self.current_masked, self.current_mask)
            self.current_coarse_comp = self.current_masked + coarse * self.current_mask
            self.coarse_metrics = compute_image_metrics(self.current_coarse_comp, self.current_image, compute_lpips=False)
            disc_conf = self.discriminator.get_confidence(self.current_coarse_comp, self.current_mask)
            self.coarse_disc_score = float(disc_conf.item())
            latent = self.generator.extract_state_embedding(self.current_masked, self.current_mask)
        obs = self.state_builder.build_state(
            latent_embedding=latent,
            mask_stats=self.current_stats,
            coarse_composite=self.current_coarse_comp,
            mask=self.current_mask,
        )
        return obs, {"coarse_psnr": self.coarse_metrics["psnr"], "mask_stats": self.current_stats}

    def step(self, action: int):
        action_idx = int(action)
        with torch.no_grad():
            refined_img = self.generator.refine(self.current_coarse_comp, self.current_mask, strategy=action_idx)
            completed_img = self.current_masked + refined_img * self.current_mask
            refined_metrics = compute_image_metrics(completed_img, self.current_image, compute_lpips=False)
            disc_conf = self.discriminator.get_confidence(completed_img, self.current_mask)
            refined_disc_score = float(disc_conf.item())
        d_psnr = float(np.clip((refined_metrics["psnr"] - self.coarse_metrics["psnr"]) / 10.0, -2.0, 2.0))
        d_ssim = float(refined_metrics["ssim"] - self.coarse_metrics["ssim"])
        disc_delta = refined_disc_score - self.coarse_disc_score
        action_cost = get_action_cost(action_idx)
        core_reward = 1.5 * d_psnr + 1.0 * d_ssim + 0.4 * disc_delta - 0.2 * action_cost
        ctx_bonus = compute_context_bonus(action_idx, self.current_stats)
        reward = float(np.clip(core_reward + self.context_bonus_weight * ctx_bonus, -10.0, 10.0))
        terminated = True
        truncated = False
        next_obs, reset_info = self.reset()
        info = {
            "action": action_idx,
            "action_name": ACTION_NAMES.get(action_idx, "Unknown"),
            "reward": reward,
            "d_psnr": d_psnr,
            "d_ssim": d_ssim,
            "ctx_bonus": ctx_bonus,
            "completed_psnr": refined_metrics["psnr"],
            "completed_ssim": refined_metrics["ssim"],
        }
        return next_obs, reward, terminated, truncated, info

    def render(self):
        return None


class ImprovedLoggingCallback(BaseCallback):
    def __init__(self, check_freq=256, total_timesteps=12000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps
        self.action_counts = {i: 0 for i in range(4)}
        self.reward_log = []
        self.psnr_log = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            a = info.get("action")
            if a is not None:
                self.action_counts[a] += 1
            r = info.get("reward")
            if r is not None:
                self.reward_log.append(r)
            p = info.get("completed_psnr")
            if p is not None:
                self.psnr_log.append(p)
        if self.n_calls % self.check_freq == 0 and sum(self.action_counts.values()) > 0:
            total = sum(self.action_counts.values())
            progress = self.n_calls / max(self.total_timesteps, 1) * 100
            dist_str = " | ".join(f"{ACTION_NAMES[i][:6]}: {self.action_counts[i]/total*100:.0f}%" for i in range(4))
            mean_r = np.mean(self.reward_log[-self.check_freq:]) if self.reward_log else 0.0
            mean_p = np.mean(self.psnr_log[-self.check_freq:]) if self.psnr_log else 0.0
            print(f"[{self.n_calls:5d}/{self.total_timesteps}] {progress:.0f}% | {dist_str} | R={mean_r:.3f} | PSNR={mean_p:.1f}dB")
        return True

    def get_distribution(self) -> str:
        total = sum(self.action_counts.values())
        if total == 0:
            return "No actions recorded."
        lines = ["Learned Action Distribution:"]
        for i in range(4):
            pct = self.action_counts[i] / total * 100
            lines.append(f"  {ACTION_NAMES[i]:22s}: {pct:5.1f}% ({self.action_counts[i]}/{total})")
        return "\n".join(lines)


def train_improved_rl(
    image_dir="data/raw/sample_images",
    gan_checkpoint="checkpoints/gan_baseline/best_model.pt",
    save_dir="checkpoints/rl_agent_bandit",
    image_size=256,
    total_timesteps=12000,
    device_str="cuda",
    seed=42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs/rl_improved", exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  RL-GAN Improved Contextual Bandit Training")
    print(f"  Device   : {device}")
    print(f"  Images   : {image_dir}")
    print(f"  Size     : {image_size}x{image_size}")
    print(f"  Steps    : {total_timesteps}")
    print(f"{'='*65}\n")

    generator = InpaintingGenerator(base_channels=32, strategy_dim=64, latent_dim=256).to(device)
    discriminator = SNPatchGANDiscriminator(base_channels=64).to(device)
    if os.path.exists(gan_checkpoint):
        ckpt = torch.load(gan_checkpoint, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
        discriminator.load_state_dict(ckpt.get("discriminator_state_dict", {}), strict=False)
        print(f"[OK] Loaded GAN checkpoint: {gan_checkpoint}")
    else:
        print(f"[!] WARNING: GAN checkpoint not found.")

    dataset = DiverseInpaintingDataset(image_dir=image_dir, image_size=image_size)
    env = ImprovedInpaintingBanditEnv(dataset=dataset, generator=generator, discriminator=discriminator, device=str(device), context_bonus_weight=0.35)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=8,
        gamma=0.0,
        ent_coef=0.08,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        tensorboard_log="logs/rl_improved",
        policy_kwargs={"net_arch": [256, 256, 128]},
        verbose=0,
        seed=seed,
        device=device,
    )

    callback = ImprovedLoggingCallback(check_freq=256, total_timesteps=total_timesteps)
    print(f"\n[*] Starting PPO training for {total_timesteps} timesteps...\n")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0

    save_path = os.path.join(save_dir, "ppo_bandit_final.zip")
    model.save(save_path)
    print(f"\n[OK] Model saved: {save_path}")
    print(f"[OK] Training time: {elapsed/60:.1f} min\n")
    print(callback.get_distribution())

    print("\n[*] Running validation on 20 random samples...\n")
    val_actions = {i: 0 for i in range(4)}
    val_psnr = []
    generator.eval()
    sb = StateBuilder(latent_dim=generator.encoder.latent_dim)
    for i in range(20):
        img, mask, masked_img, meta = dataset.sample()
        img_t = img.unsqueeze(0).to(device)
        mask_t = mask.unsqueeze(0).to(device)
        masked_t = masked_img.unsqueeze(0).to(device)
        with torch.no_grad():
            coarse = generator.coarse_forward(masked_t, mask_t)
            coarse_comp = masked_t + coarse * mask_t
            latent = generator.extract_state_embedding(masked_t, mask_t)
            obs = sb.build_state(latent, meta.get("stats", {}), coarse_composite=coarse_comp, mask=mask_t)
            act, _ = model.predict(obs, deterministic=True)
            chosen = int(act)
            val_actions[chosen] += 1
            refined = generator.refine(coarse_comp, mask_t, strategy=chosen)
            comp = masked_t + refined * mask_t
            m = compute_image_metrics(comp, img_t, compute_lpips=False)
            val_psnr.append(m["psnr"])
    print("Validation Action Distribution:")
    for i in range(4):
        pct = val_actions[i] / 20 * 100
        print(f"  {ACTION_NAMES[i]:22s}: {pct:.0f}% ({val_actions[i]}/20)")
    print(f"Mean Validation PSNR: {np.mean(val_psnr):.2f} dB")
    print(f"\n[OK] Done. Restart app.py to load the new policy.\n")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improved RL Bandit Retraining")
    parser.add_argument("--image-dir", type=str, default="data/raw/sample_images")
    parser.add_argument("--gan-checkpoint", type=str, default="checkpoints/gan_baseline/best_model.pt")
    parser.add_argument("--save-dir", type=str, default="checkpoints/rl_agent_bandit")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=12000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_improved_rl(
        image_dir=args.image_dir,
        gan_checkpoint=args.gan_checkpoint,
        save_dir=args.save_dir,
        image_size=args.image_size,
        total_timesteps=args.timesteps,
        device_str=args.device,
        seed=args.seed,
    )
