import os
import argparse
from typing import Optional, Dict
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.data.dataset import InpaintingDataset
from src.models.discriminator import SNPatchGANDiscriminator
from src.rl.actions import ACTION_NAMES
from src.rl.env_bandit import InpaintingBanditEnv
from src.training.trainer_utils import load_config, get_device, load_checkpoint

# Prefer new model; fallback to legacy
try:
    from src.models.rl_inpainting_model import RLInpaintingModel as _GeneratorCls
    _USE_RL_MODEL = True
except ImportError:
    from src.models.generator import InpaintingGenerator as _GeneratorCls
    _USE_RL_MODEL = False


class RLBanditLoggingCallback(BaseCallback):
    """Logs action distribution and reward component breakdown during PPO training."""

    def __init__(self, check_freq: int = 128, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.action_counts: Dict[int, int] = {i: 0 for i in range(4)}
        self.reward_terms: Dict[str, list] = {
            "d_psnr": [],
            "d_ssim": [],
            "d_lpips": [],
            "disc_delta": [],
            "action_cost": [],
            "reward": [],
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            action = info.get("action")
            if action is not None and action in self.action_counts:
                self.action_counts[action] += 1

            for key in self.reward_terms:
                if key in info:
                    self.reward_terms[key].append(info[key])

        if self.n_calls % self.check_freq == 0:
            total_actions = sum(self.action_counts.values())
            if total_actions > 0 and self.logger is not None:
                for act_idx, count in self.action_counts.items():
                    pct = (count / total_actions) * 100.0
                    name = ACTION_NAMES.get(act_idx, f"Action_{act_idx}")
                    self.logger.record(f"actions/{name}_pct", pct)

                for key, vals in self.reward_terms.items():
                    if len(vals) > 0:
                        self.logger.record(f"reward_components/mean_{key}", float(np.mean(vals[-self.check_freq:])))

        return True

    def get_distribution_summary(self) -> str:
        total = sum(self.action_counts.values())
        if total == 0:
            return "No actions recorded."
        lines = [f"Total Steps: {total}"]
        for act_idx, count in self.action_counts.items():
            pct = (count / total) * 100.0
            name = ACTION_NAMES.get(act_idx, f"Action_{act_idx}")
            lines.append(f"  {name:22s}: {pct:5.1f}% ({count}/{total})")
        return "\n".join(lines)


def train_rl_bandit(config_path: str, override_timesteps: Optional[int] = None) -> PPO:
    """Train PPO controller over frozen pretrained GAN (Experiment B)."""
    cfg = load_config(config_path)
    seed = cfg["rl"].get("seed", 42)
    seed_everything(seed)

    device = get_device(cfg["rl"].get("device", "cpu"))
    os.makedirs(cfg["rl"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["rl"]["log_dir"], exist_ok=True)

    logger = setup_logger("train_rl_bandit", log_file=os.path.join(cfg["rl"]["log_dir"], "train.log"))
    logger.info("Initializing Experiment B: Frozen GAN + PPO Bandit Controller")
    logger.info(f"Using generator class: {_GeneratorCls.__name__}")

    # Dataset
    train_dataset = InpaintingDataset(
        image_dir=cfg["data"].get("image_dir"),
        mask_dir=cfg["data"].get("mask_dir"),
        image_size=cfg["data"].get("image_size", 256),
        split="train",
        is_train=True,
        synthetic_size=cfg["data"].get("synthetic_size", 200),
    )

    # Models
    if _USE_RL_MODEL:
        generator = _GeneratorCls(
            cnum=cfg["model"].get("cnum", 48),
            cnum_in=cfg["model"].get("cnum_in", 5),
            strategy_dim=cfg["model"].get("strategy_dim", 64),
            latent_dim=cfg["model"].get("latent_dim", 256),
        ).to(device)
        # Load backbone
        backbone_ckpt = cfg["model"].get("backbone_checkpoint")
        if backbone_ckpt and os.path.exists(backbone_ckpt):
            logger.info(f"Loading pretrained DeepFill backbone from: {backbone_ckpt}")
            try:
                generator.load_pretrained_backbone(backbone_ckpt)
            except Exception as e:
                logger.warning(f"Backbone load failed: {e}")
        # Load adapters
        adapters_ckpt = cfg["model"].get("adapters_checkpoint")
        if adapters_ckpt and os.path.exists(adapters_ckpt):
            logger.info(f"Loading pretrained adapters from: {adapters_ckpt}")
            generator.load_adapters(adapters_ckpt)
            # Freeze adapters during PPO (primary experiment). For joint fine-tuning, they stay trainable.
            # Keep frozen as per plan
            generator.freeze_adapters() if hasattr(generator, "freeze_adapters") else None
        # Fallback legacy gan checkpoint if no backbone
        elif cfg["model"].get("checkpoint_gan") and os.path.exists(cfg["model"].get("checkpoint_gan")):
            logger.info(f"Fallback: loading legacy GAN checkpoint from {cfg['model'].get('checkpoint_gan')}")
            try:
                ckpt = load_checkpoint(cfg["model"].get("checkpoint_gan"), device=device)
                generator.load_state_dict(ckpt.get("generator_state_dict", ckpt), strict=False)
            except Exception as e:
                logger.warning(f"Legacy checkpoint load failed: {e}")
    else:
        generator = _GeneratorCls(
            base_channels=cfg["model"].get("base_channels", 32),
            strategy_dim=cfg["model"].get("strategy_dim", 64),
            latent_dim=cfg["model"].get("latent_dim", 256),
        ).to(device)
        ckpt_path = cfg["model"].get("checkpoint_gan")
        if ckpt_path and os.path.exists(ckpt_path):
            logger.info(f"Loading pretrained GAN baseline from: {ckpt_path}")
            ckpt = load_checkpoint(ckpt_path, device=device)
            generator.load_state_dict(ckpt["generator_state_dict"], strict=False)
            discriminator_state = ckpt.get("discriminator_state_dict")
        else:
            logger.warning(f"GAN checkpoint not found at '{ckpt_path}'. Training with initialized weights.")

    discriminator = SNPatchGANDiscriminator(
        base_channels=cfg["model"].get("base_channels", 32) * 2 if not _USE_RL_MODEL else 64,
    ).to(device)

    # If legacy GAN checkpoint had discriminator, load it
    if not _USE_RL_MODEL and "discriminator_state" in locals() and discriminator_state is not None:
        try:
            discriminator.load_state_dict(discriminator_state, strict=False)
        except Exception:
            pass

    # Create Gym Contextual Bandit Environment
    reward_weights = tuple(cfg["rl"].get("reward_weights", [1.0, 1.0, 0.5, 0.3, 0.1]))
    env = InpaintingBanditEnv(
        dataset=train_dataset,
        generator=generator,
        discriminator=discriminator,
        device=str(device),
        reward_weights=reward_weights,
        compute_lpips=cfg["rl"].get("compute_lpips", True),
    )

    # Instantiate PPO with gamma=0.0 for 1-step contextual bandit
    timesteps = override_timesteps if override_timesteps is not None else cfg["rl"].get("total_timesteps", 10000)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["rl"].get("learning_rate", 3e-4),
        n_steps=cfg["rl"].get("n_steps", 128),
        batch_size=cfg["rl"].get("batch_size", 32),
        n_epochs=cfg["rl"].get("n_epochs", 4),
        gamma=cfg["rl"].get("gamma", 0.0),
        ent_coef=cfg["rl"].get("ent_coef", 0.05),
        vf_coef=cfg["rl"].get("vf_coef", 0.5),
        max_grad_norm=cfg["rl"].get("max_grad_norm", 0.5),
        tensorboard_log=cfg["rl"]["log_dir"],
        verbose=1,
        seed=seed,
    )

    callback = RLBanditLoggingCallback(check_freq=cfg["rl"].get("n_steps", 128))

    logger.info(f"Training PPO for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=callback)

    save_path = os.path.join(cfg["rl"]["save_dir"], "ppo_bandit_final.zip")
    model.save(save_path)
    logger.info(f"PPO Agent saved successfully to: {save_path}")

    # Log action distribution summary
    dist_summary = callback.get_distribution_summary()
    logger.info(f"\n--- Learned Action Distribution ---\n{dist_summary}\n----------------------------------")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Contextual Bandit (Experiment B)")
    parser.add_argument("--config", type=str, default="configs/rl_agent_bandit.yaml", help="Path to config file")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()
    train_rl_bandit(args.config, args.timesteps)
