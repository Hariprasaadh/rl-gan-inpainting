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
from src.models.rl_inpainting_model import RLInpaintingModel
from src.models.discriminator import SNPatchGANDiscriminator
from src.rl.actions import ACTION_NAMES
from src.rl.env_bandit import InpaintingBanditEnv
from src.training.trainer_utils import load_config, get_device, load_checkpoint


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

    # Dataset
    train_dataset = InpaintingDataset(
        image_dir=cfg["data"].get("image_dir"),
        mask_dir=cfg["data"].get("mask_dir"),
        image_size=cfg["data"].get("image_size", 128),
        split="train",
        is_train=True,
        synthetic_size=cfg["data"].get("synthetic_size", 200),
    )

    # Models
    generator = RLInpaintingModel(
        cnum        =cfg["model"].get("cnum", 48),
        strategy_dim=cfg["model"].get("strategy_dim", 64),
        latent_dim  =cfg["model"].get("latent_dim", 256),
    ).to(device)

    # Load pretrained DeepFill-v2 backbone
    backbone_ckpt = cfg["model"].get("backbone_checkpoint")
    if backbone_ckpt and os.path.exists(backbone_ckpt):
        logger.info(f"Loading DeepFill-v2 backbone from: {backbone_ckpt}")
        generator.load_pretrained_backbone(backbone_ckpt)
    else:
        logger.warning(f"Backbone checkpoint not found at '{backbone_ckpt}'. Backbone will be random (CHECKPOINT 1 fail).")

    # Load pretrained adapter weights
    adapters_ckpt = cfg["model"].get("adapters_checkpoint")
    if adapters_ckpt and os.path.exists(adapters_ckpt):
        logger.info(f"Loading adapter weights from: {adapters_ckpt}")
        generator.load_adapters(adapters_ckpt)
    else:
        logger.warning(f"Adapter checkpoint not found at '{adapters_ckpt}'. Adapters will be random — run pretrain-adapters first.")

    # Discriminator is still used for the disc_delta reward component.
    # We use the default SN-PatchGAN with fresh random weights (no pretrained GAN).
    # Its role in the reward is downweighted by the reward_weights config.
    discriminator = SNPatchGANDiscriminator(
        base_channels=cfg["model"].get("disc_base_channels", 64),
    ).to(device)

    # Create Gym Contextual Bandit Environment
    reward_weights = tuple(cfg["rl"].get("reward_weights", [1.0, 1.0, 0.5, 0.5, 0.3]))
    env = InpaintingBanditEnv(
        dataset=train_dataset,
        generator=generator,
        discriminator=discriminator,
        device=str(device),
        reward_weights=reward_weights,
        compute_lpips=cfg["rl"].get("compute_lpips", False),
    )

    # Instantiate PPO with gamma=0.0 for 1-step contextual bandit
    timesteps = override_timesteps if override_timesteps is not None else cfg["rl"].get("total_timesteps", 5000)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["rl"].get("learning_rate", 3e-4),
        n_steps=cfg["rl"].get("n_steps", 128),
        batch_size=cfg["rl"].get("batch_size", 32),
        n_epochs=cfg["rl"].get("n_epochs", 4),
        gamma=cfg["rl"].get("gamma", 0.0),
        ent_coef=cfg["rl"].get("ent_coef", 0.01),
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
