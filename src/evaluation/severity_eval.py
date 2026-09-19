from typing import Dict, List, Any, Optional
import numpy as np
import torch
from stable_baselines3 import PPO

from src.data.mask_generator import get_severity_bucket
from src.data.dataset import InpaintingDataset
try:
    from src.models.rl_inpainting_model import RLInpaintingModel as InpaintingGenerator
except ImportError:
    from src.models.generator import InpaintingGenerator  # type: ignore
from src.rl.actions import ACTION_NAMES
from src.rl.state import StateBuilder
from src.evaluation.metrics import InpaintingMetricEvaluator


class SeverityEvaluator:
    """Evaluates and compares GAN Baseline vs RL-Guided GAN across mask severity buckets."""

    def __init__(
        self,
        generator: InpaintingGenerator,
        rl_agent: Optional[PPO] = None,
        device: str = "cpu",
        compute_lpips: bool = False,
    ):
        self.device = torch.device(device)
        self.generator = generator.to(self.device).eval()
        self.rl_agent = rl_agent
        self.compute_lpips = compute_lpips

        self.state_builder = StateBuilder(
            latent_dim=generator.encoder.latent_dim,
            include_mask_stats=True,
            include_quality_proxy=True,
        )

        self.buckets = ["10-20%", "20-40%", "40-60%", "60%+"]

    def evaluate_dataset(
        self,
        dataset: InpaintingDataset,
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run severity-bucketed evaluation across dataset.

        Returns:
            Dict containing per-bucket statistics, delta comparisons, and action breakdowns.
        """
        n = min(len(dataset), num_samples) if num_samples is not None else len(dataset)

        results: Dict[str, Dict[str, Any]] = {
            b: {
                "gan_evaluator": InpaintingMetricEvaluator(device=self.device, compute_lpips=self.compute_lpips),
                "rl_evaluator": InpaintingMetricEvaluator(device=self.device, compute_lpips=self.compute_lpips),
                "action_counts": {i: 0 for i in range(4)},
                "count": 0,
            }
            for b in self.buckets
        }

        for idx in range(n):
            item = dataset[idx]
            image = item["image"].unsqueeze(0).to(self.device)
            mask = item["mask"].unsqueeze(0).to(self.device)
            masked_image = item["masked_image"].unsqueeze(0).to(self.device)
            stats = item["stats"]

            missing_ratio = stats.get("missing_ratio", 0.3)
            bucket = get_severity_bucket(missing_ratio)
            bucket_data = results[bucket]
            bucket_data["count"] += 1

            with torch.no_grad():
                # 1. Baseline GAN (Fixed Neutral Strategy)
                gan_out = self.generator(masked_image, mask, strategy=None)
                gan_completed = gan_out["completed"]
                bucket_data["gan_evaluator"].update(gan_completed, image, mask)

                # 2. RL-Guided GAN (Adaptive Strategy Selection)
                if self.rl_agent is not None:
                    latent_emb = self.generator.extract_state_embedding(masked_image, mask)
                    coarse = self.generator.coarse_forward(masked_image, mask)
                    coarse_comp = masked_image + coarse * mask
                    obs = self.state_builder.build_state(
                        latent_embedding=latent_emb,
                        mask_stats=stats,
                        coarse_composite=coarse_comp,
                        mask=mask,
                    )
                    action, _ = self.rl_agent.predict(obs, deterministic=True)
                    action_idx = int(action)
                    bucket_data["action_counts"][action_idx] += 1

                    rl_refined = self.generator.refine(coarse_comp, mask, strategy=action_idx)
                    rl_completed = masked_image + rl_refined * mask
                else:
                    # Default neutral
                    rl_completed = gan_completed

                bucket_data["rl_evaluator"].update(rl_completed, image, mask)

        # Summarize results
        summary_table = []
        for b in self.buckets:
            b_data = results[b]
            gan_sum = b_data["gan_evaluator"].summary()
            rl_sum = b_data["rl_evaluator"].summary()

            gan_psnr = gan_sum["psnr"]["mean"]
            rl_psnr = rl_sum["psnr"]["mean"]
            delta_psnr = rl_psnr - gan_psnr

            gan_ssim = gan_sum["ssim"]["mean"]
            rl_ssim = rl_sum["ssim"]["mean"]

            summary_table.append({
                "bucket": b,
                "count": b_data["count"],
                "gan_psnr": gan_psnr,
                "rl_psnr": rl_psnr,
                "delta_psnr": delta_psnr,
                "gan_ssim": gan_ssim,
                "rl_ssim": rl_ssim,
                "action_counts": b_data["action_counts"],
            })

        return {
            "summary_table": summary_table,
            "raw_results": results,
        }

    def format_table(self, eval_results: Dict[str, Any]) -> str:
        """Format markdown severity comparison table."""
        table = eval_results["summary_table"]
        lines = [
            "| Missing Area | Samples | GAN Baseline (PSNR) | RL-GAN (PSNR) | Delta PSNR | GAN (SSIM) | RL-GAN (SSIM) |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for row in table:
            lines.append(
                f"| {row['bucket']:8s} | {row['count']:7d} | {row['gan_psnr']:6.2f} dB | {row['rl_psnr']:6.2f} dB | {row['delta_psnr']:+5.2f} dB | {row['gan_ssim']:5.3f} | {row['rl_ssim']:5.3f} |"
            )
        return "\n".join(lines)
