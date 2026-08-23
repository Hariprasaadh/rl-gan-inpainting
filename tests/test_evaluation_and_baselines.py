import os
import shutil
import tempfile
import pytest
import torch

from src.data.dataset import InpaintingDataset
from src.models.generator import InpaintingGenerator
from src.baselines.lama_inference import LaMaInference
from src.evaluation.metrics import InpaintingMetricEvaluator
from src.evaluation.severity_eval import SeverityEvaluator
from src.evaluation.evaluate import evaluate_all


def test_metric_evaluator():
    evaluator = InpaintingMetricEvaluator(device=torch.device("cpu"), compute_lpips=False)
    pred = torch.zeros(1, 3, 32, 32)
    gt = torch.zeros(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)

    item = evaluator.update(pred, gt, mask)
    assert item["l1"] == 0.0
    assert item["psnr"] > 30.0

    summary = evaluator.summary()
    assert summary["psnr"]["count"] == 1


def test_lama_inference_wrapper():
    lama = LaMaInference(device="cpu")
    img = torch.clamp(torch.randn(1, 3, 64, 64), -1.0, 1.0)
    mask = torch.zeros(1, 1, 64, 64)
    mask[:, :, 10:30, 10:30] = 1.0
    masked_img = img * (1.0 - mask)

    out = lama.inpaint(masked_img, mask)
    assert out.shape == (1, 3, 64, 64)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_severity_evaluator_smoke():
    generator = InpaintingGenerator(base_channels=16, strategy_dim=32, num_strategies=4, latent_dim=128)
    dataset = InpaintingDataset(image_dir=None, image_size=64, synthetic_size=8, is_train=False)

    sev_eval = SeverityEvaluator(generator, rl_agent=None, device="cpu", compute_lpips=False)
    res = sev_eval.evaluate_dataset(dataset, num_samples=4)

    assert "summary_table" in res
    assert len(res["summary_table"]) == 4
    table_str = sev_eval.format_table(res)
    assert "Missing Area" in table_str


def test_evaluate_all_smoke():
    temp_dir = tempfile.mkdtemp()
    try:
        generator = InpaintingGenerator(base_channels=16, strategy_dim=32, num_strategies=4, latent_dim=128)
        dataset = InpaintingDataset(image_dir=None, image_size=64, synthetic_size=6, is_train=False)
        json_res = evaluate_all(dataset, generator, rl_agent_path=None, output_dir=temp_dir, device="cpu", num_samples=3)

        assert "gan_summary" in json_res
        assert os.path.exists(os.path.join(temp_dir, "evaluation_report.md"))
        assert os.path.exists(os.path.join(temp_dir, "evaluation_summary.json"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
