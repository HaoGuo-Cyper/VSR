"""CPU diagnostic tests; fake model outputs are not experiment evidence."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from vsr.data import degrade_x4
from vsr.probes import evaluate_denoising_probe, lr_reprojection_error


class ProbeModel(nn.Module):
    def __init__(self, clean):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.clean = clean
        self.calls = []
        self.eval_child = nn.Identity().eval()

    def clear_features(self):
        pass

    def forward(self, noisy, timestep, text, low, enable_lrv=True):
        self.calls.append((noisy.clone(), timestep.clone(), low.clone(), enable_lrv, torch.is_grad_enabled()))
        sigma = timestep.view(-1, 1, 1, 1, 1) / 1000
        # Oracle velocity derived from x_sigma and known clean is intentionally
        # confined to this fake test model to establish probe arithmetic.
        return (noisy - self.clean) / sigma, None


class ProbeTests(unittest.TestCase):
    def test_oracle_prediction_and_fixed_noise_controls(self):
        clean = torch.zeros(1, 4, 2, 4, 4)
        pairs = [{"clean": clean, "low": torch.ones_like(clean) * i} for i in (1, 2)]
        clips = [{"metadata": {"record_id": str(i), "caption": "", "split": "train"}} for i in range(2)]
        model = ProbeModel(clean)
        model.train()
        model.eval_child.eval()
        with tempfile.TemporaryDirectory() as temporary:
            result = evaluate_denoising_probe(model, pairs, clips, {"": torch.zeros(1, 2, 4)}, {}, temporary)
            self.assertTrue((Path(temporary) / "denoising_probe.json").is_file())
        for sigma_results in result["aggregate_by_sigma"].values():
            for metrics in sigma_results.values():
                self.assertLess(metrics["fm_mse"], 1e-12)
                self.assertLess(metrics["x0hat_latent_mse"], 1e-12)
        self.assertTrue(model.training)
        self.assertFalse(model.eval_child.training)
        self.assertEqual(len(model.calls), 2 * 4 * 3)
        for offset in range(0, len(model.calls), 3):
            torch.testing.assert_close(model.calls[offset][0], model.calls[offset + 1][0], rtol=0, atol=0)
            torch.testing.assert_close(model.calls[offset][0], model.calls[offset + 2][0], rtol=0, atol=0)
            self.assertFalse(model.calls[offset + 1][3])
        self.assertTrue(all(not call[4] for call in model.calls))
        self.assertFalse(result["restoration_inference"])

    def test_probe_noise_ignores_global_rng_and_x0_scaling(self):
        clean = torch.ones(1, 4, 2, 4, 4) * 0.2
        pairs = [{"clean": clean, "low": clean}]
        clips = [{"metadata": {"record_id": "same", "caption": ""}}]
        class ZeroModel(ProbeModel):
            def forward(self, noisy, timestep, text, low, enable_lrv=True):
                return torch.zeros_like(noisy), None
        model = ZeroModel(clean)
        with tempfile.TemporaryDirectory() as temporary:
            first = evaluate_denoising_probe(model, pairs, clips, {"": torch.zeros(1, 2, 4)}, {}, temporary, modes=("correct",))
            torch.manual_seed(9876)
            torch.randn(1000)
            second = evaluate_denoising_probe(model, pairs, clips, {"": torch.zeros(1, 2, 4)}, {}, temporary, modes=("correct",))
        self.assertEqual(first, second)
        for sigma_string, rows in first["aggregate_by_sigma"].items():
            metrics = rows["correct"]
            self.assertAlmostEqual(metrics["x0hat_latent_mse"], float(sigma_string) ** 2 * metrics["fm_mse"], places=6)

    def test_lr_reprojection_uses_exact_shared_degradation(self):
        torch.manual_seed(1)
        hr = torch.rand(1, 3, 3, 32, 48)
        lr, _ = degrade_x4(hr)
        result = lr_reprojection_error(hr, lr)
        self.assertEqual(result["scale"], 4)
        self.assertEqual(result["lr_reprojection_mse_rgb_01"], 0)
        known = lr_reprojection_error(torch.ones(1, 3, 1, 32, 48) * 0.2, torch.ones(1, 3, 1, 8, 12) * 0.1)
        self.assertAlmostEqual(known["lr_reprojection_mse_rgb_01"], 0.01, places=7)

    def test_probe_rejects_impossible_control_and_invalid_sigma(self):
        clean = torch.zeros(1, 4, 2, 4, 4)
        pairs = [{"clean": clean, "low": clean}]
        clips = [{"metadata": {"record_id": "one", "caption": ""}}]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                evaluate_denoising_probe(ProbeModel(clean), pairs, clips, {"": torch.zeros(1, 2, 4)}, {}, temporary)
            with self.assertRaises(ValueError):
                evaluate_denoising_probe(ProbeModel(clean), pairs, clips, {"": torch.zeros(1, 2, 4)}, {}, temporary,
                                         modes=("correct",), sigmas=(0,))


if __name__ == "__main__":
    unittest.main()
