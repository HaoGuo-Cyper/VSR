"""CPU metric and control-flow tests; no training quality claims."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from vsr.evaluation import aggregate_per_video, evaluate_restoration, rgb_metrics


class FakeScheduler:
    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps, 0, -1, dtype=torch.float32, device=device)

    def step(self, velocity, timestep, latent, return_dict=False):
        return (latent - velocity * 0.1,)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()))
        self.eval_child = nn.Identity().eval()
        self.calls = []
        self.clears = 0

    def clear_features(self):
        self.clears += 1

    def forward(self, noisy, timestep, text, low, enable_lrv=True):
        self.calls.append({"initial_noise": noisy.clone(), "text": text.clone(), "low": low.clone(),
                           "enabled": enable_lrv, "grad_enabled": torch.is_grad_enabled(), "training": self.training})
        return low * (1.0 if enable_lrv else 0.0), None


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(latents_mean=[0, 0, 0], latents_std=[1, 1, 1])

    def decode(self, latent):
        return SimpleNamespace(sample=latent)


def fake_video_writer(path, video, fps):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"playback-placeholder-component-test-only")


class EvaluationTest(unittest.TestCase):
    def test_rgb_psnr_known_error_and_ssim_constant_image(self):
        target = torch.zeros(1, 3, 2, 8, 8)
        prediction = torch.full_like(target, 0.1)
        metrics = rgb_metrics(prediction, target)
        self.assertAlmostEqual(metrics["psnr_rgb_db"], 20, places=5)
        # Constant images eliminate contrast/structure terms. SSIM luminance
        # is C1/(0.1**2+C1), where C1=(0.01*data_range)**2.
        self.assertAlmostEqual(metrics["ssim_rgb"], 0.0001 / 0.0101, places=6)
        identity = rgb_metrics(target, target)
        self.assertEqual(identity["ssim_rgb"], 1)
        self.assertEqual(identity["psnr_rgb_db"], 120)
        self.assertEqual(identity["exact_match_frames"], 2)

    def test_frame_mean_and_equal_video_aggregation_are_explicit(self):
        target = torch.zeros(1, 3, 2, 8, 8)
        prediction = torch.full_like(target, 0.1)
        prediction[:, :, 1] = 0.2
        metrics = rgb_metrics(prediction, target)
        self.assertAlmostEqual(metrics["psnr_rgb_db"], (20 + 13.9794000867) / 2, places=5)
        videos = []
        for value in (10.0, 20.0):
            row = {"psnr_rgb_db": value, "ssim_rgb": value / 100, "mse_rgb_01": 0.1}
            videos.append({"bicubic": row, "modes": {"correct": {"mean": row}}})
        result = aggregate_per_video(videos, ("correct",))
        reverse = aggregate_per_video(videos[::-1], ("correct",))
        self.assertEqual(result, reverse)
        self.assertEqual(result["modes"]["correct"]["psnr_rgb_db"], 15)

    def test_sampler_noise_controls_outputs_and_restores_training_flags(self):
        model, vae = FakeModel(), FakeVAE()
        model.train()
        model.eval_child.eval()  # Preserve deliberately mixed child flags.
        clips = [{"hr": torch.full((1, 3, 2, 8, 8), value),
                  "lr_up": torch.full((1, 3, 2, 8, 8), value / 2), "fps": 24.0,
                  "metadata": {"record_id": f"video_{i}", "source_id": f"source_{i}",
                               "split": "train", "caption": ""}}
                 for i, value in enumerate((0.2, 0.6))]
        # Unreadable clean objects make accidental HR-latent conditioning fail.
        pairs = [{"low": torch.full((1, 3, 2, 8, 8), value), "clean": object()}
                 for value in (0.2, 0.6)]
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch("diffusers.UniPCMultistepScheduler.from_pretrained", side_effect=lambda *a, **k: FakeScheduler()), \
             mock.patch("vsr.evaluation.write_video", side_effect=fake_video_writer):
            summary = evaluate_restoration(model, vae, pairs, clips, {"": torch.zeros(1, 2, 4)},
                                           {"model": {"local_dir": "not-read-by-fake-scheduler"}}, temporary,
                                           seeds=(42, 43), steps=1)
            self.assertTrue((Path(temporary) / "metrics.json").is_file())
            self.assertEqual(len(summary["per_video"]), 2)
            self.assertEqual(summary["aggregate"]["video_count"], 2)
            for row in summary["per_video"]:
                self.assertEqual(row["fps"], 24)
                self.assertEqual(row["frame_count"], 2)
                for mode in ("correct", "disabled", "shuffled"):
                    for entry in row["modes"][mode]["seeds"]:
                        saved = torch.load(entry["rgb_path"], weights_only=True)
                        self.assertEqual(saved.shape, clips[0]["hr"].shape)
                self.assertGreater(row["condition_differences"]["disabled"]["mean"]["output_rgb_mae"], 0)
            # For one clip there are correct seeds42/43, disabled42/43,
            # shuffled42/43. Those controls must share exact initial noise.
            for offset in (0, 6):
                for seed_offset in (0, 1):
                    expected = model.calls[offset + seed_offset]["initial_noise"]
                    for mode_offset in (2, 4):
                        torch.testing.assert_close(expected, model.calls[offset + mode_offset + seed_offset]["initial_noise"], rtol=0, atol=0)
            self.assertFalse(torch.equal(model.calls[0]["initial_noise"], model.calls[1]["initial_noise"]))
            torch.testing.assert_close(model.calls[4]["low"], pairs[1]["low"])
            self.assertFalse(model.calls[2]["enabled"])
            self.assertTrue(all(not c["grad_enabled"] and not c["training"] for c in model.calls))
        self.assertTrue(model.training)
        self.assertFalse(model.eval_child.training)
        self.assertTrue(vae.training)
        self.assertGreater(model.clears, 1)

    def test_invalid_metrics_geometry_and_single_clip_shuffle_fail(self):
        with self.assertRaises(ValueError):
            rgb_metrics(torch.zeros(1, 3, 1, 6, 8), torch.zeros(1, 3, 1, 6, 8))
        clip = {"hr": torch.zeros(1, 3, 1, 8, 8), "lr_up": torch.zeros(1, 3, 1, 8, 8),
                "fps": 24, "metadata": {"caption": ""}}
        with self.assertRaises(ValueError), tempfile.TemporaryDirectory() as temporary:
            evaluate_restoration(FakeModel(), FakeVAE(), [{"low": clip["hr"]}], [clip], {"": torch.zeros(1, 2, 4)},
                                 {"model": {"local_dir": "unused"}}, temporary)


if __name__ == "__main__":
    unittest.main()
