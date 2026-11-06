"""CPU reference evaluation checks; synthetic fixtures are not model results."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from vsr.evaluation import rgb_metrics
from vsr.reference_evaluation import evaluate_reference_restoration, sample_reference_latent


class FakeScheduler:
    def set_timesteps(self, steps, device):
        self.timesteps = torch.arange(steps, 0, -1, device=device, dtype=torch.float32)

    def step(self, velocity, timestep, latent, return_dict=False):
        # A known reconstruction makes the reference exclusion independently
        # testable. The sampler's initial Gaussian is still audited below.
        return (velocity,)


class FakeReferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()))
        self.eval_child = nn.Identity().eval()
        self.calls = []
        self.reference_context = "stale context must be cleared"
        self.reference_clears = 0
        self.feature_clears = 0

    def clear_reference(self):
        self.reference_context = None
        self.reference_clears += 1

    def clear_features(self):
        self.feature_clears += 1

    def forward(self, noisy, timestep, text, low, enable_lrv=True, *,
                ref_latent=None, ref_index=None, frame_count=None, ref_valid=None):
        valid = bool(ref_valid[0])
        self.reference_context = ref_latent
        self.calls.append({"noisy": noisy.clone(), "text": text.clone(), "low": low.clone(),
            "valid": ref_valid.clone(), "reference": ref_latent.clone() if valid else ref_latent,
            "index": ref_index, "frame_count": frame_count, "lrv": enable_lrv,
            "grad_enabled": torch.is_grad_enabled(), "training": self.training})
        if valid:
            result = torch.ones_like(low) * ref_latent.mean()
            # Deliberately make only the supplied frame much WORSE, so a test
            # can detect accidental inclusion in the primary restoration score.
            result[:, :, ref_index] = -1
        else:
            if ref_latent is not None or ref_index is not None:
                raise AssertionError("Missing HRI must use no latent/index")
            result = low
        return result, None


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_parameter = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(latents_mean=[0, 0, 0], latents_std=[1, 1, 1])

    def decode(self, latent):
        return SimpleNamespace(sample=latent)


class PoisonReference:
    def __getattribute__(self, name):
        raise AssertionError("Disabled / unrequested HRI data must not be inspected")


class RequestedReferencesOnly(dict):
    def __iter__(self):
        raise AssertionError("Do not iterate unrequested reference values")

    def values(self):
        raise AssertionError("Do not read unrequested reference values")

    def items(self):
        raise AssertionError("Do not read unrequested reference values")


class ReferenceEvaluationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = {"model": {"local_dir": "fake-local-only-scheduler"},
                       "evaluation": {"scope": "component_test"}}

    def fixtures(self, count=2):
        clips, pairs, references = [], [], []
        for i in range(count):
            hr = torch.full((1, 3, 9, 8, 8), 0.5 + 0.05 * i)
            clips.append({"hr": hr, "lr_up": hr * .9, "fps": 24.0 + 6 * i,
                "metadata": {"record_id": f"clip_{i}", "source_id": f"source_{i}", "split": "val", "caption": ""}})
            # Reading another HR latent / teacher object inside the sampler
            # would fail; only the low latent and explicitly provided HRI exist.
            pairs.append({"low": torch.full_like(hr, .4 + .2 * i), "clean": PoisonReference(), "teacher": PoisonReference()})
            references.append(RequestedReferencesOnly({
                0: torch.full((1, 3, 1, 8, 8), .2 + .1 * i),
                4: torch.full((1, 3, 1, 8, 8), .24 + .1 * i),
                8: PoisonReference()}))
        return clips, pairs, references, {"": torch.zeros(1, 2, 4)}

    def fake_writer(self, path, video, fps):
        self.saved_video_shapes.append((tuple(video.shape), float(fps)))
        Path(path).write_bytes(b"component-playback-placeholder")

    def test_reference_every_step_noise_matched_and_full_output_preserved(self):
        model, vae = FakeReferenceModel(), FakeVAE()
        model.train()
        model.eval_child.eval()
        clips, pairs, references, text = self.fixtures()
        self.saved_video_shapes = []
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch("diffusers.UniPCMultistepScheduler.from_pretrained", side_effect=lambda *a, **k: FakeScheduler()), \
             mock.patch("vsr.reference_evaluation.write_video", side_effect=self.fake_writer):
            result = evaluate_reference_restoration(model, vae, pairs, clips, text, references, self.config, temporary,
                                                    reference_indices=(0, 4), seeds=(42, 43), steps=2)
            self.assertEqual(result["model_setting"], "reference_trained_with_dropout")
            self.assertEqual(set(result["reference_cases"]), {"0", "4"})
            self.assertEqual(result["reference_cases"]["0"]["reference_roles"], ["first_frame_main"])
            self.assertEqual(result["reference_cases"]["4"]["reference_roles"], ["middle_frame_adaptation"])
            for ref in (0, 4):
                for row in result["reference_cases"][str(ref)]["per_video"]:
                    for mode in ("hri_on", "hri_off"):
                        for entry in row["modes"][mode]["seeds"]:
                            saved = torch.load(entry["rgb_path"], weights_only=True)
                            self.assertEqual(tuple(saved.shape), (1, 3, 9, 8, 8))
                            self.assertEqual(entry["frame_count"], 9)
                            self.assertEqual(entry["fps"], clips[row["clip_index"]]["fps"])
            self.assertTrue((Path(temporary) / "reference_000/metrics.json").is_file())
            self.assertTrue((Path(temporary) / "reference_004/metrics.json").is_file())
        self.assertEqual(len(model.calls), 32)
        for offset in (0, 8, 16, 24):
            for seed_offset in (0, 2):
                on, off = model.calls[offset + seed_offset], model.calls[offset + 4 + seed_offset]
                torch.testing.assert_close(on["noisy"], off["noisy"], rtol=0, atol=0)
                torch.testing.assert_close(on["text"], off["text"], rtol=0, atol=0)
                torch.testing.assert_close(on["low"], off["low"], rtol=0, atol=0)
        for call in model.calls:
            self.assertEqual(call["frame_count"], 9)
            self.assertTrue(call["lrv"])
            self.assertFalse(call["grad_enabled"])
            self.assertFalse(call["training"])
            self.assertEqual(tuple(call["valid"].shape), (1,))
            if bool(call["valid"][0]):
                self.assertIn(call["index"], (0, 4))
                self.assertEqual(tuple(call["reference"].shape), (1, 3, 1, 8, 8))
            else:
                self.assertIsNone(call["reference"])
                self.assertIsNone(call["index"])
        self.assertEqual([shape for shape, _ in self.saved_video_shapes], [(1, 3, 9, 8, 8)] * 16)
        self.assertEqual({fps for _, fps in self.saved_video_shapes}, {24.0, 30.0})
        self.assertIsNone(model.reference_context)
        self.assertTrue(model.training)
        self.assertFalse(model.eval_child.training)
        self.assertTrue(vae.training)

    def test_both_modes_exclude_same_reference_and_distance_gain_is_nonreference(self):
        clips, pairs, references, text = self.fixtures(count=1)
        self.saved_video_shapes = []
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch("diffusers.UniPCMultistepScheduler.from_pretrained", side_effect=lambda *a, **k: FakeScheduler()), \
             mock.patch("vsr.reference_evaluation.write_video", side_effect=self.fake_writer):
            result = evaluate_reference_restoration(FakeReferenceModel(), FakeVAE(), pairs, clips, text, references,
                            self.config, temporary, reference_indices=(0, 4), seeds=(42,), steps=1)
            for ref in (0, 4):
                case = result["reference_cases"][str(ref)]
                row = case["per_video"][0]
                expected_indices = [i for i in range(9) if i != ref]
                self.assertEqual(row["evaluation_frame_indices"], expected_indices)
                for mode in ("hri_on", "hri_off"):
                    entry = row["modes"][mode]["seeds"][0]
                    self.assertEqual(entry["metrics"]["frame_indices"], expected_indices)
                    self.assertEqual(entry["metrics"]["frame_count"], 8)
                    self.assertEqual(entry["provided_reference_frame_metrics"]["frame_count"], 1)
                    saved = torch.load(entry["rgb_path"], weights_only=True)
                    expected = rgb_metrics(saved[:, :, expected_indices], clips[0]["hr"][:, :, expected_indices])
                    self.assertEqual(entry["metrics"]["psnr_rgb_db"], expected["psnr_rgb_db"])
                self.assertNotIn("0", row["gain_by_distance_frames"])
                self.assertEqual([entry["frame_index"] for entry in row["per_frame_gain"]], expected_indices)
                for entry in row["per_frame_gain"]:
                    self.assertEqual(entry["distance_frames"], abs(entry["frame_index"] - ref))
                    self.assertAlmostEqual(entry["distance_seconds"], entry["distance_frames"] / 24)
                self.assertLess(row["provided_reference_frame_gain"]["psnr_gain_db"], 0)
                self.assertGreater(row["nonreference_gain"]["mean"]["psnr_gain_db"], 0)
                self.assertEqual(case["aggregate"]["nonreference_gain"], row["nonreference_gain"]["mean"])
            primary = result["reference_cases"]["0"]["per_video"][0]["modes"]["hri_on"]["mean"]
            self.assertAlmostEqual(primary["psnr_rgb_db"], 20, places=5)

    def test_off_sampler_never_reads_placeholder_and_clears_on_solver_failure(self):
        model = FakeReferenceModel()
        low, text = torch.zeros(1, 3, 9, 8, 8), torch.zeros(1, 2, 4)
        with mock.patch("diffusers.UniPCMultistepScheduler.from_pretrained", side_effect=lambda *a, **k: FakeScheduler()):
            output = sample_reference_latent(model, low, text, self.config, steps=2, seed=42,
                         frame_count=9, ref_latent=PoisonReference(), ref_index=PoisonReference(), ref_valid=False)
        self.assertEqual(tuple(output.shape), tuple(low.shape))
        self.assertTrue(all(call["reference"] is None and call["index"] is None for call in model.calls))
        self.assertIsNone(model.reference_context)
        failing = FakeScheduler()
        failing.step = mock.Mock(side_effect=RuntimeError("synthetic solver failure"))
        with mock.patch("diffusers.UniPCMultistepScheduler.from_pretrained", return_value=failing):
            with self.assertRaisesRegex(RuntimeError, "synthetic solver failure"):
                sample_reference_latent(model, low, text, self.config, steps=1, seed=42, frame_count=9,
                         ref_latent=torch.zeros(1, 3, 1, 8, 8), ref_index=4, ref_valid=True)
        self.assertIsNone(model.reference_context)
        self.assertGreater(model.feature_clears, 1)
        self.assertGreater(model.reference_clears, 1)

    def test_invalid_reference_configuration_and_exception_restore_flags(self):
        clips, pairs, references, text = self.fixtures(count=1)
        for kwargs in ({"reference_indices": (9,)}, {"reference_indices": (0, 0)},
                       {"seeds": (True,)}, {"sample_indices": (1,)}, {"steps": 0}):
            with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                evaluate_reference_restoration(FakeReferenceModel(), FakeVAE(), pairs, clips, text, references,
                                                self.config, temporary, **kwargs)
        model, vae = FakeReferenceModel(), FakeVAE()
        model.train()
        model.eval_child.eval()
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch("diffusers.UniPCMultistepScheduler.from_pretrained", side_effect=RuntimeError("no scheduler")):
            with self.assertRaisesRegex(RuntimeError, "no scheduler"):
                evaluate_reference_restoration(model, vae, pairs, clips, text, references, self.config, temporary,
                                                reference_indices=(0,), seeds=(42,), steps=1)
        self.assertTrue(model.training)
        self.assertFalse(model.eval_child.training)
        self.assertTrue(vae.training)
        self.assertIsNone(model.reference_context)


if __name__ == "__main__":
    unittest.main()
