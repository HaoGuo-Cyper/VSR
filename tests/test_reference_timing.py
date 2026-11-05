"""CPU synthetic timing observations must preserve reference tensors and RNGs."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vsr import reference_training as training


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class TimingVAE(nn.Module):
    def __init__(self, clock, *, fail_at=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.child = nn.Identity().eval()
        self.config = SimpleNamespace(latents_mean=[0.]*3, latents_std=[1.]*3)
        self.clock, self.fail_at, self.calls = clock, fail_at, []

    def to(self, device):
        # main's real CLI requests CUDA; the test fake intentionally remains
        # CPU, so neither tensor creation nor encoder timing can use a GPU.
        self.requested_device = device
        return self

    def encode(self, frame):
        self.calls.append((frame.detach().clone(), torch.is_grad_enabled()))
        self.clock.advance(.25)
        if self.fail_at == len(self.calls):
            raise RuntimeError("synthetic encoder failure")
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: frame))


def fixtures(count=2, frames=3):
    hr = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1).expand(1, 3, frames, 8, 8) / max(1, frames-1)
    clips = [{"hr": hr.clone(), "metadata": {"record_id": f"synthetic_{i}", "source_id": f"source_{i}"},
              "binding": {"source": "synthetic CPU fixture only"}} for i in range(count)]
    config = {"model": {"local_dir": "/nonexistent/synthetic-official", "revision": "fixture"},
              "data": {"num_frames": frames}, "_sha256": "fixture-config"}
    return clips, config


class ReferenceTimingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cpu_phase_clock_success_failure_and_rng_invariance(self):
        clock = Clock()
        torch_state, py_state = torch.random.get_rng_state().clone(), random.getstate()
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(training.time, "perf_counter", clock), \
             patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")), \
             patch("torch.cuda.synchronize", side_effect=AssertionError("CPU timing must not synchronize CUDA")):
            path = Path(temporary)/"timings.json"
            timings = training._PhaseTimings(path, started=0)
            with timings.phase("cpu_work", device="cpu"):
                actual = torch.tensor([.25, .5]).square()
                clock.advance(2.5)
            self.assertEqual(timings.phases["cpu_work"]["wall_seconds"], 2.5)
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                with timings.phase("failed_work", device="cpu"):
                    clock.advance(4)
                    raise RuntimeError("controlled")
            timings.finish("failed", error="RuntimeError: controlled")
            result = json.loads(path.read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["entire_experiment_wall_seconds"], 6.5)
        self.assertEqual(result["phases"]["failed_work"]["status"], "failed")
        self.assertEqual(result["phases"]["failed_work"]["wall_seconds"], 4)
        torch.testing.assert_close(actual, torch.tensor([.0625, .25]), rtol=0, atol=0)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(py_state, random.getstate())

    def test_reference_encoding_progress_exact_values_and_unmodified_binding_schema(self):
        clock = Clock()
        vae = TimingVAE(clock)
        flags = [module.training for module in vae.modules()]
        clips, config = fixtures(count=26)
        torch_state, py_state = torch.random.get_rng_state().clone(), random.getstate()
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(training.time, "perf_counter", clock), \
             patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")), \
             patch("torch.cuda.synchronize", side_effect=AssertionError("CPU encoder must not touch CUDA")), \
             patch("builtins.print") as printer:
            result = training.encode_reference_frames(vae, clips, config, temporary)
            timing = json.loads((Path(temporary)/"reference_encoding_timings.json").read_text())
            loaded = torch.load(Path(temporary)/"clip_0/reference_latents.pt", weights_only=True)
            seal = json.loads((Path(temporary)/"reference_cache_seal.json").read_text())
        self.assertEqual(len(result), 26)
        for i, mapping in enumerate(result):
            self.assertEqual(set(mapping), {0, 1, 2})
            for index, tensor in mapping.items():
                torch.testing.assert_close(tensor, clips[i]["hr"][:, :, index:index+1]*2-1, rtol=0, atol=0)
        self.assertEqual(timing["status"], "completed")
        self.assertEqual(timing["wall_seconds"], 19.5)
        self.assertEqual(timing["completed_clips"], 26)
        self.assertEqual(timing["encoded_frames"], 78)
        self.assertEqual(timing["requested_frames"], 78)
        self.assertIsNone(timing["allocated_GPU_hours"])
        self.assertIsNone(timing["peak_cuda_allocated_bytes"])
        self.assertEqual(len(printer.call_args_list), 2)
        self.assertIn("25/26", printer.call_args_list[0].args[0])
        self.assertIn("26/26", printer.call_args_list[1].args[0])
        self.assertEqual(set(loaded["binding"]), {"source_binding", "clip_metadata", "official_revision", "vae_config_sha256",
            "encoder_source_sha256", "config_sha256", "frame_indices", "encoding", "posterior", "normalization", "vae_frozen", "role"})
        self.assertEqual(set(seal), {"new_cache", "legacy_state_loaded", "clip_count", "per_clip", "encoder_source_sha256",
                                    "single_frame_only", "video_latent_slicing"})
        self.assertEqual(flags, [module.training for module in vae.modules()])
        self.assertTrue(all(not grad for _, grad in vae.calls))
        self.assertTrue(all(frame.shape[2] == 1 for frame, _ in vae.calls))
        self.assertIsNone(vae.weight.grad)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(py_state, random.getstate())

    def test_reference_encoding_failure_keeps_partial_cost_and_restores_flags(self):
        clock = Clock()
        vae = TimingVAE(clock, fail_at=2)
        flags = [module.training for module in vae.modules()]
        clips, config = fixtures()
        rng = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(training.time, "perf_counter", clock), \
             patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
            with self.assertRaisesRegex(RuntimeError, "synthetic encoder failure"):
                training.encode_reference_frames(vae, clips, config, temporary)
            timing = json.loads((Path(temporary)/"reference_encoding_timings.json").read_text())
            self.assertFalse((Path(temporary)/"reference_cache_seal.json").exists())
        self.assertEqual(timing["status"], "failed")
        self.assertEqual(timing["wall_seconds"], .5)
        self.assertEqual(timing["encoded_frames"], 1)
        self.assertEqual(timing["completed_clips"], 0)
        self.assertIn("synthetic encoder failure", timing["error"])
        self.assertEqual(flags, [module.training for module in vae.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def _main_fixture(self, root, clock, *, fail_prepare=False):
        clips, config = fixtures()
        config.update(stage="CPU-synthetic-test", experiment_name="timing_fixture", reference={"enabled": True},
            training={"batch_size": 1, "updates": 2, "evaluate_at": [2], "variants": ["Base", "Align", "Relation", "Both"]})
        (root/"config.json").write_text(json.dumps(config))
        events, encoders = [], []
        def environment(*args):
            events.append("environment"); clock.advance(1); return {"cuda_available": True}
        def official(*args):
            events.append("official"); clock.advance(2); return {"scope": "synthetic"}
        def prepare(*args):
            events.append("prepare"); clock.advance(3)
            if fail_prepare:
                raise RuntimeError("synthetic preparation failure")
            return (clips, None, None, None, None, None, 2)
        def encoder(*args, **kwargs):
            events.append("encoder_load"); clock.advance(4)
            value = TimingVAE(clock); encoders.append(value); return value
        def release():
            events.append("release"); clock.advance(1.5)
        def supervision_seal(*args):
            events.append("supervision_seal"); clock.advance(.75)
        def variant(name, *args):
            events.append(name); clock.advance(10); return {"status": "completed", "synthetic": True}
        patches = [patch.object(training, "ROOT", root), patch.object(training.time, "perf_counter", clock),
            patch.object(training, "environment", environment), patch.object(training, "verify_official", official),
            patch.object(training, "prepare", prepare), patch.object(training, "release", release),
            patch.object(training, "train_reference_variant", variant),
            patch("vsr.reference_diagnostics.seal_reference_supervision", side_effect=supervision_seal),
            patch.object(training, "configure_reproducibility", return_value={"scope": "synthetic CPU test"}),
            patch("diffusers.AutoencoderKLWan.from_pretrained", side_effect=encoder),
            patch("sys.argv", ["reference_training", "--config", "config.json"]),
            patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")),
            patch("torch.cuda.synchronize"), patch("builtins.print")]
        return patches, events, encoders

    def test_main_records_all_success_phases_and_preserves_order_and_rng(self):
        from contextlib import ExitStack
        clock = Clock()
        before, py_before = torch.random.get_rng_state().clone(), random.getstate()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            patches, events, encoders = self._main_fixture(root, clock)
            for item in patches:
                stack.enter_context(item)
            training.main()
            run = next((root/"runs").glob("timing_fixture_*"))
            timing = json.loads((run/"phase_timings.json").read_text())
            ref_timing = json.loads((run/"reference_cache/reference_encoding_timings.json").read_text())
            self.assertEqual(json.loads((run/"status.json").read_text())["status"], "completed")
        self.assertEqual(events, ["environment", "official", "prepare", "encoder_load", "release", "supervision_seal", "Base", "Align", "Relation", "Both"])
        self.assertEqual(timing["status"], "completed")
        expected = {"environment": 1, "official_artifact_verification": 2, "shared_pair_preparation": 3,
                    "reference_vae_load": 4, "reference_single_frame_encoding": 1.5, "reference_encoder_release": 1.5,
                    "reference_supervision_seal": .75,
                    **{f"variant_{name}": 10 for name in ("Base", "Align", "Relation", "Both")}}
        self.assertEqual({k: v["wall_seconds"] for k, v in timing["phases"].items()}, expected)
        self.assertTrue(all(v["status"] == "completed" for v in timing["phases"].values()))
        self.assertEqual(timing["entire_experiment_wall_seconds"], sum(expected.values()))
        self.assertEqual(ref_timing["encoded_frames"], 6)
        self.assertEqual(len(encoders[0].calls), 6)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(py_before, random.getstate())

    def test_main_failure_persists_completed_and_failed_subphases_without_later_work(self):
        from contextlib import ExitStack
        clock = Clock()
        before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            patches, events, encoders = self._main_fixture(root, clock, fail_prepare=True)
            for item in patches:
                stack.enter_context(item)
            with self.assertRaisesRegex(RuntimeError, "synthetic preparation failure"):
                training.main()
            run = next((root/"runs").glob("timing_fixture_*"))
            timing = json.loads((run/"phase_timings.json").read_text())
            self.assertEqual(json.loads((run/"status.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((root/"runs/active_reference_training.json").read_text())["status"], "failed")
        self.assertEqual(events, ["environment", "official", "prepare"])
        self.assertFalse(encoders)
        self.assertEqual(timing["status"], "failed")
        self.assertEqual(timing["entire_experiment_wall_seconds"], 6)
        self.assertEqual(timing["phases"]["shared_pair_preparation"]["status"], "failed")
        self.assertEqual(timing["phases"]["shared_pair_preparation"]["wall_seconds"], 3)
        self.assertIn("synthetic preparation failure", timing["error"])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
