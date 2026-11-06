"""CPU reference-cache and RNG-contract checks; no full training is run."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from vsr.reference_training import draw_reference_condition, encode_reference_frames, save_reference_checkpoint


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=[1., 2., 3.], latents_std=[2., 2., 2.])
        self.inputs = []

    def encode(self, frame):
        self.inputs.append((frame.clone(), torch.is_grad_enabled()))
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: frame))


class ReferenceTrainingTests(unittest.TestCase):
    def test_all_original_frames_encoded_as_independent_single_frame_videos(self):
        vae = FakeVAE()
        hr = torch.arange(9, dtype=torch.float32).view(1, 1, 9, 1, 1).expand(1, 3, 9, 8, 8) / 8
        clip = {"hr": hr, "metadata": {"record_id": "clip", "source_id": "source", "source_indices": list(range(40, 49))},
                "binding": {"data_cache_key": "crop-and-degradation-digest", "manifest_sha256": "manifest"}}
        config = {"model": {"local_dir": "/nonexistent/fake-official-weights", "revision": "pinned-revision"},
                  "data": {"num_frames": 9}, "_sha256": "config-digest"}
        with tempfile.TemporaryDirectory() as temporary:
            result = encode_reference_frames(vae, [clip], config, temporary)
            self.assertEqual(set(result[0]), set(range(9)))
            self.assertEqual(len(vae.inputs), 9)
            for index, (input_frame, grad_enabled) in enumerate(vae.inputs):
                self.assertEqual(input_frame.shape, (1, 3, 1, 8, 8))
                self.assertFalse(grad_enabled)
                expected = (hr[:, :, index:index+1] * 2 - 1 - torch.tensor([1., 2., 3.]).view(1, 3, 1, 1, 1)) / 2
                torch.testing.assert_close(result[0][index], expected)
            loaded = torch.load(Path(temporary) / "clip_0/reference_latents.pt", weights_only=True)
            self.assertEqual(loaded["binding"]["source_binding"], clip["binding"])
            self.assertEqual(loaded["binding"]["frame_indices"], list(range(9)))
            seal = json.loads((Path(temporary) / "reference_cache_seal.json").read_text())
            self.assertTrue(seal["single_frame_only"])
            self.assertFalse(seal["video_latent_slicing"])
        self.assertTrue(vae.training)
        self.assertIsNone(vae.parameter.grad)

    def test_independent_reference_sequence_is_shared_across_variants(self):
        frames = {index: torch.full((1, 4, 1, 2, 2), float(index)) for index in range(9)}
        first_rng = torch.Generator(device="cpu").manual_seed(123 + 3)
        second_rng = torch.Generator(device="cpu").manual_seed(123 + 3)
        first = [draw_reference_condition(frames, 9, first_rng) for _ in range(30)]
        torch.manual_seed(999)
        torch.randn(500)
        second = [draw_reference_condition(frames, 9, second_rng) for _ in range(30)]
        self.assertTrue(any(record["reference_valid"] for _, record in first))
        self.assertTrue(any(not record["reference_valid"] for _, record in first))
        for (kwargs, record), (other, other_record) in zip(first, second):
            self.assertEqual(record, other_record)
            self.assertEqual(kwargs["frame_count"], 9)
            torch.testing.assert_close(kwargs["ref_valid"], other["ref_valid"], rtol=0, atol=0)
            if record["reference_valid"]:
                self.assertEqual(kwargs["ref_latent"].shape[2], 1)
                torch.testing.assert_close(kwargs["ref_latent"], other["ref_latent"], rtol=0, atol=0)
            else:
                self.assertIsNone(kwargs["ref_latent"])
                self.assertIsNone(kwargs["ref_index"])

    def test_missing_condition_never_reads_placeholder_or_changes_rng_consumption(self):
        class ForbiddenMapping:
            def __getitem__(self, _):
                raise AssertionError("Missing reference may not read a placeholder")
        missing_rng = torch.Generator(device="cpu").manual_seed(44)
        provided_rng = torch.Generator(device="cpu").manual_seed(44)
        frames = {index: torch.zeros(1, 4, 1, 2, 2) for index in range(9)}
        missing, record = draw_reference_condition(ForbiddenMapping(), 9, missing_rng, drop_probability=1)
        provided, _ = draw_reference_condition(frames, 9, provided_rng, drop_probability=0)
        self.assertFalse(record["reference_valid"])
        self.assertIsNone(missing["ref_latent"])
        self.assertIsNone(missing["ref_index"])
        self.assertFalse(bool(missing["ref_valid"][0]))
        self.assertIsNotNone(provided["ref_latent"])
        torch.testing.assert_close(missing_rng.get_state(), provided_rng.get_state(), rtol=0, atol=0)

    def test_checkpoint_retains_all_trainable_reference_parameters_and_rngs(self):
        model = nn.Module()
        model.base = nn.Parameter(torch.ones(1), requires_grad=False)
        model.reference_adapter = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        rngs = [torch.Generator(device="cpu").manual_seed(i) for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "new_checkpoint.pt"
            save_reference_checkpoint(path, model, optimizer, 12, {"reference": {"enabled": True}}, {}, *rngs)
            loaded = torch.load(path, weights_only=True)
        self.assertEqual(loaded["format"], "vsr_training_v1")
        self.assertTrue(loaded["reference_trained"])
        self.assertNotIn("base", loaded["trainable_state"])
        self.assertIn("reference_adapter.weight", loaded["trainable_state"])
        for key, rng in zip(("noise_rng", "index_rng", "reference_rng"), rngs):
            torch.testing.assert_close(loaded[key], rng.get_state(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
