"""MUSIQ contracts use synthetic images/fake scoring, not VideoLQ evaluation."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vsr.no_reference import MUSIQEvaluator, KONIQ_SHA256, preprocess_frames, verify_musiq_sources


class FakeMUSIQ(nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.head = nn.Linear(384, 1)
        self.options = options
        self.calls = []

    def forward(self, frames):
        self.calls.append((tuple(frames.shape), self.training, torch.is_grad_enabled()))
        return frames.mean((1, 2, 3)).unsqueeze(-1) * 100


def fake_evaluator(path):
    # Only fixture checkpoint I/O is mocked; vendored source integrity is real.
    def digest(file):
        if Path(file).resolve() == Path(path).resolve():
            return KONIQ_SHA256
        return hashlib.sha256(Path(file).read_bytes()).hexdigest()
    state = {"head.weight": torch.zeros(1, 384), "head.bias": torch.zeros(1)}
    with patch("vsr.no_reference._sha256_file", side_effect=digest), patch("vsr.no_reference.torch.load", return_value=state) as loader:
        evaluator = MUSIQEvaluator(path, backend_factory=FakeMUSIQ, device="cpu")
        loader.assert_called_once_with(Path(path).resolve(), map_location="cpu", weights_only=True)
    return evaluator


class NoReferenceTests(unittest.TestCase):
    def test_preprocessing_deterministic_native_and_two_aspect_scales(self):
        frames = torch.rand(1, 3, 64, 128, generator=torch.Generator().manual_seed(17))
        first, second = preprocess_frames(frames), preprocess_frames(frames)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (1, 49 + 144 + 8, 3075))
        # Scale2 is the uncut64x128 native image:2x4=8 patches.
        native = first[:, first[0, :, -2] == 2]
        self.assertEqual(native.shape[1], 8)
        self.assertTrue(bool((native[..., -1] == 1).all()))
        expected = torch.nn.functional.unfold((frames - .5) * 2, 32, stride=32).transpose(1, 2)
        self.assertTrue(torch.equal(native[..., :-3], expected))
        self.assertFalse(first.requires_grad)

    def test_local_koniq_selection_frozen_all_frames_and_video_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pth"
            path.write_bytes(b"fake score backend fixture")
            with patch("urllib.request.urlopen", side_effect=AssertionError("No network allowed")):
                evaluator = fake_evaluator(path)
                evaluator.train(True)
                video = torch.tensor([0., .25, .5, .75, 1.]).view(1, 1, 5, 1, 1).expand(1, 3, 5, 64, 128).clone().requires_grad_()
                result = evaluator(video, batch_size=2)
            self.assertEqual(result["per_frame_scores"], [0., 25., 50., 75., 100.])
            self.assertEqual(result["mean"], 50.)
            self.assertEqual(result["frame_count"], 5)
            self.assertFalse(result["quality_hypothesis_verified"])
            self.assertIsNone(video.grad)
            self.assertFalse(evaluator.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in evaluator.parameters()))
            self.assertEqual(evaluator.backend.options, {"pretrained": False, "num_class": 1, "longer_side_lengths": [224,384], "max_seq_len_from_original_res": -1})
            self.assertTrue(all(not training and not grad for _, training, grad in evaluator.backend.calls))
            self.assertEqual([shape[0] for shape, _, _ in evaluator.backend.calls], [2,2,1])

    def test_wrong_checkpoint_rejected_before_backend_or_load_and_missing_local(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "musiq_ava_wrong.pth"
            path.write_bytes(b"wrong weights")
            with patch("vsr.no_reference.torch.load") as loader, patch("urllib.request.urlopen", side_effect=AssertionError("No download")):
                with self.assertRaisesRegex(ValueError, "KonIQ checkpoint SHA256"):
                    MUSIQEvaluator(path, backend_factory=FakeMUSIQ)
                with self.assertRaises(FileNotFoundError):
                    MUSIQEvaluator(path.parent / "missing.pth", backend_factory=FakeMUSIQ)
                loader.assert_not_called()

    def test_invalid_video_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pth"
            path.write_bytes(b"fixture")
            evaluator = fake_evaluator(path)
            for value in [torch.zeros(1,3,32,64), torch.zeros(1,3,0,32,64), torch.full((1,3,1,32,64),float('nan')), torch.full((1,3,1,32,64),1.1)]:
                with self.assertRaises(ValueError):
                    evaluator(value)
            with self.assertRaises(ValueError):
                evaluator(torch.zeros(1,3,1,32,64),batch_size=0)
            self.assertEqual(verify_musiq_sources()["weights"]["sha256"], KONIQ_SHA256)


if __name__ == "__main__":
    unittest.main()
