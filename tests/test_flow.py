"""Flow contracts tested with a fake encoder; no real-flow result is implied."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from vsr.flow import FlowTeacher, OFFICIAL_SHA256_PREFIX, OFFICIAL_WEIGHTS_URL


class FakeRAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []
        self.loaded_strict = None

    def load_state_dict(self, state, strict=True):
        self.loaded_strict = strict
        return None

    def forward(self, first, second, num_flow_updates):
        self.calls.append((first.clone(), second.clone(), num_flow_updates, torch.is_grad_enabled()))
        # Signed difference proves both actual anchor order and swapped pairs.
        difference = (second - first).mean((1, 2, 3), keepdim=True)
        return [difference.expand(first.shape[0], 2, *first.shape[-2:]) * self.weight]


def make_fixture(path):
    fake = FakeRAFT()
    digest = OFFICIAL_SHA256_PREFIX + "0" * 56
    with patch("vsr.flow.sha256_file", return_value=digest), patch("vsr.flow.raft_large", return_value=fake) as builder, patch("vsr.flow.torch.load", return_value={}) as loader:
        teacher = FlowTeacher(path)
        builder.assert_called_once_with(weights=None, progress=False)
        loader.assert_called_once_with(Path(path).resolve(), map_location="cpu", weights_only=True)
    return teacher, fake


class FlowTests(unittest.TestCase):
    def test_direct_anchor_order_preprocess_shapes_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.pth"
            path.write_bytes(b"synthetic fixture only")
            teacher, encoder = make_fixture(path)
            teacher.train(True)
            self.assertFalse(teacher.training)
            self.assertFalse(encoder.training)
            self.assertTrue(encoder.loaded_strict)
            self.assertFalse(encoder.weight.requires_grad)
            frames = torch.linspace(0, 1, 9).reshape(1, 1, 9, 1, 1).expand(1, 3, 9, 128, 224).clone().requires_grad_()
            forward, backward = teacher.estimate_anchor_flows(frames, [0, 4, 8])
            self.assertEqual(forward.shape, (1, 2, 2, 128, 224))
            self.assertEqual(backward.shape, forward.shape)
            self.assertTrue(torch.allclose(forward, torch.ones_like(forward)))
            self.assertTrue(torch.allclose(backward, -torch.ones_like(backward)))
            self.assertEqual(len(encoder.calls), 4)
            # Official transform maps anchor intensity 0/.5/1 to -1/0/1.
            for call, expected in zip(encoder.calls, [(-1, 0), (0, -1), (0, 1), (1, 0)]):
                first, second, updates, grad_enabled = call
                self.assertEqual(float(first.mean()), expected[0])
                self.assertEqual(float(second.mean()), expected[1])
                self.assertEqual(updates, 12)
                self.assertFalse(grad_enabled)
                self.assertTrue(first.is_contiguous())
                self.assertTrue(second.is_contiguous())
            self.assertFalse(forward.requires_grad)
            self.assertFalse(backward.requires_grad)
            self.assertIsNone(frames.grad)
            self.assertEqual(teacher.provenance["official_url"], OFFICIAL_WEIGHTS_URL)

    def test_reject_unverified_weights_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pth"
            path.write_bytes(b"not official weights")
            with patch("vsr.flow.torch.load") as loader:
                with self.assertRaisesRegex(ValueError, "official URL hash prefix"):
                    FlowTeacher(path)
                loader.assert_not_called()

    def test_invalid_geometry_range_or_anchors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.pth"
            path.write_bytes(b"synthetic fixture")
            teacher, encoder = make_fixture(path)
            with self.assertRaisesRegex(ValueError, "divisible"):
                teacher(torch.zeros(1, 3, 9, 128, 225), [0, 4, 8])
            with self.assertRaisesRegex(ValueError, "divisible"):
                teacher(torch.zeros(1, 3, 9, 64, 128), [0, 4, 8])
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                teacher(torch.zeros(1, 3, 9, 128, 224), [0, 8, 4])
            with self.assertRaisesRegex(ValueError, "valid integer"):
                teacher(torch.zeros(1, 3, 9, 128, 224), [0, 9])
            with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
                teacher(torch.full((1, 3, 9, 128, 224), 1.1), [0, 4, 8])
            self.assertEqual(len(encoder.calls), 0)


if __name__ == "__main__":
    unittest.main()
