"""CPU HRI component checks; no reference-model training has been performed."""
import unittest

import torch
from torch import nn

from vsr.reference import HRIAdapter


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        self.adapter = HRIAdapter(dim=24, latent_channels=4, heads=2, stem_dim=8)
        self.hidden = torch.randn(2, 12, 24)
        self.reference = torch.randn(2, 4, 1, 4, 4)
        self.indices = torch.tensor([0, 4])
        self.valid = torch.tensor([True, True])

    def update(self):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=0.02)
        target = torch.randn_like(self.hidden)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            residual = self.adapter(self.hidden, self.reference, self.indices, 9, self.valid)
            (residual - target).square().mean().backward()
            optimizer.step()

    def test_zero_initialization_is_identity_and_opens_without_dead_gate(self):
        residual = self.adapter(self.hidden, self.reference, self.indices, 9, self.valid)
        self.assertEqual(torch.count_nonzero(residual), 0)
        torch.testing.assert_close(self.hidden + residual, self.hidden, rtol=0, atol=0)
        residual.sum().backward()
        self.assertGreater(float(self.adapter.output_projection.weight.grad.abs().sum()), 0)
        self.assertAlmostEqual(float(self.adapter.gate_logit.detach().sigmoid()), 0.5)

    def test_reference_changes_nonreference_tokens_after_updates(self):
        self.update()
        with torch.no_grad():
            first = self.adapter(self.hidden, self.reference, self.indices, 9, self.valid)
            other = self.adapter(self.hidden, self.reference.flip(-1), self.indices, 9, self.valid)
        # These query tokens are arbitrary video positions. No reference-token
        # copy operation is used, so changes can reach all other query positions.
        self.assertGreater(float((first[:, 4:] - other[:, 4:]).abs().sum()), 0)
        self.assertGreater(float(self.adapter.stem[0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.adapter.parameters()))

    def test_absent_reference_is_exactly_zero_and_has_no_parameter_gradient(self):
        self.update()
        self.adapter.zero_grad(set_to_none=True)
        absent = torch.tensor([False, False])
        first = self.adapter(self.hidden, self.reference, self.indices, 9, absent)
        other = self.adapter(self.hidden, torch.full_like(self.reference, float("nan")), [-1, -1], 9, absent)
        missing = self.adapter(self.hidden, None, None, 9, absent)
        self.assertEqual(torch.count_nonzero(first), 0)
        torch.testing.assert_close(first, other, rtol=0, atol=0)
        torch.testing.assert_close(first, missing, rtol=0, atol=0)
        hidden = self.hidden.clone().requires_grad_()
        (hidden + self.adapter(hidden, None, None, 9, absent)).sum().backward()
        self.assertTrue(all(p.grad is None for p in self.adapter.parameters()))
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)

    def test_mixed_missing_rows_are_ignored_and_frame_index_changes_output(self):
        self.update()
        mixed = torch.tensor([True, False])
        reference = self.reference.clone()
        reference[1] = float("nan")
        result = self.adapter(self.hidden, reference, [0, -1], [9, 0], mixed)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(torch.count_nonzero(result[1]), 0)
        changed_index = self.adapter(self.hidden, reference, [4, -1], [9, 0], mixed)
        self.assertGreater(float((result[0] - changed_index[0]).detach().abs().sum()), 0)

    def test_one_reference_and_known_valid_index_are_required(self):
        with self.assertRaises(ValueError):
            self.adapter(self.hidden, self.reference.expand(-1, -1, 2, -1, -1), self.indices, 9, self.valid)
        with self.assertRaises(ValueError):
            self.adapter(self.hidden, self.reference, [0, 9], 9, self.valid)
        with self.assertRaises(ValueError):
            self.adapter(self.hidden, self.reference, [0.0, 4.0], 9, self.valid)
        with self.assertRaises(ValueError):
            self.adapter(self.hidden, self.reference, self.indices, 9, torch.ones(2))


if __name__ == "__main__":
    unittest.main()
