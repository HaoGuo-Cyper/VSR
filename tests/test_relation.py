"""Geometry and loss tests; no estimated flow or training result is claimed."""
import unittest

import torch
import torch.nn.functional as F

from vsr.relation import relation_loss, temporal_correspondence


class RelationTests(unittest.TestCase):
    def test_constant_relation_zero_and_teacher_flow_detached(self):
        # Constant vectors differ in value between student/teacher but their
        # complete relation structures are identical (all edge cosines are 1).
        student = torch.tensor([1., 2., 3.]).view(1, 1, 1, 1, 3).expand(1, 3, 3, 4, 3).clone().requires_grad_()
        teacher = torch.tensor([3., 1., 2.]).view(1, 1, 1, 1, 3).expand_as(student).clone().requires_grad_()
        forward = torch.zeros(1, 2, 2, 12, 16, requires_grad=True)
        backward = torch.zeros_like(forward, requires_grad=True)
        loss, report = relation_loss(student, teacher, (3, 3, 4), forward, backward)
        self.assertLess(float(loss.detach()), 1e-12)
        self.assertEqual(report["temporal_edges"], 24)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNone(forward.grad)
        self.assertIsNone(backward.grad)
        self.assertIsNotNone(student.grad)

    def test_translation_hr_to_token_scales_and_pixel_centers(self):
        # HR 8x16 -> tokens 2x4: +4 HR pixels means exactly +1 token.
        forward = torch.zeros(1, 2, 8, 16)
        forward[:, 0] = 4
        backward = -forward
        result = temporal_correspondence(forward, backward, (2, 4))
        self.assertTrue(torch.equal(result["flow_tokens"][:, 0], torch.ones(1, 2, 4)))
        self.assertTrue(torch.equal(result["flow_tokens"][:, 1], torch.zeros(1, 2, 4)))
        self.assertTrue(bool(result["valid"][:, :, :3].all()))
        self.assertFalse(bool(result["valid"][:, :, 3:].any()))
        field = torch.arange(4).float().view(1, 1, 1, 4).expand(1, 1, 2, 4)
        sampled = F.grid_sample(field, result["grid"], align_corners=False)
        self.assertTrue(torch.allclose(sampled[0, 0, :, :3], torch.tensor([[1., 2., 3.], [1., 2., 3.]])))
        # Identity uses token centers (+.5), no half-pixel translation.
        identity = temporal_correspondence(forward * 0, backward * 0, (2, 4))
        unwarped = F.grid_sample(field, identity["grid"], align_corners=False)
        self.assertTrue(torch.equal(unwarped, field))

    def test_nontrivial_gradients_match_both_feature_fields(self):
        rng = torch.Generator().manual_seed(12)
        student = torch.randn(1, 3, 3, 4, 5, generator=rng, requires_grad=True)
        teacher = torch.randn(1, 3, 3, 4, 5, generator=rng, requires_grad=True)
        forward = torch.zeros(1, 2, 2, 12, 16)
        loss, report = relation_loss(student, teacher, (3, 3, 4), forward, -forward)
        self.assertEqual(report["active_edge_types"], 2)
        self.assertAlmostEqual(float(loss.detach()), (report["spatial_loss"] + report["temporal_loss"]) / 2, places=6)
        loss.backward()
        self.assertGreater(float(student.grad.norm()), 0)
        self.assertIsNone(teacher.grad)

    def test_outside_or_inconsistent_flow_has_no_temporal_edges(self):
        student = torch.randn(1, 2, 2, 3, 4, requires_grad=True)
        teacher = torch.randn_like(student)
        forward = torch.zeros(1, 1, 2, 8, 12)
        forward[:, :, 0] = 100
        loss, report = relation_loss(student, teacher, (2, 2, 3), forward, -forward)
        self.assertEqual(report["temporal_edges"], 0)
        self.assertEqual(report["active_edge_types"], 1)
        self.assertAlmostEqual(float(loss.detach()), report["spatial_loss"], places=7)
        forward[:, :, 0] = 2
        # Wrong reverse flow rejects correspondence under default threshold.
        _, report = relation_loss(student, teacher, (2, 2, 3), forward, forward)
        self.assertEqual(report["temporal_edges"], 0)

    def test_all_invalid_and_singleton_noedges_are_finite_zero(self):
        student = torch.randn(1, 2, 2, 3, 4, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        flow = torch.zeros(1, 1, 2, 8, 12)
        mask = torch.zeros(1, 2, 2, 3, dtype=torch.bool)
        loss, report = relation_loss(student, teacher, (2, 2, 3), flow, -flow, mask)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(report["active_edge_types"], 0)
        loss.backward()
        self.assertTrue(bool((student.grad == 0).all()))
        self.assertIsNone(teacher.grad)
        singleton = torch.randn(1, 1, 1, 1, 4, requires_grad=True)
        loss, report = relation_loss(singleton, singleton.detach(), (1, 1, 1))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(report["active_edge_types"], 0)

    def test_target_validity_uses_same_warp_and_flat_tokens_supported(self):
        student = torch.randn(1, 12, 4, requires_grad=True)
        teacher = torch.randn_like(student)
        flow = torch.zeros(1, 1, 2, 8, 12)
        flow[:, :, 0] = 4  # one token right
        mask = torch.ones(1, 2, 2, 3, dtype=torch.bool)
        mask[:, 1, :, 1] = False
        _, report = relation_loss(student, teacher, (2, 2, 3), flow, -flow, mask)
        # Src x0 -> invalid dst x1; src x1 -> valid dst x2; src x2 outside.
        self.assertEqual(report["temporal_edges"], 2)


if __name__ == "__main__":
    unittest.main()
