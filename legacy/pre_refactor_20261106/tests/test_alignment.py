"""CPU checks for correspondence and losses; no experimental quality claims."""

from types import SimpleNamespace
import unittest
import torch
from torch import nn

from vsr.alignment import DinoTeacher, latent_geometry, point_alignment, teacher_to_student


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.vae = {"dim_mult": [1, 2, 4, 4], "temperal_downsample": [False, True, True]}
        self.dit = {"patch_size": [1, 2, 2]}

    def test_causal_geometry_reads_configuration_and_rejects_unsupported_padding(self):
        geom = latent_geometry(self.vae, self.dit, 9, 128, 224)
        self.assertEqual(geom["temporal_stride"], 4)
        self.assertEqual(geom["spatial_stride"], 8)
        self.assertEqual(geom["anchors"], [0, 4, 8])
        self.assertEqual(geom["latent_grid"], [3, 16, 28])
        self.assertEqual(geom["token_grid"], [3, 8, 14])
        # This also rules out accidentally hardcoding a temporal stride of four.
        geom = latent_geometry(SimpleNamespace(dim_mult=[1, 2], temperal_downsample=[True]),
                               SimpleNamespace(patch_size=[1, 1, 1]), 5, 8, 12)
        self.assertEqual(geom["anchors"], [0, 2, 4])
        self.assertEqual(geom["token_grid"], [3, 4, 6])
        for args in [(8, 128, 224), (9, 127, 224), (0, 128, 224)]:
            with self.assertRaises(ValueError):
                latent_geometry(self.vae, self.dit, *args)
        with self.assertRaises(ValueError):
            latent_geometry(self.vae, {"patch_size": [2, 2, 2]}, 9, 128, 224)
        with self.assertRaises(ValueError):
            latent_geometry({}, self.dit, 9, 128, 224)

    def test_mapping_preserves_batch_anchor_and_hw_flatten_order(self):
        # B=2, Ta=3, Ht=2, Wt=4. Per-anchor constants expose accidental
        # temporal or batch mixing; a spatial ramp exposes wrong H/W order.
        source = torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8, 1)
        actual = teacher_to_student(source, (2, 4), (3, 1, 2))
        expected = torch.tensor([[[2.5], [4.5], [10.5], [12.5], [18.5], [20.5]],
                                 [[26.5], [28.5], [34.5], [36.5], [42.5], [44.5]]])
        torch.testing.assert_close(actual, expected)
        identity = teacher_to_student(source, (2, 4), (3, 2, 4))
        torch.testing.assert_close(identity, source.reshape(2, 24, 1))
        with self.assertRaises(ValueError):
            teacher_to_student(source, (2, 3), (3, 1, 2))
        with self.assertRaises(ValueError):
            teacher_to_student(source, (2, 4), (4, 1, 2))

    def test_valid_only_cosine_loss_detaches_teacher_and_masks_gradients(self):
        student = torch.tensor([[[1., 0.], [1., 0.], [float("nan"), float("nan")]]], requires_grad=True)
        teacher = torch.tensor([[[0., 1.], [1., 0.], [float("nan"), float("nan")]]], requires_grad=True)
        loss = point_alignment(student, teacher, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(loss, torch.tensor(0.5))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad[0, 0].abs().sum()), 0)
        torch.testing.assert_close(student.grad[0, 2], torch.zeros(2))
        with self.assertRaises(ValueError):
            point_alignment(student, teacher, torch.zeros(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            point_alignment(student, teacher, torch.ones(1, 3))

    def test_teacher_preprocess_keeps_full_crop_and_temporal_order(self):
        # Avoid loading a large external checkpoint in unit tests; exercise the
        # production forward with a deterministic patch reader instead.
        class PatchReader(nn.Module):
            def forward_features(self, frames):
                self.observed = frames.detach().clone()
                patches = frames.unfold(2, 14, 14).unfold(3, 14, 14).mean((-1, -2))
                tokens = patches.permute(0, 2, 3, 1).reshape(frames.shape[0], -1, 3)
                return {"x_norm_patchtokens": tokens.repeat(1, 1, 256)}

        teacher = DinoTeacher.__new__(DinoTeacher)
        nn.Module.__init__(teacher)
        teacher.model = PatchReader()
        teacher.input_range = "zero_one"
        teacher.patch_grid_multiplier = 2
        teacher.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        teacher.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        hr = torch.zeros(1, 3, 5, 28, 56, requires_grad=True)
        with torch.no_grad():
            hr[:, :, 4] = 1
            hr[:, :, 2] = 0.5
        features = teacher(hr, [0, 4], (2, 1, 2))
        self.assertEqual(features.shape, (1, 4, 768))
        self.assertFalse(features.requires_grad)
        self.assertEqual(teacher.model.observed.shape, (2, 3, 28, 56))
        torch.testing.assert_close(teacher.model.observed[0, :, 0, 0],
                                   -teacher.rgb_mean.flatten() / teacher.rgb_std.flatten())
        torch.testing.assert_close(teacher.model.observed[1, :, 0, 0],
                                   (1 - teacher.rgb_mean.flatten()) / teacher.rgb_std.flatten())
        torch.testing.assert_close(features.norm(dim=-1), torch.ones(1, 4))
        teacher.train()
        self.assertFalse(teacher.training)
        self.assertFalse(teacher.model.training)
        with self.assertRaises(ValueError):
            teacher(hr, [4, 0], (2, 1, 2))
        with self.assertRaises(ValueError):
            teacher(hr, [0, 5], (2, 1, 2))


if __name__ == "__main__":
    unittest.main()
