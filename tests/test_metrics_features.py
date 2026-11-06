"""Known-matrix CPU feature diagnostics, never restoration evidence."""
import unittest

import torch

from vsr.metrics import feature_statistics, same_anchor_patch_identity_retrieval


class FeatureDiagnosticTests(unittest.TestCase):
    def test_constant_features_have_zero_variance_and_effective_rank(self):
        features = torch.ones(1, 7, 4)
        result = feature_statistics(features, features * 5, torch.ones(1, 7, dtype=torch.bool))
        self.assertAlmostEqual(result["valid_point_cosine_error"], 0, places=14)
        self.assertEqual(result["student_mean_channel_variance"], 0)
        self.assertEqual(result["teacher_mean_channel_variance"], 0)
        self.assertEqual(result["student_effective_rank"], 0)
        self.assertEqual(result["teacher_effective_rank"], 0)
        irregular = torch.tensor([1., 2., 3., 7.]).repeat(1, 7, 1)
        irregular_result = feature_statistics(irregular, irregular, torch.ones(1, 7, dtype=torch.bool))
        self.assertEqual(irregular_result["student_mean_channel_variance"], 0)
        self.assertEqual(irregular_result["student_effective_rank"], 0)
        zeros = feature_statistics(features * 0, features * 0, torch.ones(1, 7, dtype=torch.bool))
        self.assertEqual(zeros["student_effective_rank"], 0)
        self.assertEqual(zeros["student_zero_norm_tokens"], 7)
        self.assertEqual(zeros["valid_point_cosine_error"], 1)

    def test_orthogonal_rows_have_known_centered_rank_and_population_variance(self):
        features = torch.eye(4).unsqueeze(0)
        result = feature_statistics(features, features, torch.ones(1, 4, dtype=torch.bool))
        self.assertAlmostEqual(result["student_effective_rank"], 3, places=12)
        self.assertAlmostEqual(result["student_mean_channel_variance"], 3 / 16, places=12)
        opposite = feature_statistics(features, -features, torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(opposite["valid_point_cosine_error"], 2)
        self.assertAlmostEqual(opposite["teacher_effective_rank"], 3, places=12)

    def test_masks_ignore_invalid_nan_and_no_autograd_is_retained(self):
        student = torch.tensor([[[1., 0.], [float("nan"), 0.], [0., 1.]]], requires_grad=True)
        teacher = student.detach().clone().requires_grad_()
        result = feature_statistics(student, teacher, torch.tensor([[True, False, True]]))
        self.assertEqual(result["valid_token_count"], 2)
        self.assertEqual(result["student_effective_rank"], 1)
        self.assertIsNone(student.grad)
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(not isinstance(value, torch.Tensor) for value in result.values()))
        single = feature_statistics(student, teacher, torch.tensor([[True, False, False]]))
        self.assertEqual(single["student_effective_rank"], 0)
        with self.assertRaises(ValueError):
            feature_statistics(student, teacher, torch.ones(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            feature_statistics(student, teacher, torch.zeros(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            feature_statistics(student, teacher, torch.ones(1, 3))

    def test_retrieval_candidates_stay_within_anchor(self):
        # Identical feature sets at two anchors would tie across time if the
        # implementation incorrectly searched the whole flattened video.
        features = torch.eye(3).repeat(2, 1).unsqueeze(0)
        mask = torch.ones(1, 6, dtype=torch.bool)
        result = same_anchor_patch_identity_retrieval(features, features, mask, (2, 1, 3))
        self.assertEqual(result["same_anchor_spatial_patch_top1_identity_rate"], 1)
        self.assertEqual(result["nonempty_anchor_count"], 2)
        permuted = features.reshape(1, 2, 3, 3).roll(1, dims=2).reshape(1, 6, 3)
        wrong = same_anchor_patch_identity_retrieval(features, permuted, mask, (2, 1, 3))
        self.assertEqual(wrong["same_anchor_spatial_patch_top1_identity_rate"], 0)
        negative = same_anchor_patch_identity_retrieval(features, -features, mask, (2, 1, 3))
        self.assertEqual(negative["same_anchor_spatial_patch_top1_identity_rate"], 0)

    def test_retrieval_masks_empty_anchors_and_ties_are_reported(self):
        features = torch.eye(3).repeat(2, 1).unsqueeze(0)
        features[:, :3] = float("nan")
        mask = torch.tensor([[False, False, False, True, True, True]])
        result = same_anchor_patch_identity_retrieval(features, features, mask, (2, 1, 3))
        self.assertEqual(result["empty_anchor_count"], 1)
        self.assertEqual(result["valid_query_count"], 3)
        self.assertEqual(result["same_anchor_spatial_patch_top1_identity_rate"], 1)
        constant = torch.ones(1, 4, 3)
        tied = same_anchor_patch_identity_retrieval(constant, constant, torch.ones(1, 4, dtype=torch.bool), (1, 2, 2))
        self.assertEqual(tied["same_anchor_spatial_patch_top1_identity_rate"], .25)
        self.assertEqual(tied["per_anchor"][0]["tied_top1_query_count"], 4)
        with self.assertRaises(ValueError):
            same_anchor_patch_identity_retrieval(constant, constant, torch.ones(1, 4, dtype=torch.bool), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
