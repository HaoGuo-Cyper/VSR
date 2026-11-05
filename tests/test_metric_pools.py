"""Small public metric-pool edge checks with synthetic perceptual values."""
import unittest

import torch

from vsr.metric_pools import paired_metric_pools
from vsr.perceptual_metrics import TLP_KEY, TLP100_KEY, TOF_KEY


class FixturePerceptual:
    def __init__(self):
        self.full_lengths = []

    def evaluate(self, prediction, target, include_temporal=True):
        length = prediction.shape[2]
        self.full_lengths.append(length)
        pairs = [{"indices": [index, index+1], TLP_KEY: float(index), TLP100_KEY: float(index*100), TOF_KEY: float(index+1)}
                 for index in range(2, length-3)]
        return {"lpips_per_frame": list(map(float, range(length))), "tecogan": {"per_pair": pairs}}


class MetricPoolsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_middle_frame_lpips_and_original_temporal_pair_pools(self):
        target = torch.zeros(1, 3, 9, 8, 8)
        evaluator = FixturePerceptual()
        pools = paired_metric_pools(target, target, evaluator, reference_index=4)
        self.assertEqual(evaluator.full_lengths, [9])
        self.assertEqual(pools["matched_nonreference"]["frame_indices"], [0, 1, 2, 3, 5, 6, 7, 8])
        self.assertEqual(pools["matched_nonreference"]["temporal_pair_indices"], [[2, 3], [5, 6]])
        self.assertEqual(pools["reference_frame_separate"]["lpips_alex_v0_1"], 4.0)
        self.assertEqual(pools["matched_nonreference"]["lpips_alex_v0_1"], 4.0)
        self.assertEqual(pools["reference_frame_separate"]["frame_indices"], [4])

    def test_no_remaining_time_pair_is_none_and_invalid_indices_fail(self):
        video = torch.zeros(1, 3, 6, 8, 8)
        pools = paired_metric_pools(video, video, FixturePerceptual(), reference_index=2)
        self.assertEqual(pools["all_frames"]["temporal_pair_indices"], [[2, 3]])
        self.assertEqual(pools["matched_nonreference"]["temporal_pair_indices"], [])
        self.assertIsNone(pools["matched_nonreference"][TLP_KEY])
        for reference in (-1, 6, True):
            with self.assertRaises(ValueError):
                paired_metric_pools(video, video, FixturePerceptual(), reference_index=reference)


if __name__ == "__main__":
    unittest.main()
