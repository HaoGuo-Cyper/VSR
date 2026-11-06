"""Synthetic only: supplied-frame improvements must not leak into HRI gains."""
import copy
import unittest

import torch

from vsr.perceptual_metrics import TLP_KEY, TLP100_KEY, TOF_KEY
from vsr.score_reference_saved import score_cases


class FakePerceptual:
    def evaluate(self, prediction, target, include_temporal=True):
        distances = (prediction-target).abs().mean(dim=(0, 1, 3, 4)).tolist()
        changes = prediction.mean(dim=(0, 1, 3, 4)).tolist()
        pairs = []
        for i in range(2, len(changes)-3):
            value = abs(changes[i+1]-changes[i])
            pairs.append({"indices": [i, i+1], TLP_KEY: value, TLP100_KEY: 100*value, TOF_KEY: value})
        return {"lpips_per_frame": distances, "tecogan": {"per_pair": pairs}}


def fixture():
    target = torch.zeros(1, 3, 9, 64, 64)
    clip = {"hr": target, "lr_up": target, "fps": 8.,
            "metadata": {"record_id": "video", "source_id": "upload", "split": "val"}}
    row = {"clip_index": 0, "record_id": "video", "source_id": "upload", "split": "val", "reference_index": 4,
           "evaluation_frame_indices": [i for i in range(9) if i != 4], "fps": 8., "modes": {}}
    videos = {}
    for mode in ("hri_on", "hri_off"):
        samples = []
        for seed in (42, 43):
            path = f"{mode}_{seed}"
            value = torch.full_like(target, .2)
            if mode == "hri_on":
                value[:, :, 4] = 0
            videos[path] = value
            samples.append({"seed": seed, "noise_seed": seed, "rgb_path": path, "hri_valid": mode == "hri_on",
                            "supplied_reference_index": 4 if mode == "hri_on" else None})
        row["modes"][mode] = {"seeds": samples}
    metrics = {"status": "completed", "model_setting": "reference_trained_with_dropout", "scope": "synthetic fixture only",
               "sampling": {"seeds": [42, 43], "matched_noise_across_modes": True}, "reference_cases": {"4": {"reference_index": 4, "per_video": [row]}}}
    return metrics, [clip], videos


class ReferenceScoringTest(unittest.TestCase):
    def test_provided_frame_cannot_create_primary_or_distance_gain(self):
        torch.set_num_threads(2)
        metrics, clips, videos = fixture()
        report = score_cases(metrics, clips, FakePerceptual(), videos.__getitem__)
        case = report["reference_cases"]["4"]
        delta = case["matched_nonreference_comparison"]["comparisons"]["hri_on"]
        for metric in delta.values():
            self.assertEqual(metric["delta_video_macro"], 0)
        row = case["per_video"][0]
        on = row["modes"]["hri_on"]["seeds"][0]["metrics"]
        off = row["modes"]["hri_off"]["seeds"][0]["metrics"]
        self.assertGreater(on["reference_frame_separate"]["psnr_rgb_db"], off["reference_frame_separate"]["psnr_rgb_db"])
        self.assertEqual(on["matched_nonreference"]["temporal_pair_indices"], [[2, 3], [5, 6]])
        self.assertGreater(on["all_frames"][TLP_KEY], on["matched_nonreference"][TLP_KEY])
        distances = case["gain_by_original_frame_and_distance"]["video"]
        self.assertEqual(len(distances), 8)
        self.assertTrue(all(row["psnr_gain_db"] == row["lpips_on_minus_off"] == 0 for row in distances))
        self.assertNotIn(4, [row["frame_index"] for row in distances])

    def test_refuse_incomplete_seeds_mismatched_frame_pool_and_false_validity(self):
        metrics, clips, videos = fixture()
        for kind in ("seeds", "frames", "validity", "index", "noise", "matched"):
            bad = copy.deepcopy(metrics)
            row = bad["reference_cases"]["4"]["per_video"][0]
            if kind == "seeds":
                row["modes"]["hri_on"]["seeds"].pop()
            elif kind == "frames":
                row["evaluation_frame_indices"].append(4)
            elif kind == "validity":
                row["modes"]["hri_off"]["seeds"][0]["hri_valid"] = True
            elif kind == "index":
                row["modes"]["hri_on"]["seeds"][0]["supplied_reference_index"] = 0
            elif kind == "noise":
                row["modes"]["hri_on"]["seeds"][0]["noise_seed"] += 100
            else:
                bad["sampling"]["matched_noise_across_modes"] = False
            with self.assertRaises(ValueError):
                score_cases(bad, clips, FakePerceptual(), videos.__getitem__)


if __name__ == "__main__":
    unittest.main()
