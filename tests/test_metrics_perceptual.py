"""CPU numerical checks against pinned authors' source and official weights."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from vsr.metrics import (LPIPS_KEY, TLP_KEY, TLP100_KEY, TOF_KEY,
                                   PerceptualMetrics, tecogan_crop)


ROOT = Path(__file__).resolve().parents[1]


def _author_definitions(filename, names, namespace):
    """Execute only selected author definitions, never the CLI / import code."""
    tree = ast.parse((ROOT / "vendor/metric_sources" / filename).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("Pinned author definitions missing")
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class CropProtocolTest(unittest.TestCase):
    def test_author_crop_matches_exact_geometry_and_pixel_positions(self):
        namespace = _author_definitions("tecogan_metrics.py", {"crop_8x8"}, {})
        for height, width in [(128, 224), (129, 225), (64, 96), (48, 48), (200, 300)]:
            frame = np.arange(height * width * 2).reshape(height, width, 2)
            actual, top, left = tecogan_crop(frame)
            expected, expected_top, expected_left = namespace["crop_8x8"](frame)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual((top, left), (expected_top, expected_left))
        image, top, left = tecogan_crop(np.zeros((128, 224, 3)))
        self.assertEqual((top, left, *image.shape[:2]), (16, 16, 96, 192))
        with self.assertRaises(ValueError):
            tecogan_crop(np.zeros((32, 32, 3)))


class OfficialPerceptualMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        alex = ROOT / "weights/alexnet-owt-7be5be79.pth"
        linear = ROOT / "weights/lpips_v0.1_alex.pth"
        if not alex.is_file() or not linear.is_file():
            raise unittest.SkipTest("Download pinned official perceptual weights first")
        cls.metric = PerceptualMetrics(alex, linear, device="cpu", batch_size=3)
        # Run the author's actual PNetLin class, using the byte-identical local
        # calibration and the strictly loaded official AlexNet trunk. This
        # validates the modernization separately from the temporal aggregation.
        import lpips.pretrained_networks as pn
        utilities = _author_definitions("tecogan_util.py", {"normalize_tensor", "im2tensor"},
                                       {"torch": torch, "np": np})
        namespace = _author_definitions("tecogan_networks_basic.py", {"PNetLin", "NetLinLayer"},
                         {"torch": torch, "nn": nn, "pn": pn, "util": SimpleNamespace(**utilities)})
        cls.author = namespace["PNetLin"](pnet_type="alex", pnet_rand=True, use_gpu=False, version="0.1")
        cls.author.net[0].load_state_dict(cls.metric.model.net.state_dict(), strict=True)
        cls.author.load_state_dict(torch.load(linear, map_location="cpu", weights_only=True), strict=True)
        cls.author.eval()
        cls.author_im2tensor = staticmethod(utilities["im2tensor"])
        cls.author_crop = staticmethod(_author_definitions("tecogan_metrics.py", {"crop_8x8"}, {})["crop_8x8"])

    def videos(self):
        rng = torch.Generator().manual_seed(1234)
        target = torch.rand((1, 3, 9, 64, 96), generator=rng)
        predicted = (target.roll(2, dims=-1) * 0.8 + 0.1).clamp(0, 1)
        return predicted, target

    def test_modern_official_lpips_matches_author_v01_network(self):
        prediction, target = self.videos()
        a = prediction[0].permute(1, 0, 2, 3) * 2 - 1
        b = target[0].permute(1, 0, 2, 3) * 2 - 1
        with torch.no_grad():
            actual = self.metric.model(a, b, normalize=False)
            expected = self.author(a, b)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        self.assertTrue(all(not p.requires_grad for p in self.metric.model.parameters()))
        self.assertFalse(self.metric.model.training)

    def test_temporal_metrics_match_authors_loop_and_frames(self):
        import cv2
        prediction, target = self.videos()
        actual = self.metric.evaluate(prediction, target)
        pred8 = np.rint(prediction[0].permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
        gt8 = np.rint(target[0].permute(1, 2, 3, 0).numpy() * 255).astype(np.uint8)
        expected_tlp, expected_tof = [], []
        with torch.no_grad():
            for i in range(3, len(pred8) - 2):
                # Directly reproduce metrics.py's full-frame Farneback calls,
                # center-cropped flow error, im2tensor and scalar difference.
                target_flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(gt8[i-1], cv2.COLOR_RGB2GRAY), cv2.cvtColor(gt8[i], cv2.COLOR_RGB2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                output_flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(pred8[i-1], cv2.COLOR_RGB2GRAY), cv2.cvtColor(pred8[i], cv2.COLOR_RGB2GRAY),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                target_flow, _, _ = self.author_crop(target_flow)
                output_flow, _, _ = self.author_crop(output_flow)
                diff = np.absolute(target_flow - output_flow)
                expected_tof.append(float(np.sqrt(np.sum(diff * diff, axis=-1)).mean()))
                def distance(frames):
                    first = self.author_im2tensor(self.author_crop(frames[i-1])[0])
                    second = self.author_im2tensor(self.author_crop(frames[i])[0])
                    return float(self.author(first, second).item())
                expected_tlp.append(abs(distance(gt8) - distance(pred8)))
        temporal = actual["tecogan"]
        self.assertEqual(temporal["frame_indices"], [2, 3, 4, 5, 6])
        self.assertEqual([pair["indices"] for pair in temporal["per_pair"]], [[2, 3], [3, 4], [4, 5], [5, 6]])
        self.assertAlmostEqual(temporal[TOF_KEY], float(np.mean(expected_tof)), places=7)
        np.testing.assert_allclose([pair[TLP_KEY] for pair in temporal["per_pair"]], expected_tlp, rtol=2e-4, atol=3e-7)
        self.assertAlmostEqual(temporal[TLP100_KEY], 100 * temporal[TLP_KEY], places=12)
        self.assertEqual(actual["frame_count"], 9)
        self.assertGreater(actual[LPIPS_KEY], 0)

    def test_identity_is_zero_and_allframes_lpips_uses_even_cut_frames(self):
        prediction, target = self.videos()
        identity = self.metric.evaluate(target, target)
        for value in [identity[LPIPS_KEY], identity["tecogan"][TLP_KEY], identity["tecogan"][TOF_KEY]]:
            self.assertEqual(value, 0.0)
        changed = target.clone()
        changed[:, :, 0] = prediction[:, :, 0]
        result = self.metric.evaluate(changed, target)
        self.assertGreater(result[LPIPS_KEY], 0)
        self.assertEqual(result["tecogan"][TLP_KEY], 0.0)
        self.assertEqual(result["tecogan"][TOF_KEY], 0.0)
        self.assertEqual(result, self.metric.evaluate(changed, target))

    def test_invalid_values_shapes_and_short_temporal_videos_fail(self):
        prediction, target = self.videos()
        for changed in [target * 2, target + float("nan"), target[:, :, :-1]]:
            with self.assertRaises(ValueError):
                self.metric.evaluate(changed, target)
        with self.assertRaises(ValueError):
            self.metric.evaluate(prediction[:, :, :5], target[:, :, :5])
        result = self.metric.evaluate(prediction[:, :, :1], target[:, :, :1], include_temporal=False)
        self.assertNotIn("tecogan", result)
        self.assertEqual(result["frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
