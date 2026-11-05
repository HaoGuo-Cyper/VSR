"""Tiny real-Wan CPU calibration checks; no training or quality evidence."""
import unittest

import torch
from diffusers import WanTransformer3DModel

from vsr.calibration import collect_auxiliary_gradient_scales
from vsr.model import ConditionalWan


class RecordingWan(ConditionalWan):
    def __init__(self, native, **kwargs):
        super().__init__(native, **kwargs)
        self.inputs = []

    def forward(self, noisy, timestep, text, low, enable_lrv=True):
        self.inputs.append((noisy.detach().clone(), timestep.detach().clone(), low.detach().clone()))
        return super().forward(noisy, timestep, text, low, enable_lrv)


class CalibrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        native = WanTransformer3DModel(
            patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
            in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
            num_layers=3, rope_max_seq_len=16,
        )
        self.model = RecordingWan(native, rank=2, alpha=2, injection_layers=(0, 1), alignment_layer=1,
                                  stem_channels=8, projector_hidden=32, teacher_dim=16, gradient_checkpointing=True)
        self.clean = torch.randn(1, 4, 2, 4, 4, requires_grad=True)
        self.low = torch.randn_like(self.clean, requires_grad=True)
        self.text = torch.randn(1, 4, 16, requires_grad=True)
        self.teacher = torch.randn(1, 8, 16, requires_grad=True)
        self.flow = torch.zeros(1, 1, 2, 8, 8, requires_grad=True)

    def collect(self, **kwargs):
        return collect_auxiliary_gradient_scales(self.model, self.clean, self.low, self.text, self.teacher, (2, 2, 2),
                                                flow_pair={"forward": self.flow, "backward": -self.flow}, **kwargs)

    def test_finite_scales_exclude_projection_and_clear_frozen_grads(self):
        self.model.train()
        self.model.transformer.blocks[-1].eval()
        flags = [module.training for module in self.model.modules()]
        original = {name: tensor.clone() for name, tensor in self.model.state_dict().items()}
        result = self.collect()
        self.assertEqual(result["protocol"]["model_updates"], 0)
        self.assertEqual(result["protocol"]["relation_scope"], "spatial_and_hr_anchor_flow_temporal")
        self.assertTrue(.01 <= result["suggested_lambda_R"] <= 10)
        self.assertTrue(.01 <= result["suggested_lambda_A"] <= 10)
        for row in result["rows"]:
            for objective in ("FM", "A", "R"):
                gradients = row["objectives"][objective]
                self.assertGreater(gradients["generator_l2"], 0)
                self.assertEqual(gradients["frozen_elements_with_grad"], 0)
                self.assertTrue(gradients["finite"])
                groups = gradients["groups"]
                expected = sum(groups[g]["l2"] ** 2 for g in ("lrv_stem", "lrv_residuals", "lora")) ** .5
                self.assertAlmostEqual(gradients["generator_l2"], expected, places=10)
            self.assertEqual(row["objectives"]["FM"]["projector_l2"], 0)
            self.assertGreater(row["objectives"]["A"]["projector_l2"], 0)
            expected_ratio = .1 * row["objectives"]["FM"]["generator_l2"] / row["objectives"]["R"]["generator_l2"]
            self.assertAlmostEqual(row["lambda_R"]["raw"], expected_ratio, places=10)
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
        self.assertEqual(flags, [module.training for module in self.model.modules()])
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertTrue(all(value.grad is None for value in (self.clean, self.low, self.text, self.teacher, self.flow)))

    def test_every_objective_uses_identical_noised_input_and_local_noise_rng(self):
        first = self.collect()
        self.assertEqual(len(self.model.inputs), 9)
        for start in (0, 3, 6):
            for offset in (1, 2):
                for a, b in zip(self.model.inputs[start], self.model.inputs[start + offset]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.model.inputs[0][0], self.model.inputs[3][0]))
        torch.manual_seed(123456)
        torch.randn(500)
        self.model.inputs.clear()
        second = self.collect()
        self.assertEqual(first, second)

    def test_calibration_enables_grad_in_outer_no_grad_and_validates_geometry(self):
        with torch.no_grad():
            result = self.collect(sigmas=(.5,))
        self.assertGreater(result["rows"][0]["objectives"]["FM"]["generator_l2"], 0)
        with self.assertRaises(ValueError):
            self.collect(sigmas=(0,))
        with self.assertRaises(ValueError):
            self.collect(weight_bounds=(10, .01))


if __name__ == "__main__":
    unittest.main()
