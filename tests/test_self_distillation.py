"""Tiny CPU gradient/contract fixtures; no GPU, task weights or optimizer step."""
import copy
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from vsr.self_distillation import (TrainableEMA, sample_span_mask,
    corrupt_latent_condition, temporal_token_mask, MaskedTemporalAlignment)


def rng(seed=21):
    return torch.Generator(device="cpu").manual_seed(seed)


class TinyRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4).requires_grad_(False)
        self.adapter = nn.Linear(4, 4, bias=False)

    def forward(self, value):
        return self.backbone(value) + self.adapter(value)


class EMATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.model = TinyRestorer().train()
        self.model.backbone.eval()
        self.ema = TrainableEMA(self.model)

    def test_initialized_teacher_matches_student_and_stores_only_trainables(self):
        value = torch.randn(2, 4, requires_grad=True)
        expected = self.model(value)
        actual = self.ema.teacher_forward(self.model, self.model, value)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.requires_grad)
        self.assertEqual(self.ema.names, ("adapter.weight",))
        self.assertEqual(set(self.ema.state_dict()["shadow"]), {"adapter.weight"})
        self.assertEqual(self.ema.num_updates, 0)
        self.assertTrue(self.model.training)
        self.assertFalse(self.model.backbone.training)
        self.assertTrue(self.model.adapter.training)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_ema_math_frozen_weights_and_export_ownership(self):
        initial = self.ema.state_dict()["shadow"]["adapter.weight"]
        frozen = {k: p.detach().clone() for k, p in self.model.named_parameters() if not p.requires_grad}
        with torch.no_grad():
            self.model.adapter.weight.add_(2.)
        self.ema.update(self.model)
        expected = initial * .999 + self.model.adapter.weight.detach() * .001
        torch.testing.assert_close(self.ema.state_dict()["shadow"]["adapter.weight"], expected)
        self.assertEqual(self.ema.num_updates, 1)
        self.assertTrue(all(torch.equal(p, frozen[k]) for k, p in self.model.named_parameters() if not p.requires_grad))
        exported = self.ema.state_dict(); exported["shadow"]["adapter.weight"].zero_()
        self.assertFalse(torch.equal(exported["shadow"]["adapter.weight"], self.ema.state_dict()["shadow"]["adapter.weight"]))

    def test_exception_safe_swap_preserves_student_grads_and_heterogeneous_modes(self):
        with torch.no_grad():
            self.model.adapter.weight.add_(3.)
        self.model.adapter.weight.grad = torch.full_like(self.model.adapter.weight, .75)
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        frozen_versions = {name: p._version for name, p in self.model.named_parameters() if not p.requires_grad}
        with self.assertRaisesRegex(RuntimeError, "fixture teacher failure"):
            with self.ema.teacher_parameters(self.model):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(any(module.training for module in self.model.modules()))
                self.assertTrue(torch.equal(self.model.adapter.weight, self.ema.state_dict()["shadow"]["adapter.weight"]))
                raise RuntimeError("fixture teacher failure")
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in self.model.named_parameters()))
        self.assertTrue(torch.equal(self.model.adapter.weight.grad, torch.full_like(self.model.adapter.weight, .75)))
        self.assertTrue(self.model.training and self.model.adapter.training)
        self.assertFalse(self.model.backbone.training)
        self.assertEqual(frozen_versions, {name: p._version for name, p in self.model.named_parameters() if not p.requires_grad})

    def test_teacher_result_owns_storage_after_student_restoration(self):
        shadow = self.ema.state_dict()["shadow"]["adapter.weight"]
        with torch.no_grad():
            self.model.adapter.weight.add_(5.)
        result = self.ema.teacher_forward(self.model, lambda: {"features": self.model.adapter.weight})
        self.assertTrue(torch.equal(result["features"], shadow))
        self.assertFalse(result["features"].requires_grad)
        self.assertNotEqual(result["features"].data_ptr(), self.model.adapter.weight.data_ptr())

    def test_exact_names_shapes_dtypes_and_atomic_state_load(self):
        before = self.ema.state_dict()
        wrong = copy.deepcopy(before); wrong["shadow"]["adapter.weight"] = torch.zeros(5, 4)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.ema.load_state_dict(wrong)
        wrong = copy.deepcopy(before); wrong["shadow"]["adapter.weight"] = wrong["shadow"]["adapter.weight"].double()
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            self.ema.load_state_dict(wrong)
        wrong = copy.deepcopy(before); wrong["shadow"]["extra"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "names"):
            self.ema.load_state_dict(wrong)
        self.assertTrue(torch.equal(before["shadow"]["adapter.weight"], self.ema.state_dict()["shadow"]["adapter.weight"]))
        self.model.extra = nn.Parameter(torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "names changed"):
            self.ema.update(self.model)
        self.assertEqual(self.ema.num_updates, 0)

    def test_nested_swap_and_update_during_swap_fail_with_student_restored(self):
        before = self.model.adapter.weight.detach().clone()
        with self.ema.teacher_parameters(self.model):
            with self.assertRaisesRegex(ValueError, "Nested"):
                with self.ema.teacher_parameters(self.model):
                    pass
            with self.assertRaisesRegex(ValueError, "during"):
                self.ema.update(self.model)
        self.assertTrue(torch.equal(before, self.model.adapter.weight))

    def test_nonfinite_update_fails_before_any_shadow_change(self):
        before = self.ema.state_dict()["shadow"]["adapter.weight"]
        with torch.no_grad():
            self.model.adapter.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.ema.update(self.model)
        self.assertTrue(torch.equal(before, self.ema.state_dict()["shadow"]["adapter.weight"]))
        self.assertEqual(self.ema.num_updates, 0)


class CorruptionTests(unittest.TestCase):
    def test_default_span_and_rounding_exclude_first_latent(self):
        plan = sample_span_mask(16, generator=rng())
        repeated = sample_span_mask(16, generator=rng())
        self.assertTrue(torch.equal(plan.mask, repeated.mask))
        self.assertEqual(tuple(plan.mask.shape), (16, 9))
        self.assertFalse(bool(plan.mask[:, 0].any()))
        self.assertTrue(bool((plan.mask.sum(1) == 2).all()))
        self.assertEqual(plan.realized_ratio_all_latents, 2/9)
        self.assertEqual(plan.realized_ratio_eligible_latents, .25)
        for example, spans in enumerate(plan.spans):
            self.assertEqual(len(spans), 1)
            start, stop = spans[0]
            self.assertEqual(stop-start, 2)
            self.assertEqual(plan.mask[example].nonzero().flatten().tolist(), list(range(start, stop)))
        half = sample_span_mask(1, rho=1/9, generator=rng())
        self.assertEqual(half.masked_positions_per_example, 2)  # documented half-up span rounding

    def test_maximum_feasible_mask_has_no_greedy_span_failure(self):
        plan = sample_span_mask(32, rho=1., generator=rng())
        self.assertFalse(bool(plan.mask[:, 0].any()))
        self.assertTrue(bool(plan.mask[:, 1:].all()))
        self.assertEqual(plan.realized_ratio_all_latents, 8/9)
        self.assertTrue(all(len(spans) == 4 for spans in plan.spans))

    def test_zero_ratio_keeps_values_and_rng_states(self):
        mask_rng, noise_rng = rng(3), rng(4)
        mask_state, noise_state = mask_rng.get_state(), noise_rng.get_state()
        plan = sample_span_mask(2, rho=0., generator=mask_rng)
        original = torch.randn(2, 4, 9, 2, 3)
        result = corrupt_latent_condition(original, plan.mask, generator=noise_rng)
        self.assertTrue(torch.equal(mask_state, mask_rng.get_state()))
        self.assertTrue(torch.equal(noise_state, noise_rng.get_state()))
        self.assertTrue(torch.equal(original, result))
        self.assertNotEqual(original.data_ptr(), result.data_ptr())

    def test_corruption_changes_only_masked_condition_and_never_original(self):
        original = torch.randn(2, 4, 9, 2, 3)
        before = original.clone()
        mask = sample_span_mask(2, generator=rng()).mask
        changed = corrupt_latent_condition(original, mask, generator=rng(31))
        expanded = mask[:, None, :, None, None].expand_as(original)
        self.assertTrue(torch.equal(original, before))
        self.assertTrue(torch.equal(changed[~expanded], original[~expanded]))
        self.assertFalse(torch.equal(changed[expanded], original[expanded]))
        bad = mask.clone(); bad[:, 0] = True
        with self.assertRaisesRegex(ValueError, "first latent"):
            corrupt_latent_condition(original, bad, generator=rng())

    def test_token_mask_follows_T_H_W_flattening(self):
        mask = torch.zeros(1, 9, dtype=torch.bool); mask[:, 2:4] = True
        tokens = temporal_token_mask(mask, (9, 2, 3))
        self.assertEqual(tokens.nonzero()[:, 1].tolist(), list(range(12, 24)))
        with self.assertRaisesRegex(ValueError, "B,T"):
            temporal_token_mask(mask[:, :-1], (9, 2, 3))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.grid = (9, 2, 3)
        self.mask = torch.zeros(2, 9, dtype=torch.bool); self.mask[:, 4:6] = True

    def test_geometry_and_manual_masked_cosine(self):
        layer = MaskedTemporalAlignment(4, 4, hidden_dim=8)
        layer.projector = nn.Identity()
        student, teacher = torch.randn(2, 54, 4), torch.randn(2, 54, 4)
        loss, info = layer(student, teacher, self.mask, self.grid, torch.tensor([.3, .8]))
        tokens = temporal_token_mask(self.mask, self.grid); tokens[0] = False
        target = F.layer_norm(teacher[tokens], (4,))
        expected = (1 - (F.normalize(student[tokens],dim=-1) * F.normalize(target,dim=-1)).sum(-1)).mean()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(info["selected_tokens"], 12)
        self.assertEqual(list(layer.teacher_norm.parameters()), [])

    def test_teacher_detached_projector_gradients_and_strict_timestep_window(self):
        layer = MaskedTemporalAlignment(4, 6, hidden_dim=8)
        student = torch.randn(2, 54, 4, requires_grad=True)
        teacher = torch.randn(2, 54, 6, requires_grad=True)
        loss, info = layer(student, teacher, self.mask, self.grid, torch.tensor([.3, .8]))
        loss.backward()
        self.assertEqual(info["selected_tokens"], 12)
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in layer.projector.parameters()))
        active = temporal_token_mask(self.mask, self.grid); active[0] = False
        self.assertTrue(bool((student.grad[~active] == 0).all()))
        self.assertGreater(float(student.grad[active].abs().sum()), 0)

    def test_unused_nan_tokens_do_not_contaminate_parameter_gradients(self):
        layer = MaskedTemporalAlignment(4, 4, hidden_dim=8)
        student, teacher = torch.randn(2, 54, 4), torch.randn(2, 54, 4)
        tokens = temporal_token_mask(self.mask, self.grid)
        student[~tokens] = float("nan"); teacher[~tokens] = float("nan")
        student.requires_grad_(True); teacher.requires_grad_(True)
        loss, _ = layer(student, teacher, self.mask, self.grid, .8)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in layer.projector.parameters()))
        self.assertIsNone(teacher.grad)
        with torch.no_grad():
            student[tokens] = float("nan")
        with self.assertRaisesRegex(ValueError, "Selected"):
            layer(student, teacher, self.mask, self.grid, .8)

    def test_rho_zero_auxiliary_disabled_matches_flow_only_backward_exactly(self):
        first = nn.Linear(4, 4); second = copy.deepcopy(first)
        value, target = torch.randn(2, 54, 4), torch.randn(2, 54, 4)
        plain = F.mse_loss(first(value), target)
        plain_grads = torch.autograd.grad(plain, tuple(first.parameters()))
        student = second(value)
        alignment = MaskedTemporalAlignment(4, 4, hidden_dim=8)
        zero_mask = sample_span_mask(2, rho=0., generator=rng()).mask
        aux, info = alignment(student, target, zero_mask, self.grid, .8)
        combined = F.mse_loss(student, target) + aux
        combined_grads = torch.autograd.grad(combined, tuple(second.parameters()))
        self.assertTrue(torch.equal(plain, combined))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(plain_grads,combined_grads)))
        self.assertEqual(info["selected_tokens"], 0)
        self.assertTrue(all(p.grad is None for p in alignment.parameters()))

    def test_invalid_grid_or_native_timestep_units_fail(self):
        layer = MaskedTemporalAlignment(4, 4)
        features = torch.randn(2, 54, 4)
        with self.assertRaisesRegex(ValueError, "token count"):
            layer(features, features, self.mask, (9, 1, 3), .8)
        with self.assertRaisesRegex(ValueError, "normalized"):
            layer(features, features, self.mask, self.grid, 800.)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
