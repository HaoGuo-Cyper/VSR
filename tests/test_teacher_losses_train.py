"""Tiny CPU gradient/contract fixtures; no GPU, task weights or optimizer step."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from vsr.teacher import TrainableEMA
from vsr.data import sample_span_mask, corrupt_latent_condition
from vsr.losses import temporal_token_mask, MaskedTemporalAlignment


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

    def test_all_sources_and_shadows_share_one_device_verdict(self):
        # Count actual Tensor->host boolean reads, rather than timing a CPU
        # fixture and extrapolating GPU performance. All 128 values are checked.
        model = nn.ParameterList([nn.Parameter(torch.randn(index % 4 + 1)) for index in range(64)])
        ema = TrainableEMA(model)
        original = torch.Tensor.__bool__
        reads = []
        def counted(value):
            reads.append(value.device)
            return original(value)
        with patch.object(torch.Tensor, "__bool__", counted):
            ema._validate(model)
        self.assertEqual(reads, [torch.device("cpu")])

    def test_late_nonfinite_source_or_shadow_rejects_without_partial_mutation(self):
        for location in ("source", "shadow"):
            with self.subTest(location=location):
                model = TinyRestorer()
                model.late_adapter = nn.Linear(4, 4)
                ema = TrainableEMA(model)
                last = ema.names[-1]
                self.assertEqual(last, "late_adapter.bias")
                with torch.no_grad():
                    # If an earlier shadow were updated before full validation,
                    # this deliberately changed source would reveal it.
                    model.adapter.weight.add_(7.)
                    target = dict(model.named_parameters())[last] if location == "source" else ema._shadow[last]
                    target[0] = float("nan") if location == "source" else float("inf")
                before_model = {name: p.detach().clone() for name, p in model.named_parameters()}
                versions = {name: p._version for name, p in model.named_parameters()}
                before_shadow = ema.state_dict()["shadow"]
                message = "Nonfinite student trainable: " if location == "source" else "Invalid EMA shadow: "
                with self.assertRaisesRegex(ValueError, message + last):
                    ema.update(model)
                with self.assertRaisesRegex(ValueError, message + last):
                    with ema.teacher_parameters(model):
                        self.fail("A corrupt late value must never enter the teacher context")
                # Byte comparisons also preserve a deliberately injected NaN.
                self.assertTrue(all(torch.equal(p.detach().view(torch.uint8), before_model[name].view(torch.uint8))
                                    for name, p in model.named_parameters()))
                self.assertTrue(all(torch.equal(value.view(torch.uint8), before_shadow[name].view(torch.uint8))
                                    for name, value in ema.state_dict()["shadow"].items()))
                self.assertEqual(versions, {name: p._version for name, p in model.named_parameters()})
                self.assertEqual(ema.num_updates, 0)
                self.assertFalse(ema._swapping)

    def test_late_nonfinite_initialization_and_state_load_are_atomic(self):
        model = TinyRestorer()
        model.late_adapter = nn.Linear(4, 4)
        ema = TrainableEMA(model)
        last = ema.names[-1]
        before = ema.state_dict()
        bad = ema.state_dict()
        bad["num_updates"] = 23
        bad["shadow"][ema.names[0]].add_(5.)
        bad["shadow"][last][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "value mismatch: " + last):
            ema.load_state_dict(bad)
        self.assertEqual(ema.num_updates, 0)
        self.assertTrue(all(torch.equal(value, before["shadow"][name])
                            for name, value in ema.state_dict()["shadow"].items()))
        with torch.no_grad():
            dict(model.named_parameters())[last][0] = float("inf")
        versions = {name: p._version for name, p in model.named_parameters()}
        with self.assertRaisesRegex(ValueError, "initialization.*" + last):
            TrainableEMA(model)
        self.assertEqual(versions, {name: p._version for name, p in model.named_parameters()})


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




"""Real ConditionalWan/tiny random Wan CPU fixtures; no experimental training.

Small synthetic AdamW trajectories are allowed integration tests. No official
weights, task checkpoint payload, video, GPU or production optimizer is loaded.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from diffusers import WanTransformer3DModel

from vsr.model import ConditionalWan
from vsr.train import (NoRefContinuationStep, MatchedStepRNG,
    capture_block_output, load_matched_base_continuation)


def tiny_model(seed=123, checkpointing=True):
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        native = WanTransformer3DModel(patch_size=(1,2,2), num_attention_heads=2,
            attention_head_dim=12, in_channels=16, out_channels=16, text_dim=16,
            freq_dim=16, ffn_dim=48, num_layers=30, rope_max_seq_len=16)
        model = ConditionalWan(native, rank=2, alpha=2, injection_layers=(0,5,9),
            alignment_layer=9, stem_channels=8, projection=False,
            gradient_checkpointing=checkpointing)
        # A trained Base has an effective LR branch. Give this synthetic fixture
        # nonzero residuals so the corruption checks cannot pass vacuously.
        with torch.no_grad():
            for layer in model.lrv_residuals.values():
                layer.weight.normal_(std=.02)
    return model.train()


def inputs():
    generator = torch.Generator().manual_seed(97)
    return (torch.randn(1,16,9,4,4,generator=generator),
            torch.randn(1,16,9,4,4,generator=generator),
            torch.randn(1,4,16,generator=generator))


def schedule(sigma=.8):
    return SimpleNamespace(timesteps=torch.linspace(999.,0.,1000),
                           sigmas=torch.full((1001,),sigma))


def named_values(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


class WanIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.clean, self.low, self.text = inputs()

    def test_equal_EMA_no_corruption_matches_at_the_same_layer(self):
        model = tiny_model()
        engine = NoRefContinuationStep(model, arm="A2", training_seed=5, hidden_dim=16)
        time = torch.tensor([500.])
        teacher9 = engine.teacher_features(self.clean,time,self.text,self.low,layer=9)
        _, student9 = capture_block_output(model,9,self.clean,time,self.text,self.low)
        torch.testing.assert_close(teacher9,student9,rtol=0,atol=0)
        self.assertFalse(teacher9.requires_grad)
        self.assertTrue(student9.requires_grad)
        # Different blocks are intentionally not asserted equal.
        model.clear_features()
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_teacher_first_same_noisy_HR_sigma_and_only_LR_asymmetry(self):
        model = tiny_model()
        engine = NoRefContinuationStep(model,arm="A2",training_seed=5,hidden_dim=16)
        frozen = {name:value for name,value in named_values(model).items()
                  if not dict(model.named_parameters())[name].requires_grad}
        flags = [(module,module.training) for module in model.modules()]
        before_low = self.low.clone()
        events=[];forward=model.forward
        def spy(noisy,timestep,text,low,*args,**kwargs):
            events.append({"grad":torch.is_grad_enabled(),"training":model.training,
                           "noisy":noisy.detach().clone(),"timestep":timestep.detach().clone(),
                           "low":low.detach().clone()})
            return forward(noisy,timestep,text,low,*args,**kwargs)
        with patch.object(model,"forward",side_effect=spy):
            result=engine.step(self.clean,self.low,self.text,schedule())
        self.assertEqual(len(events),2)
        self.assertEqual([event['grad'] for event in events],[False,True])
        self.assertEqual([event['training'] for event in events],[False,True])
        self.assertTrue(torch.equal(events[0]['noisy'],events[1]['noisy']))
        self.assertTrue(torch.equal(events[0]['timestep'],events[1]['timestep']))
        self.assertTrue(torch.equal(events[0]['low'],before_low))
        self.assertTrue(torch.equal(events[1]['low'][:,:,0],before_low[:,:,0]))
        self.assertFalse(torch.equal(events[1]['low'],before_low))
        self.assertTrue(torch.equal(self.low,before_low))
        self.assertTrue(result['teacher_forward'])
        self.assertEqual(result['selected_tokens'],8)  # 2 masked times × 2 × 2
        self.assertEqual(result['EMA_updates'],1)
        self.assertEqual((result['student_layer'],result['teacher_layer']),(9,21))
        self.assertTrue(all(torch.equal(parameter,frozen[name]) for name,parameter in model.named_parameters() if not parameter.requires_grad))
        self.assertTrue(all(module.training==training for module,training in flags))
        self.assertTrue(all(parameter.grad is None for parameter in engine.parameters))
        self.assertIsNone(model._condition_tokens);self.assertIsNone(model._captured)
        self.assertTrue(all('projector' not in name for name in engine.ema.names))
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)

    def test_rho_zero_matches_core_Base_full_tiny_AdamW_trajectory(self):
        reference=tiny_model()
        arms=[NoRefContinuationStep(tiny_model(),arm=arm,training_seed=19,rho=0.,hidden_dim=16,
                                    learning_rate=.002) for arm in ('flow_only','C1','A2')]
        params=[parameter for parameter in reference.parameters() if parameter.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.002,weight_decay=.01,foreach=False)
        streams=MatchedStepRNG(19,'cpu')
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            noisy,target,time,_,index=streams.flow_inputs(self.clean,schedule())
            flow,_=reference(noisy,time,self.text,self.low)
            loss=torch.nn.functional.mse_loss(flow.float(),target)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step();reference.clear_features();optimizer.zero_grad(set_to_none=True)
            for arm in arms:
                result=arm.step(self.clean,self.low,self.text,schedule())
                self.assertEqual(result['fm'],float(loss.detach()))
                self.assertEqual(result['timestep_index'],index)
                self.assertFalse(result['teacher_forward'])
                self.assertEqual(result['sd'],0.)
                for name,parameter in reference.named_parameters():
                    other=dict(arm.model.named_parameters())[name]
                    torch.testing.assert_close(parameter,other,rtol=0,atol=0,msg=f'{arm.arm}/{step}/{name}')
                    if parameter.requires_grad:
                        self.assertEqual(set(optimizer.state[parameter]),set(arm.optimizer.state[other]))
                        for key in optimizer.state[parameter]:
                            torch.testing.assert_close(optimizer.state[parameter][key],arm.optimizer.state[other][key],rtol=0,atol=0)
            self.assertEqual(arms[0].data_index(2),(step+1)%2)
        self.assertTrue(all(parameter.grad is None for arm in arms for parameter in arm.parameters))

    def test_gamma_zero_corrupted_A2_matches_C1_not_clean_flow(self):
        control=NoRefContinuationStep(tiny_model(),arm='C1',training_seed=11,learning_rate=.002,hidden_dim=16)
        disabled=NoRefContinuationStep(tiny_model(),arm='A2',training_seed=11,gamma=0.,learning_rate=.002,hidden_dim=16)
        clean=NoRefContinuationStep(tiny_model(),arm='flow_only',training_seed=11,learning_rate=.002,hidden_dim=16)
        for _ in range(2):
            c1=control.step(self.clean,self.low,self.text,schedule())
            a2=disabled.step(self.clean,self.low,self.text,schedule())
            clean.step(self.clean,self.low,self.text,schedule())
            self.assertEqual(c1['fm'],a2['fm'])
            self.assertFalse(a2['teacher_forward'])
            for name,parameter in control.model.named_parameters():
                torch.testing.assert_close(parameter,dict(disabled.model.named_parameters())[name],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(parameter,dict(clean.model.named_parameters())[name])
                            for name,parameter in control.model.named_parameters() if parameter.requires_grad))

    def test_teacher_exception_cleans_hooks_restores_student_and_never_steps_optimizer(self):
        model=tiny_model().eval()
        engine=NoRefContinuationStep(model,arm='A2',training_seed=5,hidden_dim=16)
        before=named_values(model)
        def fail(_module,_args,_output):
            if not torch.is_grad_enabled():raise RuntimeError('teacher fixture failure')
        handle=model.transformer.blocks[21].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError,'teacher fixture failure'):
                engine.step(self.clean,self.low,self.text,schedule())
        finally:handle.remove()
        self.assertEqual(engine.steps,0);self.assertEqual(len(engine.optimizer.state),0)
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(parameter,before[name]) for name,parameter in model.named_parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in engine.parameters))
        self.assertIsNone(model._condition_tokens)
        self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks),0)

    def test_late_nonfinite_gradients_reject_before_clipping_or_any_update(self):
        for location, invalid in (("restoration", float("nan")), ("student_projector", float("inf"))):
            with self.subTest(location=location):
                model = tiny_model().eval()
                engine = NoRefContinuationStep(model, arm="A2", training_seed=5, hidden_dim=16)
                name, target = [(name, p) for name, p in engine._named_optimizer_parameters
                                if name.startswith(location + ".")][-1]
                before = named_values(model)
                projector_before = named_values(engine.alignment.projector)
                ema_before = engine.ema.state_dict()
                frozen_versions = {name: p._version for name, p in model.named_parameters() if not p.requires_grad}
                handle = target.register_hook(lambda gradient: torch.full_like(gradient, invalid))
                try:
                    with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
                        with patch.object(engine.optimizer, "step", wraps=engine.optimizer.step) as update:
                            with self.assertRaisesRegex(ValueError, "Nonfinite continuation gradients: " + name):
                                engine.step(self.clean, self.low, self.text, schedule())
                        update.assert_not_called()
                    clip.assert_not_called()
                finally:
                    handle.remove()
                self.assertEqual(engine.steps, 0)
                self.assertEqual(len(engine.optimizer.state), 0)
                self.assertEqual(engine.ema.num_updates, 0)
                self.assertFalse(model.training)
                self.assertTrue(all(torch.equal(p, before[name]) for name, p in model.named_parameters()))
                self.assertTrue(all(torch.equal(p, projector_before[name]) for name, p in engine.alignment.projector.named_parameters()))
                self.assertTrue(all(torch.equal(value, ema_before["shadow"][name])
                                    for name, value in engine.ema.state_dict()["shadow"].items()))
                self.assertEqual(frozen_versions, {name: p._version for name, p in model.named_parameters() if not p.requires_grad})
                self.assertTrue(all(p.grad is None for p in engine.parameters))
                self.assertIsNone(model._condition_tokens)
                self.assertEqual(sum(len(block._forward_hooks) for block in model.transformer.blocks), 0)

    def test_disabled_auxiliary_rejects_late_shadow_before_optimizer_mutation(self):
        model = tiny_model().eval()
        engine = NoRefContinuationStep(model, arm="A2", training_seed=5, gamma=0., hidden_dim=16)
        last = engine.ema.names[-1]
        engine.ema._shadow[last].view(-1)[-1] = float("inf")
        before = named_values(model)
        versions = {name: p._version for name, p in model.named_parameters()}
        with patch.object(model, "forward", wraps=model.forward) as forward:
            with self.assertRaisesRegex(ValueError, "Invalid EMA shadow: " + last):
                engine.step(self.clean, self.low, self.text, schedule())
        forward.assert_not_called()
        self.assertEqual(engine.steps, 0)
        self.assertEqual(len(engine.optimizer.state), 0)
        self.assertEqual(engine.ema.num_updates, 0)
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(p, before[name]) for name, p in model.named_parameters()))
        self.assertEqual(versions, {name: p._version for name, p in model.named_parameters()})

    def test_sigma_window_skips_teacher_without_altering_flags(self):
        model=tiny_model()
        engine=NoRefContinuationStep(model,arm='A2',training_seed=5,hidden_dim=16)
        before=(torch.is_grad_enabled(),torch.are_deterministic_algorithms_enabled(),
                torch.backends.cuda.matmul.allow_tf32,torch.cuda.is_initialized())
        result=engine.step(self.clean,self.low,self.text,schedule(sigma=.3))
        self.assertFalse(result['teacher_forward'])
        self.assertEqual(result['selected_tokens'],0)
        self.assertEqual(before,(torch.is_grad_enabled(),torch.are_deterministic_algorithms_enabled(),
                                torch.backends.cuda.matmul.allow_tf32,torch.cuda.is_initialized()))

    def test_future_loader_requires_matched_Base_and_fresh_optimizer(self):
        model=tiny_model()
        config={'seed':20261102,'training':{'seed':20261103,'learning_rate':1e-4,'weight_decay':.01,'max_grad_norm':1.},
                'data':{'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4}}
        audit={'checkpoint_sha256':'a'*64,'step':8192,'variant':'Base','config':config,
               'optimizer_state_restored':False,'scheduler_state_restored':False,'rng_state_restored':False}
        with patch('vsr.checkpoints.load_training_checkpoint',return_value=(model,audit)) as loader:
            arm=load_matched_base_continuation('/new/runs/base/Base/checkpoint_008192.pt',expected_checkpoint_sha256='a'*64,
                expected_training_seed=20261103,allowed_root='/new/runs',arm='A2',hidden_dim=16)
            loader.assert_called_once_with('/new/runs/base/Base/checkpoint_008192.pt',device='cpu',allowed_root=Path('/new/runs'),variant='Base')
        self.assertEqual(len(arm.optimizer.state),0)
        self.assertEqual(arm.ema.num_updates,0)
        self.assertEqual(arm.base_load_audit,audit)
        with patch('vsr.checkpoints.load_training_checkpoint',return_value=(tiny_model(),audit)):
            with self.assertRaisesRegex(ValueError,'seed/protocol'):
                load_matched_base_continuation('unused',expected_checkpoint_sha256='a'*64,expected_training_seed=20261104,
                                              allowed_root='/new/runs',arm='flow_only')



from vsr import train as worker
import tempfile
import json
import sys
import weakref
from contextlib import contextmanager, nullcontext

def timing_rows(*,a2_seconds=1.5,sigma=.8):
    result = []
    for role in worker.ROLES:
        for ordinal in range(worker.WARMUP+worker.TIMED):
            active = role == 'A2' and sigma > worker.SIGMA_BOUNDARY
            result.append({'role':role,'ordinal':ordinal,'phase':'warmup' if ordinal<2 else 'timed',
                'seconds':a2_seconds if role == 'A2' else 1.,'fm':2.,'sd':.2 if active else 0.,
                'total':2.16 if active else 2.,'sigma':sigma,'teacher_forward':active,
                'mask_spans':[] if role == 'flow_only' else [[[2,4]]],
                'matched_corruption_RNG':{'mask':'same','corruption':'same'},
                'matched_FM':{'data_index':ordinal,'record_id':str(ordinal),'noise':'same','sigma':sigma}})
    return result


class UnifiedResourceTests(unittest.TestCase):
    def test_budget_default_and_whole_timed_gate_boundary(self):
        self.assertEqual(worker.G3_BUDGET['benchmark_updates'],3*(2+20))
        self.assertEqual(worker.G3_BUDGET['total_diagnostic_updates'],66+4)
        self.assertEqual((worker.METHOD['frames'],worker.METHOD['latent_frames'],worker.METHOD['rho'],worker.METHOD['span']),(33,9,.25,2))
        result = worker.summarize_roles(timing_rows())
        self.assertEqual(result['status'],'passed_G3_resource_cost')
        self.assertEqual(result['primary_ratio_A2_over_flow_all20'],1.5)
        self.assertEqual(result['secondary']['teacher_active_count_timed'],20)
        self.assertEqual(result['role_timing']['A2']['n'],20)
        self.assertEqual(worker.summarize_roles(timing_rows(a2_seconds=1.500001))['status'],'failed_G3_overhead_stop')


    def test_all_outliers_retained_and_sample_SD(self):
        rows = timing_rows(a2_seconds=1.)
        rows[-1]['seconds'] = 21.
        result = worker.summarize_roles(rows)
        self.assertEqual(result['status'],'failed_G3_overhead_stop')
        self.assertEqual(result['primary_ratio_A2_over_flow_all20'],2.)
        self.assertAlmostEqual(result['role_timing']['A2']['sample_SD_seconds'],20/(20**.5))
        self.assertEqual(result['outliers_removed'],0)
        self.assertEqual(len(rows),66)


    def test_no_active_sigma_is_null_and_float32_boundary_is_excluded(self):
        result = worker.summarize_roles(timing_rows(sigma=worker.SIGMA_BOUNDARY))
        self.assertEqual(result['secondary']['teacher_active_count_timed'],0)
        self.assertIsNone(result['secondary']['A2_over_flow_same_active_indices'])
        self.assertIsNone(result['secondary']['active_A2_timing']['mean_seconds'])
        self.assertIsNone(result['secondary']['active_A2_timing']['sample_SD_seconds'])


    def test_mixed_default_window_keeps_inactive_steps_in_primary_cost(self):
        rows = timing_rows(a2_seconds=1.)
        for row in rows:
            sigma = .8 if row['ordinal'] in (2,3) else .1
            row['sigma'] = row['matched_FM']['sigma'] = sigma
            active = row['role']=='A2' and sigma>.3
            row.update(teacher_forward=active,sd=.2 if active else 0.,total=2.16 if active else 2.)
            if active:row['seconds'] = 4.
        result = worker.summarize_roles(rows)
        self.assertEqual(result['status'],'passed_G3_resource_cost')
        self.assertEqual(result['primary_ratio_A2_over_flow_all20'],1.3)
        self.assertEqual(result['secondary']['A2_over_flow_same_active_indices'],4.)
        self.assertEqual(result['secondary']['teacher_active_count_timed'],2)


    def test_rejects_missing_or_mismatched_data_stream_window_or_finite(self):
        changes = [lambda rows:rows.pop(),
            lambda rows:rows[-1]['matched_FM'].update(noise='different'),
            lambda rows:rows[-1].update(phase='warmup'),
            lambda rows:rows[-1].update(teacher_forward=False),
            lambda rows:rows[-1].update(seconds=float('nan')),
            lambda rows:rows[-1].update(sd=float('inf')),
            lambda rows:rows[-1]['matched_corruption_RNG'].update(mask='different'),
            lambda rows:rows[-1].update(mask_spans=[[[6,8]]])]
        for change in changes:
            with self.subTest(change=change):
                rows = timing_rows();change(rows)
                with self.assertRaises(ValueError):worker.summarize_roles(rows)


    def test_successful_optimizer_call_count_survives_later_failure(self):
        optimizer = SimpleNamespace(step=lambda:None)
        counter = worker.UpdateCounter();counter.attach(optimizer,'A2')
        with self.assertRaisesRegex(ValueError,'post-update EMA'):
            optimizer.step()
            raise ValueError('post-update EMA fixture failure')
        report = counter.snapshot()
        self.assertEqual(report['diagnostic_optimizer_updates_applied'],1)
        self.assertEqual(report['diagnostic_optimizer_updates_attempted'],1)
        self.assertEqual(report['completed_step_returns'],0)
        self.assertEqual(report['scientific_training_updates'],0)


    def test_optimizer_throw_records_unknown_partial_effect_and_preserves_original_error(self):
        def fail():raise RuntimeError('AdamW fixture OOM')
        optimizer = SimpleNamespace(step=fail)
        counter = worker.UpdateCounter();counter.attach(optimizer,'A2')
        with self.assertRaisesRegex(RuntimeError,'AdamW fixture OOM'):optimizer.step()
        self.assertEqual(counter.snapshot()['optimizer_calls_not_returned_partial_effect_unknown'],1)
        self.assertEqual(counter.snapshot()['diagnostic_optimizer_updates_applied'],0)


    def test_failure_traceback_reference_does_not_mask_original_OOM(self):
        class Engine:
            def fail(self):raise RuntimeError('original fixture OOM')
        torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))
        engine = Engine();reference = weakref.ref(engine);counter = worker.UpdateCounter()
        with self.assertRaisesRegex(RuntimeError,'original fixture OOM'):
            try:engine.fail()
            finally:
                successful = sys.exc_info()[0] is None
                engine = None
                worker._check_release(torch,reference,successful,counter,'failure_fixture')
        self.assertTrue(counter.cleanup_events[0]['release_assert_skipped_during_original_exception'])
        self.assertTrue(counter.cleanup_events[0]['model_reference_alive_via_possible_traceback'])
        held = Engine()
        with self.assertRaisesRegex(ValueError,'remains resident'):
            worker._check_release(torch,weakref.ref(held),True,counter,'success_fixture')



class UnifiedRunnerContractTests(unittest.TestCase):
    def test_memory_uses_explicit_cap_and_actual_selected_device(self):
        GiB=1024**3
        def fake(allocated,reserved,physical=80*GiB):
            return SimpleNamespace(cuda=SimpleNamespace(current_device=lambda:3,
                max_memory_allocated=lambda device:allocated,max_memory_reserved=lambda device:reserved,
                get_device_properties=lambda device:SimpleNamespace(total_memory=physical)))
        self.assertEqual(worker._memory(fake(4*GiB,5*GiB),80*GiB)['physical_device_bytes'],80*GiB)
        for hardware,cap in [(fake(17*GiB,17*GiB),16*GiB),(fake(4*GiB,17*GiB),16*GiB),
                             (fake(4*GiB,5*GiB,4*GiB),80*GiB)]:
            with self.assertRaisesRegex(ValueError,'memory ceiling'):worker._memory(hardware,cap)

    def test_prepare_is_metadata_only_and_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            args=SimpleNamespace(run_dir=root/'runs/new',run_root=root/'runs')
            protocol={'mode':'resource','budget':dict(worker.G3_BUDGET)}
            with patch.object(worker,'collect_protocol',return_value=protocol):
                with patch('torch.load',side_effect=AssertionError('prepare deserialized a tensor')):
                    result=worker.prepare(args)
                    self.assertEqual(worker.read(args.run_dir/'protocol.json'),protocol)
                    self.assertFalse(worker.read(args.run_dir/'status.json')['GPU_loaded'])
                    self.assertEqual(result['budget']['total_diagnostic_updates'],70)
                    with self.assertRaisesRegex(ValueError,'Fresh-only'):worker.prepare(args)
            self.assertFalse(torch.cuda.is_initialized())

    def test_stopped_or_completed_run_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'runs/stopped';run.mkdir(parents=True)
            worker.save(run/'protocol.json',{})
            sha=worker.file_sha(run/'protocol.json')
            worker.save(run/'status.json',{'status':'stopped_by_user','protocol_sha256':sha})
            args=SimpleNamespace(run_dir=run,run_root=root/'runs',expected_protocol_sha=sha)
            with patch('vsr.checkpoints.gpu_lease') as lease:
                with self.assertRaisesRegex(ValueError,'never-started'):worker.execute(args)
                lease.assert_not_called()
            self.assertFalse((run/'execute.lock').exists())

    def test_changed_source_or_prerequisite_rejects_before_GPU_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'runs/new';run.mkdir(parents=True)
            protocol={'mode':'resource','prepared_arguments':{}}
            worker.save(run/'protocol.json',protocol);sha=worker.file_sha(run/'protocol.json')
            worker.save(run/'status.json',{'status':'planned','protocol_sha256':sha})
            args=SimpleNamespace(run_dir=run,run_root=root/'runs',expected_protocol_sha=sha)
            with patch.object(worker,'collect_protocol',return_value={'changed':True}):
                with patch('vsr.checkpoints.gpu_lease') as lease:
                    with self.assertRaisesRegex(ValueError,'changed'):worker.execute(args)
                    lease.assert_not_called()
            self.assertFalse((run/'execute.lock').exists())

    def test_failure_receipt_written_inside_lease_and_original_error_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'runs/new';run.mkdir(parents=True)
            protocol={'mode':'resource','prepared_arguments':{},'gpu_lock':str(root/'gpu.lock'),
                      'device':'cuda:0','admission':{'expected_gpu_uuid':'GPU-fixture'},
                      'config':{'reproducibility':{'mode':'strict'}},'strict_environment':worker.STRICT_ENVIRONMENT}
            worker.save(run/'protocol.json',protocol);sha=worker.file_sha(run/'protocol.json')
            worker.save(run/'status.json',{'status':'planned','protocol_sha256':sha})
            args=SimpleNamespace(run_dir=run,run_root=root/'runs',expected_protocol_sha=sha)
            events=[]
            @contextmanager
            def lease(*args):
                events.append('lease_enter')
                try:yield {'fixture':'admitted'}
                finally:
                    self.assertTrue((run/'failure.json').exists())
                    self.assertEqual(worker.read(run/'status.json')['status'],'failed_stop_no_retry')
                    events.append('lease_exit')
            with patch.object(worker,'collect_protocol',return_value=protocol):
                with patch('vsr.checkpoints.gpu_lease',lease):
                    with patch('torch.cuda.device',return_value=nullcontext()):
                        with patch('torch.cuda.is_available',return_value=True):
                            with patch.object(worker,'_load_inputs',side_effect=RuntimeError('fixture input failure')), \
                                 patch.object(worker,'_configure_runtime',return_value={'CPU_fixture':True}), \
                                 patch.dict(worker.os.environ,worker.STRICT_ENVIRONMENT):
                                with self.assertRaisesRegex(RuntimeError,'fixture input failure'):worker.execute(args)
            self.assertEqual(events,['lease_enter','lease_exit'])
            self.assertEqual(worker.read(run/'failure.json')['diagnostic_optimizer_updates_applied'],0)
            with self.assertRaisesRegex(ValueError,'never-started'):worker.execute(args)
            self.assertFalse(torch.cuda.is_initialized())

    def test_main_help_does_not_initialize_CUDA_or_download(self):
        with self.assertRaises(SystemExit) as result:
            worker.main(['--help'])
        self.assertEqual(result.exception.code,0)
        self.assertFalse(torch.cuda.is_initialized())


class CompletedResourceEvidenceTests(unittest.TestCase):
    def test_native_training_scheduler_explicitly_sets_original_grid(self):
        from unittest.mock import MagicMock
        scheduler=MagicMock()
        scheduler.timesteps=list(range(1000));scheduler.sigmas=list(range(1001))
        with patch('diffusers.UniPCMultistepScheduler.from_pretrained',return_value=scheduler) as loader:
            actual=worker._training_scheduler('/private/official','cuda:2')
        self.assertIs(actual,scheduler)
        loader.assert_called_once_with('/private/official',subfolder='scheduler',local_files_only=True)
        scheduler.set_timesteps.assert_called_once_with(1000,device='cuda:2')
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self,root):
        g2={'status':'passed_G2','protocol_sha256':'a'*64}
        sources={'fixture':'CPU only'}
        checkpoint={'path':'/private/base/Base/checkpoint_008192.pt','sha256':'b'*64}
        protocol={'schema':'vsr_training_protocol_v2','mode':'resource','sources':sources,
                  'G2_evidence':g2,'prepared_arguments':{'mode':'resource'},'matched_Base':{'checkpoint':checkpoint}}
        rows=timing_rows()
        semantics={'status':'passed_exact_GPU_semantics','diagnostic_optimizer_updates':4,
           'checks':{'same_layer':{'optimizer_updates':0,'exact_equal':True,'teacher_detached':True,'student_restored':True},
                     'semantic_direct_core':{'updates':2},'semantic_A2_rho0':{'updates':2,'exact_trajectory_equal':True}}}
        roles={role:{'returned_updates':22,'optimizer_updates_applied':22,'fresh_Base_SHA256':checkpoint['sha256'],
                     'frozen_identity_versions_unchanged':True,'mode_and_gradient_cleanup_passed':True} for role in worker.ROLES}
        root.mkdir(parents=True)
        worker.save(root/'protocol.json',protocol);worker.save(root/'benchmark_rows.json',rows)
        psha=worker.file_sha(root/'protocol.json')
        summary={'status':'completed','protocol_sha256':psha,'G3_passed':True,
                 'diagnostic_optimizer_updates_applied':70,'scientific_training_updates':0,
                 'diagnostic_optimizer_updates_attempted':70,'completed_step_returns':70,
                 'optimizer_calls_not_returned_partial_effect_unknown':0,'semantics':semantics,
                 'resource_gate':worker.summarize_roles(rows),'roles':roles,
                 'benchmark_rows_sha256':worker.file_sha(root/'benchmark_rows.json')}
        worker.save(root/'summary.json',summary)
        worker.save(root/'status.json',{'status':'completed','protocol_sha256':psha,'summary_sha256':worker.file_sha(root/'summary.json')})
        return protocol,g2,sources

    def reseal_summary(self,root,summary):
        (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        status=worker.read(root/'status.json');status['summary_sha256']=worker.file_sha(root/'summary.json')
        (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')

    def test_complete_resource_evidence_recomputed_and_semantic_failure_cannot_be_resealed_away(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'G3'
            protocol,g2,sources=self.fixture(directory)
            with patch.object(worker,'_source_bindings',return_value=sources):
                with patch.object(worker,'collect_protocol',return_value=protocol):
                    result=worker._passed_g3(directory,g2)
                    self.assertEqual(result['status'],'passed_G3')
                    summary=worker.read(directory/'summary.json')
                    summary['semantics']['checks']['same_layer']['exact_equal']=False
                    self.reseal_summary(directory,summary)
                    with self.assertRaisesRegex(ValueError,'semantic evidence'):worker._passed_g3(directory,g2)
            self.assertFalse(torch.cuda.is_initialized())

    def test_partial_benchmark_or_changed_prerequisite_cannot_unlock_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'G3'
            protocol,g2,sources=self.fixture(directory)
            with patch.object(worker,'_source_bindings',return_value=sources):
                with patch.object(worker,'collect_protocol',return_value={'changed':True}):
                    with self.assertRaisesRegex(ValueError,'inputs no longer match'):worker._passed_g3(directory,g2)
                with patch.object(worker,'collect_protocol',return_value=protocol):
                    rows=worker.read(directory/'benchmark_rows.json');rows.pop()
                    (directory/'benchmark_rows.json').write_text(json.dumps(rows)+'\n')
                    summary=worker.read(directory/'summary.json')
                    summary['benchmark_rows_sha256']=worker.file_sha(directory/'benchmark_rows.json')
                    self.reseal_summary(directory,summary)
                    with self.assertRaisesRegex(ValueError,'Incomplete'):worker._passed_g3(directory,g2)


class OptimizerConfigurationTests(unittest.TestCase):
    def base_config(self):
        return {'seed':20261102,'model':{'local_dir':'/private/official'},'conditioning':{},
                'reproducibility':{'mode':'strict'},
                'data':{'num_frames':33,'temporal_stride':1,'height':128,'width':224,'scale':4},
                'training':{'seed':20261102,'updates':8192,'variants':['Base'],
                            'learning_rate':.001,'weight_decay':.02,'max_grad_norm':2.}}

    def test_recorded_optimizer_and_method_config_use_same_matched_Base_values(self):
        config=self.base_config()
        options={key:config['training'][key] for key in ('learning_rate','weight_decay','max_grad_norm')}
        base={'config':config,'checkpoint':{'path':'/private/runs/Base/checkpoint_008192.pt','sha256':'a'*64}}
        protocol={'matched_Base':base,'method':worker.METHOD,'optimizer':options,
                  'allowed_checkpoint_root':'/private/runs','device':'cpu','model_directory':'/private/official'}
        engine=SimpleNamespace(base_load_audit={'config':config},optimizer=SimpleNamespace(state={},param_groups=[{'foreach':False}]))
        with patch.object(worker,'load_matched_base_continuation',return_value=engine) as loader:
            self.assertIs(worker._new_engine(protocol,'C1'),engine)
        recorded=worker._method_config(config,'C1')
        for key,value in options.items():
            self.assertEqual(loader.call_args.kwargs[key],value)
            self.assertEqual(recorded['training'][key],value)
        self.assertEqual(config['training']['updates'],8192)
        self.assertEqual(recorded['training']['updates'],2048)

    def test_changed_optimizer_config_rejected_before_resource_or_pilot_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=self.base_config();source=copy.deepcopy(config)
            source['training']['learning_rate']=.002
            config_path=root/'config.json';worker.save(config_path,config)
            admission=root/'admission.json';worker.save(admission,{'expected_gpu_uuid':'GPU-fixture'})
            checkpoint={'path':str(root/'base/Base/checkpoint_008192.pt'),'bytes':1,'sha256':'a'*64}
            base={'config':source,'checkpoint':checkpoint}
            evaluation={'roles':[{'training_seed':20261102,'variant':'Base','checkpoint':checkpoint}]}
            args=SimpleNamespace(mode='resource',config=config_path,cache_root=root/'cache',model_directory=None,
                base_checkpoint=Path(checkpoint['path']),expected_base_sha='a'*64,allowed_checkpoint_root=root,
                g2_run=root/'G2',g3_run=None,g4_definitions=None,device='cuda:0',gpu_admission=admission,
                gpu_lock=root/'gpu.lock',memory_cap_gib=80.)
            with patch.object(worker,'_cache_spec',return_value=({},[])):
                with patch.object(worker,'_official_bindings',return_value={}):
                    with patch.object(worker,'_dependency_bindings',return_value={}):
                        with patch.object(worker,'_passed_g2',return_value=(evaluation,{'protocol_sha256':'b'*64})):
                            with patch.object(worker,'_base_spec',return_value=base):
                                with self.assertRaisesRegex(ValueError,'must equal its matched Base source config'):
                                    worker.collect_protocol(args)
            self.assertFalse(torch.cuda.is_initialized())


class StrictRuntimeTests(unittest.TestCase):
    def runtime(self):
        enabled=('deterministic_algorithms','cudnn_deterministic','sdp_math')
        disabled=('deterministic_warn_only','cudnn_benchmark','cudnn_allow_tf32','matmul_allow_tf32',
                  'matmul_allow_fp16_reduced_precision_reduction','matmul_allow_bf16_reduced_precision_reduction',
                  'sdp_flash','sdp_memory_efficient','sdp_cudnn','sdp_math_allow_fp16_bf16_reduction')
        return {'mode':'strict','flags':{**{k:True for k in enabled},**{k:False for k in disabled},
                 'CUBLAS_WORKSPACE_CONFIG':':4096:8','cuda_initialized':False}}

    def test_runtime_setup_requires_fresh_context_and_all_strict_flags(self):
        config={'reproducibility':{'mode':'strict'}}
        with patch('vsr.model.configure_reproducibility',return_value=self.runtime()) as configure:
            self.assertEqual(worker._configure_runtime(config),self.runtime())
            configure.assert_called_once_with(config)
        bad=self.runtime();bad['flags']['matmul_allow_tf32']=True
        with patch('vsr.model.configure_reproducibility',return_value=bad):
            with self.assertRaisesRegex(ValueError,'Strict deterministic'):worker._configure_runtime(config)
        with patch('torch.cuda.is_initialized',return_value=True):
            with patch('vsr.model.configure_reproducibility') as configure:
                with self.assertRaisesRegex(ValueError,'fresh process'):worker._configure_runtime(config)
                configure.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())

    def test_missing_offline_or_strict_env_refuses_before_GPU_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'runs/new';run.mkdir(parents=True)
            protocol={'mode':'resource','prepared_arguments':{},'strict_environment':worker.STRICT_ENVIRONMENT}
            worker.save(run/'protocol.json',protocol);sha=worker.file_sha(run/'protocol.json')
            worker.save(run/'status.json',{'status':'planned','protocol_sha256':sha})
            args=SimpleNamespace(run_dir=run,run_root=root/'runs',expected_protocol_sha=sha)
            with patch.object(worker,'collect_protocol',return_value=protocol):
                with patch.dict(worker.os.environ,{'CUBLAS_WORKSPACE_CONFIG':'wrong'}):
                    with patch('vsr.checkpoints.gpu_lease') as lease:
                        with self.assertRaisesRegex(ValueError,'launch environment missing'):worker.execute(args)
                        lease.assert_not_called()
            self.assertFalse((run/'execute.lock').exists())
            self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
