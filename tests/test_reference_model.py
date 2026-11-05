"""Tiny random Wan CPU integration checks; no pretrained/HRI training claims."""
import copy
import unittest

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel

from vsr.model import ConditionalWan
from vsr.reference_model import ReferenceConditionalWan


def tiny_wan():
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
        num_layers=3, rope_max_seq_len=16,
    )


def options(projection=True, checkpointing=True):
    return dict(rank=2, alpha=2, injection_layers=(0, 1), alignment_layer=1,
                stem_channels=8, projector_hidden=32, teacher_dim=16,
                projection=projection, gradient_checkpointing=checkpointing)


class ReferenceModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        self.noisy = torch.randn(2, 4, 2, 4, 4)
        self.low = torch.randn_like(self.noisy)
        self.text = torch.randn(2, 4, 16)
        self.t = torch.tensor([500., 500.])
        self.ref = torch.randn(2, 4, 1, 4, 4)
        self.ref_args = dict(ref_latent=self.ref, ref_index=torch.tensor([0, 4]),
                             frame_count=5, ref_valid=torch.tensor([True, True]))

    def make(self, native=None, projection=True, checkpointing=True):
        return ReferenceConditionalWan(
            tiny_wan() if native is None else native, reference_layer=1,
            reference_stem_dim=8, reference_seed=20261202,
            **options(projection, checkpointing),
        )

    def test_zero_branch_matches_plain_conditional_wan(self):
        native = tiny_wan()
        torch.manual_seed(55)
        plain = ConditionalWan(copy.deepcopy(native), **options()).eval()
        torch.manual_seed(55)
        reference = self.make(copy.deepcopy(native)).eval()
        with torch.no_grad():
            expected, expected_tokens = plain(self.noisy, self.t, self.text, self.low)
            actual, actual_tokens = reference(self.noisy, self.t, self.text, self.low, **self.ref_args)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)

    def test_reference_initialization_independent_of_projector_rng(self):
        base = self.make(projection=False)
        aligned = self.make(projection=True)
        for key, tensor in base.reference_adapter.state_dict().items():
            torch.testing.assert_close(tensor, aligned.reference_adapter.state_dict()[key], rtol=0, atol=0)
        # Constructing an HRIAdapter may not consume the surrounding CPU RNG.
        torch.manual_seed(70)
        native = tiny_wan()
        torch.manual_seed(71)
        plain = ConditionalWan(copy.deepcopy(native), **options())
        expected = torch.get_rng_state()
        torch.manual_seed(71)
        wrapped = self.make(copy.deepcopy(native))
        torch.testing.assert_close(torch.get_rng_state(), expected, rtol=0, atol=0)

    def test_gradients_reach_reference_lr_and_lora_after_updates(self):
        model = self.make().train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.02)
        target = F.normalize(torch.randn(2, 8, 16), dim=-1)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            flow, tokens = model(self.noisy, self.t, self.text, self.low, **self.ref_args)
            loss = flow.square().mean() + (1 - (F.normalize(tokens, dim=-1) * target).sum(-1)).mean()
            loss.backward()
            optimizer.step()
            model.clear_features()
            model.clear_reference()
        for parameter in (model.reference_adapter.stem[0].weight, model.lrv_stem[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                               if ".lora_" in n and p.grad is not None), 0)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertIsNone(model._reference_context)

    def test_missing_branch_ignores_nan_and_next_no_ref_clears_stale_input(self):
        model = self.make(projection=False).eval()
        with torch.no_grad():
            model.reference_adapter.output_projection.weight.normal_(std=.02)
            active, _ = model(self.noisy, self.t, self.text, self.low, **self.ref_args)
            absent, _ = model(self.noisy, self.t, self.text, self.low)
            explicit, _ = model(self.noisy, self.t, self.text, self.low,
                                ref_latent=torch.full_like(self.ref, float("nan")), ref_valid=torch.tensor([False, False]))
        torch.testing.assert_close(absent, explicit, rtol=0, atol=0)
        self.assertGreater(float((active - absent).abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        flow, _ = model(self.noisy, self.t, self.text, self.low)
        flow.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.reference_adapter.parameters()))

    def test_mixed_missing_and_checkpoint_gradients_match(self):
        direct = self.make(checkpointing=False).train()
        replay = self.make(checkpointing=True).train()
        with torch.no_grad():
            direct.reference_adapter.output_projection.weight.normal_(std=.02)
            for layer in direct.lrv_residuals.values():
                layer.weight.normal_(std=.02)
        replay.load_state_dict(direct.state_dict())
        ref = self.ref.clone()
        ref[1] = float("nan")
        kwargs = dict(ref_latent=ref, ref_index=[0, -1], frame_count=[5, 0], ref_valid=[True, False])
        for model in (direct, replay):
            flow, tokens = model(self.noisy, self.t, self.text, self.low, **kwargs)
            self.assertTrue(bool(torch.isfinite(flow).all()))
            self.assertTrue(bool(torch.isfinite(tokens).all()))
            (flow.square().mean() + tokens.square().mean()).backward()
        for (name, parameter), (other_name, other) in zip(direct.named_parameters(), replay.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(parameter.grad is None, other.grad is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=2e-7, msg=name)


if __name__ == "__main__":
    unittest.main()
