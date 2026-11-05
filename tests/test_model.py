"""CPU component checks using a tiny random Wan, never experimental results."""

import copy
import unittest

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel

from vsr.model import ConditionalWan, gradient_norms


def tiny_wan():
    return WanTransformer3DModel(
        patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48,
        num_layers=3, rope_max_seq_len=16,
    )


class ConditionalWanTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(2)
        self.noisy = torch.randn(1, 4, 2, 4, 4)
        self.lr = torch.randn_like(self.noisy)
        self.time = torch.tensor([500.0])
        self.text = torch.randn(1, 4, 16)

    def make_model(self, native=None, **kwargs):
        return ConditionalWan(
            native if native is not None else tiny_wan(),
            rank=2, alpha=2, injection_layers=(0, 1), alignment_layer=1,
            stem_channels=8, projector_hidden=32, teacher_dim=16, **kwargs,
        )

    def test_zero_initialization_keeps_native_output_and_base_has_no_projection(self):
        native = tiny_wan().eval()
        original = copy.deepcopy(native)
        model = self.make_model(native, projection=False, gradient_checkpointing=False).eval()
        with torch.no_grad():
            expected = original(self.noisy, self.time, self.text, return_dict=False)[0]
            actual, tokens = model(self.noisy, self.time, self.text, self.lr)
            disabled, _ = model(self.noisy, self.time, self.text, self.lr, enable_lrv=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(disabled, expected, rtol=0, atol=0)
        self.assertIsNone(tokens)
        self.assertEqual(model.token_grid, (2, 2, 2))

    def test_alignment_only_gradients_reach_generator_with_checkpointing(self):
        model = self.make_model(gradient_checkpointing=True).train()
        teacher = F.normalize(torch.randn(1, 8, 16), dim=-1)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        records = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            _, student = model(self.noisy, self.time, self.text, self.lr)
            loss = (1 - (F.normalize(student, dim=-1) * teacher).sum(-1)).mean()
            loss.backward()
            records.append(gradient_norms(model))
            optimizer.step()
            model.clear_features()
        # Stem and LoRA-A can initially have zero gradients due to zero branch
        # maps / LoRA-B. They must receive gradients after the branch updates.
        for group in ("lrv_stem", "lrv_residuals", "lora", "projector"):
            self.assertGreater(records[-1][group]["l2"], 0, group)
            self.assertTrue(records[-1][group]["finite"], group)
        self.assertEqual(records[-1]["frozen"]["elements_with_grad"], 0)
        self.assertIsNone(model._condition_tokens)
        self.assertIsNone(model._captured)

    def test_condition_changes_output_after_updates_and_missing_branch_is_exact(self):
        model = self.make_model(projection=False, gradient_checkpointing=True).train()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            flow, _ = model(self.noisy, self.time, self.text, self.lr)
            flow.square().mean().backward()
            optimizer.step()
            model.clear_features()
        model.eval()
        with torch.no_grad():
            correct, _ = model(self.noisy, self.time, self.text, self.lr)
            shuffled, _ = model(self.noisy, self.time, self.text, self.lr.flip(-1))
            disabled1, _ = model(self.noisy, self.time, self.text, self.lr, enable_lrv=False)
            disabled2, _ = model(self.noisy, self.time, self.text, self.lr * 100, enable_lrv=False)
        self.assertGreater(float((correct - shuffled).abs().max()), 0)
        torch.testing.assert_close(disabled1, disabled2, rtol=0, atol=0)

    def test_checkpoint_replay_preserves_external_lrv_gradient(self):
        direct = self.make_model(gradient_checkpointing=False).train()
        replay = self.make_model(gradient_checkpointing=True).train()
        # Nonzero residuals expose the complete LRV stem gradient path, including
        # the condition tensor captured by block hooks during checkpoint replay.
        with torch.no_grad():
            for layer in direct.lrv_residuals.values():
                layer.weight.normal_(std=0.02)
        replay.load_state_dict(direct.state_dict())
        teacher = F.normalize(torch.randn(1, 8, 16), dim=-1)
        for model in (direct, replay):
            flow, student = model(self.noisy, self.time, self.text, self.lr)
            loss = flow.square().mean() + (1 - (F.normalize(student, dim=-1) * teacher).sum(-1)).mean()
            loss.backward()
        for (name, parameter), (other_name, other) in zip(direct.named_parameters(), replay.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(parameter.grad is None, other.grad is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-7, msg=name)


if __name__ == "__main__":
    unittest.main()
