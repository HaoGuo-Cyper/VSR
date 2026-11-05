"""CPU fake-VAE/tiny-real-Wan checks; the real GPU smoke is not run here."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F
from diffusers import WanTransformer3DModel

from scripts.smoke_reference import load_v2_training_clip, run_graph_smoke
from vsr.checkpoints import OFFICIAL_REPO_ID, OFFICIAL_REVISION
from vsr.data import _digest
from vsr.reference_model import ReferenceConditionalWan


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(latents_mean=[0.]*4, latents_std=[1.]*4)
        self.encoded_shapes = []

    def encode(self, frame):
        self.encoded_shapes.append(tuple(frame.shape))
        pooled = F.avg_pool2d(frame[:, :, 0], 8)
        latent = torch.cat((pooled, pooled.mean(1, keepdim=True)), 1).unsqueeze(2)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))


def fixture(root):
    run = root/"frozen_v2"
    directory = run/"clip_0"; directory.mkdir(parents=True)
    config = {"seed": 42, "model": {"repo_id": OFFICIAL_REPO_ID, "revision": OFFICIAL_REVISION},
              "data": {"num_frames": 5, "height": 32, "width": 32, "temporal_stride": 4},
              "training": {"variants": ["Both"], "learning_rate": .02, "weight_decay": .01,
                           "lambda_align": .1, "lambda_relation": .1, "max_grad_norm": 1},
              "flow": {"fb_alpha": .01, "fb_beta": .5}, "reproducibility": {"mode": "strict", "relation_loss_device": "cpu"}}
    metadata = {"data_version": "hq-vsr-paired-v2", "record_id": "clip", "split": "train", "caption": "", "seed": 42,
                "temporal_stride": 4, "source_indices": [0,4,8,12,16], "original_fps": 24., "output_fps": 6.,
                "fps_policy": "preserve_sampled_input_fps"}
    binding = {"manifest_sha256": "manifest", "data_cache_key": _digest(metadata), "official_revision": OFFICIAL_REVISION}
    for path, obj in ((run/"config.json", config), (run/"selection.json", {"training_ids": ["clip"], "manifest_sha256": "manifest", "test_accessed": False}),
                      (run/"correspondence.json", {"token_grid": [2,2,2], "latent_grid": [2,4,4], "anchors": [0,4]}),
                      (directory/"metadata.json", {**metadata, "binding": binding})):
        path.write_text(json.dumps(obj))
    torch.save({"hr": torch.rand(1,3,5,32,32), "lr_up": torch.rand(1,3,5,32,32), "lr": torch.rand(1,3,5,8,8)}, directory/"paired_rgb.pt")
    torch.save({"clean": torch.randn(1,4,2,4,4), "low": torch.randn(1,4,2,4,4), "binding": binding}, directory/"fresh_latents.pt")
    torch.save({"": torch.randn(1,4,16)}, run/"fresh_text_embeddings.pt")
    torch.save([torch.randn(1,8,16)], run/"fresh_teacher_features.pt")
    torch.save([{"forward": torch.zeros(1,1,2,32,32), "backward": torch.zeros(1,1,2,32,32)}], run/"fresh_anchor_flows.pt")
    return run


class SmokeReferenceTests(unittest.TestCase):
    def test_v2_training_cache_one_clip_loading_no_task_state_and_path_guards(self):
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
            root = Path(temporary); run = fixture(root)
            config, inputs = load_v2_training_clip(run, allowed_root=root)
            self.assertEqual(inputs["clean"].shape, (1,4,2,4,4))
            self.assertFalse(inputs["source_audit"]["task_checkpoint_loaded"])
            self.assertEqual(inputs["source_audit"]["source_status"], "preparing_or_running")
            self.assertEqual(config["model"]["revision"], OFFICIAL_REVISION)
            with self.assertRaises(ValueError):
                load_v2_training_clip(run, allowed_root=root/"other")
            with self.assertRaises(FileNotFoundError):
                load_v2_training_clip(run, 1, allowed_root=root)

    def test_cpu_graph_checks_reference_checkpoint_replay_and_strict_missing_path(self):
        torch.manual_seed(12); torch.set_num_threads(2)
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
                root = Path(temporary); run = fixture(root)
                config, inputs = load_v2_training_clip(run, allowed_root=root)
                native = WanTransformer3DModel(patch_size=(1,2,2), num_attention_heads=2, attention_head_dim=12,
                    in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48, num_layers=3, rope_max_seq_len=16)
                model = ReferenceConditionalWan(native, reference_layer=1, reference_stem_dim=8,
                    rank=2, alpha=2, injection_layers=(0,1), alignment_layer=1, stem_channels=8,
                    projector_hidden=32, teacher_dim=16, gradient_checkpointing=True)
                vae = FakeVAE()
                summary = run_graph_smoke(model, vae, inputs, config, root/"smoke", reference_index=4)
                self.assertEqual(summary["status"], "passed")
                self.assertTrue(summary["checkpoint_gradients_exact"])
                self.assertTrue(summary["missing_hri_parameter_gradients_none"])
                self.assertEqual(vae.encoded_shapes, [(1,3,1,32,32)])
                self.assertTrue(summary["initial_controls"]["on_off_exact"])
                self.assertGreater(summary["final_controls"]["flow_mae_on_minus_off"], 0)
                self.assertTrue(all(row["reference_adapter_calls_forward_and_replay"] >= 2 for row in summary["rows"]))
                self.assertTrue(all(row["relation_diagnostics"]["strict_cpu_backward"] for row in summary["rows"]))
                self.assertGreater(summary["rows"][-1]["hri_stem_gradient_l2"], 0)
                self.assertEqual(summary["reference_source_frame_index"], 16)
                self.assertIsNone(model._reference_context)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertFalse((root/"active_training.json").exists())
        finally:
            torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)


if __name__ == "__main__":
    unittest.main()
