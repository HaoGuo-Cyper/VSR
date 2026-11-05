"""Tiny/fake CPU fixtures only, never actual validation/test pixels or GPU."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch
from torch import nn
from diffusers import WanTransformer3DModel

from vsr import noref_feature_diagnostics as diagnostics
from vsr.audit import file_hash
from vsr.data import _digest
from vsr.model import ConditionalWan
from tests.test_reference_diagnostics import loader_fixture as reference_loader_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


class RecordingWan(ConditionalWan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed, self.fail = [], False

    def forward(self, noisy, timestep, text, low, *args, **kwargs):
        self.observed.append({"noisy": noisy.detach().clone(), "timestep": timestep.detach().clone(),
            "grad_enabled": torch.is_grad_enabled(), "low_requires_grad": low.requires_grad,
            "text_requires_grad": text.requires_grad})
        flow, student = super().forward(noisy, timestep, text, low, *args, **kwargs)
        return flow, student*float("nan") if self.fail else student


def model():
    native = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12,
        in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=48, num_layers=3, rope_max_seq_len=16)
    return RecordingWan(native, rank=2, alpha=2, injection_layers=(0, 1), alignment_layer=1,
        stem_channels=8, projector_hidden=32, teacher_dim=16, gradient_checkpointing=True)


CONFIG = {"seed": 42, "data": {"num_frames": 5, "height": 32, "width": 32, "temporal_stride": 4},
    "conditioning": {"teacher_dim": 16}, "reproducibility": {"mode": "strict"},
    "teacher": {"source_dir": "fake", "weights": "fake", "weights_sha256": "f"*64, "revision": "fake-revision"},
    "flow": {"fb_alpha": .01, "fb_beta": .5, "weights": "fake", "updates": 12, "weights_sha256": "e"*64}}


def sample(index, *, large_motion=False):
    flows = {"forward": torch.zeros(1, 1, 2, 32, 32), "backward": torch.zeros(1, 1, 2, 32, 32)}
    if large_motion:
        flows["forward"] += 1000; flows["backward"] -= 1000
    hr = torch.rand(1, 3, 5, 32, 32)
    entry = {"clip_index": index, "metadata": {"record_id": f"val{index}", "source_id": f"source{index}", "split": "val"},
        "hr": hr, "clean": torch.randn(1, 4, 2, 4, 4, requires_grad=True),
        "low": torch.randn(1, 4, 2, 4, 4, requires_grad=True), "text": torch.randn(1, 4, 16, requires_grad=True),
        "teacher": torch.randn(1, 8, 16, requires_grad=True), "flows": flows}
    entry["HR_descriptors"] = diagnostics.describe_hr(hr, flows, {"anchors": [0, 4], "token_grid": [2, 2, 2]}, CONFIG)
    return entry


def inputs(*, large_motion=False):
    return {"geometry": {"anchors": [0, 4], "latent_grid": [2, 4, 4], "token_grid": [2, 2, 2]},
            "validation": [sample(17, large_motion=large_motion), sample(23, large_motion=large_motion)]}


def loader_fixture(directory):
    root, run, source, cfg = reference_loader_fixture(directory)
    cfg = copy.deepcopy(cfg); cfg["reference"] = {"enabled": False}; cfg["conditioning"] = {"teacher_dim": 16}
    cfg["teacher"].update(source_dir="fake", weights="fake")
    cfg["flow"].update(fb_alpha=.01, fb_beta=.5, weights="fake")
    write(run/"config.json", cfg); write(run/"status.json", {"status": "completed"})
    write(Path(cfg["model"]["local_dir"])/"text_encoder/config.json", {"d_model": 16})
    manifest = json.loads((root/cfg["data"]["manifest"]).read_text()); records = {row["id"]: row for row in manifest["records"]}
    seal = json.loads((source/"cache_asset_seal.json").read_text())
    captions = set()
    for clip in sorted(source.glob("clip_*")):
        raw = json.loads((clip/"metadata.json").read_text()); metadata = {k: v for k, v in raw.items() if k != "binding"}
        metadata["caption"] = records[metadata["record_id"]]["caption"]
        metadata["caption_status"] = records[metadata["record_id"]]["caption_status"]
        binding = raw["binding"]; binding["data_cache_key"] = _digest(metadata)
        write(clip/"metadata.json", {**metadata, "binding": binding})
        payload = torch.load(clip/"fresh_latents.pt", weights_only=True); payload["binding"] = binding
        torch.save(payload, clip/"fresh_latents.pt"); captions.add(metadata["caption"])
        for name in ("metadata.json", "paired_rgb.pt", "fresh_latents.pt"):
            seal["artifact_sha256"][str((clip/name).relative_to(source))] = file_hash(clip/name)
    torch.save({caption: torch.zeros(1, 512, 16) for caption in captions}, source/"fresh_text_embeddings.pt")
    for name in ("config.json", "environment.json", "selection.json", "correspondence.json", "fresh_text_embeddings.pt"):
        seal["artifact_sha256"][name] = file_hash(source/name)
    write(source/"status.json", {"status": "interrupted"})
    seal.update(format="vsr_prepared_cache_seal_v1", preparation_completed=True, contains_task_initialization=False,
        source_run=str(source), origin_training_status="interrupted", origin_status_sha256=file_hash(source/"status.json"))
    write(source/"cache_asset_seal.json", seal)
    write(run/"cache_reuse.json", {"source_run": str(source), "source_asset_seal_sha256": file_hash(source/"cache_asset_seal.json")})
    # These intentionally cannot be deserialized. The new diagnostic must not
    # inspect any old teacher/flow or top cache merely because files exist.
    for path in (source/"fresh_training_cache.pt", source/"fresh_teacher_features.pt", source/"fresh_anchor_flows.pt",
                 run/"fresh_training_cache.pt", run/"fresh_teacher_features.pt", run/"fresh_anchor_flows.pt"):
        path.write_bytes(b"POISON old supervision payload - do not load")
    return root, run, source, cfg


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, hr, anchors, grid):
        assert not torch.is_grad_enabled() and not self.training and not self.weight.requires_grad
        values = hr[:, :, anchors].mean(dim=(1, 3, 4)).mean(dim=1, keepdim=True)
        return (torch.arange(16, device=hr.device).float().view(1, 1, 16)+values[:, :, None]).expand(1, int(np.prod(grid)), 16).clone()


class FakeFlow(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.zeros(())); self.provenance = {"scope": "synthetic CPU test only"}

    def estimate_anchor_flows(self, hr, anchors):
        assert not torch.is_grad_enabled() and not self.training and not self.weight.requires_grad
        shape = (1, len(anchors)-1, 2, *hr.shape[-2:])
        return torch.zeros(shape, device=hr.device), torch.zeros(shape, device=hr.device)


class NorefDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(22); torch.set_num_threads(2)

    def test_oracle_noise_uses_source_cache_index_and_no_updates_or_rng_flag_changes(self):
        net, data = model(), inputs(); net.train(); net.transformer.blocks[-1].eval()
        before = {name: value.clone() for name, value in net.state_dict().items()}
        flags = [m.training for m in net.modules()]; rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        with patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")), patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer permitted")):
            result = diagnostics.collect_noref_feature_diagnostics(net, data, CONFIG, "Both")
        self.assertTrue(all(result["cleanup"].values()))
        self.assertEqual(result["protocol"]["backward_calls"], 0)
        self.assertFalse(result["protocol"]["restoration_quality_evidence"])
        for sample, call, row in zip(data["validation"], net.observed, result["per_video"]):
            seed = CONFIG["seed"]+91+sample["clip_index"]
            noise = torch.randn(sample["clean"].shape, generator=torch.Generator().manual_seed(seed))
            torch.testing.assert_close(call["noisy"], .5*sample["clean"].detach()+.5*noise, rtol=0, atol=0)
            self.assertEqual(row["noise_seed"], seed)
            self.assertEqual(call["timestep"].tolist(), [500.])
            self.assertFalse(call["grad_enabled"] or call["low_requires_grad"] or call["text_requires_grad"])
            self.assertIn("spatial", row["relation_objectives"])
            self.assertIn("student_effective_rank", row["statistics"])
            self.assertTrue(all(sample[key].grad is None for key in ("clean", "low", "text", "teacher")))
        self.assertEqual(flags, [m.training for m in net.modules()]); self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate()); np.testing.assert_equal(numpy_rng, np.random.get_state())
        for name, value in net.state_dict().items():
            torch.testing.assert_close(before[name], value, rtol=0, atol=0)

    def test_absent_temporal_edges_are_null_and_combined_uses_spatial_only(self):
        result = diagnostics.collect_noref_feature_diagnostics(model(), inputs(large_motion=True), CONFIG, "Relation")
        self.assertEqual(result["aggregate"]["relation_objectives"]["temporal"], {"video_count_with_edges": 0, "video_mean": None})
        for row in result["per_video"]:
            self.assertIsNone(row["relation_objectives"]["temporal"])
            self.assertEqual(row["relation_diagnostics"]["active_edge_types"], 1)
            self.assertEqual(row["relation_objectives"]["spatial"], row["relation_objectives"]["combined"])

    def test_failure_cleans_model_context_and_Base_is_explicitly_not_applicable(self):
        net, data = model(), inputs(); net.fail = True; net.train(); net.transformer.blocks[0].eval()
        flags = [m.training for m in net.modules()]
        with self.assertRaises(ValueError):
            diagnostics.collect_noref_feature_diagnostics(net, data, CONFIG, "Align")
        self.assertEqual(flags, [m.training for m in net.modules()]); self.assertIsNone(net._condition_tokens); self.assertIsNone(net._captured)
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        class Poison:
            def __getattribute__(self, key):
                raise AssertionError("Base must not request projected features")
        result = diagnostics.collect_noref_feature_diagnostics(Poison(), Poison(), CONFIG, "Base")
        self.assertEqual(result["status"], "not_applicable")

    def test_descriptors_are_fixed_model_independent_motion_and_laplacian_quantities(self):
        hr = torch.zeros(1, 3, 5, 8, 8); flow = {key: torch.zeros(1, 1, 2, 8, 8) for key in ("forward", "backward")}
        geometry = {"anchors": [0, 4], "token_grid": [2, 2, 2]}
        flat = diagnostics.describe_hr(hr, flow, geometry, CONFIG)
        self.assertEqual(flat["forward_flow_norm_hr_pixels"], {"mean": 0., "p95": 0.})
        self.assertEqual(flat["token_boundary_valid_fraction"], 1)
        self.assertEqual(flat["token_forward_backward_consistent_fraction"], 1)
        self.assertEqual(flat["token_motion_visibility_valid_fraction"], 1)
        self.assertEqual(flat["HR_anchor_luma_laplacian_energy"], 0)
        checker = ((torch.arange(8)[:, None]+torch.arange(8)[None, :])%2).float()
        textured = diagnostics.describe_hr(checker.view(1, 1, 1, 8, 8).expand_as(hr), flow, geometry, CONFIG)
        self.assertAlmostEqual(textured["HR_anchor_luma_laplacian_energy"], 16.)
        flow["forward"][:, :, 0] = 3; flow["forward"][:, :, 1] = 4
        flow["backward"] = -flow["forward"]
        moving = diagnostics.describe_hr(hr, flow, geometry, CONFIG)
        self.assertEqual(moving["forward_flow_norm_hr_pixels"], {"mean": 5., "p95": 5.})
        self.assertEqual(moving["backward_flow_norm_hr_pixels"], {"mean": 5., "p95": 5.})
        self.assertEqual(moving["token_boundary_valid_fraction"], .25)
        self.assertEqual(moving["token_motion_visibility_valid_fraction"], .25)

    def test_sealed_interrupted_source_loads_only_val_RGB_perclip_latents_text(self):
        with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
            root, run, source, cfg = loader_fixture(Path(temporary))
            real_load, opened = torch.load, []
            forbidden = {"fresh_training_cache.pt", "fresh_teacher_features.pt", "fresh_anchor_flows.pt"}
            def observed(path, **kwargs):
                path = Path(path)
                if path.name in forbidden:
                    raise AssertionError("No saved teacher/flow/top payload may be read")
                opened.append(path); return real_load(path, **kwargs)
            with patch("vsr.noref_feature_diagnostics.torch.load", side_effect=observed):
                result = diagnostics.load_noref_validation(run, cfg, project_root=root, expected_validation_count=1)
            self.assertEqual(len(result["validation"]), 1)
            sample = result["validation"][0]
            self.assertEqual(sample["metadata"]["split"], "val")
            self.assertGreater(sample["clip_index"], 0)
            self.assertNotIn("teacher", sample); self.assertNotIn("flows", sample)
            self.assertTrue(all(path.name not in forbidden for path in opened))
            self.assertFalse(result["source_audit"]["teacher_or_flow_payload_read"])
            self.assertEqual(result["source_audit"]["source_training_status"], "interrupted")

    def test_loader_rejects_changed_pixels_latents_order_or_incomplete_interrupted_seal(self):
        for kind in ("RGB", "latent", "order", "seal"):
            with tempfile.TemporaryDirectory() as temporary:
                root, run, source, cfg = loader_fixture(Path(temporary))
                selection = json.loads((source/"selection.json").read_text()); index = len(selection["training_ids"])
                if kind in ("RGB", "latent"):
                    filename = "paired_rgb.pt" if kind == "RGB" else "fresh_latents.pt"
                    path = source/f"clip_{index}"/filename; payload = torch.load(path, weights_only=True)
                    payload["hr" if kind == "RGB" else "clean"] += 1
                    torch.save(payload, path)
                elif kind == "order":
                    changed = {**selection, "validation_ids": [selection["training_ids"][0]]}
                    write(run/"selection.json", changed)
                else:
                    path = source/"cache_asset_seal.json"; seal = json.loads(path.read_text()); seal["preparation_completed"] = False; write(path, seal)
                with self.assertRaises(ValueError):
                    diagnostics.load_noref_validation(run, cfg, project_root=root, expected_validation_count=1)

    def test_loader_rejects_source_change_after_RGB_has_been_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, run, source, cfg = loader_fixture(Path(temporary))
            selection = json.loads((source/"selection.json").read_text()); index = len(selection["training_ids"])
            real_load = torch.load
            def changed_during_load(path, **kwargs):
                result = real_load(path, **kwargs)
                if Path(path) == source/f"clip_{index}/fresh_latents.pt":
                    (source/f"clip_{index}/paired_rgb.pt").write_bytes(b"changed after initial RGB verification")
                return result
            with patch("vsr.noref_feature_diagnostics.torch.load", side_effect=changed_during_load):
                with self.assertRaisesRegex(ValueError, "changed during loading"):
                    diagnostics.load_noref_validation(run, cfg, project_root=root, expected_validation_count=1)

    def test_fresh_targets_release_teachers_and_ignore_existing_target_objects(self):
        data = inputs(); refs, events = {}, []
        class Poison:
            def __getattribute__(self, key):
                raise AssertionError("Old targets must not be used")
        for entry in data["validation"]:
            entry["teacher"], entry["flows"] = Poison(), Poison()
        def teacher_factory(config):
            events.append("teacher"); obj = FakeTeacher(); refs["teacher"] = weakref.ref(obj); return obj
        def flow_factory(config):
            self.assertIsNone(refs["teacher"]()); events.append("flow")
            obj = FakeFlow(); refs["flow"] = weakref.ref(obj); return obj
        with patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
            fresh = diagnostics.extract_fresh_targets(data, CONFIG, device="cpu", teacher_factory=teacher_factory, flow_factory=flow_factory)
        self.assertEqual(events, ["teacher", "flow"]); self.assertIsNone(refs["teacher"]()); self.assertIsNone(refs["flow"]())
        self.assertTrue(fresh["fresh_target_audit"]["freshly_extracted"])
        for entry in fresh["validation"]:
            self.assertEqual(entry["teacher"].shape, (1, 8, 16)); self.assertEqual(entry["teacher"].device.type, "cpu")
            self.assertFalse(entry["teacher"].requires_grad)
            self.assertEqual(entry["flows"]["forward"].shape, (1, 1, 2, 32, 32))

    def test_main_guard_precedes_checkpoint_or_data_access(self):
        with patch("sys.argv", ["diagnostics", "--checkpoint", "no-read"]), \
             patch.object(diagnostics, "assert_gpu_idle", side_effect=RuntimeError("training active")), \
             patch("vsr.infer_video.preview_training_config", side_effect=AssertionError("No checkpoint access")):
            with self.assertRaisesRegex(RuntimeError, "training active"):
                diagnostics.main()

    def test_main_targets_precede_task_restore_and_changed_checkpoint_is_rejected(self):
        for changed in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                root, run, source, cfg = loader_fixture(Path(temporary)); directory = run/"Both"; directory.mkdir()
                checkpoint = directory/"checkpoint_008192.pt"; checkpoint.write_bytes(b"synthetic checkpoint preview only")
                write(directory/"summary.json", {"status": "completed", "variant": "Both", "updates": 8192})
                preview = {"config": cfg, "variant": "Both", "step": 8192, "checkpoint_sha256": file_hash(checkpoint)}
                refs, events = {}, []
                def teacher(*args, **kwargs):
                    events.append("teacher"); obj = FakeTeacher(); refs["teacher"] = weakref.ref(obj); return obj
                def flow(*args, **kwargs):
                    self.assertIsNone(refs["teacher"]()); events.append("flow")
                    obj = FakeFlow(); refs["flow"] = weakref.ref(obj); return obj
                def restore(*args, **kwargs):
                    self.assertIsNone(refs["teacher"]()); self.assertIsNone(refs["flow"]()); events.append("task")
                    return model(), {**preview, "checkpoint_sha256": "changed" if changed else preview["checkpoint_sha256"],
                                     "optimizer_state_restored": False, "rng_state_restored": False}
                real_inputs = diagnostics.load_noref_validation
                def load_inputs(*args, **kwargs):
                    return real_inputs(*args, **kwargs, expected_validation_count=1)
                output = root/"runs/new_diagnostic"
                with patch.object(diagnostics, "ROOT", root), patch("sys.argv", ["diagnostics", "--checkpoint", str(checkpoint), "--output", str(output), "--device", "cpu"]), \
                     patch.object(diagnostics, "assert_gpu_idle", return_value={"GPU_idle_verified": True}), \
                     patch("vsr.infer_video.preview_training_config", return_value=preview), \
                     patch.object(diagnostics, "verify_official", return_value={"scope": "synthetic CPU test"}), \
                     patch.object(diagnostics, "environment", return_value={"scope": "synthetic CPU test"}), \
                     patch.object(diagnostics, "verify_numeric_runtime", return_value={"scope": "synthetic CPU test"}), \
                     patch.object(diagnostics, "configure_reproducibility", return_value={"scope": "synthetic CPU test"}), \
                     patch.object(diagnostics, "load_noref_validation", side_effect=load_inputs), \
                     patch.object(diagnostics, "DinoTeacher", side_effect=teacher), patch.object(diagnostics, "FlowTeacher", side_effect=flow), \
                     patch.object(diagnostics, "load_training_checkpoint", side_effect=restore), \
                     patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")), patch("builtins.print"):
                    if changed:
                        with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                            diagnostics.main()
                    else:
                        diagnostics.main()
                self.assertEqual(events, ["teacher", "flow", "task"])
                status = json.loads((output/"status.json").read_text())
                self.assertEqual(status["status"], "failed" if changed else "completed")
                self.assertEqual(status["optimizer_updates"], 0)


if __name__ == "__main__":
    unittest.main()
