"""Task-only checkpoint restoration with isolated CPU factories and fixtures."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vsr.checkpoints import load_training_checkpoint, OFFICIAL_REPO_ID, OFFICIAL_REVISION


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Parameter(torch.full((2,), 11.), requires_grad=False)
        self.lora = nn.Parameter(torch.zeros(2, 3))
        self.lrv = nn.Parameter(torch.zeros(3))


def config():
    value = {"seed": 42, "model": {"repo_id": OFFICIAL_REPO_ID, "revision": OFFICIAL_REVISION}, "training": {"variants": ["Base", "Align"]}}
    value["_sha256"] = hashlib.sha256((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    return value


def make_factory(model=None):
    def factory(cfg, variant):
        return model if model is not None else FakeModel(), {"official_revision": OFFICIAL_REVISION, "fresh_official_initialization": True, "legacy_state_loaded": False}
    return factory


def fixture(root):
    path = root / "fresh_run/Base/checkpoint_000012.pt"
    path.parent.mkdir(parents=True)
    payload = {"format": "vsr_training_v1", "step": 12, "config": config(),
               "trainable_state": {"lora": torch.arange(6).reshape(2, 3).float() / 7, "lrv": torch.tensor([-2., 0., 5.])},
               # These values must never be restored into the evaluation model.
               "optimizer": {"do_not_restore": 123}, "noise_rng": torch.zeros(4, dtype=torch.uint8), "index_rng": torch.ones(4, dtype=torch.uint8)}
    torch.save(payload, path)
    (path.parent.parent / "config.json").write_text(json.dumps(payload["config"]))
    return path, payload


class CheckpointTests(unittest.TestCase):
    def test_task_only_float_restore_frozen_unchanged_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, payload = fixture(root)
            model = FakeModel()
            original_rng = torch.random.get_rng_state().clone()
            real_load = torch.load
            with patch("vsr.checkpoints.torch.load", wraps=real_load) as loader:
                restored, meta = load_training_checkpoint(path, device="cpu", allowed_root=root, model_factory=make_factory(model))
                loader.assert_called_once_with(path, map_location="cpu", weights_only=True)
            self.assertIs(restored, model)
            self.assertTrue(torch.equal(model.lora, payload["trainable_state"]["lora"]))
            self.assertTrue(torch.equal(model.lrv, payload["trainable_state"]["lrv"]))
            self.assertTrue(torch.equal(model.frozen, torch.full((2,), 11.)))
            self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
            self.assertFalse(model.training)
            self.assertFalse(meta["optimizer_state_restored"])
            self.assertFalse(meta["scheduler_state_restored"])
            self.assertFalse(meta["rng_state_restored"])
            self.assertEqual(meta["task_initialization_origin"], "new VSR run checkpoint")
            self.assertEqual(meta["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(meta["variant"], "Base")

    def test_all_keys_shapes_and_dtypes_checked_before_any_copy(self):
        changes = [lambda x: x.pop("lrv"), lambda x: x.update(frozen=torch.zeros(2)),
                   lambda x: x.update(lrv=torch.zeros(4)), lambda x: x.update(lrv=torch.ones(3, dtype=torch.int64)),
                   lambda x: x.update(lrv=torch.full((3,), float("nan"))),
                   lambda x: x.update(lrv=torch.full((3,), 1e100, dtype=torch.float64))]
        for mutate in changes:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, payload = fixture(root)
                mutate(payload["trainable_state"])
                torch.save(payload, path)
                model = FakeModel()
                with self.assertRaises(ValueError):
                    load_training_checkpoint(path, device="cpu", allowed_root=root, model_factory=make_factory(model))
                self.assertTrue(torch.equal(model.lora, torch.zeros(2, 3)))
                self.assertTrue(torch.equal(model.lrv, torch.zeros(3)))

    def test_format_step_pinned_repo_and_config_hash(self):
        mutations = [lambda p: p.update(format="old-format"), lambda p: p.update(step=0),
                     lambda p: p["config"]["model"].update(revision="old-finetune"),
                     lambda p: p["config"].update(seed=99)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, payload = fixture(root)
                mutate(payload)
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    load_training_checkpoint(path, device="cpu", allowed_root=root, model_factory=make_factory())

    def test_legacy_and_path_escape_rejected_before_torch_load(self):
        with patch("vsr.checkpoints.torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_training_checkpoint('/home/hugo/Projects/ReprVSR-pretrained/outputs/old.pt', device="cpu")
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, _ = fixture(root)
                with self.assertRaisesRegex(ValueError, "escapes"):
                    load_training_checkpoint(path, device="cpu", model_factory=make_factory())
            loader.assert_not_called()

    def test_explicit_variant_and_factory_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = fixture(root)
            _, meta = load_training_checkpoint(path, variant="Align", device="cpu", allowed_root=root, model_factory=make_factory())
            self.assertEqual(meta["variant"], "Align")
            with self.assertRaisesRegex(ValueError, "variant"):
                load_training_checkpoint(path, variant="Both", device="cpu", allowed_root=root, model_factory=make_factory())
            def unverified_factory(cfg, variant):
                return FakeModel(), {"official_revision": OFFICIAL_REVISION, "fresh_official_initialization": False, "legacy_state_loaded": True}
            with self.assertRaisesRegex(ValueError, "fresh pinned"):
                load_training_checkpoint(path, device="cpu", allowed_root=root, model_factory=unverified_factory)


if __name__ == "__main__":
    unittest.main()
