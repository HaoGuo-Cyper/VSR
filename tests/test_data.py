"""Protocol tests use synthetic fixtures only; they are not model evidence."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from vsr.data import build_manifest, degrade_x4, decode_clip, validate_manifest, write_video


class DataProtocolTests(unittest.TestCase):
    def test_group_split_sealed_and_debug_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source in range(12):
                source_id = f"src_{source:07d}"
                for segment in range(2):
                    (root / f"{source_id}_{segment}_0to10.mp4").write_bytes(f"{source}:{segment}".encode())
            (root / "._src_0000000_0_0to10.mp4").write_bytes(b"ignored")
            first = build_manifest(root)
            second = build_manifest(root)
            self.assertEqual(first, second)
            self.assertEqual(first["source_count"], 12)
            seen = {}
            for record in first["records"]:
                self.assertEqual(seen.setdefault(record["source_id"], record["split"]), record["split"])
            self.assertEqual(set(first["split_source_counts"]), {"train", "val", "test"})
            self.assertTrue(all(r["split"] == "train" for r in first["debug_records"]))
            self.assertEqual(len(first["debug_records"]), 2)
            corrupt = copy.deepcopy(first)
            corrupt["records"][0]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_manifest(corrupt)

    def test_duplicate_bytes_cannot_leak_across_source_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(4):
                (root / f"{i:011d}_0_0to10.mp4").write_bytes(b"duplicate" if i < 2 else str(i).encode())
            manifest = build_manifest(root)
            self.assertEqual(len(manifest["rejected"]), 2)
            self.assertEqual(len(manifest["records"]), 2)

    def test_shared_deterministic_degradation(self):
        frame = torch.rand((1, 3, 1, 32, 48), generator=torch.Generator().manual_seed(11))
        hr = frame.expand(-1, -1, 9, -1, -1).contiguous()
        lr, up = degrade_x4(hr)
        lr2, up2 = degrade_x4(hr)
        self.assertTrue(torch.equal(lr, lr2))
        self.assertTrue(torch.equal(up, up2))
        self.assertTrue(torch.equal(lr[:, :, 0], lr[:, :, -1]))
        self.assertEqual(lr.shape, (1, 3, 9, 8, 12))
        self.assertEqual(up.shape, hr.shape)
        self.assertGreaterEqual(float(lr.min()), 0)
        self.assertLessEqual(float(up.max()), 1)

    def test_decode_window_preserves_fps_and_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "abc_defghij_0_0to13.mp4"
            frames = torch.rand((1, 3, 13, 48, 80), generator=torch.Generator().manual_seed(7))
            write_video(path, frames, 25.0)
            manifest = build_manifest(root)
            record = manifest["records"][0]
            first = decode_clip(record, num_frames=9, height=32, width=48)
            second = decode_clip(record, num_frames=9, height=32, width=48)
            self.assertEqual(first["fps"], 25.0)
            self.assertEqual(first["metadata"]["source_indices"], list(range(first["metadata"]["source_indices"][0], first["metadata"]["source_indices"][0] + 9)))
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertTrue(torch.equal(first["hr"], second["hr"]))
            self.assertTrue(bool(first["valid_mask"].all()))
            changed = decode_clip({**record, "split": "test"}, num_frames=9, height=32, width=48)
            self.assertNotEqual(first["cache_key"], changed["cache_key"])
            with self.assertRaisesRegex(ValueError, "Undersized"):
                decode_clip(record, height=96, width=160)


if __name__ == "__main__":
    unittest.main()
