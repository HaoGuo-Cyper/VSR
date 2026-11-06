"""CPU synthetic lossless video tests; no real HQ or external pixels."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from vsr.data import _digest, select_debug_records
from vsr.frozen_evaluation import stable_hq_input_contract
from vsr.native_qa_reader import NativeQAReader


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    path = root/"synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25., (64, 32))
    if not writer.isOpened():
        raise RuntimeError("Synthetic CPU QA tests require local FFV1 lossless video support")
    rgb_frames = []
    yy, xx = np.mgrid[:32, :64]
    try:
        for frame in range(19):
            rgb = np.stack(((xx*3+frame*7)%256, (yy*5+frame*11)%256, (xx+yy+frame*13)%256), axis=-1).astype(np.uint8)
            rgb_frames.append(rgb)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    rid = "00000000001_0_0to19"
    record = {"id": rid, "source_id": "00000000001", "split": "val", "dataset": "HQ-VSR",
              "path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
    train = {"id": "00000000000_0_0to19", "source_id": "00000000000", "split": "train", "dataset": "HQ-VSR",
             "path": str(root/"metadata_only_train.avi"), "bytes": 1, "sha256": hashlib.sha256(b"train").hexdigest()}
    manifest = {"schema_version": 1, "dataset": "HQ-VSR", "seed": 20261102, "records": [train, record]}
    manifest["debug_records"] = select_debug_records(manifest)
    manifest["manifest_sha256"] = _digest(manifest)
    (root/"manifests").mkdir()
    (root/"manifests/hq_vsr.json").write_text(json.dumps(manifest))
    ordered = {"manifest_sha256": manifest["manifest_sha256"], "training_ids": [train["id"]],
               "validation_ids": [rid], "test_accessed": False}
    selection = root/"runs/formal_hq_v2_noref_20261102T142214.588571Z/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps(ordered))
    header = {"opened": True, "fps": 25., "frame_count": 19, "height": 32, "width": 64, "backend": "FFMPEG"}
    inventory = root/"manifests/hq_temporal_stride4_inventory.json"
    inventory.write_text(json.dumps({"records": [{"id": rid, "split": "val", "raw_sha256": record["sha256"], "header": header}]}))
    plans = []
    for full in (False, True):
        height, width = (32, 64) if full else (16, 32)
        plan = {"schema": "vsr_native_fullframe_operator_qa_predeclaration_v1" if full else "vsr_native_crop_operator_qa_predeclaration_v1",
            "scope": "Synthetic CPU test: explicit17frame native pixel QA",
            "selection": {"manifest_sha256": manifest["manifest_sha256"], "split": "val", "record_id": rid,
                "source_id": record["source_id"], "source_path": str(path), "source_bytes": record["bytes"],
                "source_sha256": record["sha256"], "ordered_validation_selection_sha256": _sha(selection),
                "header_inventory_sha256": _sha(inventory), "header_inventory_observation": header},
            "input_preparation": {"source_hw": [32,64], "source_frame_indices": list(range(17)), "num_frames": 17,
                "temporal_stride": 1, "scale": 4, "HR_shape": [1,3,17,height,width], "LR_shape": [1,3,17,height//4,width//4],
                "source_fps_observed": 25., "LR_degradation": {"kernel": "torch bicubic", "antialias": True,
                    "align_corners": False, "clamp_01": True}}}
        if not full:
            plan["input_preparation"]["native_center_crop_tlhw"] = [8,16,16,32]
        plans.append(plan)
    return rid, path, plans, rgb_frames


class NativeQAReaderTests(unittest.TestCase):
    def test_constructor_reads_metadata_only_and_policy_is_one_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rid, path, plans, _ = fixture(root)
            with patch("cv2.VideoCapture", side_effect=AssertionError("Constructor decoded pixels")), \
                 patch("vsr.native_qa_reader.sha256_file", side_effect=AssertionError("Constructor hashed source pixels")), \
                 patch("torch.cuda.init", side_effect=AssertionError("GPU forbidden")):
                reader = NativeQAReader(root, plans=plans)
            self.assertEqual(set(reader.records), {"HQ-val-nativecrop", "HQ-val-nativefull"})
            self.assertEqual(reader.metadata["selected_record_ids"], [rid])
            self.assertFalse(reader.metadata["pixel_decode_in_constructor"])
            self.assertFalse(reader.metadata["training_allowed"])
            self.assertEqual(reader.metadata["noise_seed_by_record_id"], {rid:0})
            for dataset, expected in (("HQ-val-nativecrop",(16,32)), ("HQ-val-nativefull",(32,64))):
                policy=reader.policy(dataset)
                self.assertEqual(policy.family, "hq")
                self.assertEqual(policy.allowed_datasets, (dataset,))
                self.assertEqual((policy.reader_options["height"],policy.reader_options["width"]), expected)
                single=NativeQAReader(root, **policy.reader_options)
                self.assertEqual(list(single.records), [dataset])
            self.assertIsNone(reader.close())

    def test_lossless_synthetic_native_pixels_and_x4_shapes_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rid, _, plans, frames = fixture(root)
            reader=NativeQAReader(root,plans=plans)
            full=reader.load("HQ-val-nativefull",rid)
            crop=reader.load("HQ-val-nativecrop",rid)
            expected=torch.from_numpy(np.stack(frames[:17])).permute(3,0,1,2).unsqueeze(0).float()/255
            torch.testing.assert_close(full["hr"],expected,rtol=0,atol=0)
            torch.testing.assert_close(crop["hr"],expected[:,:,:,8:24,16:48],rtol=0,atol=0)
            self.assertEqual(tuple(full["lr"].shape),(1,3,17,8,16))
            self.assertEqual(tuple(crop["lr"].shape),(1,3,17,4,8))
            self.assertEqual(full["fps"],25.)
            self.assertFalse(full["metadata"]["HR_resized"])
            self.assertEqual(crop["metadata"]["source_frame_indices"],list(range(17)))
            self.assertEqual(crop["metadata"]["resize_hw"],[32,64])
            self.assertNotEqual(full["cache_key"],crop["cache_key"])

    def test_input_binding_is_repeatable_and_accepted_by_shared_engine_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rid, _, plans, _=fixture(root)
            reader=NativeQAReader(root,plans=[plans[0]])
            first=reader.load("HQ-val-nativecrop",rid)
            second=reader.load("HQ-val-nativecrop",rid)
            def contract(value):
                return stable_hq_input_contract(value["metadata"],value["lr"],value["hr"],fps=value["fps"],frames=17)
            self.assertEqual(contract(first),contract(second))
            self.assertEqual(first["metadata"]["input_binding"]["robustness_mode"],"nativecrop")
            first["lr"][0,0,0,0,0]+=0.01
            with self.assertRaisesRegex(ValueError,"float bytes differ"):
                contract(first)

    def test_changed_source_fails_before_decoder_and_wrong_requests_never_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rid, path, plans, _=fixture(root)
            reader=NativeQAReader(root,plans=[plans[0]])
            with patch("cv2.VideoCapture",side_effect=AssertionError("Should fail before pixels")):
                with self.assertRaises(ValueError):reader.load("HQ-val-nativecrop",rid,num_frames=16)
                with self.assertRaises(ValueError):reader.load("HQ-val-nativecrop",rid,num_frames=17.0)
                with self.assertRaises(ValueError):reader.load("HQ-val-nativecrop",rid,start=1)
                with self.assertRaises(KeyError):reader.load("HQ-val-nativecrop","unselected")
                path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError,"Raw source changed"):
                    reader.load("HQ-val-nativecrop",rid)

    def test_declared_geometry_fps_selection_and_source_binding_cannot_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); _, _, plans, _=fixture(root)
            mutations=[
                lambda d:d["selection"].update(source_sha256="0"*64),
                lambda d:d["selection"].update(record_id="not-first-val"),
                lambda d:d["input_preparation"].update(source_frame_indices=list(range(1,18))),
                lambda d:d["input_preparation"].update(native_center_crop_tlhw=[0,0,16,32]),
                lambda d:d["input_preparation"].update(source_fps_observed=24.),
                lambda d:d["selection"].update(header_inventory_sha256="0"*64),
                lambda d:d["selection"].update(ordered_validation_selection_sha256="0"*64),
            ]
            for mutate in mutations:
                wrong=copy.deepcopy(plans[0]);mutate(wrong)
                with self.subTest(plan=wrong), self.assertRaises(ValueError):NativeQAReader(root,plans=[wrong])
            with self.assertRaises(ValueError):NativeQAReader(root,plans=plans,height=16,width=32)
            with self.assertRaises(ValueError):NativeQAReader(root,plans=[plans[0]],split="test")
            with self.assertRaises(ValueError):NativeQAReader(root,plans=[plans[0]],temporal_stride=4)

    def test_partial_real_decode_is_a_failure_not_a_shorter_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rid, path, plans, _=fixture(root)
            reader=NativeQAReader(root,plans=[plans[0]])
            actual=cv2.VideoCapture(str(path))
            class TruncatedCapture:
                calls=0
                def isOpened(self):return actual.isOpened()
                def get(self,key):return actual.get(key)
                def release(self):actual.release()
                def read(self):
                    self.calls+=1
                    return actual.read() if self.calls<8 else (False,None)
            with patch("cv2.VideoCapture",return_value=TruncatedCapture()):
                with self.assertRaisesRegex(ValueError,"failed to decode original frame7"):
                    reader.load("HQ-val-nativecrop",rid)

    def test_file_declaration_is_copied_and_runtime_header_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rid, _, plans, _=fixture(root)
            declaration=root/"declaration.json";declaration.write_text(json.dumps(plans[0]))
            reader=NativeQAReader(root,plans=[declaration])
            self.assertEqual(reader.metadata["cases"]["HQ-val-nativecrop"]["declaration_file_sha256"],_sha(declaration))
            wrong=copy.deepcopy(reader._cases["nativecrop"]["header"]);wrong["fps"]=24.
            reader._cases["nativecrop"]["header"]=wrong
            with self.assertRaisesRegex(ValueError,"Actual native source FPS"):
                reader.load("HQ-val-nativecrop",rid)


if __name__=="__main__":
    unittest.main()
