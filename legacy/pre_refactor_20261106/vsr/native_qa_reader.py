"""Predeclared native-pixel validation inputs for structural operator QA.

Construction reads JSON metadata only. ``load`` is the only pixel entry point;
it decodes exactly the declared first 17 original frames, without HR resizing.
The reader never selects a checkpoint or changes a failed input geometry.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import sha256_file, validate_manifest


_SELECTION = "runs/formal_hq_v2_noref_20261102T142214.588571Z/selection.json"
_INVENTORY = "manifests/hq_temporal_stride4_inventory.json"
_SCHEMAS = {
    "vsr_native_crop_operator_qa_predeclaration_v1": "nativecrop",
    "vsr_native_fullframe_operator_qa_predeclaration_v1": "nativefull",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tensor_sha256(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _integers(value, count, label, *, minimum=0):
    if (not isinstance(value, list) or len(value) != count
            or any(not isinstance(x, int) or isinstance(x, bool) or x < minimum for x in value)):
        raise ValueError(f"{label} must contain {count} integers >= {minimum}")
    return value


def _json_copy(value):
    # A finite JSON snapshot also prevents caller mutation after construction.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class NativeQAReader:
    """One fixed val source with crop and/or full-frame QA declarations.

    ``policy(dataset)`` returns engine inputs for ONE spatial case per run.
    Pass ``factories={"reader": NativeQAReader, ...}`` to the shared evaluator;
    its policy reader_options carry the selected declaration and exact geometry.
    Multiple declarations may be catalogued here, but a single HQ engine run
    has one height/width and therefore must use one policy/dataset at a time.
    """

    def __init__(self, project_root, *, plans, height=None, width=None,
                 split="val", temporal_stride=1):
        if (split != "val" or not isinstance(temporal_stride, int)
                or isinstance(temporal_stride, bool) or temporal_stride != 1):
            raise ValueError("Native operator QA accepts val and original contiguous frames only")
        if not isinstance(plans, (list, tuple)) or not plans:
            raise ValueError("Provide at least one explicit native QA declaration")
        self.root = Path(project_root).resolve()
        manifest_path = (self.root / "manifests/hq_vsr.json").resolve()
        if not manifest_path.is_relative_to(self.root):
            raise ValueError("HQ manifest must remain inside the selected independent project")
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        validate_manifest(manifest)
        if manifest.get("dataset") != "HQ-VSR":
            raise ValueError("Native QA requires the sealed HQ-VSR catalogue")
        self.manifest_sha256 = manifest["manifest_sha256"]
        self.manifest_file_sha256 = hashlib.sha256(raw).hexdigest()
        self._cases, self.records, selected_ids = {}, {}, []
        for value in plans:
            path = None if isinstance(value, dict) else Path(value).expanduser().resolve()
            declaration_raw = path.read_bytes() if path is not None else None
            plan = _json_copy(value if path is None else json.loads(declaration_raw))
            mode = _SCHEMAS.get(plan.get("schema"))
            if mode is None or mode in self._cases:
                raise ValueError("Native QA declarations require distinct known crop/full schemas")
            case = self._validate_plan(plan, manifest)
            dataset = "HQ-val-" + mode
            case.update(mode=mode, dataset=dataset, plan=plan, plan_sha256=_digest(plan),
                        declaration_path=str(path) if path is not None else None,
                        declaration_file_sha256=hashlib.sha256(declaration_raw).hexdigest() if declaration_raw is not None else None)
            self._cases[mode] = case
            record = case["record"]
            self.records[dataset] = {record["id"]: copy.deepcopy(record)}
            if record["id"] not in selected_ids:
                selected_ids.append(record["id"])
            for requested, actual, label in ((height, case["height"], "height"), (width, case["width"], "width")):
                if requested is not None and (not isinstance(requested, int) or isinstance(requested, bool) or requested != actual):
                    raise ValueError(f"Reader {label} differs from native QA declaration")
        if len(selected_ids) != 1:
            raise ValueError("Both native QA cases must use the same first sealed validation source")
        self.metadata = {
            "reader": "NativeQAReader", "source_dataset": "HQ-VSR", "split": "val",
            "manifest_path": str(manifest_path), "manifest_sha256": self.manifest_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "selected_record_ids": selected_ids, "selected_record_count": 1,
            "validation_selection": "first record in bound sealed formal validation order; no result ranking",
            "noise_seed_by_record_id": {rid: 0 for rid in selected_ids},
            "noise_seed_rule": "zero binding sentinel; no additional LR noise is applied",
            "cases": {case["dataset"]: self._case_metadata(case) for case in self._cases.values()},
            "pixel_decode_in_constructor": False, "training_allowed": False,
            "parameters_selected_on_test": False, "checkpoint_selection_performed": False,
            "metrics_computed": False, "official_test_split": False,
            "scope": "predeclared native-pixel validation operator QA; structural acceptance, no quality threshold tuning",
        }

    def _inside_metadata(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("QA metadata must remain inside the selected independent project")
        return path

    def _validate_plan(self, plan, manifest):
        selection, prep = plan.get("selection", {}), plan.get("input_preparation", {})
        if selection.get("manifest_sha256") != self.manifest_sha256 or selection.get("split") != "val":
            raise ValueError("Native QA declaration catalogue/split differs from sealed validation")
        selection_path = self._inside_metadata(selection.get("ordered_validation_selection_path", _SELECTION))
        selection_raw = selection_path.read_bytes()
        if hashlib.sha256(selection_raw).hexdigest() != selection.get("ordered_validation_selection_sha256"):
            raise ValueError("Native QA validation selection SHA256 mismatch")
        ordered = json.loads(selection_raw)
        val = sorted((row for row in manifest["records"] if row["split"] == "val"),
                     key=lambda row: hashlib.sha256(f"val:{manifest['seed']}:{row['id']}".encode()).hexdigest())
        ids = ordered.get("validation_ids")
        if (not isinstance(ids, list) or not ids or ids != [row["id"] for row in val[:len(ids)]]
                or ordered.get("manifest_sha256") != self.manifest_sha256 or ordered.get("test_accessed") is not False):
            raise ValueError("Native QA needs the actual sealed ordered validation selection")
        record = copy.deepcopy(val[0])
        if (selection.get("record_id") != ids[0] or selection.get("source_id") != record["source_id"]
                or selection.get("source_sha256") != record["sha256"]
                or selection.get("source_bytes") != record["bytes"]
                or Path(selection.get("source_path", "")).resolve() != Path(record["path"]).resolve()):
            raise ValueError("Native QA must identify the first sealed validation source exactly")
        inventory_path = self._inside_metadata(selection.get("header_inventory_path", _INVENTORY))
        inventory_raw = inventory_path.read_bytes()
        if hashlib.sha256(inventory_raw).hexdigest() != selection.get("header_inventory_sha256"):
            raise ValueError("Native QA header inventory SHA256 mismatch")
        entries = [row for row in json.loads(inventory_raw).get("records", []) if row.get("id") == record["id"]]
        if (len(entries) != 1 or entries[0].get("split") != "val"
                or entries[0].get("raw_sha256") != record["sha256"]
                or entries[0].get("header") != selection.get("header_inventory_observation")):
            raise ValueError("Native QA source/header binding differs from its declared inventory")
        header = entries[0]["header"]
        source_hw = _integers(prep.get("source_hw"), 2, "source_hw", minimum=1)
        if (source_hw != [header.get("height"), header.get("width")]
                or header.get("opened") is not True or not isinstance(header.get("frame_count"), int)
                or header["frame_count"] < 17 or not math.isfinite(float(header.get("fps", 0))) or header["fps"] <= 0):
            raise ValueError("Native QA header geometry/FPS/frame count is not eligible")
        if (prep.get("source_frame_indices") != list(range(17)) or prep.get("num_frames") != 17
                or prep.get("temporal_stride") != 1 or prep.get("scale") != 4):
            raise ValueError("Native QA requires explicit original frames0..16 and exact x4 scale")
        hr_shape = _integers(prep.get("HR_shape"), 5, "HR_shape", minimum=1)
        lr_shape = _integers(prep.get("LR_shape"), 5, "LR_shape", minimum=1)
        h, w = hr_shape[-2:]
        if (hr_shape[:3] != [1, 3, 17] or h % 4 or w % 4
                or lr_shape != [1, 3, 17, h//4, w//4]):
            raise ValueError("Native QA HR/LR shapes do not define paired x4 RGB")
        full = plan["schema"] == "vsr_native_fullframe_operator_qa_predeclaration_v1"
        crop = [0, 0, *source_hw] if full else _integers(prep.get("native_center_crop_tlhw"), 4, "native crop")
        expected = [0, 0, *source_hw] if full else [(source_hw[0]-h)//2, (source_hw[1]-w)//2, h, w]
        if crop != expected or crop[-2:] != [h, w] or min(crop) < 0 or h > source_hw[0] or w > source_hw[1]:
            raise ValueError("Native QA must use the declared original whole frame or exact integer center crop")
        degradation = prep.get("LR_degradation", {})
        if (degradation.get("kernel") != "torch bicubic" or degradation.get("antialias") is not True
                or degradation.get("align_corners") is not False or degradation.get("clamp_01") is not True):
            raise ValueError("Native QA LR degradation must be the fixed antialiased bicubic x4 rule")
        observed_fps = float(prep.get("source_fps_observed", 0))
        if not math.isfinite(observed_fps) or not math.isclose(observed_fps, header["fps"], rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError("Native QA FPS declaration differs from the bound source header")
        return {"record": record, "source_hw": source_hw, "crop": crop, "height": h, "width": w,
                "header": copy.deepcopy(header), "selection_sha256": hashlib.sha256(selection_raw).hexdigest(),
                "inventory_sha256": hashlib.sha256(inventory_raw).hexdigest()}

    @staticmethod
    def _case_metadata(case):
        return {"plan": copy.deepcopy(case["plan"]), "plan_sha256": case["plan_sha256"],
                "declaration_path": case["declaration_path"], "declaration_file_sha256": case["declaration_file_sha256"],
                "source_hw": case["source_hw"], "crop_tlhw": case["crop"], "target_hw": [case["height"], case["width"]],
                "selection_sha256": case["selection_sha256"], "header_inventory_sha256": case["inventory_sha256"]}

    def policy(self, dataset):
        """Return one-case BenchmarkPolicy usable with reader factory=self class."""
        from .frozen_evaluation import BenchmarkPolicy
        if dataset not in self.records:
            raise KeyError(dataset)
        case = self._cases[dataset.rsplit("-", 1)[1]]
        return BenchmarkPolicy(family="hq", allowed_datasets=(dataset,), run_prefix="native_qa_"+case["mode"],
            manifest_relative="manifests/hq_vsr.json", manifest_hash_field="hq_manifest_file_sha256",
            reader_options={"plans": [copy.deepcopy(case["plan"])], "height": case["height"], "width": case["width"],
                            "split": "val", "temporal_stride": 1},
            freeze_reader_catalogue=True, preserve_reader_order=True,
            no_reference_primary_pool="all_frames", reference_primary_pool="matched_nonreference",
            selection_policy="predeclared own checkpoints for structural validation QA; no quality threshold or test selection",
            summary_scope=case["plan"]["scope"],
            primary_policy_description="all frames for no-reference; identical non-reference pool for HRI on/off; descriptive QA metrics only")

    def load(self, dataset, record_id, num_frames=17, start=0):
        if (not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames != 17
                or not isinstance(start, int) or isinstance(start, bool) or start != 0):
            raise ValueError("Native QA permits only the predeclared17-frame original prefix at start0")
        if dataset not in self.records or record_id not in self.records[dataset]:
            raise KeyError("Source is outside the predeclared native validation case")
        import cv2
        import numpy as np
        case = self._cases[dataset.rsplit("-", 1)[1]]
        record = case["record"]
        path = Path(record["path"]).resolve()
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("Raw source changed since the sealed native QA declaration")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("Cannot open the declared native QA source")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_hw = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))]
            if (not math.isfinite(fps) or fps <= 0 or not math.isclose(fps, case["header"]["fps"], rel_tol=1e-4, abs_tol=1e-5)
                    or not math.isfinite(count) or count != case["header"]["frame_count"] or actual_hw != case["source_hw"]):
                raise ValueError("Actual native source FPS/frame count/geometry differs from frozen header")
            frames = []
            top, left, h, w = case["crop"]
            for index in range(17):
                okay, bgr = capture.read()
                if not okay or bgr is None:
                    raise ValueError(f"Native QA source failed to decode original frame{index}")
                if bgr.dtype != np.uint8 or bgr.shape != (*case["source_hw"], 3):
                    raise ValueError("Native QA source changes RGB geometry or decoder dtype")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb[top:top+h, left:left+w].copy())
        finally:
            capture.release()
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError("Native QA source changed during decoding")
        hr = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).contiguous().float().div_(255)
        lr_frames = F.interpolate(hr[0].permute(1, 0, 2, 3), size=(h//4, w//4),
                                  mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        lr = lr_frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        if not bool(torch.isfinite(hr).all()) or not bool(torch.isfinite(lr).all()):
            raise ValueError("Native QA decoder/degradation produced nonfinite pixels")
        lr_hash, hr_hash = _tensor_sha256(lr), _tensor_sha256(hr)
        degradation = {"scale": 4, "kernel": "torch_bicubic", "antialias": True, "align_corners": False,
                       "clamp_01": True, "temporal_parameters": "shared_for_all_frames", "additional_noise": False}
        protocol_hash = _digest({"native_case": case["mode"], "plan_sha256": case["plan_sha256"], "degradation": degradation})
        cache_key = _digest({"manifest_sha256": self.manifest_sha256, "source_sha256": record["sha256"],
            "plan_sha256": case["plan_sha256"], "source_indices": list(range(17)), "crop_tlhw": case["crop"],
            "hr_rgb_f32_sha256": hr_hash, "lr_rgb_f32_sha256": lr_hash, "fps": fps})
        binding = {"manifest_sha256": self.manifest_sha256, "base_pair_cache_key": cache_key,
            "record_id": record_id, "robustness_mode": case["mode"], "noise_seed": 0,
            "robustness_protocol_sha256": protocol_hash, "base_lr_rgb_sha256": lr_hash, "robust_lr_rgb_sha256": lr_hash}
        metadata = {"dataset": dataset, "source_dataset": "HQ-VSR", "record_id": record_id, "sequence_id": record_id,
            "source_id": record["source_id"], "source_path": str(path), "source_sha256": record["sha256"], "split": "val",
            "manifest_sha256": self.manifest_sha256, "manifest_file_sha256": self.manifest_file_sha256,
            "source_frame_indices": list(range(17)), "source_indices": list(range(17)), "temporal_stride": 1,
            "original_fps": fps, "output_fps": fps, "fps_source": "actual original source FPS; no temporal resampling",
            "source_hw": case["source_hw"], "resize_hw": case["source_hw"], "HR_resized": False,
            "crop_tlhw": case["crop"], "prepared_lr_hw": [h//4, w//4], "prepared_hr_hw": [h, w],
            "native_case": case["mode"], "native_QA_scope": case["plan"]["scope"],
            "benchmark_prefix_explicit": True, "requested_num_frames": 17, "start": 0,
            "valid_region": "entire native-pixel crop, no padding", "degradation": degradation,
            "base_pair_cache_key": cache_key, "input_binding": binding, "native_plan_sha256": case["plan_sha256"],
            "hr_rgb_f32_sha256": hr_hash, "source_header_frame_count": case["header"]["frame_count"],
            "training_allowed": False, "parameters_selected_on_test": False, "metrics_computed": False,
            "checkpoint_selection_performed": False, "official_test_split": False}
        return {"lr": lr, "hr": hr, "fps": fps, "metadata": metadata, "cache_key": cache_key}

    def close(self):
        return None
