#!/usr/bin/env python3
"""Audit HQ-VSR header eligibility without decoding any video pixels.

The original sealed catalog and source-group assignments are read unchanged.
Eligibility indicates only sufficient header-reported temporal/spatial geometry;
it does not measure motion, quality, successful full decoding or model scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from vsr.data import _digest, validate_manifest


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def inspect_headers(catalog, *, frames, temporal_stride, height, width):
    """Read VideoCapture properties only; never call read/grab/retrieve."""
    required_frames = 1 + (frames - 1) * temporal_stride
    records = []
    for index, original in enumerate(catalog["records"]):
        path = Path(original["path"])
        reasons, header = [], {"opened": False, "fps": None, "frame_count": None,
                               "width": None, "height": None, "backend": None}
        actual_bytes = path.stat().st_size if path.is_file() else None
        if actual_bytes is None:
            reasons.append("source_file_missing")
        elif actual_bytes != original["bytes"]:
            reasons.append("source_file_size_differs_from_sealed_catalog")
        capture = None
        try:
            if actual_bytes is not None:
                capture = cv2.VideoCapture(str(path))
                header["opened"] = bool(capture.isOpened())
                if header["opened"]:
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    source_width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    source_height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    header["fps"] = fps if math.isfinite(fps) else None
                    header["frame_count"] = int(count) if math.isfinite(count) and count >= 0 and count.is_integer() else None
                    header["width"] = int(source_width) if math.isfinite(source_width) and source_width >= 0 and source_width.is_integer() else None
                    header["height"] = int(source_height) if math.isfinite(source_height) and source_height >= 0 and source_height.is_integer() else None
                    header["backend"] = capture.getBackendName()
                else:
                    reasons.append("video_header_cannot_open")
        except Exception as error:
            reasons.append("header_inspection_error:" + type(error).__name__)
        finally:
            if capture is not None:
                capture.release()
        if header["fps"] is None or header["fps"] <= 0:
            reasons.append("fps_not_positive_finite")
        if header["frame_count"] is None or header["frame_count"] < required_frames:
            reasons.append("insufficient_header_frame_count")
        if header["width"] is None or header["height"] is None or header["width"] < width or header["height"] < height:
            reasons.append("spatial_size_requires_hr_upscaling")
        records.append({"id": original["id"], "source_id": original["source_id"], "split": original["split"],
                        "path": original["path"], "raw_sha256": original["sha256"],
                        "raw_sha256_status": "copied_from_sealed_catalog_not_rehashed_in_header_only_audit",
                        "catalog_bytes": original["bytes"], "actual_bytes": actual_bytes,
                        "header": header, "eligible": not reasons, "ineligible_reasons": reasons})
        if (index + 1) % 250 == 0:
            print(f"Inspected {index+1}/{len(catalog['records'])} video headers", flush=True)
    return records


def build_inventory(catalog, catalog_path, catalog_file_sha, *, frames, temporal_stride, height, width):
    validate_manifest(catalog)
    records = inspect_headers(catalog, frames=frames, temporal_stride=temporal_stride, height=height, width=width)
    # Preserve record order and source assignments verbatim; filtering is a
    # future protocol operation, not a new split-allocation step.
    if [(r["id"], r["source_id"], r["split"], r["raw_sha256"]) for r in records] != [
        (r["id"], r["source_id"], r["split"], r["sha256"]) for r in catalog["records"]]:
        raise RuntimeError("Original catalog identities or split assignments changed")
    summaries = {}
    for split in ("train", "val", "test"):
        subset = [record for record in records if record["split"] == split]
        eligible = [record for record in subset if record["eligible"]]
        source_records = defaultdict(list)
        for record in subset:
            source_records[record["source_id"]].append(record)
        summaries[split] = {"catalog_clips": len(subset), "catalog_sources": len(source_records),
                            "eligible_clips": len(eligible), "ineligible_clips": len(subset) - len(eligible),
                            "eligible_sources": len({record["source_id"] for record in eligible}),
                            "fully_eligible_sources": sum(all(r["eligible"] for r in group) for group in source_records.values()),
                            "ineligible_reasons": dict(Counter(reason for r in subset for reason in r["ineligible_reasons"]))}
    payload = {"schema_version": 1, "kind": "hq_vsr_header_temporal_eligibility_inventory",
               "source_catalog": {"path": str(catalog_path), "manifest_sha256": catalog["manifest_sha256"],
                                  "file_sha256": catalog_file_sha, "source_group_assignments_unchanged": True},
               "settings": {"frames": frames, "temporal_stride": temporal_stride,
                            "required_source_frames": 1 + (frames - 1) * temporal_stride,
                            "height": height, "width": width, "allow_hr_upscale": False,
                            "fps_requirement": "positive finite original FPS header; inventory does not set output FPS"},
               "inspection": {"opencv_version": cv2.__version__, "method": "VideoCapture.isOpened/get properties/getBackendName only",
                              "pixel_data_decoded": False, "metrics_computed": False, "raw_video_hashes_recomputed": False,
                              "test_headers_inspected": True, "test_pixels_decoded": False,
                              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "limitations": ["Header frame counts can be estimates; later selected clips still require decode validation",
                                              "No motion/visibility/quality eligibility was inferred from headers",
                                              "No clip/source was reassigned between original splits"]},
               "catalog_record_count": len(records), "eligible_record_count": sum(r["eligible"] for r in records),
               "summary_by_original_split": summaries, "records": records}
    payload["manifest_sha256"] = _digest(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=positive_integer, default=9)
    parser.add_argument("--temporal-stride", type=positive_integer, default=4)
    parser.add_argument("--height", type=positive_integer, default=128)
    parser.add_argument("--width", type=positive_integer, default=224)
    args = parser.parse_args()
    manifest_path, output = args.manifest.resolve(), args.output.resolve()
    if manifest_path == output:
        parser.error("--output must not overwrite the original sealed catalog")
    started = time.perf_counter()
    original_bytes = manifest_path.read_bytes()
    catalog_file_sha = hashlib.sha256(original_bytes).hexdigest()
    catalog = json.loads(original_bytes)
    inventory = build_inventory(catalog, manifest_path, catalog_file_sha, frames=args.frames,
                                temporal_stride=args.temporal_stride, height=args.height, width=args.width)
    if manifest_path.read_bytes() != original_bytes:
        raise RuntimeError("Original catalog changed during the audit; no inventory written")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "inventory_sha256": inventory["manifest_sha256"],
                      "source_catalog_sha256": catalog["manifest_sha256"], "source_catalog_file_unchanged": True,
                      "summary_by_original_split": inventory["summary_by_original_split"],
                      "eligible_record_count": inventory["eligible_record_count"], "elapsed_seconds": time.perf_counter()-started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
