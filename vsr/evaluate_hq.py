"""Frozen internal HQ-VSR val/test and predeclared unseen-LR-stress evaluation.

This is the project's source-grouped internal holdout, not an official HQ-VSR
test benchmark. The test reader always exposes all 211 held-out clips; val uses
the fixed SHA-sorted 32. Checkpoints are explicit and never chosen using test
scores. Saved PNG/q255 measurements are distinct from training-time float RGB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .frozen_evaluation import ROOT, BenchmarkPolicy, run_frozen_evaluation
from .robustness import MODES


def _hq_reader(root, **options):
    from .hq_evaluation_data import HQEvaluationReader
    return HQEvaluationReader(root, **options)


def run_hq_evaluation(project_root, checkpoints, *, split="test", modes=MODES,
                      frames=9, height=128, width=224, temporal_stride=4,
                      seed=20261102, seeds=(42, 43), steps=20,
                      tile_hw=(256, 448), overlap=64, device="cuda", metric_device="cpu",
                      run_dir=None, caption="A high quality video.", preview_max_side=None,
                      selection_note=None, factories=None):
    """Freeze catalogue/config before load(); evaluate every selected record.

    No test-limit or arbitrary degradation parameter is accepted. Factory
    overrides are the same explicit synthetic-test interfaces as the shared
    external engine. Default real-reader construction decodes no video pixels.
    """
    if split not in ("test", "val"):
        raise ValueError("HQ evaluation accepts only val/test; train is forbidden")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes)-set(MODES):
        raise ValueError("Select distinct fixed HQ identity/blur/noise/compression/combined modes")
    for value, label in ((height, "height"), (width, "width"), (temporal_stride, "temporal_stride")):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if height % 4 or width % 4:
        raise ValueError("HQ paired target geometry must be divisible by x4")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("data seed must lie in [0,2**63)")
    datasets = tuple(f"HQ-{split}-{mode}" for mode in modes)
    reader_options = {"split": split, "val_limit": 32, "height": height, "width": width,
                      "temporal_stride": temporal_stride, "seed": seed, "modes": modes}
    policy = BenchmarkPolicy(family="hq", allowed_datasets=datasets, run_prefix=f"hq_{split}",
        manifest_relative="manifests/hq_vsr.json", manifest_hash_field="hq_manifest_file_sha256",
        reader_options=reader_options, freeze_reader_catalogue=True, preserve_reader_order=True,
        no_reference_primary_pool="all_frames", reference_primary_pool="matched_nonreference",
        selection_policy="caller checkpoint list fixed before pixels; validation may inform selection, internal test must never select checkpoints/parameters",
        summary_scope="fixed seeded HQ-VSR paired crops with recorded temporal stride; project-created source-grouped internal holdout, not official test",
        primary_policy_description="no-reference-trained: all_frames primary; reference-trained on/off: identical matched_nonreference primary; all groups additionally retain both pools")
    return run_frozen_evaluation(project_root, checkpoints, datasets=datasets, frames=frames,
        seeds=seeds, steps=steps, tile_hw=tile_hw, overlap=overlap, device=device,
        metric_device=metric_device, run_dir=run_dir, caption=caption,
        preview_max_side=preview_max_side, selection_note=selection_note, factories=factories,
        policy=policy, reader_factory=_hq_reader)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261102, help="data/window/robustness seed; sampling seeds are separate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tile", default="256x448")
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, help="resume only an identical frozen HQ evaluation run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default="cpu")
    parser.add_argument("--caption", default="A high quality video.")
    parser.add_argument("--preview-max-side", type=int, help="optional playback-only downscaled MP4; PNG remains authoritative")
    parser.add_argument("--selection-note", help="record checkpoint selection using internal validation only")
    return parser


def main():
    args = build_parser().parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("Use --checkpoint LABEL=PATH")
        label, path = value.split("=", 1)
        if label in checkpoints:
            raise ValueError("Checkpoint labels must be unique")
        checkpoints[label] = path
    if not re.fullmatch(r"\d+x\d+", args.tile):
        raise ValueError("--tile must be HEIGHTxWIDTH")
    result = run_hq_evaluation(args.project_root, checkpoints, split=args.split, modes=args.modes,
        frames=args.frames, height=args.height, width=args.width, temporal_stride=args.temporal_stride,
        seed=args.seed, seeds=args.seeds, steps=args.steps, tile_hw=tuple(map(int, args.tile.split("x"))),
        overlap=args.overlap, device=args.device, metric_device=args.metric_device, run_dir=args.run_dir,
        caption=args.caption, preview_max_side=args.preview_max_side, selection_note=args.selection_note)
    print(json.dumps({"status": result["status"], "run": result["run"], "protocol_sha256": result["protocol_sha256"]}))


if __name__ == "__main__":
    main()
