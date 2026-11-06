"""Frozen external evaluation CLI; shared engine preserves native PNG protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .frozen_evaluation import (ROOT, ALLOWED_DATASETS, quantize_rgb, paired_metrics,
                                run_frozen_evaluation)


def run_external_evaluation(project_root, checkpoints, *, datasets=ALLOWED_DATASETS, frames=17,
                            seeds=(42, 43), steps=20, tile_hw=(256, 448), overlap=64,
                            device="cuda", metric_device="cpu", run_dir=None, preview_fps=30.0,
                            caption="A high quality video.", preview_max_side=None,
                            selection_note=None, factories=None):
    """Original public external API; all dataset names/defaults remain unchanged."""
    return run_frozen_evaluation(project_root, checkpoints, datasets=datasets, frames=frames,
        seeds=seeds, steps=steps, tile_hw=tile_hw, overlap=overlap, device=device,
        metric_device=metric_device, run_dir=run_dir, preview_fps=preview_fps,
        caption=caption, preview_max_side=preview_max_side, selection_note=selection_note,
        factories=factories)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--datasets", nargs="+", choices=ALLOWED_DATASETS, default=list(ALLOWED_DATASETS))
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tile", default="256x448", help="native HR tile height x width")
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, help="existing own-run directory to resume with the identical protocol")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metric-device", default="cpu")
    parser.add_argument("--preview-fps", type=float, default=30.0, help="REDS4 preview FPS only; original PNG FPS unknown")
    parser.add_argument("--caption", default="A high quality video.")
    parser.add_argument("--preview-max-side", type=int, help="optional explicitly downscaled MP4, never used for metrics")
    parser.add_argument("--selection-note", help="record prior internal-validation-only checkpoint selection evidence")
    return parser


def main():
    args = build_parser().parse_args()
    checkpoints = {}
    for entry in args.checkpoint:
        if "=" not in entry:
            raise ValueError("Use --checkpoint LABEL=PATH")
        label, path = entry.split("=", 1)
        if label in checkpoints:
            raise ValueError("Checkpoint labels must be unique")
        checkpoints[label] = path
    if not re.fullmatch(r"\d+x\d+", args.tile):
        raise ValueError("--tile must be HEIGHTxWIDTH")
    summary = run_external_evaluation(args.project_root, checkpoints, datasets=args.datasets, frames=args.frames,
        seeds=args.seeds, steps=args.steps, tile_hw=tuple(map(int, args.tile.split("x"))), overlap=args.overlap,
        device=args.device, metric_device=args.metric_device, run_dir=args.run_dir, preview_fps=args.preview_fps,
        caption=args.caption, preview_max_side=args.preview_max_side, selection_note=args.selection_note)
    print(json.dumps({"status": summary["status"], "run": summary["run"], "protocol_sha256": summary["protocol_sha256"]}))


if __name__ == "__main__":
    main()
