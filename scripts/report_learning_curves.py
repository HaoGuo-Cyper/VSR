"""Plot fixed-seed paired validation curves from own SHA-bound saved scores."""
from pathlib import Path
import argparse
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vsr-matplotlib")

from vsr.audit import dump, file_hash
from vsr.summarize_saved import load_saved_result, METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, help="LABEL=run/variant directory")
    parser.add_argument("--steps", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True, help="New prefix for JSON/PNG/PDF")
    args = parser.parse_args()
    if args.seed < 0 or not args.steps or args.steps != sorted(set(args.steps)) or min(args.steps) <= 0:
        raise ValueError("Use one nonnegative fixed seed and increasing positive checkpoints")
    targets = [args.output.with_suffix(extension) for extension in (".json", ".png", ".pdf")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Preserve existing plots; choose a new prefix")
    methods, reference_bindings, record_set = {}, None, None
    for item in args.method:
        label, text = item.split("=", 1); directory = Path(text).resolve()
        if not label or label in methods or not directory.is_relative_to(ROOT / "runs"):
            raise ValueError("Each label must be unique and inputs must be own VSR experiments")
        costs = {}
        with (directory / "training.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["step"] in args.steps:
                    value = row["optimizer_updates_seconds"]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("Update cost must be a positive finite observation")
                    costs[row["step"]] = value
        points = []
        for step in args.steps:
            result = load_saved_result(directory / f"step_{step:06d}")
            rows = [row for row in result["observations"] if row["seed"] == args.seed]
            keys = {(row["record_id"], row["source_id"]) for row in rows}
            if not rows or len(keys) != len(rows):
                raise ValueError("Every video requires the selected predeclared seed exactly once")
            if reference_bindings is None:
                reference_bindings, record_set = result["bindings"], keys
            elif result["bindings"] != reference_bindings or keys != record_set:
                raise ValueError("Curve points changed videos, crop, degradation, FPS or source identity")
            points.append({"step": step, "sampling_seed": args.seed, "video_count": len(rows),
                "metrics": {key: sum(row["metrics"][key] for row in rows)/len(rows) for key in METRICS},
                "optimizer_update_seconds": costs[step], "optimizer_update_GPU_hours": costs[step]/3600,
                "provenance": result["provenance"]})
        methods[label] = points
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ("PSNR (dB), higher", "SSIM, higher", "LPIPS, lower", "tLP100, lower", "tOF, lower")
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 7.1), constrained_layout=True)
    for label, points in methods.items():
        steps = [point["step"] for point in points]
        for axis, key, title in zip(axes.flat, METRICS, labels):
            axis.plot(steps, [point["metrics"][key] for point in points], marker="o", label=label)
            axis.set_title(title, fontsize=11)
        axes.flat[5].plot(steps, [point["optimizer_update_GPU_hours"] for point in points], marker="o", label=label)
    axes.flat[5].set_title("Optimizer update hours (excludes preparation)", fontsize=10)
    for axis in axes.flat:
        axis.set_xlabel("Optimizer updates"); axis.grid(alpha=.25)
    axes.flat[0].legend(fontsize=9)
    figure.suptitle(f"HQ-VSR fixed validation / sampling seed {args.seed} only / {len(record_set)} videos\n"
                   "Descriptive single-training-run curves; final quality tables average all preset seeds", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(targets[1], dpi=160); figure.savefig(targets[2]); plt.close(figure)
    dump(targets[0], {"sampling_seed": args.seed, "steps": args.steps, "methods": methods,
        "video_count": len(record_set), "aggregation": "same fixed sampling seed, frame/pair mean per video then equal video weights",
        "training_repeat_uncertainty_included": False, "confidence_intervals_plotted": False,
        "cost_scope": "synchronized optimizer-update blocks only; excludes target preparation, loads and evaluation",
        "final_quality_protocol": "separate final tables use all preset seeds; no best-seed selection",
        "matplotlib_version": matplotlib.__version__, "script_sha256": file_hash(Path(__file__))})
    print(json.dumps({"status": "completed", "outputs": [str(path) for path in targets]}))


if __name__ == "__main__":
    main()
