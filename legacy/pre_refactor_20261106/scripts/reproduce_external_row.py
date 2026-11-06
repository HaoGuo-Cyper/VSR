#!/usr/bin/env python3
"""Prepare, then execute one frozen external-row reproduction; never trains.

prepare uses only the standard library, hashes existing files, and creates a
fresh planned run. execute requires its exact protocol SHA and planned state.
No resume, retry, caption encoder fallback, or numerical-path substitution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
OLD = Path("/home/hugo/Projects/VSR")
OLD_CHILD = OLD / "runs/external_paired_frozen_20261103T125240.867596Z/external"
LABEL = "NoRef_Base_seed20261102_step8192"
ROW = OLD_CHILD / "results" / LABEL / "REDS4/000"
CHECKPOINT = OLD / "runs/formal_hq_v2_noref_20261102T142214.588571Z/Base/checkpoint_008192.pt"
CHECKPOINT_SHA = "8f3d1e19edc24b8ad72219ef207b85c4c798f241f6f204fac57e57c11960806c"
OLD_METRICS_SHA = "f1ebe19ccbe606990bb96dfd0857b093e482a34dbfb96fd5e404a9a86528684f"
HELPERS = {
    "external": ("work/run_frozen_external_T1_policy_desktop_v2.py", "bcdbf8ee9c741afc8db57064c67d8fcd3bb61702073310c287529d6b944095f6"),
    "hq": ("work/run_frozen_HQ_test_T1_policy.py", "6304f49d5a30ad0a26fd8ae5e8a52d782dd12ee5dd6148b87b099f03d93c50c7"),
    "policy": ("work/t1_vae_cache_policy.py", "d94340b33868912f43633c71ef4a34a97b2dd7eb7b74e9d40ded363b09417270"),
}
ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
POOLS = ("all_frames", "matched_nonreference", "reference_frame_separate")
ATOL, RTOL = 1e-10, 1e-8


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def binding(path, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    need(expected is None or value == expected, "Changed frozen file: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": value}


def runtime_versions(expected):
    actual = {name: importlib.metadata.version(name) for name in expected}
    need(actual == expected, "Runtime package versions differ from the historical row: " + str(actual))
    return actual


def safe_run(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    need(root != OLD and not root.is_relative_to(OLD), "Use the new project for all outputs")
    need(path != root / "runs" and path.is_relative_to(root / "runs"), "Run must be a child of the selected new project's runs/")
    need(not path.is_relative_to(OLD), "Historical assets are read-only")
    return path


def scalar_comparison(actual, expected, *, label):
    """Compare all scalar fields in every paired pool, including None/counts."""
    rows = []
    for key in ("kind", "primary_pool", "excluded_reference_index"):
        if key in expected:
            need(actual.get(key) == expected[key], f"{label}: metric metadata differs: {key}")
    for pool in POOLS:
        left, right = actual[pool], expected[pool]
        keys = {k for k, value in right.items() if not isinstance(value, (list, dict))}
        actual_keys = {k for k, value in left.items() if not isinstance(value, (list, dict))}
        need(keys == actual_keys, f"{label}/{pool}: scalar field set changed")
        for key in sorted(keys):
            a, b = left[key], right[key]
            numeric = type(a) in (int, float) and type(b) in (int, float)
            if numeric:
                passed = math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=ATOL, rel_tol=RTOL)
                delta = a - b
            else:
                passed, delta = type(a) is type(b) and a == b, None
            rows.append({"pool": pool, "metric": key, "actual": a, "expected": b,
                         "difference": delta, "passed": passed})
    return {"label": label, "passed": all(row["passed"] for row in rows),
            "atol": ATOL, "rtol": RTOL, "scalar_count": len(rows), "comparisons": rows}


def collect_protocol(root):
    """Only JSON, package metadata, stat and file hashing; no tensor imports."""
    root = Path(root).resolve()
    child = read(OLD_CHILD / "protocol.json")
    outer = read(OLD_CHILD.parent / "execution_policy.json")
    old_row, complete = read(ROW / "metrics.json"), read(ROW / "complete.json")
    need(sha(ROW / "metrics.json") == OLD_METRICS_SHA == complete["metrics_sha256"], "Historical row seal changed")
    contract = digest({"protocol": digest(child), "label": LABEL, "dataset": "REDS4", "id": "000"})
    need(old_row["status"] == "completed" and old_row["contract_sha256"] == complete["contract_sha256"] == contract,
         "Historical row is not a correctly sealed completed row")
    need(old_row["label"] == LABEL and old_row["dataset"] == "REDS4" and old_row["sequence_id"] == "000"
         and old_row["prefix_frames"] == 17 and old_row["checkpoint_sha256"] == CHECKPOINT_SHA,
         "Historical row identity changed")
    need(child["frames"] == 17 and child["start"] == 0 and child["seeds"] == [42, 43]
         and child["steps"] == 20 and child["tile_hw"] == [256, 448] and child["overlap"] == 64
         and child["caption"] == "A high quality video." and child["metric_device"] == "cpu",
         "Historical child numerical protocol changed")
    spec = child["checkpoints"][LABEL]
    need(spec["path"] == str(CHECKPOINT) and spec["checkpoint_sha256"] == CHECKPOINT_SHA
         and spec["variant"] == "Base" and spec["step"] == 8192
         and spec["config"]["reproducibility"]["mode"] == "strict", "Historical checkpoint specification changed")
    code = child["code_sha256"]
    need(set(code) == {path.name for path in (root / "vsr").glob("*.py")}, "Current core module set changed")
    source_files = {str(root / "vsr" / name): value for name, value in code.items()}
    need(all(sha(path) == expected for path, expected in source_files.items()), "Current production core differs from historical evaluator")
    helper_specs = {name: binding(root / "scripts/frozen_external" / relative, expected)
                    for name, (relative, expected) in HELPERS.items()}
    need(outer["worker_sha256"] == helper_specs["external"]["sha256"]
         and outer["policy_sha256"] == helper_specs["policy"]["sha256"], "Archived helper bindings disagree with actual execution")
    metadata_files = {}
    for key in ("manifest", "stream_metadata", "provenance"):
        item = outer["root_protocol"]["external_native_prefix"][key]
        metadata_files[key] = binding(item["path"], item["sha256"])
    vae_source = outer["root_protocol"]["bindings"]["Wan_VAE_source"]
    metadata_files["Wan_VAE_source"] = binding(vae_source["path"], vae_source["sha256"])
    caption_doc = outer["root_protocol"]["bindings"]["caption_preflight"]
    metadata_files["caption_preflight"] = binding(caption_doc["path"], caption_doc["sha256"])
    caption = read(caption_doc["path"])["entries"][LABEL]
    need(caption["checkpoint_path"] == str(CHECKPOINT) and caption["checkpoint_sha256"] == CHECKPOINT_SHA,
         "Caption preflight belongs to a different checkpoint")
    caption_binding = binding(caption["cache_path"], caption["cache_file_sha256"])
    config_binding = binding(caption["config_path"], caption["config_sha256"])
    need(read(config_binding["path"]) == spec["config"], "Originating checkpoint config changed")
    model_manifest = OLD / "docs/official-model-manifest.json"
    manifest = read(model_manifest)
    need(manifest["sha"] == spec["config"]["model"]["revision"], "Official model revision changed")
    official_files = []
    model_root = Path(spec["config"]["model"]["local_dir"])
    for item in manifest["siblings"]:
        relative = item["rfilename"]
        if not relative.startswith(("transformer/", "vae/", "scheduler/")):
            continue
        path = model_root / relative
        if "lfs" in item:
            official_files.append(binding(path, item["lfs"]["sha256"]))
        else:
            data = path.read_bytes()
            git_sha = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            need(git_sha == item["blobId"], "Official model configuration/blob changed: " + relative)
            official_files.append(binding(path))
    metric_files = [binding(OLD / "weights/alexnet-owt-7be5be79.pth", child["metric_versions"]["lpips_alexnet_sha256"]),
                    binding(OLD / "weights/lpips_v0.1_alex.pth", child["metric_versions"]["lpips_linear_sha256"])]
    entries = old_row["modes"]["no_reference"]["seeds"]
    need([entry["seed"] for entry in entries] == [42, 43], "Historical seed order differs")
    reference_pngs = {}
    by_path = {item["path"]: item for item in old_row["artifacts"]}
    for entry in entries:
        expected = []
        for index in range(17):
            relative = entry["png_directory"] + f"/frame_{index:08d}.png"
            item = by_path[relative]
            need((OLD_CHILD / relative).resolve().is_relative_to(OLD_CHILD), "Reference PNG escaped child run")
            actual = binding(OLD_CHILD / relative, item["sha256"])
            need(actual["bytes"] == item["bytes"], "Historical PNG byte count changed")
            expected.append({"frame": index, "bytes": actual["bytes"], "sha256": actual["sha256"]})
        reference_pngs[str(entry["seed"])] = expected
    need(old_row["source"]["source_frame_indices"] == list(range(17))
         and old_row["source"]["native_lr_hw"] == [180, 320]
         and old_row["source"]["native_hr_hw"] == [720, 1280], "Original row native geometry changed")
    deps = root / ".deps"
    need(deps.resolve() == (OLD / ".deps").resolve(), "Use the same pinned local dependency directory")
    original_path = list(sys.path)
    try:
        # The unchanged production frozen_evaluation module prepends this same
        # .deps directory. Metadata discovery must use its LPIPS installation.
        sys.path.insert(0, str(deps))
        runtime_versions(child["runtime_versions"])
    finally:
        sys.path[:] = original_path
    return {
        "schema": "vsr_single_external_row_reproduction_v1", "project_root": str(root),
        "harness": binding(__file__), "old_project_root": str(OLD), "old_child_run": str(OLD_CHILD),
        "reference_files": {"metrics": binding(ROW / "metrics.json", OLD_METRICS_SHA),
                            "complete": binding(ROW / "complete.json"), "child_protocol": binding(OLD_CHILD / "protocol.json"),
                            "execution_policy": binding(OLD_CHILD.parent / "execution_policy.json"),
                            "official_model_manifest": binding(model_manifest)},
        "reference_row_contract_sha256": contract, "historical_child_protocol_digest": digest(child),
        "checkpoint": binding(CHECKPOINT, CHECKPOINT_SHA), "checkpoint_spec": spec,
        "caption": {"text": child["caption"], "cache": caption_binding, "config": config_binding,
                    "embedding_sha256": caption["embedding_sha256"], "shape": caption["shape"],
                    "dtype": caption["dtype"], "T5_fallback": False},
        "scope": {"label": LABEL, "dataset": "REDS4", "sequence_id": "000", "frames": 17,
                  "start": 0, "temporal_stride": 1, "seeds": [42, 43], "steps": 20,
                  "tile_hw": [256, 448], "overlap": 64, "cfg": 1.0, "reference_image": None,
                  "native_LR_hw": [180, 320], "native_HR_hw": [720, 1280], "tiles_per_restore": 16},
        "production_code_sha256": source_files, "helper_bindings": helper_specs,
        "metadata_bindings": metadata_files, "official_files": official_files, "metric_weights": metric_files,
        "runtime_versions": child["runtime_versions"], "runtime_environment": ENVIRONMENT,
        "runtime_flags_expected": outer["runtime_before_GPU_load"]["flags"],
        "VAE_policy_metadata": {"phase": "paired", "records": [{"LR_hw": [180, 320], "HR_hw": [720, 1280], "tile_count": 16}]},
        "VAE_lifecycle": outer["lifecycle"], "reference_pngs": reference_pngs,
        "pass_rule": {"all_34_saved_PNG_files_byte_identical": True, "all_pool_scalar_metrics": True,
                      "include_bicubic_baseline_and_seed_mean": True, "metric_atol": ATOL, "metric_rtol": RTOL,
                      "seed42_failure_stops_before_seed43": True, "retries": 0},
        "timing_budget": {"expected_restore_seconds_per_seed": 120, "expected_metric_seconds_per_seed": 11,
                          "expected_total_minutes_approx": 5, "kind": "historical estimate, not a timeout or guarantee"},
        "training_updates": 0, "parameters_selected_on_test": False,
    }


def import_helper(path, name):
    # Archived entry points insert OLD into sys.path at import. Restore our
    # current-core search path; reuse only their exact class/ledger/policy code.
    original = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original


def prepare(args):
    started = time.perf_counter()
    root = args.project_root.resolve()
    run = safe_run(args.run_dir, root)
    need(not run.exists() or not any(run.iterdir()), "Prepare requires a new empty run; preserve existing runs")
    protocol = collect_protocol(root)
    need("torch" not in sys.modules, "prepare must not import Torch or initialize CUDA")
    run.mkdir(parents=True, exist_ok=True)
    save(run / "protocol.json", protocol)
    for name in ("metrics.json", "complete.json"):
        (run / ("reference_" + name)).write_bytes((ROW / name).read_bytes())
    save(run / "preparation.json", {"status": "planned", "CUDA_initialized": False, "torch_imported": False,
         "model_or_checkpoint_tensor_loaded": False, "source_pixels_decoded": False,
         "file_hashes_verified": True, "cpu_wall_seconds": time.perf_counter() - started,
         "created_utc": datetime.now(timezone.utc).isoformat(), "training_updates": 0})
    protocol_sha = sha(run / "protocol.json")
    save(run / "status.json", {"status": "planned", "protocol_sha256": protocol_sha, "training_updates": 0})
    print(json.dumps({"status": "planned", "run": str(run), "protocol_sha256": protocol_sha}))


def execute(args):
    run = safe_run(args.run_dir, args.project_root)
    protocol_path = run / "protocol.json"
    protocol = read(protocol_path)
    need(args.expected_protocol_sha and sha(protocol_path) == args.expected_protocol_sha,
         "Execution requires the exact prepared protocol raw-file SHA")
    state = read(run / "status.json")
    need(state["status"] == "planned" and state["protocol_sha256"] == args.expected_protocol_sha,
         "Execute accepts only a never-started planned run; no automatic retry or resume")
    save(run / "status.json", {"status": "running", "protocol_sha256": args.expected_protocol_sha,
                              "started_utc": datetime.now(timezone.utc).isoformat(), "training_updates": 0})
    started, stages, seed_results = time.perf_counter(), {}, []
    current_stage, wrapper, vae, model, reader, torch = "pre_GPU_binding_checks", None, None, None, None, None
    try:
        need(protocol["project_root"] == str(args.project_root.resolve()) and protocol == collect_protocol(args.project_root),
             "Prepared source, assets, runtime or harness changed; preserve run and prepare a new one")
        need(all(os.environ.get(key) == value for key, value in ENVIRONMENT.items()), "Launch with the exact frozen environment")
        need(sha(run / "reference_metrics.json") == OLD_METRICS_SHA
             and sha(run / "reference_complete.json") == protocol["reference_files"]["complete"]["sha256"], "Reference snapshot changed")
        sys.path.insert(0, protocol["project_root"])
        import torch as torch_module
        torch = torch_module
        need(not torch.cuda.is_initialized(), "Strict runtime must be configured before CUDA initialization")
        torch.set_num_threads(2)
        from vsr.reproducibility import configure_reproducibility, reproducibility_flags
        from vsr.external_data import ExternalReader
        from vsr.checkpoints import load_training_checkpoint
        from vsr.infer_video import load_frozen_vae
        from vsr.inference import restore_video
        from vsr.frozen_evaluation import quantize_rgb, _save_pngs, _average_scores
        from vsr.metric_pools import paired_metrics
        from vsr.perceptual_metrics import PerceptualMetrics
        from vsr.native_qa_restore import _contexts_empty
        import torch.nn.functional as F
        import inspect
        from diffusers import AutoencoderKLWan
        actual_vae_source = inspect.getfile(AutoencoderKLWan)
        need(sha(actual_vae_source) == protocol["metadata_bindings"]["Wan_VAE_source"]["sha256"], "Imported Wan VAE source differs")
        for name, module in list(sys.modules.items()):
            if name == "vsr" or name.startswith("vsr."):
                path = Path(module.__file__).resolve()
                need(path.is_relative_to(Path(protocol["project_root"]) / "vsr"), "Historical core shadowed current core: " + str(path))
        cfg = protocol["checkpoint_spec"]["config"]
        runtime = configure_reproducibility(cfg)
        expected_flags = protocol["runtime_flags_expected"]
        need(all(runtime["flags"][key] == value for key, value in expected_flags.items()),
             "Strict runtime flags differ from historical run")
        save(run / "runtime.json", {"runtime": runtime, "actual_VAE_source": actual_vae_source})
        helper = {key: import_helper(item["path"], "row_reproduction_" + key)
                  for key, item in protocol["helper_bindings"].items()}
        wrapper = helper["external"].RotatingExternalRestore(restore_video, helper["policy"].install,
            helper["hq"].validate_ledger, protocol["VAE_policy_metadata"],
            finite=lambda value: bool(torch.isfinite(value).all()), contexts_empty=_contexts_empty)

        @contextmanager
        def phase(name, cuda=False):
            nonlocal current_stage
            current_stage = name
            if cuda:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            before = time.perf_counter()
            observation = {"status": "running"}
            stages[name] = observation
            try:
                yield
                if cuda:
                    torch.cuda.synchronize()
                observation["status"] = "completed"
            except BaseException:
                observation["status"] = "failed"
                raise
            finally:
                observation["wall_seconds"] = time.perf_counter() - before
                observation["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
                observation["peak_scope"] = "phase peak including already resident weights" if cuda else "running peak, not reset for CPU phase"
                save(run / "timings.json", stages)

        old_row = read(run / "reference_metrics.json")
        with phase("reader_metadata"):
            reader = ExternalReader(OLD, preview_fps=30.0)
        with phase("cached_caption"):
            cache = torch.load(protocol["caption"]["cache"]["path"], map_location="cpu", weights_only=True)
            text = cache.get(protocol["caption"]["text"]) if isinstance(cache, dict) else None
            need(isinstance(text, torch.Tensor) and list(text.shape) == protocol["caption"]["shape"]
                 and str(text.dtype) == protocol["caption"]["dtype"] and bool(torch.isfinite(text).all()), "Invalid cached text; T5 fallback forbidden")
            header = json.dumps([str(text.dtype), list(text.shape)], separators=(",", ":")).encode() + b"\n"
            actual = hashlib.sha256(header + text.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            need(actual == protocol["caption"]["embedding_sha256"], "Cached embedding tensor digest changed")
            text = text.detach().cpu()
            del cache
        with phase("model_load", cuda=True):
            model, audit = load_training_checkpoint(CHECKPOINT, device="cuda", allowed_root=OLD / "runs")
            need(audit["checkpoint_sha256"] == CHECKPOINT_SHA and audit["config"] == cfg
                 and audit["step"] == 8192 and audit["variant"] == "Base", "Loaded checkpoint identity changed")
            save(run / "checkpoint_load_audit.json", audit)
        with phase("VAE_load_register", cuda=True):
            vae = load_frozen_vae(cfg, device="cuda")
            need(all(p.device.type == "cuda" and p.dtype == torch.float32 for p in vae.parameters()), "Original CUDA/FP32 VAE required")
            wrapper.register(vae, cfg)
            model.eval(); vae.requires_grad_(False).eval()
        with phase("original_source_pixels"):
            source = reader.load("REDS4", "000", num_frames=17, start=0)
            need(source["metadata"] == old_row["source"], "ExternalReader source metadata differs from original row")
            lr, hr = source["lr"], source["hr"]
            need(list(lr.shape) == [1, 3, 17, 180, 320] and list(hr.shape) == [1, 3, 17, 720, 1280], "Source geometry changed")
            _, measured_hr = quantize_rgb(hr)
        with phase("baseline_metrics"):
            perceptual = PerceptualMetrics(protocol["metric_weights"][0]["path"], protocol["metric_weights"][1]["path"], device="cpu")
            up = F.interpolate(lr[0].permute(1, 0, 2, 3).float().cpu(), size=(720, 1280), mode="bicubic",
                               align_corners=False, antialias=True).clamp(0, 1)
            _, baseline = quantize_rgb(up.permute(1, 0, 2, 3).unsqueeze(0))
            baseline_metrics = paired_metrics(baseline, measured_hr, perceptual, excluded_reference=0)
            baseline_metrics["primary_pool"] = old_row["baseline"]["primary_pool"]
            comparison = scalar_comparison(baseline_metrics, old_row["baseline"], label="baseline")
            save(run / "baseline_metrics.json", baseline_metrics)
            save(run / "baseline_comparison.json", comparison)
            need(comparison["passed"], "Bicubic baseline metric reproduction failed; stop before sampling")
            del baseline, up
        expected_entries = old_row["modes"]["no_reference"]["seeds"]
        for seed, expected in zip((42, 43), expected_entries):
            need(seed == expected["seed"], "Historical seed order changed")
            with phase(f"seed_{seed}_restore", cuda=True):
                video, restoration = wrapper(model, vae, lr, text, cfg, steps=20, seed=seed,
                    tile_hw=(256, 448), overlap=64, reference_image=None, reference_index=None)
                save(run / f"seed_{seed}_restoration.json", restoration)
            with phase(f"seed_{seed}_PNG"):
                quantized, measured = quantize_rgb(video)
                artifacts = _save_pngs(quantized, run / f"seed_{seed}/png", run)
                expected_pngs = protocol["reference_pngs"][str(seed)]
                png_comparisons = [{"frame": i, "actual": actual, "expected": reference,
                                    "passed": actual["sha256"] == reference["sha256"] and actual["bytes"] == reference["bytes"]}
                                   for i, (actual, reference) in enumerate(zip(artifacts, expected_pngs))]
                png_passed = len(artifacts) == 17 and len(png_comparisons) == 17 and all(x["passed"] for x in png_comparisons)
                save(run / f"seed_{seed}_PNG_comparison.json", {"passed": png_passed, "frames": png_comparisons})
                need(png_passed, f"Seed {seed} PNG byte mismatch; no subsequent seed will run")
            with phase(f"seed_{seed}_metrics"):
                metrics = paired_metrics(measured, measured_hr, perceptual, excluded_reference=0)
                metrics["primary_pool"] = expected["metrics"]["primary_pool"]
                comparison = scalar_comparison(metrics, expected["metrics"], label=f"seed_{seed}")
                save(run / f"seed_{seed}_metrics.json", metrics)
                save(run / f"seed_{seed}_metric_comparison.json", comparison)
                need(comparison["passed"], f"Seed {seed} scalar metric mismatch; no subsequent seed will run")
            seed_results.append({"seed": seed, "PNG_count": 17, "PNG_byte_identical": True,
                                 "scalar_comparisons_passed": comparison["scalar_count"], "metrics": metrics})
            save(run / "progress.json", {"status": "running", "completed_seeds": [x["seed"] for x in seed_results], "training_updates": 0})
            del video, quantized, measured
        with phase("scope_completion"):
            mean = _average_scores([x["metrics"] for x in seed_results])
            mean_comparison = scalar_comparison(mean, old_row["modes"]["no_reference"]["mean"], label="seed_mean")
            save(run / "seed_mean_comparison.json", mean_comparison)
            need(mean_comparison["passed"], "Seed-mean scalar metrics differ")
            role = wrapper.roles[0]
            need(len(wrapper.roles) == 1 and role["restores"] == role["rotations"] == 2
                 and role["T1"] == 0 and role["original_delegate"] == role["events"] == 32,
                 "Single-row policy ledger budget differs")
            need(_contexts_empty(model), "Model forward context leaked")
            vae._vsr_t1_cache_policy.remove()
            need(wrapper.removed(vae), "Final policy removal failed")
            need(sha(protocol_path) == args.expected_protocol_sha
                 and sha(__file__) == protocol["harness"]["sha256"]
                 and all(sha(path) == expected for path, expected in protocol["production_code_sha256"].items())
                 and all(sha(item["path"]) == item["sha256"] for item in protocol["helper_bindings"].values())
                 and sha(ROW / "metrics.json") == OLD_METRICS_SHA
                 and sha(ROW / "complete.json") == protocol["reference_files"]["complete"]["sha256"],
                 "Frozen source or original row changed during reproduction")
            save(run / "policy_role_counters.json", role)
        with phase("release_model_VAE_and_reader", cuda=True):
            reader.close(); reader = None
            model = vae = None
            gc.collect(); torch.cuda.empty_cache()
            need(all(ref() is None for ref in wrapper.refs), "Single-row VAE leaked after release")
        summary = {"status": "passed", "protocol_sha256": args.expected_protocol_sha,
                   "training_updates": 0, "checkpoint": str(CHECKPOINT), "dataset": "REDS4", "sequence_id": "000",
                   "seeds": [42, 43], "PNG_files_byte_identical": 34, "all_pool_scalar_metrics_passed": True,
                   "baseline_and_seed_mean_passed": True, "metric_atol": ATOL, "metric_rtol": RTOL,
                   "policy_role_counters": role, "timings": stages, "wall_seconds": time.perf_counter() - started,
                   "runtime_flags": reproducibility_flags(), "quality_conclusion": None,
                   "scope": "one original external row only; does not complete external evaluation"}
        save(run / "summary.json", summary)
        save(run / "status.json", {"status": "passed", "protocol_sha256": args.expected_protocol_sha,
                                  "summary_sha256": sha(run / "summary.json"), "training_updates": 0})
        print(json.dumps({"status": "passed", "run": str(run), "protocol_sha256": args.expected_protocol_sha}))
    except BaseException as error:
        failure = {"status": "failed", "stage": current_stage, "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc(), "completed_seeds": [x["seed"] for x in seed_results],
                   "last_policy_audit": wrapper.last_audit if wrapper else None,
                   "policy_roles": wrapper.roles if wrapper else [], "timings": stages,
                   "wall_seconds": time.perf_counter() - started, "automatic_retry": False, "training_updates": 0}
        save(run / "failure.json", failure)
        save(run / "status.json", {"status": "failed", "protocol_sha256": args.expected_protocol_sha,
                                  "stage": current_stage, "failure_sha256": sha(run / "failure.json"), "training_updates": 0})
        raise
    finally:
        if vae is not None:
            policy = getattr(vae, "_vsr_t1_cache_policy", None)
            if policy is not None:
                policy.remove()
        if reader is not None:
            reader.close()
        model = vae = wrapper = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        need(args.expected_protocol_sha is None, "prepare creates a new protocol; do not supply an expected SHA")
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
