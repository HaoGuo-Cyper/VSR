"""Frozen REDS4/000 full-resolution VAE encoding comparison; no DiT/training.

The two encode paths share an untiled FP32 decoder. Preparation only inspects
metadata/files. Execution stops on OOM/failure, preserves the attempt, and never
changes precision, geometry, device or tiling to retry automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import audit_vae_reconstruction as common

CODE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vsr_vae_full_resolution_encode_comparison_v1"
HEIGHT, WIDTH, FRAMES = 720, 1280, 9
TILING = {"tile_sample_min_height": 256, "tile_sample_min_width": 448,
          "tile_sample_stride_height": 192, "tile_sample_stride_width": 384}


def prepare(args):
    root = args.read_root.resolve(strict=True)
    config_path = root / "configs/formal_hq_v2_noref.json"
    config = common.read_json(config_path)
    manifest_path = root / "manifests/external.json"
    manifest = common.read_json(manifest_path)
    row = next(x for x in manifest["datasets"]["REDS4"]["records"] if x["id"] == "000")
    gt = row["gt"]
    common.require([gt["height"], gt["width"]] == [HEIGHT, WIDTH] and row["frame_count"] >= FRAMES, "Expected native REDS4/000 720x1280 and >=9 frames")
    images = []
    for ordinal, item in enumerate(gt["frames"][:FRAMES]):
        actual = common.binding(common.relative_input(root, item["path"]))
        common.require(actual["bytes"] == item["bytes"] and actual["sha256"] == item["sha256"], "Original REDS PNG bytes changed")
        images.append({"ordinal": ordinal, "source_name": Path(item["path"]).name, "file": actual})
    official_path = root / "docs/official-model-manifest.json"
    official = common.read_json(official_path)
    common.require(official["id"] == config["model"]["repo_id"] and official["sha"] == config["model"]["revision"], "Official model identity mismatch")
    catalogue = {x["rfilename"]: x for x in official["siblings"]}
    model_root = Path(config["model"]["local_dir"]).resolve(strict=True)
    vae_config_path = model_root / "vae/config.json"
    raw_config = vae_config_path.read_bytes()
    common.require(hashlib.sha1(b"blob " + str(len(raw_config)).encode() + b"\0" + raw_config).hexdigest() == catalogue["vae/config.json"]["blobId"], "VAE config differs from official pinned blob")
    vae_config = json.loads(raw_config)
    vae_weight = common.binding(model_root / "vae/diffusion_pytorch_model.safetensors")
    common.require(vae_weight["sha256"] == catalogue["vae/diffusion_pytorch_model.safetensors"]["lfs"]["sha256"], "VAE weights differ from official pinned LFS hash")
    frozen = common.read_json(root / "docs/final_evaluation_v1/protocol.json")
    vae_source = common.binding(frozen["bindings"]["Wan_VAE_source"]["path"])
    common.require(vae_source["sha256"] == frozen["bindings"]["Wan_VAE_source"]["sha256"], "VAE implementation source differs")
    metric_weights = []
    for name, digest in common.LPIPS_HASHES.items():
        item = common.binding(root / "weights" / name)
        common.require(item["sha256"] == digest, "Pinned metric weights differ")
        metric_weights.append(item)
    sources = {name: common.sha(CODE_ROOT / name) for name in (*common.CORE_FILES, "scripts/audit_vae_reconstruction.py", "scripts/audit_vae_full_resolution.py")}
    protocol = {
        "schema": SCHEMA, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "preparation": "file_hashes_and_metadata_only_no_torch_no_pixels_no_GPU", "read_root": str(root),
        "run_root": str(args.run_root.resolve()), "scope": "REDS4_000_T9_stride1_native_encoding_only_comparison",
        "case": {"dataset": "REDS4", "record_id": "000", "frames": FRAMES, "stride": 1, "source_start": 0,
                 "source_indices": list(range(FRAMES)), "source_hw": [HEIGHT, WIDTH],
                 "hr_shape": [1, 3, FRAMES, HEIGHT, WIDTH], "raw_latent_shape": [1, 16, 3, HEIGHT // 8, WIDTH // 8],
                 "source_role": "GT", "resize": False, "crop": False, "pad": False,
                 "frames_bound": images, "preview_fps": 30.0, "capture_timing_known": False},
        "model": {"local_dir": str(model_root), "repo_id": config["model"]["repo_id"], "revision": config["model"]["revision"],
                  "class": "AutoencoderKLWan_only", "dtype": "float32", "device": "cuda", "eval": True,
                  "posterior": "mode_no_sampling", "latents_mean": vae_config["latents_mean"], "latents_std": vae_config["latents_std"],
                  "DiT_loaded": False, "T5_loaded": False, "optimizer_updates": 0},
        "paths": [{"id": "untiled_encode", "encoder_tiled": False, "decoder_tiled": False},
                  {"id": "native_tiled_encode", "encoder_tiled": True, "encoder_tile_parameters": TILING, "decoder_tiled": False}],
        "boundary_diagnostics": {"outer_border_pixels": 16, "seam_band_half_width_pixels": 16,
                                 "seam_y": list(range(192, HEIGHT, 192)), "seam_x": list(range(384, WIDTH, 384)),
                                 "regions": ["all_pixels", "outer_border", "tile_start_bands", "interior_excluding_border_and_bands"],
                                 "mask_rule": "tile-start bands [start-16,start+16), clipped to image; intersection with outer border retained in band region; no pixel excluded from all_pixels"},
        "metrics": {"pixel_contract": "float32_RGB01_spatial; historical TecoGAN uint8 conversion temporal",
                    "implementations": "existing rgb_metrics + PerceptualMetrics CPU, all9 frames; TecoGAN cutfr2, four adjacent pairs",
                    "comparisons": ["untiled reconstruction versus GT", "tiled-encode/common-untiled-decode reconstruction versus GT",
                                    "raw and normalized tiled-minus-untiled latent", "reconstruction difference and GT error by frozen spatial region"],
                    "no_training_or_sampler_seed": True, "pending": ["E*warp", "separately defined tLPIPS"],
                    "DOVE_source": f"https://github.com/zhengchen1999/DOVE/blob/{common.DOVE_COMMIT}/metrics/raft/ewarp.py",
                    "DOVE_caveat": "Requires original raft-things.pth and CUDA Resample2d, half-resolution RAFT20 with masks; existing torchvision RAFT weights are different. This script does not implement, rename or download E*warp."},
        "bindings": {"external_manifest": common.binding(manifest_path), "config": common.binding(config_path),
                     "official_model_manifest": common.binding(official_path), "vae_config": common.binding(vae_config_path),
                     "vae_weights": vae_weight, "vae_source": vae_source, "metric_weights": metric_weights, "source_files": sources},
        "execution": {"fresh_empty_run_only": True, "resume": False, "overwrite": False,
                      "stop_on_any_failure_OOM": True, "fallback": False, "full_resolution_fit_not_previously_verified": True,
                      "reproducibility": config.get("reproducibility", {}),
                      "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                                      "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                      "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}},
        "completion_boundary": "Passing this fixed comparison does not complete all T1.1 cells or settle D1-D4; an untiled OOM means no completed comparison."
    }
    common.write_json(args.protocol, protocol)
    print(json.dumps({"status": "prepared", "protocol": str(args.protocol.resolve()), "protocol_sha256": common.sha(args.protocol),
                      "shape": protocol["case"]["hr_shape"], "paths": 2, "GPU_used": False, "pixels_decoded": False}))


def validate_protocol(protocol):
    common.require(protocol["schema"] == SCHEMA, "Wrong full-resolution protocol")
    common.require(protocol["case"]["hr_shape"] == [1, 3, 9, 720, 1280] and protocol["case"]["source_indices"] == list(range(9)), "Full-resolution frame/shape contract changed")
    common.require(protocol["paths"] == [{"id": "untiled_encode", "encoder_tiled": False, "decoder_tiled": False},
                                         {"id": "native_tiled_encode", "encoder_tiled": True, "encoder_tile_parameters": TILING, "decoder_tiled": False}], "Encode/decode isolation contract changed")
    common.require(protocol["model"]["dtype"] == "float32" and protocol["model"]["posterior"] == "mode_no_sampling", "VAE numerical contract changed")
    for name, digest in protocol["bindings"]["source_files"].items():
        common.require(common.sha(CODE_ROOT / name) == digest, f"Source changed: {name}")
    for name in ("external_manifest", "config", "official_model_manifest", "vae_config", "vae_weights", "vae_source"):
        common.validate_binding(protocol["bindings"][name])
    for item in protocol["bindings"]["metric_weights"]:
        common.validate_binding(item)
    for item in protocol["case"]["frames_bound"]:
        common.validate_binding(item["file"])


def load_hr(protocol):
    import numpy as np
    import torch
    from PIL import Image
    arrays = []
    for item in protocol["case"]["frames_bound"]:
        with Image.open(item["file"]["path"]) as image:
            common.require(image.format == "PNG" and (image.height, image.width) == (HEIGHT, WIDTH), "Native source PNG shape/format differs")
            arrays.append(np.array(image.convert("RGB"), dtype=np.uint8, copy=True))
    hr = torch.from_numpy(np.stack(arrays)).permute(3, 0, 1, 2).unsqueeze(0).contiguous().float().div_(255)
    common.require(list(hr.shape) == protocol["case"]["hr_shape"], "Native source tensor geometry differs")
    return hr


def region_masks(height, width, diagnostics):
    import torch
    border, band = diagnostics["outer_border_pixels"], diagnostics["seam_band_half_width_pixels"]
    outer = torch.zeros(height, width, dtype=torch.bool)
    outer[:border] = True
    outer[-border:] = True
    outer[:, :border] = True
    outer[:, -border:] = True
    seams = torch.zeros_like(outer)
    for y in diagnostics["seam_y"]:
        seams[max(0, y - band):min(height, y + band)] = True
    for x in diagnostics["seam_x"]:
        seams[:, max(0, x - band):min(width, x + band)] = True
    return {"all_pixels": torch.ones_like(outer), "outer_border": outer, "tile_start_bands": seams,
            "interior_excluding_border_and_bands": ~(outer | seams)}


def region_errors(prediction, reference, masks):
    import torch
    common.require(prediction.shape == reference.shape and prediction.shape[0:2] == (1, 3), "Region comparison geometry mismatch")
    records = []
    for frame in range(prediction.shape[2]):
        difference = prediction[0, :, frame].double() - reference[0, :, frame].double()
        for name, mask in masks.items():
            count = int(mask.sum())
            common.require(count > 0, "Frozen diagnostic region has no pixels")
            selected = difference[:, mask]
            mse = float(selected.square().mean())
            records.append({"sampled_frame_index": frame, "region": name, "spatial_pixel_count": count,
                            "RGB_scalar_count": count * 3, "mse_rgb_01": mse, "rmse_rgb_01": mse ** 0.5,
                            "mae_rgb_01": float(selected.abs().mean()), "max_abs_rgb_01": float(selected.abs().max())})
    return records


def latent_difference(untiled, tiled, mean, std):
    import torch
    common.require(untiled.shape == tiled.shape, "Encoded latent shapes differ")
    a, b = untiled.float().cpu(), tiled.float().cpu()
    mean = a.new_tensor(mean).view(1, -1, 1, 1, 1)
    std = a.new_tensor(std).view(1, -1, 1, 1, 1)
    differences = {"raw": b.double() - a.double(), "normalized": ((b - mean) / std).double() - ((a - mean) / std).double()}
    result = {"direction": "native_tiled_encode_minus_untiled_encode", "shape": list(a.shape),
              "normalization_arithmetic": "float32_then_float64_statistics", "per_channel": [], "per_latent_time": []}

    def stats(value):
        return {"mean": float(value.mean()), "rmse": float(value.square().mean().sqrt()),
                "mae": float(value.abs().mean()), "max_abs": float(value.abs().max())}

    result["global"] = {kind: stats(value) for kind, value in differences.items()}
    for channel in range(a.shape[1]):
        result["per_channel"].append({"channel": channel, **{kind: stats(value[:, channel]) for kind, value in differences.items()}})
    for t in range(a.shape[2]):
        result["per_latent_time"].append({"latent_time": t, **{kind: stats(value[:, :, t]) for kind, value in differences.items()}})
    return result


def execute(args):
    raw = args.protocol.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    common.require(digest == args.expected_protocol_sha, "Frozen protocol SHA mismatch")
    protocol = json.loads(raw)
    validate_protocol(protocol)
    run = args.run_dir.resolve()
    common.require(run.parent == Path(protocol["run_root"]).resolve(), "Run must be a direct child of the frozen new output root")
    common.require(not run.is_relative_to(Path(protocol["read_root"]) / "runs"), "Cannot write inside original run tree")
    common.require(not run.exists() or (run.is_dir() and not any(run.iterdir())), "Fresh empty run only; no resume/overwrite")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "protocol.json").open("xb") as stream:
        stream.write(raw)
    status = {"status": "starting", "protocol_sha256": digest, "completed_encode_decode_paths": 0,
              "training_updates": 0, "T1_1_complete": False, "started_utc": datetime.now(timezone.utc).isoformat()}
    common.write_json(run / "status.json", status)
    started, stage, vae = time.perf_counter(), "imports", None
    resources = {}
    try:
        os.environ.update(protocol["execution"]["environment"])
        os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
        sys.path.insert(0, str(Path(protocol["read_root"]) / ".deps"))
        sys.path.insert(0, str(CODE_ROOT))
        import torch
        import diffusers
        from diffusers import AutoencoderKLWan
        from vsr.reproducibility import configure_reproducibility
        from vsr.perceptual_metrics import PerceptualMetrics

        common.require(not torch.cuda.is_initialized(), "Fresh process required")
        common.require(Path(inspect.getsourcefile(AutoencoderKLWan)).resolve() == Path(protocol["bindings"]["vae_source"]["path"]), "Imported VAE source path differs")
        flags = configure_reproducibility({"reproducibility": protocol["execution"]["reproducibility"]})
        common.require(torch.cuda.is_available(), "CUDA required; no CPU/precision fallback")
        environment = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                       "diffusers": diffusers.__version__, "torch_cuda": torch.version.cuda,
                       "device_name": torch.cuda.get_device_name(0), "reproducibility": flags, "packages": {}}
        for package in ("lpips", "numpy", "Pillow", "scikit-image", "opencv-python-headless"):
            try:
                environment["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                environment["packages"][package] = None
        common.write_json(run / "environment.json", environment)
        stage = "source_png_decode_cpu"
        hr = load_hr(protocol)
        common.write_json(run / "input.json", {"source_indices": list(range(9)), "shape": list(hr.shape),
                          "hr_rgb_f32_sha256": hashlib.sha256(hr.numpy().tobytes()).hexdigest(), "resize_crop_padding": False})
        stage = "load_vae_only"
        stamp = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(protocol["model"]["local_dir"], subfolder="vae", dtype=torch.float32,
                                             local_files_only=True, use_safetensors=True).requires_grad_(False).eval().to("cuda")
        common.require(all(p.dtype == torch.float32 and p.device.type == "cuda" and not p.requires_grad for p in vae.parameters()), "Expected frozen CUDA FP32 VAE")
        common.require(list(vae.config.latents_mean) == protocol["model"]["latents_mean"] and list(vae.config.latents_std) == protocol["model"]["latents_std"], "VAE normalization differs")
        vae.disable_tiling()
        vae.disable_slicing()
        torch.cuda.synchronize()
        common.write_json(run / "vae_initialization.json", {"seconds": time.perf_counter() - stamp, "FP32": True, "DiT_loaded": False, "T5_loaded": False})
        paths = {}
        def phase(name, function):
            nonlocal stage
            stage = name
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            stamp = time.perf_counter()
            try:
                return function()
            finally:
                torch.cuda.synchronize()
                resources[name] = {"seconds": time.perf_counter() - stamp, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "scope": "including resident VAE and current GPU inputs/latents"}
                common.write_json(run / "resource_progress.json", resources, replace=True)
                status.update(status="running", stage=stage)
                common.write_json(run / "status.json", status, replace=True)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            for path in protocol["paths"]:
                name = path["id"]
                if path["encoder_tiled"]:
                    vae.enable_tiling(**TILING)
                else:
                    vae.disable_tiling()
                encoded = phase(name + "_encode", lambda: vae.encode((hr * 2 - 1).to("cuda", dtype=torch.float32)).latent_dist.mode())
                common.require(list(encoded.shape) == protocol["case"]["raw_latent_shape"] and bool(torch.isfinite(encoded).all()), "Encoded latent shape/nonfinite failure")
                latent_cpu = encoded.detach().float().cpu()
                vae.disable_tiling()
                common.require(not vae.use_tiling and not vae.use_slicing, "Both paths require the same untiled decoder")
                reconstructed = phase(name + "_common_untiled_decode", lambda: (vae.decode(encoded).sample.float() / 2 + 0.5).clamp(0, 1).cpu())
                common.require(list(reconstructed.shape) == protocol["case"]["hr_shape"] and bool(torch.isfinite(reconstructed).all()), "Reconstructed shape/nonfinite failure")
                paths[name] = {"latent": latent_cpu, "reconstruction": reconstructed}
                del encoded
                vae.clear_cache()
                torch.cuda.empty_cache()
                status.update(completed_encode_decode_paths=len(paths))
                common.write_json(run / "status.json", status, replace=True)
                print(json.dumps({"path": name, "status": "encode_decode_passed", "completed_paths": len(paths)}), flush=True)
        stage = "release_vae_before_metrics"
        vae.clear_cache()
        del vae
        vae = None
        torch.cuda.empty_cache()
        stage = "metrics_cpu"
        metric_weights = {Path(row["path"]).name: row["path"] for row in protocol["bindings"]["metric_weights"]}
        perceptual = PerceptualMetrics(metric_weights["alexnet-owt-7be5be79.pth"], metric_weights["lpips_v0.1_alex.pth"], device="cpu", batch_size=1)
        masks = region_masks(HEIGHT, WIDTH, protocol["boundary_diagnostics"])
        result_paths, frame_rows, region_rows, pair_rows = {}, [], [], []
        for name, data in paths.items():
            stamp = time.perf_counter()
            metrics = common.metric_records(hr, data["reconstruction"], perceptual)
            errors = region_errors(data["reconstruction"], hr, masks)
            result_paths[name] = {"metrics": metrics, "GT_region_errors": errors, "metrics_seconds": time.perf_counter() - stamp}
            for row in metrics["per_frame"]:
                frame_rows.append({"path": name, **row})
            for row in errors:
                region_rows.append({"comparison": name + "_versus_GT", **row})
            for row in metrics["perceptual"]["tecogan"]["per_pair"]:
                pair_rows.append({"path": name, "first": row["indices"][0], "second": row["indices"][1], **{k:v for k,v in row.items() if k != "indices"}})
        stage = "encoding_difference_cpu"
        difference = latent_difference(paths["untiled_encode"]["latent"], paths["native_tiled_encode"]["latent"], protocol["model"]["latents_mean"], protocol["model"]["latents_std"])
        recon_difference = region_errors(paths["native_tiled_encode"]["reconstruction"], paths["untiled_encode"]["reconstruction"], masks)
        region_rows.extend({"comparison": "tiled_encode_minus_untiled_encode_reconstruction", **row} for row in recon_difference)
        result = {"status": "passed", "protocol_sha256": digest, "case": {k:v for k,v in protocol["case"].items() if k != "frames_bound"},
                  "paths": result_paths, "latent_difference": difference, "reconstruction_difference_by_region": recon_difference,
                  "resources": resources, "wall_seconds": time.perf_counter() - started,
                  "both_decoders_untiled": True, "binary_latents_saved": False, "T1_1_complete": False,
                  "pending_metrics": protocol["metrics"]["pending"], "training_updates": 0}
        common.write_json(run / "result.json", result)
        for name, rows in (("per_frame.csv", frame_rows), ("region_errors.csv", region_rows), ("temporal_pairs.csv", pair_rows)):
            common.write_csv(run / name, rows)
        common.write_json(run / "complete.json", {"status": "passed", "protocol_sha256": digest, "result_sha256": common.sha(run / "result.json"),
                          "table_sha256": {name: common.sha(run / name) for name in ("per_frame.csv", "region_errors.csv", "temporal_pairs.csv")}, "T1_1_complete": False})
        status.update(status="passed", completed_encode_decode_paths=2, wall_seconds=result["wall_seconds"], complete_sha256=common.sha(run / "complete.json"))
        common.write_json(run / "status.json", status, replace=True)
        print(json.dumps({"status": "passed", "run": str(run), "both_decoders_untiled": True, "T1_1_complete": False}), flush=True)
    except BaseException as error:
        failure = {"status": "failed", "protocol_sha256": digest, "stage": stage, "type": type(error).__name__,
                   "message": str(error), "traceback": traceback.format_exc(), "resources": resources,
                   "wall_seconds": time.perf_counter() - started, "no_fallback_or_retry": True, "T1_1_complete": False}
        if "torch" in locals() and torch.cuda.is_initialized():
            failure["cuda_memory"] = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        common.write_json(run / "failure.json", failure)
        status.update(status="failed", stage=stage, failure_sha256=common.sha(run / "failure.json"), wall_seconds=failure["wall_seconds"])
        common.write_json(run / "status.json", status, replace=True)
        print(json.dumps({"status": "failed", "stage": stage, "run": str(run)}), flush=True)
        raise
    finally:
        if vae is not None:
            try:
                vae.clear_cache()
            except BaseException:
                pass


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze native first-nine-frame input and two encoder paths without GPU")
    prep.add_argument("--read-root", type=Path, default=Path("/home/hugo/Projects/VSR"))
    prep.add_argument("--run-root", type=Path, default=Path("/home/hugo/Projects/VSR_/runs"))
    prep.add_argument("--protocol", type=Path, required=True)
    run = sub.add_parser("execute", help="Run both fixed paths with the common untiled decoder; stop on any OOM")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--expected-protocol-sha", required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    return root


def main():
    args = parser().parse_args()
    prepare(args) if args.command == "prepare" else execute(args)


if __name__ == "__main__":
    main()
