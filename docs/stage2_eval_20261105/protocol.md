# Stage2 frame-varying crop validation: protocol and implementation boundary

This is the frozen preparation for T2.2/T2.3 and the internal G2 pilot. No real validation pixels, new model scores, internal-test pixels or external pixels were opened while creating the metadata seal. No GPU, training or model inference was run by these scripts. The three protocol-matched Base training seeds remain required before G2.

## Cohort and pixel protocol

Use the existing source-held-out validation prefix32 under `SHA256('val:20261102:record_id')`, exactly matching the new Base configs in `configs/stage2_20261105/`. Keep data seed20261102, 33 ordinal consecutive frames, HR128×224, LR32×56, ×4 antialiased bicubic, original FPS and the existing deterministic source window/crop. Validate source-ID and exact-file-hash split isolation. The complete internal test211 and external sets remain for final selected models under a separately frozen protocol; this helper accepts validation only.

Each clip receives one precomputed8-pixel-frame burst. A namespace/seed/record hash chooses its start from4/8/12/16/20. The exact33-bit mask, original source indices, eight selected positions, two immediately adjacent clean neighbor positions and per-frame noise seeds are in `cohort_seal.json`. All models, training seeds, sampler seeds and five degradation conditions reuse those masks. The realized pixel fraction is8/33. These are pixel evaluation masks: do not call them two isolated VAE latents. The separate latent-level A2 implementation excludes latent0 by default; this protocol does not implement latent A2 or change its mask rule.

The five equally weighted conditions are fixed before new scores: LR Gaussian blur sigma2/kernel13/reflection; independent local-CPU Gaussian noise sigma25/255; H.264 CRF40/medium/yuv420p encoded one selected frame per independent all-intra stream; repetition of the original clean LR frame immediately before the burst; and RGB-zero blank frames. The first pixel frame is always clean. No severity was selected using a restoration result.

All nonselected LR frames must be bit-identical float32 values to the clean counterpart. Preserve the clean bicubic LR float32 input; do not quantize all clean/context LR frames to PNG. Authoritative CPU inputs and HR targets are `.npy` float32 arrays. Only selected codec frames require an RGB24 quantization/decode. The actual changed indices are logged even if a constant frame is unchanged by a chosen operation; never resample a mask to force a change. FFmpeg/libx264 executable digest, version and exact encode/decode commands are bound. Codec failure stops; it cannot silently switch to JPEG, another codec, weaker settings or a missing condition.

## Inference and metrics

A later, separately reviewed evaluation worker must freeze the explicit three new Base checkpoint paths/hashes and config identities before loading predictions. Keep20-step official-config UniPC, CFG1, sampling seeds42/43, the same Gaussian noise for clean/corrupt pairs, no temporal windowing, and the declared256×448 tile request/64 overlap (one spatial tile for this crop). Save lossless restored PNGs. Score restored RGB and HR from the same PNG-equivalent `round(clamp(rgb,0,1)*255)/255` values, with no MP4 decode and no Y-channel alias.

`score_pools` consumes the existing per-frame RGB metrics and full-sequence LPIPS/TecoGAN output. It reports RGB PSNR/SSIM and LPIPS on the eight masked frames, the two clean neighbors and all33 frames. Temporal evaluation first runs on the complete original33-frame sequence using the existing TecoGAN cutfr2/crop definition. Then retain the nine original adjacent pairs touching the mask, including its two boundary pairs. All28 original retained pairs are supplementary. Never delete frames and join across a corruption span; tLP100 is the existing100×TecoGAN LPIPS-distance-difference metric, not E*warp or an unimplemented tLPIPS alias.

## G2, fixed before scores

For PSNR, harm-gap means `PSNR(neighbors) − PSNR(masked)`. For LPIPS it means `LPIPS(masked) − LPIPS(neighbors)`. Within the same model/sampler seed, subtract the clean counterpart's gap from the corrupted gap to obtain paired excess gap. This prevents intrinsically difficult masked content from masquerading as corruption damage.

Aggregate in this exact order: mean the two sampling seeds within each clip; equal clips within training seed and degradation type; equal all five types within each training seed; equal the three training seeds. For each metric, compare the mean paired excess gap to the sample SD (ddof1) of the three clean corresponding harm-gap means. **Both PSNR and LPIPS excess gaps must strictly exceed their respective clean-gap seed spread.** Keep all raw clean/corrupt gaps, per-type and per-clip appendices. Do not choose favorable types or treat sampler seeds/clips as training repeats. No p-value or Stage5 significance decision is made by G2. Failure stops for a report; it does not authorize in-place harder masks, a severity sweep or test-set tuning.

## New-only helpers

- `scripts/stage2_framewise.py seal`: metadata-only cohort/config/code/codec freeze.
- `materialize`: explicitly requested CPU-only validation crop/float32-asset generation into a fresh new-project `runs/` child. It rehashes raw sources using the existing `decode_clip`, checks the sealed source indices/FPS and preserves old files. A failed partial output is not resumed or overwritten.
- `FramewiseReader`: JSON-only construction and later hash-verified CPU NPY loads; exposes stable HQ input bindings for a new evaluation adapter. It chooses no checkpoint and runs no model.
- `scripts/stage2_metrics.py`: pure-stdlib original-frame/pair pooling and full1152-observation G2 completeness checks (3 training seeds×32 clips×2 sampler seeds×6 clean/degraded conditions).
- `selftest`: synthetic CPU tensors and synthetic metric records only, including actual independent-frame libx264 encode/decode, unmasked equality, temporal pair identity, clean-gap subtraction, nonzero three-seed SD thresholds, both-metric requirement and missing-row rejection.

The existing8-model external worker/seal and the old9-frame HQ metrics do not cover this new protocol. No GPU evaluation queue has been added here. A future worker must use this reader/metric contract and independently bind its model/runtime/output files.

## Calls

Run from the staging project with the existing Python environment:

```bash
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python scripts/stage2_framewise.py seal --output docs/stage2_eval_20261105/cohort_seal.json
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python scripts/stage2_framewise.py selftest --output docs/stage2_eval_20261105/cpu_synthetic_final.json
# Later explicit CPU asset generation, after review; not run in this preparation:
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python scripts/stage2_framewise.py materialize --seal docs/stage2_eval_20261105/cohort_seal.json --output runs/stage2_val_assets_NEW_UTC
```

All outputs refuse overwrites. Seals bind absolute project/source paths and exact code/config hashes; after deployment to a different project root, create a new metadata seal there before pixel generation. The frozen cohort selection and corruption settings must remain unchanged.
