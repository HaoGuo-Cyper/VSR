# Full-resolution VAE attempt — capacity boundary, 2026-11-05

The fixed REDS4/000 native 720×1280, nine-contiguous-frame, FP32 VAE-only comparison stopped with CUDA OOM while decoding the **first, untiled-encoding path**. Untiled encoding had completed. No tiled encoding, complete reconstruction, quality scoring or tile-difference measurement was reached. This is a capacity failure of the specified execution path; it does not reject the research method.

## Exact execution boundary

| Phase | Recorded outcome |
| --- | --- |
| Original GT PNG input, indices 0–8 | Prepared at `[1,3,9,720,1280]`, no resize/crop/padding |
| Untiled FP32 encoding | Completed; the subsequent shape/finite checks passed for `[1,16,3,90,160]` |
| First common untiled decode | OOM inside a later native temporal chunk (`i > 0`); exact chunk index was not recorded |
| Native tiled encoding (256×448; strides 192×384) | Not reached |
| Second common untiled decode | Not reached |
| CPU LPIPS/TecoGAN loading/scoring | Not reached |
| Latent/reconstruction/edge difference tables | Not reached |

`completed_encode_decode_paths=0` is correct: that counter advances only after an entire encode-plus-decode path finishes. The recorded phase name `untiled_encode_common_untiled_decode` denotes decoding of the untiled-encoding result; it does not mean encoding failed. There is no `result.json`, `complete.json`, `summary.json` or quality CSV for this attempt.

## Recorded resources

| Phase | Seconds before completion/failure | Peak CUDA allocated | Peak CUDA reserved |
| --- | ---: | ---: | ---: |
| Untiled encode | 3.526 | 13,616,806,400 bytes (12.682 GiB) | 14,109,638,656 bytes |
| Common untiled decode, failed | 2.390 | 14,594,624,000 bytes (13.592 GiB) | 14,862,516,224 bytes |

The OOM message reports a further 1.99 GiB allocation request, on a 15.47 GiB device with 660.31 MiB free. It reports 13.25 GiB allocated by PyTorch, 261.78 MiB reserved but unallocated, and 13.85 GiB held by this process including non-PyTorch memory. The recorded phase peak and the post-exception current allocation are distinct observations; the failed requested allocation never became a successful allocation. Total recorded attempt wall time was 9.153 seconds, not a completed-comparison time.

## Cache and harness review

Only JSON metadata, the recorded traceback and exact text sources were inspected. Their protocol/source hashes match the frozen bindings. No GPU job, pixel/pt load, code edit or retry was performed for this review.

- `scripts/audit_vae_full_resolution.py:240–247` constructs one FP32 frozen VAE and disables tiling/slicing. No DiT, T5, teacher or optimizer is constructed.
- Lines 264–276 run under `no_grad` with autocast disabled, encode once, verify latent geometry/finite values, copy the small latent to CPU and call the untiled decoder.
- Lines 278–284 retain a completed path only after decode succeeds. At this failure, the paths dictionary contains no previous full-resolution reconstruction.
- In the pinned native VAE source, `_encode` clears encoder/decoder caches before and after its loop (lines 1137–1162); `_decode` clears caches at entry and retains its required causal decoder feature cache across chunks (lines 1191–1216).
- The traceback ends at native `WanCausalConv3d.forward` / `F.pad` (line 176), reached through the later-chunk branch of `_decode` (line 1208). It is not a shape-check, tiling-adapter, metric or source-loader exception.

No accidental second model, autograd graph or retained whole encode cache was identified in the inspected harness. The trace is consistent with the required native untiled FP32 decoder activations/caches exceeding capacity under this protocol. At the reported failure, free device memory plus the reserved-but-unallocated PyTorch pool is below the requested 1.99 GiB; clearing that pool alone is not established as a fix. This does not prove that every streaming implementation, precision setting, device or alternative decoder policy would fail. None was substituted or tried.

## Outcome and next boundary

The attempt preserved failure/resource metadata and stopped without fallback or automatic retry. The tiled-encoding difference remains **unmeasured**, and T1.1 remains partial. The completed 58 crop cases still provide their separate VAE reconstruction and condition-gap evidence. This capacity failure does not evaluate A2, C1 or any training benefit. Report both records to the human and wait for D1–D4 decisions; no further GPU work was performed by this review.

## Provenance

- Run: `/home/hugo/Projects/VSR_/runs/vae_full_resolution_20261105T115318Z`
- Protocol SHA256: `31dc4f55199794f06fea8658722bf741a9caa3cc3d0bca50b9e672fdde9c6420`
- Failure SHA256: `6f8f09a09eee4fff204a7d87e3d05a57d559af2c9dd5024ad13eb476f44c3dc1`
- Status SHA256: `0a6edb6141ab5e0124e8991245798b9c972a32a58637166f6c8b3d6749590bdd`
- Script SHA256: `a21f32175bec1191c4b9ac786087c2cbede26411cf75a64d40d7713b2f7755cd`
- Pinned native VAE source: `/home/hugo/miniconda3/envs/reprvsr-pretrained/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl_wan.py`, SHA256 `29ea101ecd11d5886e21c75fa6c27cca38b026d48a0cbe8f061db9b77bbfa9dd`
- Machine-readable details: `full_resolution_attempt.json`.
