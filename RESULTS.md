# Results

Append-only. Do not edit or delete earlier entries; add corrections as new entries.

## Inference protocol (fill in during T0.2; frozen for each comparison)

| Setting | Value |
| --- | --- |
| Sampler | |
| Sampling steps | |
| Sampler seed | |
| Spatial tiling (tile size, overlap) | |
| Temporal window (frames, stride, overlap) | |
| Metric implementations (PSNR/SSIM on Y, LPIPS version, RAFT model for E*warp) | |

## Run log

| Run id | Date | Stage / task | Git commit | Config | Seed | Start checkpoint | Updates | Wall-clock | Peak VRAM | Final L_flow | Final L_sd | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Stage summaries

<!-- For each stage: what was run, the results table (mean ± std over seeds, bootstrap p-values), the gate verdict, and questions for the human. -->


## 2026-11-05 — Stage 0 code map and frozen-protocol inspection (T0.1 / T0.2)

This entry records read-only code, configuration, protocol and completion-marker inspection. No training, GPU inference, VAE reconstruction audit or new quality measurement was run. Existing checkpoints, configurations, results and partial outputs remain unchanged. Detailed evidence is in `docs/stage0-code-audit-20261105.md`; the source code was copied from `/home/hugo/Projects/VSR` at producer commit `04228644605f5a3d96834ba05dde636636400817` together with the separately inventoried working-copy documentation.

### Frozen external inference protocol recovered from existing files

| Setting | Existing protocol |
| --- | --- |
| Cohort | Eight original seed-20261102 checkpoints: NoRef/HRI × Base/Align/Relation/Both, each at step 8192 |
| Datasets | REDS4 IDs 000/011/015/020 and YouHQ40 IDs 000–039 |
| Sampler | Local official-config `UniPCMultistepScheduler`; CFG 1.0 |
| Sampling steps / seeds | 20 steps; both sampling seeds 42 and 43 |
| Input interval | First 17 contiguous ordinal PNG frames, indices 0–16, stride 1; these are prefix results |
| Temporal window / overlap | One 17-frame clip; no temporal windowing or temporal overlap |
| Spatial operator | Native ×4; one whole-input bicubic interpolation before spatial tiling |
| Spatial tiles / overlap | HR 256×448; overlap 64; positive raised-cosine blend with an axis-wise 1e-3 floor |
| Noise | One CPU float32 Gaussian field per video/seed; corresponding spatial crop reused by each tile |
| Text | `A high quality video.` from the existing accepted text-cache path |
| Execution | CUDA restoration, CPU metrics; original PNG source pixels; no metric resize |
| Metric pixels | Saved uint8 PNG after `round(clamp(RGB01)*255)`; metric input is exactly q/255 |
| Spatial metrics | `psnr_rgb_db`, `ssim_rgb`, `mse_rgb_01`, `lpips_alex_v0_1`; zero border shaving |
| Temporal metrics | `tLP_alex_v0_1_TecoGAN_crop`, `tLP100_alex_v0_1_TecoGAN_crop`, `tOF_Farneback_TecoGAN_crop`; cutfr=2 and original adjacent pairs |
| Reference / pools | NoRef no-reference mode; HRI planned on/off with supplied GT frame 0; paired primary pool `matched_nonreference` for every model/mode |
| FPS | REDS4 30 is declared playback FPS only; YouHQ40 PNG timing inherits supplied MKV metadata; original capture timing is not independently established |

This records the historical protocol; it is not a reproduction certificate. T0.2 remains in progress because no checkpoint/sequence was rerun. The original blank template above mentions Y-channel PSNR/SSIM and RAFT E*warp; those names do not describe these historical frozen metrics. Existing tOF must not be relabeled as RAFT E*warp, and TecoGAN tLP must retain its implementation name.

### Checkpoint inventory and actual external progress

There are 16 preserved final checkpoints: the eight original seed-20261102 models, plus Base/Align repeats for both families at training seeds 20261103 and 20261104. The frozen external cohort contains only the original eight. Sampling seeds 42/43 are not independent training replicates. T0.3's requested all-16 coverage is therefore broader than the existing frozen eight-model cohort and needs an explicit compatible evaluation plan.

| Original checkpoint | REDS4 completion markers | YouHQ40 completion markers | Total markers |
| --- | --- | --- | --- |
| NoRef Base, training seed 20261102 | 4/4 | 40/40 | 44/44 |
| NoRef Align, training seed 20261102 | 4/4 | 40/40 | 44/44 |
| NoRef Relation, training seed 20261102 | 4/4 | 24/40 | 28/44 |
| NoRef Both, training seed 20261102 | 0/4 | 0/40 | 0/44 |
| HRI Base / Align / Relation / Both, each | 0/4 | 0/40 | 0/44 each |

The 116 `complete.json` markers were counted in the existing run; they are not a new strict artifact/semantic acceptance or a quality comparison. The pause record states that Base/Align execution artifacts were reviewed earlier, while Relation markers still need strict verification. NoRef Relation last completed YouHQ40/023; YouHQ40/024 has preserved seed-42 PNGs but no complete marker and is excluded from the 116 count. The original phase planned 352 model/sequence rows and 1056 restores, owing to HRI on/off modes. No quality values were opened for new benefit claims during this audit.

Existing run: `/home/hugo/Projects/VSR/runs/external_paired_frozen_20261103T125240.867596Z`; rows under `external/results/`; log `/home/hugo/Projects/VSR/runs/console_external_paired_frozen.log`. Preservation receipt: `/home/hugo/Documents/Codex/2026-11-02/files-pasted-by-the-user-wan2/outputs/VSR_paused_by_user_20261104T050113Z/PAUSE_AND_RESUME.txt`.

### Gate status and next decision

T0.1 is complete once the code map and D1–D3 evidence are recorded. T0.2 is partially documented and not reproduced; T0.3 is blocked pending an explicit fresh-output continuation/re-evaluation plan. G0 has not passed: external coverage, statistics and loss diagnostics remain unfinished. G1 has not passed: D1–D4 remain human decisions and Stage 1 measurements have not been performed.

The audit found conflicts with the handoff assumptions: training uses 9 stride-4 frames while the frozen external protocol uses 17 contiguous frames; historical external testing covers prefixes, not full sequences; the frozen cohort has eight models rather than all 16 checkpoints; and HRI external Base/Align have not run. The original worker is fresh-only/no-resume and binds original paths, source bytes and protocol seals. Under AGENTS.md rule 8, report these discrepancies and wait for the human's next GPU evaluation/profiling plan. Do not resume the old partial run, overwrite its outputs, or silently change its protocol. No new training or GPU evaluation was launched.

### 2026-11-05 — Source import validation

The current source snapshot and original handoff were committed before the audit edits (`c95495b`). The existing CPU suite ran with `CUDA_VISIBLE_DEVICES=''`: 230 tests in 9.729 seconds, no failures/errors, one skipped because Node is unavailable (inline viewer JavaScript parse-only check). See `docs/cpu-tests-import-20261105.txt` and `docs/import-validation-20261105.json`. All 212 inventoried original files still match their pre-import SHA256 values and the old repository's Git status is unchanged. Copied code, configs, manifests and tests are byte-identical; README and ignore rules have new checkout instructions. This test run does not fulfill the T0.2 GPU reproduction criterion.

The seven actual frozen external workers/helpers and four directly referenced JSON records were archived byte-for-byte under `scripts/frozen_external/`, with SHA256 provenance. Historical execution and model/data bindings remain in force; no archived worker was run. Local binaries, checkpoints and datasets are excluded from the public code import. GitHub synchronization remains pending write authentication: the connected integration returned HTTP 403, and the local Git client had no authenticated username. No remote branch was changed by these attempts.

## 2026-11-05 — GitHub publication completed

The human explicitly authorized publishing the 239 text files, including local paths, experiment metadata and dataset manifests, to the public repository `HaoGuo-Cyper/VSR`. Weights, checkpoints, original videos, cached tensors, credentials and local symlinks remain excluded.

The GitHub connection published the two prepared snapshots as `2a697995d24483c66c38ab67b8d72793eaf2abf6` and `fe8b581e57a662028a9ac56a7246b7ce04a1ce36`, preserving the initial remote commit `468db7a927a528f2bf8a80b9dee5e0336d955743`. The connection generated new commit identities; their file trees are byte-identical to local `c95495b` and `dbedaaf` respectively. Verified tree SHAs are `b72c6e473343d426420369d67b14628de4fbcad0` and `ddc2a8f264de985cb97c4c2c673270bb5697d422`. The local pre-publication history is preserved in an archive ref and the earlier Git bundle.

A recursive, non-truncated GitHub tree read and independent agent review confirmed 239 regular UTF-8 text blobs, no symlink/gitlink or excluded binary asset, and the original six handoff files in the first snapshot. This append-only entry records the completed publication; the earlier pending-authentication entry remains historical. No training, GPU evaluation or numerical protocol change was performed during synchronization. T0.2 reproduction and the open research gates remain unfinished.
