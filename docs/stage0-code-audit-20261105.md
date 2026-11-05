# Stage 0 code and protocol audit — 2026-11-05

## Scope and provenance

Read-only inspection of `/home/hugo/Projects/VSR` and the six handoff files in `/home/hugo/Projects/VSR_/vsr-handoff`. The existing source checkout identifies producer commit `04228644605f5a3d96834ba05dde636636400817`; additional working-copy documentation was present and must be inventoried separately from that commit. No checkpoint, old config, old result or partial output was modified. No model training, GPU inference, VAE reconstruction or new quality measurement was run.

References below use repository-relative file names and one-based line numbers of the copied source/protocol files. Original local asset paths are stated explicitly. Paths embedded in immutable historical protocols are provenance records; moving the source tree does not automatically make the historical worker portable.

## D1 — Training video encoding

| Finding | Evidence |
| --- | --- |
| Main training selects 9 frames at stride 4, spanning 33 source frames | `configs/formal_hq_v2_noref.json:13–23`; `vsr/data.py:159–169` |
| The decoder chooses a deterministic start and retains source indices `start, start+4, …, start+32` | `vsr/data.py:187–197` |
| Frames are stacked as `[1,3,T,H,W]`; the sampled playback FPS is original FPS/4 | `vsr/data.py:204–211` |
| The training data preparation passes the configured stride into decoding and then calls video encoding | `vsr/train.py:144–145,159–162` |
| HR and upsampled LR are each passed to the Wan VAE as a whole video; mode is used instead of a posterior sample | `vsr/preflight.py:76–99`, especially `:83–87` |
| The model checks the actual VAE latent geometry | `vsr/preflight.py:89–94` |
| Separate T=1 VAE encoding exists for the supplied HRI reference branch, not as the main training video's encoding scheme | `vsr/reference_training.py:137–159` |

The main training path is whole-clip encoding: 9 input frames yield 3 latent time positions under Wan's temporal compression factor 4. It is not nine independent T=1 encodes of the main video. Exact temporal receptive-field behavior and stride-1 versus stride-4 reconstruction quality remain Stage 1 measurements; no VAE quality conclusion is asserted from this inspection alone. The HRI single-frame cache must not be mistaken for the main LR/HR video path.

## D2 — Train/test frame spacing and interval

| Path | Actual temporal protocol | Evidence |
| --- | --- | --- |
| Formal training | 9 frames, stride 4, 33-frame source span | `configs/formal_hq_v2_noref.json:13–23`; `vsr/data.py:187–211` |
| Internal frozen HQ test | 9 frames, stride 4, deterministic clip start, source FPS/4 | `docs/final_evaluation_v1/protocol.json:711–755` |
| Frozen external REDS4 / YouHQ40 | First 17 contiguous ordinal PNG frames, indices 0–16, stride 1 | `docs/final_evaluation_v1/protocol.json:1038–1072` |
| External reader | Ordered PNG sources sliced by `start:start+num_frames`, then stacked as one video | `vsr/external_data.py:277–295,315–340,347–368` |
| Evaluation engine | Calls the reader with `num_frames=frames, start=0` | `vsr/frozen_evaluation.py:464–471` |
| Native restore | One whole temporal clip per spatial tile, no temporal windows or frame reordering | `vsr/inference.py:141–173,195–205,219–237` |

There is a temporal-spacing mismatch between training and external testing. On an individual source at fixed capture FPS, contiguous frames represent one quarter of the elapsed time of stride-4 adjacent training samples. This does not prove that motion magnitude is exactly four times smaller across different datasets. No protocol was changed to address the mismatch. These external results cover 17-frame prefixes; they are not full-sequence or official whole-benchmark scores. The generic restore API can accept longer clips, but the historical external run explicitly selected 17.

## D3 — LR branch injection and frame separation

| Finding | Evidence |
| --- | --- |
| LR residual injections are before zero-based blocks 0, 5 and 9 (one-based blocks 1, 6 and 10) | `configs/formal_hq_v2_noref.json:25–31`; `vsr/model.py:30–41,99–100,111–120` |
| Existing alignment capture is after zero-based block 9 | `vsr/model.py:30–31,101–102,122–125` |
| Stem uses a Conv3d whose kernel and stride equal the official transformer patch size | `vsr/model.py:78–84` |
| Local Wan transformer patch size is `[1,2,2]`, and depth is 30 | `/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers/transformer/config.json:13–19` |
| Stem tokens and per-block residual projection preserve token positions | `vsr/model.py:84,111–119,132–145` |

Despite using Conv3d, the actual stem temporal kernel and temporal stride are both 1: it does not mix different latent time positions before injection. The per-token Linear projection also does not mix them. The preceding whole-video VAE encoding already contains temporal context; later transformer computation can also mix tokens. Thus latent-frame separation in the residual stem does not imply pixel-frame independence throughout the conditioning path. Future corruption must specify whether it is applied before VAE encoding or to latent conditions. Student/teacher layer choices remain an open human decision; the existing alignment output is not automatically the correct self-distillation choice.

## Historical external inference protocol (T0.2 static evidence)

| Setting | Value | Evidence |
| --- | --- | --- |
| Sampler | `UniPCMultistepScheduler` from the local official `scheduler` subfolder | `vsr/inference.py:85–112` |
| Steps | 20 | `docs/final_evaluation_v1/protocol.json:1063` |
| Sampling seeds | 42 and 43, with identical rules across models | protocol `:1059–1062`; `vsr/inference.py:172–173` |
| CFG | 1.0 | `vsr/inference.py:234–237` |
| Spatial tiles | HR 256×448, overlap 64 | protocol `:1065–1069` |
| Whole-input interpolation | Bicubic ×4, antialias=True, align_corners=False, clamp [0,1], before tiling | `vsr/inference.py:165–171` |
| Noise | One CPU float32 Gaussian field per video/seed, shared by corresponding tile coordinates | `vsr/inference.py:172–173,203–204,236–237` |
| Blending | Positive raised cosine; each axis has 1e-3 floor | `vsr/inference.py:232–233` |
| Temporal handling | 17 contiguous frames as one clip; no temporal window or overlap | protocol `:1038–1072`; `vsr/inference.py:197–205,229` |
| Source pixels | Original REDS4 PNG files / YouHQ40 ZIP PNG members | protocol `:1071`; `vsr/external_data.py:277–313` |
| Caption | `A high quality video.` | actual worker `run_frozen_external_T1_policy_desktop_v2.py:540–544`; existing child protocol `caption` |
| Devices | CUDA restoration, CPU metrics; real CUDA FP32 VAE | original worker `:533–544` |
| FPS | REDS4 playback 30 only; YouHQ40 inherits corresponding MKV metadata, with original capture timing unverified | protocol `:1212–1215`; `vsr/external_data.py:363–367` |
| Reference / pools | Planned HRI on/off using supplied GT frame0; paired primary pool `matched_nonreference` for all models/modes | protocol `:1211`; `vsr/frozen_evaluation.py:366,392` |

The actual historical child protocol is at `/home/hugo/Projects/VSR/runs/external_paired_frozen_20261103T125240.867596Z/external/protocol.json`. Static field inspection confirmed frames=17, start=0, seeds=[42,43], steps=20, tile_hw=[256,448], overlap=64, and the original dataset IDs. No sequence was rerun: T0.2's reproduction criterion remains unmet.

### Metric names must preserve the implementation

The initial handoff `RESULTS.md` template refers to Y-channel PSNR/SSIM and RAFT E*warp. Historical frozen evaluation instead defines RGB PSNR/SSIM, LPIPS Alex0.1 and TecoGAN-style tLP/tOF with Farneback. Do not silently rename or combine them with planned future metrics.

- Saved metrics use quantized uint8 PNG pixels, exactly q/255; zero border shaving for spatial metrics: `docs/final_evaluation_v1/protocol.json:1278–1301`; `vsr/frozen_evaluation.py:120–130`.
- The declared spatial keys are `psnr_rgb_db`, `ssim_rgb`, `mse_rgb_01`, `lpips_alex_v0_1`: protocol `:1278–1283`.
- LPIPS package 0.1.4, Alex network version 0.1: protocol `:1296–1301`.
- Temporal keys retain `tLP_alex_v0_1_TecoGAN_crop`, `tLP100_alex_v0_1_TecoGAN_crop`, `tOF_Farneback_TecoGAN_crop`: protocol `:1284–1292`.
- cutfr=2, original adjacent pairs, TecoGAN crop and Farneback parameters: protocol `:1303–1387`. tLP100 is exactly 100×tLP, not a distinct hypothesis. The original frame0 is outside the retained temporal interval.

No new metric values, seed confidence intervals or benefit claims were produced by this audit.

## Sixteen checkpoints versus the frozen eight-model cohort

The preserved set has eight original seed-20261102 checkpoints (two families × four variants), plus eight Base/Align repeat checkpoints (two families × two variants × training seeds 20261103 and 20261104). The historical external cohort contains only the eight original seed-20261102 checkpoints, all at step 8192.

Evidence: `docs/final_evaluation_v1/protocol.json:150–222` lists the eight exact paths and checkpoint SHA256 values. Its cohort rule `:238–239` explicitly retains repeats separately and forbids best-repeat substitution. The pause inventory at `/home/hugo/Documents/Codex/2026-11-02/files-pasted-by-the-user-wan2/outputs/VSR_paused_by_user_20261104T050113Z/checkpoint_preservation_inventory.json` preserves all 16. T0.3's all-16 external CSV criterion is broader than this historical cohort.

Sampling seeds 42/43 do not turn Relation or Both into multi-training-seed results. Base/Align training stability must be summarized over the true three training seeds, keeping sampling variation and clip uncertainty distinguishable. Historical protocol `:1433–1453` also labels the frozen cohort as one training seed per variant.

## Actual external completion markers and pause

Read-only path enumeration under the original `external/results` directory found:

| Checkpoint | REDS4 | YouHQ40 | Marker total |
| --- | --- | --- | --- |
| NoRef Base seed20261102 | 4/4 | 40/40 | 44 |
| NoRef Align seed20261102 | 4/4 | 40/40 | 44 |
| NoRef Relation seed20261102 | 4/4 | 24/40 | 28 |
| NoRef Both seed20261102 | 0/4 | 0/40 | 0 |
| HRI Base / Align / Relation / Both, each | 0/4 | 0/40 | 0 each |

There are 116 `complete.json` markers out of 352 planned model/sequence rows. This enumeration is not a new strict acceptance of their artifact hashes, metrics, source contracts or PNG semantics. The pause record explicitly says earlier Base/Align execution artifacts were reviewed, while Relation completion marks alone do not establish full semantic/PNG acceptance.

Original run: `/home/hugo/Projects/VSR/runs/external_paired_frozen_20261103T125240.867596Z`; original log: `/home/hugo/Projects/VSR/runs/console_external_paired_frozen.log`.

Pause evidence: `/home/hugo/Documents/Codex/2026-11-02/files-pasted-by-the-user-wan2/outputs/VSR_paused_by_user_20261104T050113Z/PAUSE_AND_RESUME.txt:8–17` records NoRef Relation / YouHQ40 last complete sequence 023. Sequence 024 has preserved seed42 PNGs but no complete marker; its seed43 call was interrupted. Lines 19–23 explain that the outer failure followed user SIGINT and the child's `running` value is stale. This was not all-eight completion. `docs/continuation-state.json:14229–14244` records the pause and absence of automatic resume authorization.

## Frozen worker dependency closure and relocation limits

The actual phase entry point lives outside the old project code tree. The exact entry and helpers are part of the historical evidence and should be archived without source edits. Static AST inspection of imports and `.py` dynamic-import path literals found this seven-file local closure in the old Codex `work/` directory:

| File | Local Python dependencies |
| --- | --- |
| `run_frozen_external_T1_policy_desktop_v2.py` | `run_frozen_HQ_test_T1_policy.py`, `probe_global_HRI_encode_allocator.py`, `run_native_QA_T1_policy.py`, `t1_vae_cache_policy.py`, `external_startup_gpu_guard_v2.py` |
| `external_startup_gpu_guard_v2.py` | `gpu_process_classification.py` |
| `run_frozen_HQ_test_T1_policy.py` | `probe_global_HRI_encode_allocator.py`, `run_native_QA_T1_policy.py`, `t1_vae_cache_policy.py` |
| `run_native_QA_T1_policy.py` | `probe_global_HRI_encode_allocator.py`, `t1_vae_cache_policy.py` |
| `probe_global_HRI_encode_allocator.py` | No additional old-work Python files |
| `t1_vae_cache_policy.py` | No additional old-work Python files |
| `gpu_process_classification.py` | No additional old-work Python files |

The archive should retain `scripts/frozen_external/work/<files>` and `scripts/frozen_external/outputs/<evidence>` if the files remain byte-identical. Helpers derive W as `Path(__file__).resolve().parents[1]` and then use W/work and W/outputs; flat placement directly under `scripts/frozen_external/` would point at the wrong sibling directories. This path layout preserves relative discovery only. Hardcoded P=`/home/hugo/Projects/VSR`, original absolute paths inside seals and predecessor runs, package-source bindings and raw file SHA constraints still apply. Editing them creates a new worker requiring its own review/seals; it must not inherit the old worker's verification status.

Key source references: original entry `:23–40,48,79–82,175–182,432–458,490–544`; phase seal `docs/final_evaluation_v1/external_paired_execution_seal_desktop_v2.json:10–23`. The actual outer run records real restore/vae factory overrides and the lifecycle `public_remove_reinstall_same_VAE_after_each_successful_restore`; these are not synthetic fixtures.

Minimum supporting metadata includes:

- Project protocol and provenance: `docs/final_evaluation_v1/protocol.json`, `external_paired_execution_seal_desktop_v2.json`, `HQ_phase_completion_review_v1.json`, external readiness/seal files, `docs/external-data-provenance.json`, `docs/external-streams.json`, `manifests/external.json`, `manifests/hq_vsr.json`, `manifests/hq_temporal_stride4_inventory.json`.
- Old outputs explicitly bound by the entry: `VSR_evaluation_primary_checkpoint_cohort_acceptance_v1.json`, `VSR_final_evaluation_text_cache_preflight.json`, `VSR_external_metadata_capacity_audit.json`.
- QA helper binds `VSR_T1_VAE_cache_policy_CPU_parity.json`, the CUDA parity run, and original QA predecessor protocols/summaries; the common helper additionally binds the original producer environment and failed resource-probe metadata. These immutable run files and model/data assets remain at their original local locations and are not made portable by copying seven Python files.
- Pause/resume record, checkpoint preservation inventory, original outer/child protocols and status files should remain available as provenance; partial output PNGs and checkpoint binaries stay local.

No archival source was executed during this audit.

## Task and gate disposition

- T0.1: code/assets mapped and D1–D3 supported with file/line evidence; findings do not settle human decisions.
- T0.2: static protocol recovered; sequence reproduction not performed.
- T0.3: blocked pending an explicit new-output continuation/re-evaluation plan. The frozen worker requires a fresh empty own run (`run_frozen_external_T1_policy_desktop_v2.py:444–445`) and records `fresh_only_no_resume=True` (`:525–526`). The old pause record `:36–40` also requires explicit handling of the preserved partial run.
- T0.4/T0.5: not performed in this audit; no consolidated seed statistics or new loss-diagnostic conclusions.
- Stage 1: no reconstruction, condition-gap or resource measurements performed.
- G0: not passed; Stage 0 is unfinished. G1: not passed; D1–D4 require human settlement.

The temporal-spacing mismatch, prefix-versus-full-sequence scope, eight-versus-sixteen cohort and HRI progress discrepancy contradict assumptions in the handoff. Apply AGENTS.md rule 8: report and await the next GPU evaluation/profiling plan. No new GPU evaluation or training was started, and no old partial run was resumed.
