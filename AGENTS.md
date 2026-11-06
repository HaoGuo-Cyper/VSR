# Agent brief: temporal self-distillation for diffusion VSR

Read this file first, then `DECISIONS.md`, then `TASKS.md`. `RESEARCH_PLAN.md` is the technical reference for the method, baselines, metrics and controls.

## Goal

Test whether temporally structured self-distillation improves a Wan2.1-T2V-1.3B video super-resolution (VSR) model. An EMA copy of the model (the teacher) sees frames the student only sees in corrupted form, and the student learns to rebuild those frames from neighbouring frames. The target gains are better restoration of badly degraded frames and better temporal consistency, at zero inference cost.

The comparison that matters most is against **C1**: the same corruption used as plain data augmentation, without the alignment loss. A gain that C1 also gets is not a gain from self-distillation.

## Current state (as of 2026-11-05)

**Hardware and assets**
- One NVIDIA RTX 5080, 16 GB VRAM.
- HQ-VSR dataset stored locally; 1,641 training clips used so far.
- Wan2.1-T2V-1.3B weights and its VAE stored locally.

**Existing models: 8 configurations, 16 final checkpoints**
- Backbone frozen. Trained parts: rank-8 attention LoRA plus a learned LR residual branch injected into several transformer blocks. There is no widened 32-channel input embedding.
- Every run started from the official pretrained weights and ran 8,192 optimizer updates.
- Training data: 4× bicubic degradation, HR crops of 128×224, 9 frames sampled at temporal stride 4 (spanning 33 original frames), one fixed generic text condition, and the same degradation on every frame.
- Two families:
  - **NoRef**: conditioned on the LR video only.
  - **HRI**: LR video plus a high-resolution reference image, through an extra trained reference adapter.
- Four variants per family:

| Variant | Training objective | Seeds per family |
| --- | --- | --- |
| Base | Flow-matching restoration loss only | 3 |
| Align | Flow matching + pointwise alignment to frozen DINOv2-B/14 features of HR frames | 3 |
| Relation | Flow matching + matching of spatial and motion-corresponding temporal relations between DINOv2 features | 1 |
| Both | Flow matching + Align and Relation losses | 1 |

**Evaluation status**
- Internal HQ evaluation is done for all 8 configurations. Results were mixed, with no stable overall gain from the auxiliary losses.
- External REDS4 + YouHQ40 testing is paused. Only NoRef training seed 20261102 has external outputs: Base and Align each have all 44 sequence summaries; Relation has 28/44; Both has none. These are 17-frame prefixes with sampler seeds 42/43, not full-source-video evaluation. HRI and the other training seeds have no external sequence summaries in this run. Presence of an output is not a fresh reproduction or a passed quality gate.

**Not done yet:** EMA self-distillation, per-frame-timestep models, VAE-latent alignment, one-step distillation.

**Continuation audit, 2026-11-05.** T0.2 now passed a fresh NoRef Base / REDS4 / 000 reproduction: 34 saved PNG files byte-identical across both sampler seeds. Internal statistics (including a separate crossed training-seed/source bootstrap) and all 16 final-run loss diagnostics are recorded in `docs/stage0_analysis_20261105/`; external coverage is still incomplete. Stage 1 completed 58 supported VAE crop cases and condition-gap measurements plus six synthetic no-parameter-update resource cases. The independent native720p VAE attempt stopped with OOM in its common untiled decoder, before a tiled-path comparison. Reports are in `docs/stage1_audit_20261105/`. **D1–D4 are PROPOSED, not SETTLED. GPU work is stopped awaiting the human; no training run or optimizer update has occurred.**

**Latest human decision, 2026-11-05.** D1–D4 are now **SETTLED**: 33 contiguous frames / stride1 / 9 latent positions; unchanged 128×224 ×4 bicubic and source split; future student9/teacher21 and trainable-only EMA0.999. G1 passed by explicit human acceptance. T2.1 is authorized: three independently initialized NoRef Base runs, seeds20261102/03/04, 8192 updates each. Cached text only and streaming RGB preparation are required. Normal planned continuation after passing gates is authorized without repeated confirmation. Historical audit statements above describe the earlier pause; The subsequent human-approved review closes G0 in a reduced internal-evidence scope, descopes the old external backlog and native untiled/E*warp audit requirements, and selects latent-level A2 masking excluding position0; the failed native untiled audit is not retried.

**Latest execution,2026-11-06.** All three protocol33 NoRef Bases completed8192 updates each, with diagnostic8 updates separate. Read `docs/stage2_20261106/three_base_completion_audit.json` for the complete data/source/cache/checkpoint/seed-statistics audit. Actual source remains41407b0. Current evaluation declaration is staging `runs/active_stage2_evaluation.json`; read its run status/progress and latest `docs/stage2_20261106/live/progress.json` before any action. Never duplicate a planned/running matrix. Once prepared, every numeric source/dependency bound by that protocol is frozen through all1152 restores; only documentation/logs and unbound workspace scratch may change. G2 and real G3 require their measured gate verdicts. The human explicitly requested stopping execution and a timeline on2026-11-06. The evaluator is stopped and automationvsr is PAUSED; preserve all completed/partial artifacts and do not restart experiments or autonomous development without a new human instruction. Read the user_stop receipt for the final committed-row count.

## Where things are

**Current planning entry,2026-11-06:** read `docs/H100_MAIN_EXPERIMENT_PLAN_20261106.md`. The human assigned a remote H10080GB to main experiments and RTX5080 16GB to development/validation, requesting a new plan **without starting experiments**. H100 access/environment/assets are not yet verified or migrated. Experiments remain stopped and automation PAUSED; this planning instruction does not authorize GPU execution or restart. Keep D1–D4, existing three Bases and scientific gates; prepare a new prospective H100 protocol before future execution.

**Latest source refactor,2026-11-06:** the human explicitly authorized actual10-module consolidation, preservation of old code/results, relatedCPU tests, README update and GitHub sync, while forbidding GPU experiments/H100 migration. The staging `vsr/` now has exactly data/vae/model/teacher/losses/train/infer/evaluate/metrics/checkpoints plus __init__.py, with real implementations and no runtime legacy import. Oldvsr/scripts/tests144 files are byte-identical under `legacy/pre_refactor_20261106/`, manifest/source commit60e2b4b. All130CPU tests passed,0 failures/0 skips,CUDA never initialized; see `docs/code_refactor_20261106/cpu_validation.json`. Read README and `docs/CLI_GUIDE.md` for the real unifiedCLI. Current scientific gates and GPU/H100 validation remain pending; automation stays PAUSED. Old stopped protocol sources are preserved in the archive/original commit and actual41407b0; never resume that oldrun using new code.

Verified by read-only inspection on 2026-11-05. The new source root is `/home/hugo/Projects/VSR_`; the original `/home/hugo/Projects/VSR` remains the read-only asset and historical-results source. Existing configs retain their original paths and hashes. Do not redirect outputs into the historical runs.

| Item | Path |
| --- | --- |
| Training code | Current10-module staging: `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/VSR-sync/vsr/train.py`; original actual runtime stays `/home/hugo/Projects/VSR_/` at41407b0. Historical HRI/DINO/phase sources are archived in staging `legacy/pre_refactor_20261106/`. |
| Evaluation code | Current staging `vsr/evaluate.py`, `vsr/infer.py`, `vsr/metrics.py`; old actual/original implementations remain unchanged in `/home/hugo/Projects/VSR_/` and `/home/hugo/Projects/VSR/`. |
| Configs for the 8 existing configurations | `/home/hugo/Projects/VSR_/configs/formal_hq_v2_noref.json`, `formal_hq_v2_remaining_noref.json`, `formal_hq_v2_reference.json`; Base/Align repeats in `formal_hq_v2_{noref,reference}_repeat_seed2026110{3,4}.json`. Variants share family configs; original immutable copies are in `/home/hugo/Projects/VSR/configs/`. |
| The 16 checkpoints | `/home/hugo/Projects/VSR/runs/` — exact run directories and variant mapping below; each filename is `checkpoint_008192.pt`. Eight per family: Base ×3, Align ×3, Relation ×1, Both ×1. |
| HQ-VSR data (train / val split) | Raw `/home/hugo/Projects/ReprVSR-pretrained/data/HQ-VSR/`; split `/home/hugo/Projects/VSR/manifests/hq_vsr.json`; stride-4 eligibility `/home/hugo/Projects/VSR/manifests/hq_temporal_stride4_inventory.json`. Manifest copies are also in the new source root. |
| REDS4, YouHQ40 | `/home/hugo/Projects/VSR/data/external/{REDS4,YouHQ40}/`, source archives `/home/hugo/Projects/VSR/data/external/archives/`; manifest `/home/hugo/Projects/VSR/manifests/external.json`; provenance `/home/hugo/Projects/VSR/docs/external-data-provenance.json`. |
| Wan2.1-T2V-1.3B weights and VAE | `/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers/{transformer,vae}/` (official revision recorded in configs); cached tokenizer/text encoder are in the same model root. |
| Cached text embedding, cached DINOv2 features | `/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/{fresh_text_embeddings.pt,fresh_teacher_features.pt,fresh_training_cache.pt}`. DINO source `/home/hugo/Projects/VSR/vendor/dinov2/`, weights `/home/hugo/Projects/VSR/weights/dinov2_vitb14_pretrain.pth`; shared frozen-cache reuse is validated by `vsr/train.py:97–139`. |
| Training logs | `/home/hugo/Projects/VSR/runs/<formal-run>/<variant>/training.jsonl`, sibling `summary.json`, `initialization.json` and `evaluation_history.json`; console logs `/home/hugo/Projects/VSR/runs/console_*.log`. |
| Existing evaluation outputs | Internal `/home/hugo/Projects/VSR/runs/HQ_test_frozen_T1_policy_20261103T022832.538880Z/`; external `/home/hugo/Projects/VSR/runs/external_paired_frozen_20261103T125240.867596Z/external/`; supporting reports `/home/hugo/Projects/VSR/docs/final_evaluation_v1/` and `/home/hugo/Documents/Codex/2026-11-02/files-pasted-by-the-user-wan2/outputs/`. |

All checkpoint paths below are relative to `/home/hugo/Projects/VSR/runs/`; append `/<variant>/checkpoint_008192.pt`. These files remain outside Git.

| Family / training seed | Run directory | Final variants |
| --- | --- | --- |
| NoRef / 20261102 | `formal_hq_v2_noref_20261102T142214.588571Z` | Base |
| NoRef / 20261102 | `formal_hq_v2_remaining_noref_20261102T153108.569879Z` | Align, Relation, Both |
| NoRef / 20261103 | `formal_hq_v2_noref_repeat_seed20261103_20261102T194730.270735Z` | Base, Align |
| NoRef / 20261104 | `formal_hq_v2_noref_repeat_seed20261104_20261102T204607.424405Z` | Base, Align |
| HRI / 20261102 | `formal_hq_v2_reference_20261102T171545.936007Z` | Base, Align, Relation, Both |
| HRI / 20261103 | `formal_hq_v2_reference_repeat_seed20261103_20261102T214631.884279Z` | Base, Align |
| HRI / 20261104 | `formal_hq_v2_reference_repeat_seed20261104_20261102T225815.432804Z` | Base, Align |

The 2026-11-05 T0.1 evidence, bounded Stage 1 proposals and subsequent human settlement of D1–D4 are in `DECISIONS.md`. The earlier read-only-audit outputs are under `/home/hugo/Projects/VSR_/runs/{stage0_reproduction_20261105T113721Z,vae_crop_full_20261105T114559Z,profile_protocol_resources_20261105T114931Z,vae_full_resolution_20261105T115318Z}/`. New Stage2 runs use separate dated directories and the accepted protocol33 configs.

## Working rules

1. **No new training before the open decisions are settled.** Stages 0 and 1 need no decisions and can start immediately. Stage 2 onward waits until the human marks D1–D4 in `DECISIONS.md` as SETTLED.
2. **Never modify existing checkpoints, configs or result files.** Write new outputs to new, dated directories.
3. **One fixed inference protocol per comparison.** Fix the sampler, number of steps, sampler seed, spatial tiling and temporal windowing. Record the protocol in `RESULTS.md` before evaluating, and never change it partway through a comparison.
4. **Every claim needs seed statistics.** Report mean ± std over seeds and a paired bootstrap over clips. Label single-seed results "single seed, indicative".
5. **Primary metrics are fixed before Stage 5** and recorded in `DECISIONS.md`. Never pick metrics after seeing results.
6. **Log every training run** in `RESULTS.md`: run id, git commit, config path, seed, starting checkpoint, number of updates, wall-clock time, peak VRAM, final loss values, evaluation numbers.
7. **Keep `TASKS.md` current.** Mark each task `[~]` when started and `[x]` when done (or `[!]` when blocked), with a date and a one-line note.
8. **Stop at failed gates.** If a gate fails or a result contradicts the plan, report it and wait. Do not improvise a new research direction.
9. **Ask the human first** before you delete anything, change the data split, change the degradation used by existing baselines, or spend more than 2 run-equivalents on anything not in `TASKS.md`.
10. **Hardware roles.** The original16GB limit remains the RTX5080 development/validation target; the remote H10080GB is the future main experiment platform. Preserve first-round scientific settings and review device assumptions before migration. Use precomputed text embedding, never keep T5 loaded during training, and retain declared bf16/checkpointing settings. Historical DINO targets are cached offline. Neither hardware availability nor this planning update authorizes GPU startup.
11. **Public GitHub synchronization is authorized.** On 2026-11-05 the human authorized this and future new/updated project text files to be synchronized to the public `HaoGuo-Cyper/VSR` repository, including project-local paths, experiment metadata and data manifests. No repeated publication confirmation is needed for that scope. Keep weights, checkpoints, raw media, tensor caches and credentials outside Git. This permission does not settle D1–D4, waive failed gates or authorize training.

## Vocabulary

- **RE (run-equivalent):** one 8,192-update training run. A 2,048-update continuation is 0.25 RE.
- **A2:** the main method. Frame-selective corruption of the LR condition, plus alignment of student features to EMA-teacher features on the corrupted frames.
- **C1:** the same corruption with no alignment loss (augmentation-only control).
- **Base / Align / Relation / Both:** the existing variants above.
- **A0–A4, B0–B4, C1–C5:** defined in `RESEARCH_PLAN.md`.

## Reporting

At the end of each stage, add a short section to `RESULTS.md` containing:
- what was run;
- the results table;
- the gate verdict (pass or fail, and why);
- any questions for the human.

Human authorization on 2026-11-05 permits automatic continuation through the existing plan after successful checks. Report each stage, continue when its gate passes, and ask only for failed gates, problematic results or exceptional changes; do not repeatedly request permission for already-authorized training.
