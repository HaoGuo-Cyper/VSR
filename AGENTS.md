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

## Where things are

Verified by read-only inspection on 2026-11-05. The new source root is `/home/hugo/Projects/VSR_`; the original `/home/hugo/Projects/VSR` remains the read-only asset and historical-results source. Existing configs retain their original paths and hashes. Do not redirect outputs into the historical runs.

| Item | Path |
| --- | --- |
| Training code | New source: `/home/hugo/Projects/VSR_/vsr/train.py` (NoRef), `/home/hugo/Projects/VSR_/vsr/reference_training.py` (HRI); original copies under `/home/hugo/Projects/VSR/vsr/` |
| Evaluation code | `/home/hugo/Projects/VSR_/vsr/{evaluate_hq,evaluate_external,frozen_evaluation,inference,evaluation,perceptual_metrics}.py`; original copies under `/home/hugo/Projects/VSR/vsr/` |
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

The 2026-11-05 T0.1 evidence and bounded Stage 1 proposals are in `DECISIONS.md`. D1–D4 are PROPOSED and await human settlement; this audit does not authorize new training. New read-only-audit outputs are under `/home/hugo/Projects/VSR_/runs/{stage0_reproduction_20261105T113721Z,vae_crop_full_20261105T114559Z,profile_protocol_resources_20261105T114931Z,vae_full_resolution_20261105T115318Z}/`.

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
10. **Fit in 16 GB.** Use the precomputed text embedding and never load the T5 encoder during training. Train in bf16 with gradient checkpointing. Cache DINOv2 features offline.

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

Then stop and wait for confirmation before starting a stage that trains models.
