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
- External REDS4 + YouHQ40 testing is paused. Base and Align finished all sequences, Relation finished part of them, and Both has not been run.

**Not done yet:** EMA self-distillation, per-frame-timestep models, VAE-latent alignment, one-step distillation.

## Where things are

Fill this in during task T0.1, then keep it current.

| Item | Path |
| --- | --- |
| Training code | |
| Evaluation code | |
| Configs for the 8 existing configurations | |
| The 16 checkpoints | |
| HQ-VSR data (train / val split) | |
| REDS4, YouHQ40 | |
| Wan2.1-T2V-1.3B weights and VAE | |
| Cached text embedding, cached DINOv2 features | |
| Training logs | |
| Existing evaluation outputs | |

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
