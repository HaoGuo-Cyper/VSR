# Independent eight-update resource review — 2026-11-06

**The repaired, bounded Base resource diagnostic passes this independent metadata review.** Eight real optimizer updates completed on two training-only clips. This pass checked JSON/JSONL records and exact text-source hashes; it did not execute GPU work, deserialize model/latent tensors or modify the actual runtime checkout.

## Runtime and data contract

The recorded runtime is clean commit `41407b0e62748d879d2d9e1d8c5b58fa5202d09f`. Queue, streaming preparation, train, preflight, data, model and reproducibility sources match both its Git tree and `environment.json`. The environment's free/total memory tuple is canonically stored as a list before persistence. Preparation still enforces exact config/environment equality, and the effective config hash follows the existing checkpoint-loader convention.

The explicit diagnostic uses two train clips, no validation/test data,33 contiguous frames,128×224 crops and data/training seed20261102. The prepared metadata records33 consecutive source indices for both clips and an FP32 frozen posterior-mode VAE roundtrip. The latent grid is`[9,16,28]`, token grid`[9,8,14]`. `correspondence.temporal_stride=4` denotes VAE temporal compression; input sampling stride is1.

Preparation records T5/DINO/task-model/optimizer loading as false; only cached text and a frozen VAE are used there. The real Base/AdamW model is constructed afterwards, with an FP32 evaluation VAE resident during the update profile.

## Initialization, optimizer and gradients

The initialization records a fresh pinned official Wan base, empty missing/unexpected/mismatched/error loading lists, a fresh optimizer and no legacy state. There are 6,500,992 trainable elements: 602,752 in the LR branch and 5,898,240 in LoRA, all FP32. Projector, HRI and other trainable groups have zero elements.

The recorded initial trainable hash exactly matches the initialization record and changes after eight updates. AdamW has 13,002,472 state elements, matching two moment arrays for every trainable element plus one step scalar for each of the 488 trainable parameter tensors. All eight loss/gradient/norm/time rows are finite; frozen-gradient element counts are zero at every step. Final-step gradients are nonzero in LR stem, LR residuals and LoRA. No projector/HRI gradient appears. This review did not independently hash every frozen backbone value after the diagnostic; frozen parameters are excluded from the optimizer and have no recorded gradients.

## Resource observation

| Quantity | Recorded and independently recomputed value |
| --- | --- |
| Real diagnostic optimizer updates | 8; zero formal Base budget updates credited |
| Peak CUDA allocated | 3,845,157,888 bytes = 3.581 GiB |
| Timing sample | Seven updates after excluding the first warm-up observation |
| Mean ± sample SD | 0.342009058 ± 0.057117525 s/update |
| First update, excluded from timing summary | 0.531579525 s |
| Eighth update, retained | 0.470780601 s |
| Update-only8192 projection | 0.7783 h |

The eighth update is slower than updates2–7 and is retained in both mean and sample SD. The peak includes the resident Base, VAE, AdamW and update tensors; it is CUDA allocated memory, not total device usage. The projection excludes official-artifact checks, full preprocessing, validation/evaluation, checkpoint writes, startup and any EMA/A2/projector cost. It is not a whole-run ETA.

## Completion boundary

These are eight updates on a discarded diagnostic model, separately counted from the three planned8192-update Base runs. The record explicitly says trained diagnostic parameters are not reused. Different noise/timesteps and alternating clips also prevent interpreting the first-to-last loss decrease as convergence on a fixed objective. Formal runs require their own successful checkpoints and validation records.

No actual EMA teacher/A2/projector integration or overhead test is covered. **G3 has not passed**, and no3RE completion is claimed.

## Provenance

- Run: `/home/hugo/Projects/VSR_/runs/stage2_protocol33_optimizer_probe_20261106T025808.644791Z`
- Resource-profile SHA256: `fbad09b32556a821dba42003dd130f492925c36b73612556ad025937ba85269b`
- Status SHA256: `4da173d4ce619b4b4c5c74d0a15a975adbc10003110353dcaee9eddf1b0dc3c8`
- Environment SHA256: `0b35817a139df0bf60970c4224353d005ba9cab04fef19cbcf6706693054b106`
- Initialization SHA256: `de1284630b4fb45f63cfea76c7c9cd63218b80fabbf9521c1e9bf373a5dbe5f9`
- `resource_review.json` preserves exact source/artifact hashes, configuration checks, geometry, per-step finite audit, gradient groups and timing arithmetic.
