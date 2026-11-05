# Stage2 val32 CPU asset materialization (2026-11-05)

Completed all32 fixed validation clips, clean plus five predeclared pixel-frame degradation conditions. No GPU, restoration model, checkpoint or new prediction/metric score was read or run.

- Assets: `/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work/VSR-sync/runs/stage2_val_assets_20261105T132924Z` (ignored runs directory).
- 224 float32 NPY files: 32 HR targets, 32 clean LR clips, 160 degraded LR clips.
- 33 consecutive frames, HR 128×224/LR 32×56, data seed 20261102, original FPS retained.
- All 192 reader views passed file/float-pixel SHA bindings, source/window/stride/FPS and original stable-HQ-input-contract checks.
- HR target bytes match across conditions; every unmasked LR frame is identical to the clean counterpart.
- 256 independent single-frame libx264 all-intra encodes completed; no fallback or severity change.
- Frozen scripts, protocol config and cohort seal remain byte-identical.

Dataset manifest SHA256: `355060708dd3ac3203801764a202ac124999cfa07bb317a45db2a493fa669004`.
Cohort seal digest: `9c5e938d8b2c9b5e7d965fca2f6b3925557848235d6b9d404b6231bffcbc36e8`.

## Actual pixel support

| Condition | Changed selected frames | Min/max per clip | Intended frames per clip |
| --- | ---: | ---: | ---: |
| blur | 256 | 8/8 |8 |
| noise | 256 | 8/8 |8 |
| codec_allintra | 256 | 8/8 |8 |
| drop_repeat | 256 | 8/8 |8 |
| drop_black | 256 | 8/8 |8 |

All changes remain within the frozen mask. A constant frame is allowed to remain unchanged by an operation; no mask was resampled based on pixel effects.

## Remaining boundary

This completes the authorized validation-asset generation and reader checks only. It does not produce SR quality results, pass G2, replace the three new Base seeds, or open the heldout test211/external evaluation cohorts. Per-file and per-codec-frame provenance remains in the ignored dataset manifest and clip records.

The current NPY paths are project-relative; invoke the reader from the recorded staging project root. Do not move arrays or rewrite the frozen materialization records. No actual runtime checkout or frozen training code was changed.
