# Existing training loss diagnostics

Analysis date: 2026-11-05.

Read-only JSON/JSONL analysis of 16 final-checkpoint runs. The comparison uses the first and last **512 recorded loss observations**, fixed before calculating the results. JSON records the actual count and step range for each window, source hashes, final values, inactive placeholders and binned observations. No checkpoint payload, GPU, model forward pass or optimizer was used.

Coverage: 16/16 selected logs contain every update from 1 through 8192. Observed window step ranges: early 1–512, late 7681–8192. This coverage was checked from the logs, not inferred from checkpoint names. Inactive auxiliary fields are excluded from decrease counts; all their logged values are zero: True.

## Log coverage

| Family | Variant | Training seed | Records | Logged steps | Run |
| --- | --- | --- | ---: | --- | --- |
| HRI | Base | 20261102 | 8192 | 1–8192 | `formal_hq_v2_reference_20261102T171545.936007Z` |
| HRI | Base | 20261103 | 8192 | 1–8192 | `formal_hq_v2_reference_repeat_seed20261103_20261102T214631.884279Z` |
| HRI | Base | 20261104 | 8192 | 1–8192 | `formal_hq_v2_reference_repeat_seed20261104_20261102T225815.432804Z` |
| HRI | Align | 20261102 | 8192 | 1–8192 | `formal_hq_v2_reference_20261102T171545.936007Z` |
| HRI | Align | 20261103 | 8192 | 1–8192 | `formal_hq_v2_reference_repeat_seed20261103_20261102T214631.884279Z` |
| HRI | Align | 20261104 | 8192 | 1–8192 | `formal_hq_v2_reference_repeat_seed20261104_20261102T225815.432804Z` |
| HRI | Relation | 20261102 | 8192 | 1–8192 | `formal_hq_v2_reference_20261102T171545.936007Z` |
| HRI | Both | 20261102 | 8192 | 1–8192 | `formal_hq_v2_reference_20261102T171545.936007Z` |
| NoRef | Base | 20261102 | 8192 | 1–8192 | `formal_hq_v2_noref_20261102T142214.588571Z` |
| NoRef | Base | 20261103 | 8192 | 1–8192 | `formal_hq_v2_noref_repeat_seed20261103_20261102T194730.270735Z` |
| NoRef | Base | 20261104 | 8192 | 1–8192 | `formal_hq_v2_noref_repeat_seed20261104_20261102T204607.424405Z` |
| NoRef | Align | 20261102 | 8192 | 1–8192 | `formal_hq_v2_remaining_noref_20261102T153108.569879Z` |
| NoRef | Align | 20261103 | 8192 | 1–8192 | `formal_hq_v2_noref_repeat_seed20261103_20261102T194730.270735Z` |
| NoRef | Align | 20261104 | 8192 | 1–8192 | `formal_hq_v2_noref_repeat_seed20261104_20261102T204607.424405Z` |
| NoRef | Relation | 20261102 | 8192 | 1–8192 | `formal_hq_v2_remaining_noref_20261102T153108.569879Z` |
| NoRef | Both | 20261102 | 8192 | 1–8192 | `formal_hq_v2_remaining_noref_20261102T153108.569879Z` |

## Flow loss

Raw objective values; auxiliary coefficients are not applied in this table.

| Family | Variant | Training seed | Early mean | Late mean | Change % | Final observed value | Window samples (early / late) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HRI | Base | 20261102 | 0.2059 | 0.103614 | -49.68% | 0.0696327 | 512 / 512 |
| HRI | Base | 20261103 | 0.200144 | 0.0925752 | -53.75% | 0.0475658 | 512 / 512 |
| HRI | Base | 20261104 | 0.21155 | 0.0952698 | -54.97% | 0.0401995 | 512 / 512 |
| HRI | Align | 20261102 | 0.205961 | 0.103349 | -49.82% | 0.0678371 | 512 / 512 |
| HRI | Align | 20261103 | 0.200164 | 0.0926179 | -53.73% | 0.0452354 | 512 / 512 |
| HRI | Align | 20261104 | 0.215847 | 0.0952932 | -55.85% | 0.0395219 | 512 / 512 |
| HRI | Relation | 20261102 | 0.206279 | 0.103476 | -49.84% | 0.0703548 | 512 / 512 |
| HRI | Both | 20261102 | 0.205499 | 0.103708 | -49.53% | 0.0698526 | 512 / 512 |
| NoRef | Base | 20261102 | 0.210585 | 0.102946 | -51.11% | 0.070107 | 512 / 512 |
| NoRef | Base | 20261103 | 0.19955 | 0.0918613 | -53.97% | 0.0454914 | 512 / 512 |
| NoRef | Base | 20261104 | 0.211949 | 0.0951062 | -55.13% | 0.0388401 | 512 / 512 |
| NoRef | Align | 20261102 | 0.208094 | 0.102799 | -50.60% | 0.0676758 | 512 / 512 |
| NoRef | Align | 20261103 | 0.19948 | 0.0918619 | -53.95% | 0.0469059 | 512 / 512 |
| NoRef | Align | 20261104 | 0.215085 | 0.0949296 | -55.86% | 0.0404244 | 512 / 512 |
| NoRef | Relation | 20261102 | 0.210485 | 0.102897 | -51.11% | 0.0694437 | 512 / 512 |
| NoRef | Both | 20261102 | 0.208325 | 0.102884 | -50.61% | 0.0690112 | 512 / 512 |

## Alignment loss

Raw objective values; auxiliary coefficients are not applied in this table.

| Family | Variant | Training seed | Early mean | Late mean | Change % | Final observed value | Window samples (early / late) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HRI | Align | 20261102 | 0.680597 | 0.411416 | -39.55% | 0.474975 | 512 / 512 |
| HRI | Align | 20261103 | 0.675603 | 0.414314 | -38.67% | 0.488358 | 512 / 512 |
| HRI | Align | 20261104 | 0.677948 | 0.412323 | -39.18% | 0.527484 | 512 / 512 |
| HRI | Both | 20261102 | 0.82897 | 0.496511 | -40.11% | 0.590281 | 512 / 512 |
| NoRef | Align | 20261102 | 0.680035 | 0.411815 | -39.44% | 0.463164 | 512 / 512 |
| NoRef | Align | 20261103 | 0.670796 | 0.410477 | -38.81% | 0.478675 | 512 / 512 |
| NoRef | Align | 20261104 | 0.674496 | 0.410155 | -39.19% | 0.542106 | 512 / 512 |
| NoRef | Both | 20261102 | 0.827973 | 0.497358 | -39.93% | 0.59212 | 512 / 512 |

## Relation loss

Raw objective values; auxiliary coefficients are not applied in this table.

| Family | Variant | Training seed | Early mean | Late mean | Change % | Final observed value | Window samples (early / late) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HRI | Relation | 20261102 | 0.0189477 | 0.00746478 | -60.60% | 0.00720053 | 512 / 512 |
| HRI | Both | 20261102 | 0.0189966 | 0.00738812 | -61.11% | 0.00710048 | 512 / 512 |
| NoRef | Relation | 20261102 | 0.0192514 | 0.00743059 | -61.40% | 0.00702831 | 512 / 512 |
| NoRef | Both | 20261102 | 0.0193222 | 0.00738809 | -61.76% | 0.0074873 | 512 / 512 |

## Seed summaries

Sample standard deviation across training seeds; these summarize loss trajectories, not evaluation quality or independent per-update replicates.

| Family | Variant | Loss | Seeds | Early mean ± SD | Late mean ± SD | Change % mean ± SD |
| --- | --- | --- | ---: | --- | --- | --- |
| HRI | Align | alignment | 3 | 0.678049 ± 0.00249876 | 0.412684 ± 0.00148213 | -39.1355 ± 0.439629 |
| HRI | Align | flow | 3 | 0.207324 ± 0.00793006 | 0.0970866 ± 0.00558572 | -53.1339 ± 3.05885 |
| HRI | Base | flow | 3 | 0.205865 ± 0.00570275 | 0.0971529 ± 0.00575517 | -52.7964 ± 2.76899 |
| HRI | Both | alignment | 1 (indicative) | 0.82897 ± n/a | 0.496511 ± n/a | -40.1051 ± n/a |
| HRI | Both | flow | 1 (indicative) | 0.205499 ± n/a | 0.103708 ± n/a | -49.5335 ± n/a |
| HRI | Both | relation | 1 (indicative) | 0.0189966 ± n/a | 0.00738812 ± n/a | -61.1082 ± n/a |
| HRI | Relation | flow | 1 (indicative) | 0.206279 ± n/a | 0.103476 ± n/a | -49.8371 ± n/a |
| HRI | Relation | relation | 1 (indicative) | 0.0189477 ± n/a | 0.00746478 ± n/a | -60.6032 ± n/a |
| NoRef | Align | alignment | 3 | 0.675109 ± 0.00464947 | 0.410816 ± 0.000880648 | -39.1468 ± 0.31956 |
| NoRef | Align | flow | 3 | 0.207553 ± 0.00781643 | 0.0965301 ± 0.00564146 | -53.471 ± 2.66459 |
| NoRef | Base | flow | 3 | 0.207361 ± 0.00679879 | 0.096638 ± 0.00569912 | -53.4025 ± 2.06534 |
| NoRef | Both | alignment | 1 (indicative) | 0.827973 ± n/a | 0.497358 ± n/a | -39.9307 ± n/a |
| NoRef | Both | flow | 1 (indicative) | 0.208325 ± n/a | 0.102884 ± n/a | -50.6137 ± n/a |
| NoRef | Both | relation | 1 (indicative) | 0.0193222 ± n/a | 0.00738809 ± n/a | -61.7637 ± n/a |
| NoRef | Relation | flow | 1 (indicative) | 0.210485 ± n/a | 0.102897 ± n/a | -51.1142 ± n/a |
| NoRef | Relation | relation | 1 (indicative) | 0.0192514 ± n/a | 0.00743059 ± n/a | -61.4024 ± n/a |

## Exclusions and interpretation

- Excluded `/home/hugo/Projects/VSR/runs/formal_hq_v2_noref_20261102T142214.588571Z/Align/training.jsonl`: 3849 recorded updates, steps 1–3849; no matching final checkpoint. Excluded attempts are not concatenated with the selected completed logs.

The first/last-window mean decreased for 16/16 flow objectives, 8/8 active alignment objectives, and 4/4 active relation objectives. The auxiliary objectives therefore show optimization progress in these logged windows; this does not establish any quality gain.

The records sample different training clips, noise and timesteps; these are descriptive first/last-window comparisons, not paired held-out tests, statistical significance tests, or evidence of monotonic loss decrease. The final single observation can differ from the late-window mean. Lower alignment/relation training loss does not establish better restoration, temporal consistency, or gains over the augmentation-only C1 control. No feature-similarity diagnostic was run.

Reproduce into a fresh output directory (the script refuses to overwrite results):

```bash
python scripts/analyze_existing_losses.py --runs-root /home/hugo/Projects/VSR/runs --window-records 512 --analysis-date 2026-11-05 --output-dir docs/stage0_analysis_20261105_rerun
```
