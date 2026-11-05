# T0.3 external completion coverage/cost plan (2026-11-05)

Read-only plan; no GPU queue was implemented or started. T0.3 remains incomplete. The 704-row CSV/JSON maps all16 checkpoints (NoRef/HRI Base3 + Align3 + Relation1 + Both1) to the44 frozen REDS4/YouHQ40 sequence prefixes.

The old116 complete.json files are markers requiring separate strict acceptance review. They are not116 newly accepted rows. Base44 + Align44 + Relation28 belong only to NoRef training seed20261102; HRI and the other training seeds have no historical markers here. The separate successful REDS4/000 reproduction is a calibration/validation of one original row, not automatic acceptance of an expanded16-model cohort.

| Family | Model-sequence rows | Required restores | Old markers (unaccepted) | Optimistic minimum pending restores | Pending tile-restores | NoRef-rate restore-only hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NoRef | 352 | 704 | 116 | 472 | 13192 | 27.250 |
| HRI | 352 | 1408 | 0 | 1408 | 39296 | 81.173 |
| Total | 704 | 2112 | 116 | 1880 | 52488 | 108.424 |

NoRef requires two restores per row (sampling seeds42/43). HRI requires four (both seeds × reference on/off) and two full-reference T1 encodes per row. Native capacity metadata sums to1228 spatial tiles for one restore across all44 sequences. Total scope is58,944 tile-restores; old markers cover6,456 tile-restores/232 restores without acceptance. At least588 rows/1880 restores remain if all old markers eventually pass review; unaccepted markers may increase the work.

## Restore-component extrapolation

The passed NoRef REDS4/000 reproduction recorded restores of 118.863962s and 119.102613s, each16 tiles. Their mean rate is 7.436455474s per tile.

Applying that NoRef rate to52,488 minimum pending tile-restores gives 108.424 hours of restore computation alone. If all704 rows need fresh restores, the same extrapolation is 121.760 hours. These are optimistic component estimates, not rigorous wall-time bounds or completion-date promises. The two calibration-rate range is not a confidence or forecast interval.

The calibration row's stored wall time was 295.721s, including 57.755s outside the two restore phases. That overhead is disclosed, not multiplied across different sequences. Sequence-dependent throughput, HRI forward/T1 encoding, all CPU metrics, PNG/disk I/O, cold starts, review/reruns and failures remain unpriced. HRI uses a NoRef rate only as a lower-component proxy; its actual runtime has not been measured for this plan.

## Protocol and acceptance requirements

Preserve the existing labeled17-frame prefix, start0/stride1, native ×4 geometry,20-step official-config UniPC/CFG1, sampler seeds42/43,256×448 HR tiles with64 overlap. This comparison does not become full-source-sequence testing, and it does not resolve the training stride4 versus external stride1 mismatch.

Before execution, independently review the original116 rows and freeze a new explicit16-checkpoint cohort protocol. The old8-model sealed worker and its role budget do not cover an expanded16-model run. Decide acceptance/reuse under the new protocol explicitly; use fresh reviewed output directories and never append to or overwrite the paused historical run. No completion status or queue is created here.

Only existing JSON and marker/metrics-path presence were inspected. No checkpoints, PNGs, source videos or tensor payloads were opened; no file hashes were recomputed. Stored marker digests and reproduction status are provenance declarations pending the stated acceptance review.
