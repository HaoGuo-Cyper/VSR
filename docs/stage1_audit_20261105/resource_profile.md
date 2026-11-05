# Synthetic no-update resource profile

The sealed run passed its declared synthetic partial T1.3 checks. Independent CPU-only review verified both seals, all six cases and 24 forward/backward cycles, recomputed timings and peaks, and confirmed consistent zero-update and parameter-preservation records. No checkpoint tensor or GPU was opened for this review.

Source run: `/home/hugo/Projects/VSR_/runs/profile_protocol_resources_20261105T114931Z`.
Summary SHA-256: `0678eb7b152779b721c4da24d846bb6c31471967c891df10bd5feddcc669d36b`.
Protocol SHA-256: `0ef963d980e3208916b79ac84d68cb918f850a458fdaff9b30aefc64b634c4dc`.

Each case has one excluded warmup and three timed repetitions. The SD is the sample SD of those three cycle times, not training-seed spread. Peaks are the maximum of the three timed cycles, including resident model and inputs. GiB uses 2^30 bytes.

| Nominal pixel / latent frames | Condition | Cycle seconds, mean ± sample SD | Allocated peak GiB | Reserved peak GiB | Extra-forward / Base mean |
| --- | --- | ---: | ---: | ---: | ---: |
| 9 / 3 | Base forward/backward | 0.143702 ± 0.001235 | 2.846835 | 2.871094 | 1.000000× |
| 9 / 3 | One no-grad forward + Base forward/backward | 0.224949 ± 0.081070 | 2.846875 | 2.871094 | 1.565390× |
| 17 / 5 | Base forward/backward | 0.162595 ± 0.000042 | 2.908384 | 2.966797 | 1.000000× |
| 17 / 5 | One no-grad forward + Base forward/backward | 0.214272 ± 0.000686 | 2.908451 | 2.966797 | 1.317831× |
| 33 / 9 | Base forward/backward | 0.288841 ± 0.000633 | 3.059916 | 3.144531 | 1.000000× |
| 33 / 9 | One no-grad forward + Base forward/backward | 0.379086 ± 0.000193 | 3.060036 | 3.144531 | 1.312440× |

The 9-frame extra-forward condition contains a 0.318560346 s second timed cycle; the other two were approximately 0.178 s. All three remain included, giving its larger SD and 1.5654× mean ratio. The 17- and 33-frame ratios are approximately 1.318× and 1.312×. These are ratios of means without confidence intervals.

The run reports zero optimizer steps and zero parameter updates, unchanged parameter object/storage identities and version counters, and matching trainable parameter hashes before and after. The loaded-Base audit matches the same initial hash. Source inspection confirms a train-mode model, no optimizer `.step()` call, teacher `no_grad`, teacher-output retention through student backward, and gradient/feature cleanup. The report verifies these sealed records and source checks; it does not reread model tensor contents.

Trainable SHA-256 before and after: `75ed2c4429558607753fb43bdef7aa7fcf9f37130534f5ba79fcd644c5bffc0e`.

The teacher estimate reuses the same resident Base parameters in train mode. Synthetic independent random latents cover the 9→3, 17→5 and 33→9 geometries at nominal 128×224 HR size; they contain no real motion or stride-specific information. No separate EMA state, feature projector, alignment objective, optimizer state/update, gradient clipping, VAE/T5 processing or input generation is included. Timing covers synchronized forward, flow MSE, backward, zero_grad and feature clearing. These values are cycle costs; they cannot be reported as full training-step time, formal RE cost or completed T1.3. No G3 verdict or protocol decision follows from them.

GPU work remains stopped following the separately reported full-resolution OOM. This report only analyzes the already completed profile and does not propose a retry.

Files checked: `status.json`, `protocol.json`, `summary.json`, `loaded_Base.json`, `runtime.json`, `profile_progress.json`, the bound profile/helper scripts, and all 46 production-core source hashes. Exact references, raw timed values and verification scope are in `resource_profile.json`.
