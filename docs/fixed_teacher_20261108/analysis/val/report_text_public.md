# 固定teacher对照：baseline / HR-VAE / HR-DINO

阶段：**val**；完整1024恢复，5条训练轨迹共10240次更新=1.25RE。方法训练seed只有20261102。

**主指标：all33 LPIPS；表内SD为32个clip sample SD，不是训练seed SD。** 每clip先平均采样42/43，再等权平均不同source。

| Family / setting / λ | 更新 | 已提交step墙钟秒 | LPIPS↓ | PSNR↑ | SSIM↑ | tLP100↓ | tOF↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline/baseline/0.0 | 0 | 0.000 | 0.11774 ± 0.03853 | 22.35231 ± 1.94968 | 0.69836 ± 0.09303 | 0.32106 ± 0.20122 | 0.17590 ± 0.09711 |
| baseline/baseline/0.0 | 512 | 176.067 | 0.12683 ± 0.03743 | 21.96511 ± 1.84702 | 0.68789 ± 0.09416 | 0.37149 ± 0.19071 | 0.18062 ± 0.09947 |
| baseline/baseline/0.0 | 1024 | 352.115 | 0.11782 ± 0.03661 | 22.23105 ± 2.00990 | 0.69763 ± 0.09378 | 0.33907 ± 0.19726 | 0.17501 ± 0.09677 |
| baseline/baseline/0.0 | 2048 | 704.994 | 0.11681 ± 0.03758 | 22.71339 ± 1.96300 | 0.71286 ± 0.08964 | 0.34924 ± 0.21128 | 0.17399 ± 0.09595 |
| hr_vae/low/0.11 | 0 | 0.000 | 0.11774 ± 0.03853 | 22.35231 ± 1.94968 | 0.69836 ± 0.09303 | 0.32106 ± 0.20122 | 0.17590 ± 0.09711 |
| hr_vae/low/0.11 | 512 | 177.084 | 0.12519 ± 0.03618 | 22.01320 ± 1.88589 | 0.69029 ± 0.09397 | 0.36682 ± 0.19130 | 0.18095 ± 0.10016 |
| hr_vae/low/0.11 | 1024 | 353.936 | 0.11737 ± 0.03629 | 22.28939 ± 2.00952 | 0.69888 ± 0.09309 | 0.34826 ± 0.19891 | 0.17604 ± 0.09680 |
| hr_vae/low/0.11 | 2048 | 708.138 | 0.11644 ± 0.03862 | 22.68758 ± 1.96437 | 0.70969 ± 0.09086 | 0.34913 ± 0.20951 | 0.17529 ± 0.09671 |
| hr_vae/high/1.1 | 0 | 0.000 | 0.11774 ± 0.03853 | 22.35231 ± 1.94968 | 0.69836 ± 0.09303 | 0.32106 ± 0.20122 | 0.17590 ± 0.09711 |
| hr_vae/high/1.1 | 512 | 177.218 | 0.12637 ± 0.03618 | 22.06196 ± 1.92784 | 0.69170 ± 0.09424 | 0.34547 ± 0.19028 | 0.18232 ± 0.10188 |
| hr_vae/high/1.1 | 1024 | 354.180 | 0.12230 ± 0.03788 | 22.16606 ± 2.03749 | 0.69128 ± 0.09595 | 0.35146 ± 0.18742 | 0.17692 ± 0.09803 |
| hr_vae/high/1.1 | 2048 | 708.564 | 0.11998 ± 0.04061 | 22.65082 ± 1.94481 | 0.70751 ± 0.09075 | 0.35809 ± 0.20930 | 0.17868 ± 0.09993 |
| hr_dino/low/0.094 | 0 | 0.000 | 0.11774 ± 0.03853 | 22.35231 ± 1.94968 | 0.69836 ± 0.09303 | 0.32106 ± 0.20122 | 0.17590 ± 0.09711 |
| hr_dino/low/0.094 | 512 | 178.145 | 0.12738 ± 0.03753 | 21.94396 ± 1.86449 | 0.68726 ± 0.09414 | 0.37923 ± 0.19357 | 0.18212 ± 0.10113 |
| hr_dino/low/0.094 | 1024 | 356.239 | 0.11981 ± 0.03730 | 22.19840 ± 1.99943 | 0.69560 ± 0.09421 | 0.35335 ± 0.20357 | 0.17674 ± 0.09845 |
| hr_dino/low/0.094 | 2048 | 712.342 | 0.11955 ± 0.03962 | 22.62146 ± 1.97905 | 0.70714 ± 0.09137 | 0.35352 ± 0.20324 | 0.17770 ± 0.09906 |
| hr_dino/high/0.94 | 0 | 0.000 | 0.11774 ± 0.03853 | 22.35231 ± 1.94968 | 0.69836 ± 0.09303 | 0.32106 ± 0.20122 | 0.17590 ± 0.09711 |
| hr_dino/high/0.94 | 512 | 178.010 | 0.13069 ± 0.03877 | 22.02282 ± 1.93339 | 0.68824 ± 0.09378 | 0.36743 ± 0.19580 | 0.18427 ± 0.10251 |
| hr_dino/high/0.94 | 1024 | 355.708 | 0.12447 ± 0.03903 | 22.15022 ± 2.02126 | 0.69402 ± 0.09311 | 0.36260 ± 0.20958 | 0.18395 ± 0.10182 |
| hr_dino/high/0.94 | 2048 | 711.588 | 0.12693 ± 0.04066 | 22.34458 ± 1.96075 | 0.69824 ± 0.09119 | 0.36293 ± 0.21749 | 0.18370 ± 0.10185 |

## 各family的validation选择

- hr_vae：vae_low，λ=0.11；最终2048 val LPIPS=0.1164383。各family独立选最低值，精确并列选小weight。
- hr_dino：dino_low，λ=0.094；最终2048 val LPIPS=0.1195549。各family独立选最低值，精确并列选小weight。

**test阶段不重新选择权重或checkpoint。原test来源被历史模型评过，不能称pristine/blind历史test。**
正式test RGB评价资产与模型评分在val选择后开启；选择前允许绑定记录的自动完整性缩略图筛查，无人工质量阅图、无模型分数。不能声称本轮此前从未访问test像素。

## Final2048相对baseline的配对描述区间

LPIPS取baseline−method，PSNR/SSIM取method−baseline；负PSNR/SSIM明确表示保真tradeoff。10000次PCG64配对clip、95%linear-percentile CI，仅描述本训练seed。

| Arm | LPIPS收益 [95%CI] | PSNR差 [95%CI] | SSIM差 [95%CI] | 结论 |
| --- | ---: | ---: | ---: | --- |
| vae_low | 0.0003675 [-0.0006596, 0.0013534] | -0.0258105 [-0.0449866, -0.0064387] | -0.0031667 [-0.0040404, -0.0023210] | 证据不足 |
| vae_high | -0.0031704 [-0.0055638, -0.0010638] | -0.0625732 [-0.1125242, -0.0116941] | -0.0053513 [-0.0072292, -0.0033470] | 未见增益 |
| dino_low | -0.0027491 [-0.0047657, -0.0008951] | -0.0919332 [-0.1173292, -0.0674312] | -0.0057154 [-0.0071109, -0.0044104] | 未见增益 |
| dino_high | -0.0101256 [-0.0124844, -0.0078176] | -0.3688161 [-0.4346037, -0.3044289] | -0.0146164 [-0.0170797, -0.0120714] | 未见增益 |

## 质量与效率分开解释

LPIPS固定目标=baseline@2048=0.1168058；只认最早实际checkpoint，不插值。
- baseline：最早实测2048步，704.994同步训练秒。
- vae_low：最早实测2048步，708.138同步训练秒。
- vae_high：已测点均未达到目标。
- dino_low：已测点均未达到目标。
- dino_high：已测点均未达到目标。

**时间轴只表示已提交日志中的同步训练step墙钟秒，既不是纯GPU elapsed/kernel时间，也不是总实验花费。**
离线teacher/cache、校准、模型/数据载入、checkpoint与评价I/O、丢弃或重放attempt另列；未知不记为0。以下wall总量有包含关系，不能与其子项相加。

- 绑定训练attempt成本：`{"committed_training_step_seconds": 3545.626707339543, "convergence_curve_time_basis": "committed synchronized training-step seconds; wasted work reported separately", "known_discarded_completed_step_returns": 0, "known_discarded_completed_step_seconds": 0, "successful_step_returns_this_attempt": 10240, "unreturned_failed_step_compute": "not in committed curve; included in attempt wall time, pure GPU duration unknown", "wall_seconds_all_training_attempts": 3705.88041072106}`。
- 校准preflight墙钟秒：35.94639151694719；各method梯度校准子项：`{"vae": 3.1860333040822297, "dino": 2.314769815071486}`。
- 离线DINO/cache时序：`{"feature_scope": "encode_hr wall: CPU resize/normalize + H2D + FP32 DINO forward + blocking CPU copy + time grouping; not pure GPU compute", "other_io_binding_seconds": 77.54389563563745, "other_scope": "source/teacher binding, model construction/loading, input/output IO, hashing and publication through cache seal", "pure_GPU_encoder_seconds": null, "teacher_feature_extraction_seconds": 96.91283185733482, "total_cache_build_seconds": 174.45672749297228, "total_scope": "cache builder entry through completed seal; excludes caller metadata construction, GPU admission and final cleanup"}`；receipt中的scope决定覆盖范围。
- 本阶段评价execute墙钟秒：3806.6004366920097；Monotonic child process start through normal exit; includes execute model loading, restore, metrics, PNG/MP4 and cleanup; excludes prepare/verify; not pure GPU kernel time

## 逐视频表与完整交付

- [逐clip五指标CSV](per_clip_metrics.csv)：每个实际role每clip先平均采样42/43；共享step0仅一份，不复制为五条独立结果。
- [逐clip配对差CSV](paired_clip_differences.csv)：保留全部候选/更新及所选VAE对DINO；PSNR/SSIM为method−reference，其余为reference−method，正值均有利于method。
- [role汇总CSV](summary.csv)、[全部配对区间CSV](paired_summary.csv)、[全部候选设置CSV](candidate_settings.csv)。
- [成本CSV](costs.csv)与[校准/梯度/资源/重建诊断CSV](calibration_diagnostics.csv)：保留原绑定来源和覆盖范围；包含关系的总量与子项不能相加。
- [JSON](analysis.json)保留完整数组与统计；[交付hash清单](delivery_manifest.json)绑定全部CSV、JSON、报告及曲线。



## 局限

- Weights are chosen only on the study-bound validation cohort. Evaluation sources excluded or replaced before training/scoring are determined by the bound data-integrity audit, not model quality. Formal test RGB evaluation assets and model scores are opened after this-round selection. A bound preselection exception permits automated integrity-only thumbnails without human quality review or model scores; this is not a claim of no earlier test-pixel access. Old models previously evaluated this test split, so it is not a historically pristine holdout.
- One training seed; paired clip intervals and sampler repetitions are not independent training repeats or a multi-seed significance test.
- Full selected33-frame128x224 crops,x4 clean only, not entire original source videos or full-resolution/external evaluation.
- Validation efficiency uses only observed checkpoints, no interpolation. Shared initial already meeting the target makes acceleration uninterpretable.
- Time axis is committed synchronized training-step wall seconds, not pure GPU elapsed/kernel time or total experiment cost. Offline DINO/cache preparation, calibration, model/data loading, checkpoint/evaluation IO and discarded/replayed work are reported separately where a bound receipt covers them; missing costs are unknown, not zero.
- LPIPS gains and PSNR/SSIM fidelity tradeoffs are separately reported; no old EMA gate is inherited.

Protocol SHA：`b638fd326211724d9f3b561d2bc1d0fa87c3deb478aff4ca2d1810d2acb64c0a`；study SHA：`13206686e3c6513867f25131967d48949d75cdb0f958f08e716ecdf79bb61216`。完整逐step五指标区间、全部候选和source记录见[JSON](analysis.json)。

曲线PNG与完整视频保存在本机outputs，未公开；delivery_manifest.json仍绑定原始本机交付完整文件（包含未上传图片）的SHA。
