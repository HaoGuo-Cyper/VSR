# 固定teacher对照：baseline / HR-VAE / HR-DINO

阶段：**test**；完整192恢复，5条训练轨迹共10240次更新=1.25RE。方法训练seed只有20261102。

**主指标：all33 LPIPS；表内SD为32个clip sample SD，不是训练seed SD。** 每clip先平均采样42/43，再等权平均不同source。

| Family / setting / λ | 更新 | 已提交step墙钟秒 | LPIPS↓ | PSNR↑ | SSIM↑ | tLP100↓ | tOF↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline/baseline/0.0 | 2048 | 704.994 | 0.11482 ± 0.03779 | 22.68875 ± 2.21099 | 0.69887 ± 0.09203 | 0.37465 ± 0.26740 | 0.24037 ± 0.32029 |
| hr_vae/low/0.11 | 2048 | 708.138 | 0.11355 ± 0.03761 | 22.66739 ± 2.20929 | 0.69612 ± 0.09189 | 0.37477 ± 0.26696 | 0.22519 ± 0.25029 |
| hr_dino/low/0.094 | 2048 | 712.342 | 0.11466 ± 0.03812 | 22.62003 ± 2.24835 | 0.69335 ± 0.09354 | 0.38639 ± 0.27848 | 0.22852 ± 0.25742 |

## 各family的validation选择

- hr_vae：vae_low，λ=0.11；最终2048 val LPIPS=0.1164383。各family独立选最低值，精确并列选小weight。
- hr_dino：dino_low，λ=0.094；最终2048 val LPIPS=0.1195549。各family独立选最低值，精确并列选小weight。

**test阶段不重新选择权重或checkpoint。原test来源被历史模型评过，不能称pristine/blind历史test。**
正式test RGB评价资产与模型评分在val选择后开启；选择前允许绑定记录的自动完整性缩略图筛查，无人工质量阅图、无模型分数。不能声称本轮此前从未访问test像素。

## Final2048相对baseline的配对描述区间

LPIPS取baseline−method，PSNR/SSIM取method−baseline；负PSNR/SSIM明确表示保真tradeoff。10000次PCG64配对clip、95%linear-percentile CI，仅描述本训练seed。

| Arm | LPIPS收益 [95%CI] | PSNR差 [95%CI] | SSIM差 [95%CI] | 结论 |
| --- | ---: | ---: | ---: | --- |
| vae_low | 0.0012671 [0.0002218, 0.0024026] | -0.0213664 [-0.0443282, 0.0043332] | -0.0027471 [-0.0040666, -0.0013438] | 本训练seed有LPIPS改善 |
| dino_low | 0.0001608 [-0.0012579, 0.0015052] | -0.0687181 [-0.1070413, -0.0151363] | -0.0055189 [-0.0066790, -0.0043201] | 证据不足 |

## 质量与效率分开解释

Test仅测冻结后的最终3个checkpoint，不用test推断收敛速度或重新调参；训练效率见val checkpoint曲线。

**时间轴只表示已提交日志中的同步训练step墙钟秒，既不是纯GPU elapsed/kernel时间，也不是总实验花费。**
离线teacher/cache、校准、模型/数据载入、checkpoint与评价I/O、丢弃或重放attempt另列；未知不记为0。以下wall总量有包含关系，不能与其子项相加。

- 绑定训练attempt成本：`{"committed_training_step_seconds": 3545.626707339543, "convergence_curve_time_basis": "committed synchronized training-step seconds; wasted work reported separately", "known_discarded_completed_step_returns": 0, "known_discarded_completed_step_seconds": 0, "successful_step_returns_this_attempt": 10240, "unreturned_failed_step_compute": "not in committed curve; included in attempt wall time, pure GPU duration unknown", "wall_seconds_all_training_attempts": 3705.88041072106}`。
- 校准preflight墙钟秒：35.94639151694719；各method梯度校准子项：`{"vae": 3.1860333040822297, "dino": 2.314769815071486}`。
- 离线DINO/cache时序：`{"feature_scope": "encode_hr wall: CPU resize/normalize + H2D + FP32 DINO forward + blocking CPU copy + time grouping; not pure GPU compute", "other_io_binding_seconds": 77.54389563563745, "other_scope": "source/teacher binding, model construction/loading, input/output IO, hashing and publication through cache seal", "pure_GPU_encoder_seconds": null, "teacher_feature_extraction_seconds": 96.91283185733482, "total_cache_build_seconds": 174.45672749297228, "total_scope": "cache builder entry through completed seal; excludes caller metadata construction, GPU admission and final cleanup"}`；receipt中的scope决定覆盖范围。
- 本阶段评价execute墙钟秒：750.5623760569142；Monotonic child process start through normal exit; includes execute model loading, restore, metrics, PNG/MP4 and cleanup; excludes prepare/verify; not pure GPU kernel time

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

Protocol SHA：`d1572aeb22f736d1c1823aae858f8c1aaba838a748c4875610507aaa2c011869`；study SHA：`13206686e3c6513867f25131967d48949d75cdb0f958f08e716ecdf79bb61216`。完整逐step五指标区间、全部候选和source记录见[JSON](analysis.json)。

曲线PNG与完整视频保存在本机outputs，未公开；delivery_manifest.json仍绑定原始本机交付完整文件（包含未上传图片）的SHA。
