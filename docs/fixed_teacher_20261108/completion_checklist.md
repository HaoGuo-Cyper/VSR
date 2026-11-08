# 本轮完成前必须做的检查

这些工具都已CPU验收，但真实输出必须等相应完整阶段后生成，不能把工具就绪当研究完成。所有数值源码/六个已绑定pipeline helper保持冻结。

1. 先读原workspace/work/active_fixed_teacher.json和continuation.status_file；当前训练unit034113、等待父v2 unit042551（旧034645已在0评价前superseded）。只等待的父服务绝不启动第二个训练。训练最终必须退出0/MainPID0、10240有限连续行/20CP/相同Base初始化，随后parent串行val1024/test192/分析。
2. 训练complete10240后立即运行下面CPU diagnostics（即使val已开始），验证全部数据/σ序列及相同CP的四RNG精确相同，检查head梯度/参数更新和分噪声区间损失。若出现真实公平性或非有限问题，保留记录并停止依赖GPU，不能静默改数值协议。
3. 完整val1024和核心ft-verify后运行shared_initial_regression，核31保留窗口/2046PNG与旧Base01/clean完全相同；1替换源不混入历史匹配。
4. 父pipeline必须完整test192及ft-verify成功、selection只来自最终2048valLPIPS；所有候选、512/1024/2048曲线、CSV/成本和单seed描述CI完整。
5. 完整test后运行video_delivery；查看固定3clip×sampler42/43的6份视频及contactsheet，记录细节/失真/运动与闪烁的实际观察。可另给后验失败诊断，但不可替换预声明样本或据test重选设置。
6. 完成综合报告，分别回答相同步数质量、已测训练时间下质量及是否有加速证据；把一次性缓存/校准/评估/浪费cost分开，保留未知项，不用辅助loss下降代替正常推理质量。明确单seed、历史test暴露、screen非穷尽、128×224/33帧crop、不同head大小和LoRA续训的结论边界。给第二轮建议，不能自动启动。
7. 完整文本日志/配置/CSV/报告同步GitHub；权重/CP/tensor/原视频/PNG/MP4不公开。用户交付在原workspace/outputs，真正齐备后才标F6完成、汇报，并结束本轮监控。

## CPU执行命令

```bash
FT_WORK=/home/hugo/Documents/Codex/2026-11-05/home-hugo-projects-vsr-home-hugo/work
FT_REPO=/home/hugo/.codex/worktrees/vsr-gamma-ablation/VSR-sync
FT_PYTHON=/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python
FT_TRAIN=$FT_WORK/fixed_teacher_runs/paired_training_20261108T033800Z
FT_PIPELINE=$FT_WORK/fixed_teacher_runs/pipeline_paired_training_20261108T033800Z
# 每条命令都使用CUDA_VISIBLE_DEVICES=''、OMP_NUM_THREADS=2、MKL_NUM_THREADS=2、PYTHONDONTWRITEBYTECODE=1
# 并设置PYTHONPATH=/home/hugo/Projects/VSR/.deps:$FT_REPO
$FT_PYTHON $FT_WORK/fixed_teacher_training_diagnostics.py --run $FT_TRAIN --source-root $FT_REPO --output-dir $FT_WORK/../outputs/FixedTeacherTrainingDiagnostics_20261108
$FT_PYTHON $FT_WORK/fixed_teacher_shared_initial_regression.py --historical-run $FT_WORK/VSR-sync/runs/base_evaluation_local_20261107T093027Z --validation-run $FT_PIPELINE/validation --output $FT_WORK/../outputs/FixedTeacher_shared_initial_regression_20261108.json
$FT_PYTHON $FT_WORK/fixed_teacher_video_delivery.py --source-root $FT_REPO --dependency-root /home/hugo/Projects/VSR/.deps --test-run $FT_PIPELINE/test --output-dir $FT_WORK/../outputs/video_compare_fixed_teacher_20261108
```

命令必须等上文完成门槛。每项输出使用新目录，既有报告不能覆盖；失败记录不可丢弃，恢复必须保留来源并说明。完整管线会把val/test主分析写到outputs/FixedTeacher_20261108/val和test。
