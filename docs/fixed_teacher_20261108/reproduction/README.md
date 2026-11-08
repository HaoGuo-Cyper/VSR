# 本轮复现和审计工具

主运行实现仍在vsr的10个业务模块；本目录只保留本次研究的CPU数据完整性、配置生成、串行调度和分析工具副本，训练/推理不导入本目录。文件与工作区实际运行helper逐字节相同，见helper_manifest.json。

当前实验仍运行中，不能执行本目录启动重复队列。主配置和资产绑定含这台机器的显式绝对路径；在其他机器复现须准备私有数据/权重，创建对应目录并生成新的路径绑定协议，不直接重用现有run。没有权重、媒体或tensor缓存随GitHub提供。

原工作区helper集中在workspace/work，交付必须在workspace/outputs；pipeline默认after-training、拒绝正在running/planned的训练，只有显式include-training才可执行全新planned训练。waiter只等待已有服务，不启动第二个训练。每GPU子进程独立取得同一租约并校验设备/桌面身份，失败停止，没有自动加预算或改变科学设置。

先读上层plan.md、preflight_and_budget.md和CLI_GUIDE，再按实际新资产配置核心ft命令。完整试验结束后将补实际复现/恢复命令与结果位置。

## 文件用途

| 工具 | 用途 |
| --- | --- |
| fixed_teacher_data.py | 数据完整性筛查、固定val/test名单与选后RGB资产 |
| fixed_teacher_eval_configs.py | 从完整训练证据生成固定评价角色配置 |
| fixed_teacher_pipeline.py | 串行val→选择→test→分析，不自动重试 |
| wait_fixed_teacher_training_then_evaluate.py | 只等待已有训练服务正常退出再接续 |
| analyze_fixed_teacher_study.py | 全候选/逐clip/成本CSV及JSON、描述CI和两种曲线 |
| analyze_hr_vae_study.py | 上述工具复用的原CPU统计/绘图基础，不启动旧HR-only实验 |
| export_fixed_teacher_progress.py | 仅allowlist文本的增量导出 |
| fixed_teacher_training_diagnostics.py | 完成后核对数据/σ/四RNG、学生与head更新、分噪声区间诊断 |
| fixed_teacher_shared_initial_regression.py | 完整验证后核对31保留窗口的2046PNG与旧Base推理精确相同 |
| fixed_teacher_video_delivery.py | 完整test后用原PNG生成预先固定片段的五panel视频/contactsheets |
| test_*.py | 相应工具的CPU合成回归，不能替代真实实验结果 |
