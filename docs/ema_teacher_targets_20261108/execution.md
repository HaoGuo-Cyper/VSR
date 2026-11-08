# Wan TRD 教师目标实验执行记录

2026-11-08，用户在方案交付后明确指示“好的，按你这个思路开始吧”。本轮按原计划实施，不改变margin0.1、共同lambda规则、三组预算或数据。新源码位于 `/home/hugo/.codex/worktrees/vsr-wan-trd/VSR-sync`；上一轮gamma工作树和actual源码保持原样。

## 当前进度

已实现full-Gram关系损失、冻结/EMA FP32教师、半噪声native时间配对、三组训练与完整checkpoint恢复、val640/test192原生评价入口。仍为10个业务模块加init。初次组合70项CPU检查0失败/0跳过/CUDA未初始化；随后评价配置生成器已收进evaluate.py，并通过12项相关CPU检查。预飞协议与真实目标检查尚未运行。

## 入口

所有命令从新源码目录运行，使用声明的Python环境及依赖路径。每次GPUexecute必须使用protocol内固定环境、共享GPU锁及本机桌面身份检查。

- `python -m vsr.train trd-prepare --config configs/wan_trd_design_20261108.json --run-dir <new_calibration_run>` 只准备元数据。
- `python -m vsr.train trd-preflight --run-dir <calibration_run> --expected-protocol-sha <SHA>` 先做15个训练样本/噪声组合的零更新目标校准。目标不通过即停止；通过后才执行41次独立诊断更新。
- `python -m vsr.train trd-prepare --config <final_common_lambda_study.json> --run-dir <new_training_run>` 要求完整通过且同源码的预飞记录。
- `python -m vsr.train trd-execute --run-dir <training_run> --expected-protocol-sha <SHA>` 完成三组2048步；同协议恢复另加`--resume`并保留中断记录。
- `python -m vsr.evaluate trd-build-config --training-run <completed_training> --phase val --output <val_config> --run-root <root>` 生成全640矩阵配置。
- 依次使用`trd-prepare / trd-execute / trd-verify`完成原生评价；`trd-lock-finals --validation-run <val_run> --output <receipt>`锁定全部三组最终2048，不挑分数。
- 再用`trd-build-config --phase test --validation-run <val_run> --finals-receipt <receipt> ...`生成全192测试矩阵；三组全部保留。

本轮测试来源已经历史使用，公开记录这一限制。任何目标门槛/非有限/OOM失败均封存，停止依赖GPU，不自动改参数使其通过。
