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


## 真实预飞通过并进入训练

预飞run `preflight_20261108T123515Z`、protocol `a74e1a44f362a569d934696ff171f71cb26a5351beab35b0b567d361ceb6c650` 已退出0/MainPID0，41 attempted/applied/returned。15/15例主干辅助梯度为正，覆盖三个σ档；共同λ=0.3。相同输入features逐字相等、零TRD、零权重基线轨迹和两教师完整恢复通过；实际预飞独立复算0flag。

6个计时样本含慢值：baseline0.349884±0.057714秒，frozen0.494330±0.063966，EMA0.529744±0.057220；峰值allocated约3.58/3.64/3.64GiB。比值1.413/1.514只是短测成本，不继承旧G3的1.5倍门槛，不是质量结果。

唯一训练run `paired_training_20261108T124055Z`、protocol `6d5221ce42f2264e10a78c51c04b4d833a3137e146aa7aa54d2c176151b4f072` 已启动三组各2048。baseline已完整完成，初始和最终学生参数哈希均匹配上一轮同条件baseline；当前继续frozen与EMA。不能从训练损失判断最终超分收益。

串行父进程只等待现有训练退出0，再完整训练审计、val640、verify、锁定三组最终2048、test192、verify和分析；任何一步失败即停止，不重启训练或调参。运行声明见原工作区 `work/active_wan_trd.json` 与其continuation/pipeline_status。全11数值源码及父流程绑定的统计helper从现在保持不变。

公开训练日志按512完整行无损分块，`training_parts_manifest.json`给出每块及canonical总SHA；按顺序拼接可以恢复原JSONL完整字节，不丢步、不删行。权重、图像视频和tensor缓存保持本机。
