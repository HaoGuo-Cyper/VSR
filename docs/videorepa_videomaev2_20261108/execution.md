# VideoMAEv2 / Wan 超分执行记录

2026-11-08。上一轮 Wan TRD 已完整交付并以 c4b66a1e 发布；本轮在独立工作树 `/home/hugo/.codex/worktrees/vsr-videomaev2/VSR-sync` 执行用户后续授权。旧TRD、gamma、canonical和actual源码/结果保持不变。

## 实现和正确性

仍为10个业务模块加init。teacher.py严格加载官方VideoMAEv2-B的158个encoder键、冻结在线HR特征及原位置插值；losses.py采用原三层SiLU投影和全时空Gram marginmean；data.py提供与同一FM样本绑定的封存HR；train.py提供独立26更新预飞和两组4096科学训练；checkpoints.py保存学生/6个head张量/AdamW/4显式与全部全局RNG/游标/累计同步时间；evaluate.py提供完整448val/128test和双方final锁定。推理复用原student-only入口，教师/head不进入恢复。

科学设置见不可变的[前瞻方案](plan.md)，其中必须适配和资源选择明确分列。block17、MLP1536→2048→2048→768、margin.1/lambda.5/eps1e-12、teacher FP32/student relation bf16、drop首位置和8→16映射均固定。没有EMA、额外候选、教师缓存或质量出分后调参。

已通过原语/对应关系相关72项、checkpoint相关21项、完整评价相关27项、新训练10项、独立统计10项、串行parent5项CPU检查（分组可能覆盖共同依赖，不冒充互不重叠全库计数）。真实官方资源strict-load已完成；移到独立资源harness以免普通仓库测试依赖本机权重。mock预飞核26次更新/31teacher forward/30VJP；tinyWan零lambda与原NoRef三步参数/AdamW/4RNG完全相同，真实头完整恢复与全局RNG异常恢复通过。CPU不等于GPU通过。

独立审查补齐了timm实际函数源绑定、AdamW更新后有限性和native构造Python/NumPy/Torch全局RNG隔离。真实CPU metadata核1641份HR与FM/source链、正式权重/教师依赖，无CUDA初始化。预留磁盘充足；host读查旧服务MainPID0/exit0、共享锁空闲和完整桌面身份。此处仅声明实现/CPU与元数据完成；GPU状态以 `work/active_videomaev2.json` 和实际run为准。

## 明确入口

在源码目录、声明环境运行；所有GPU调用先prepare并审核protocol，唯一共享锁、Restart=no。

```bash
python -m vsr.train vmae-prepare --config configs/videorepa_videomaev2_design_20261108.json --run-dir <new_preflight_run>
python -m vsr.train vmae-preflight --run-dir <preflight_run> --expected-protocol-sha <SHA>
python -m vsr.train vmae-prepare --config <fixed_training_config_with_passed_preflight.json> --run-dir <new_training_run>
python -m vsr.train vmae-execute --run-dir <training_run> --expected-protocol-sha <SHA>
```

预飞先做4次真实教师重复提取与15个目标梯度案例；全部有限且至少8例正梯度覆盖2sigma档才继续26诊断预算的剩余检查。完整通过后新鲜Base/freshAdamW正式训练B/R各2048；诊断参数不复用。可恢复工程故障须先诊断，再显式同protocol `vmae-execute --resume`；保留attempt/rollback和浪费成本。非有限/OOM/科学失败停报，不自动重试或改强度。

完整训练审计之后：`vsr.evaluate vmae-build-config / vmae-prepare / vmae-execute / vmae-verify`；val448完整核验后用 `vmae-lock-finals` 保留双方2048，不挑分数，再完整test128。parent只等待已启动训练，不启动第二个训练。完成还需原生核验、独立统计、两条曲线、6套固定四panel视频实际复核、结果/复现及纯文本GitHub同步。

计时区分含teacher的同步Step、外置HR读取、整attempt、预飞和评价；包含关系不能重复加总。单训练seed、历史使用的source-heldout、小裁剪/LoRA续训均披露，不是原论文数值复现或新盲测。
