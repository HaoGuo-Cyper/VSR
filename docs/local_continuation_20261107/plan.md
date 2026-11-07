# 本地继续计划

2026-11-07。用户决定继续在本地 RTX5080 完成既定工程，并授权自行处理正常推进和工程选择；H100 迁移暂缓。

已完成三个 Base 各8192步、原协议数据/缓存审计、10模块源码整理及本地开发GPU闭环。当前位置是 Stage2/T2.4，完整评价后才能判断 G2。

| 顺序 | 工作 | 状态 |
| --- | --- | --- |
| 1 | 本地环境、资产、40项相关CPU检查及独立审查 | 已通过 |
| 2 | 新目录完整评价3Base：3seed×32clip×6condition×2sample | 已完成1152次恢复、0训练更新 |
| 3 | 完整输出/来源/hash复核，重算G2并报告seed统计 | G2通过，38016PNG及1152行链通过；seed统计已发布 |
| 4 | G2通过后运行G3：70诊断更新，验证真实语义/显存/同设备开销 | 已通过，A2/flow1.406≤1.5，0科学更新 |
| 5 | G3通过后flow-only/C1/A2各2048步，再完整pilot评价及G4 | 训练6144步与完整1152评价均完成；G4因两个clean保真项失败而停止后续GPU |

已完成的Base评价run：`runs/base_evaluation_local_20261107T093027Z`。协议SHA256：`e63681c96413edddefb3a7077679e9d3d3077fdceb4a6f80135732a703c2aea7`。服务：`vsr-base-evaluation-local-20261107t093027z.service`，Restart=no。第一轮现场检查为active/running，7/1152已提交完成；该数字只是初始快照，后续以本目录run子目录的status/progress快照或实际runs声明为准。

旧run `stage2_base_eval_20261106T081159Z` 的228行和先前0行失败记录完整保留；新矩阵不resume、不复用旧行。原生UniPC20/CFG1/CPU噪声42/43、33帧stride1、128×224/x4、FP32冻结VAE、原模型/三Base、224固定NPY、mask/pairs、量化与指标均不变。必须完成全部1152行才讨论G2。

准备时发现Nautilus使用同UID的系统FileManager1 D-Bus激活路径，原“gnome”子串检查误拒绝。此时未创建正式失败run、未加载模型、0恢复/0训练。新增精确FileManager1兼容支持，同时将cgroup检查收紧到完整同UID统一路径、父子bus/endpoint一致、Ptyxis scopePID一致和最多一个gjs；完整pin、系统文件身份和256MiB上限保持。40项CPU检查和独立AST审查通过；其他数值函数/模块未变。记录见本目录的desktop相关JSON。

本run准备后，10个核心模块及绑定依赖冻结。开发仅可写未绑定scratch，文档/日志可以更新；所有GPU任务共用原actual共享锁，禁止第二队列。按既定门槛正常继续，无需重复询问；科学门槛失败、非有限、OOM或新的科学改变要保留记录、停止依赖GPU工作并报告，不自动改协议/重试。诊断权重不用于正式训练，G4失败不自动续4096。

代码和完整实验文本继续同步公开GitHub；权重、checkpoint、原视频、PNG/MP4和tensor cache保持私有。actual旧checkout仍固定41407b0且clean。

## 最新门槛结果

G2全量输出复核与独立公式审查通过，详见base_evaluation_completion_audit.json和base_seed_statistics.md。真实G3 run为resource_gate_local_20261107T105105Z，协议c38559f4ff0a4dfa66e3b71dc98de416424b74efae8245c4380fab406b5375ee；70诊断、0科学更新，精确GPU语义/显存通过，A2/flow全20平均比1.4061998794≤1.5，无删慢值。

Pilot已启动：pilot_local_20261107T105630Z，SHA2c0badf42a34c83f7c67cd1b5d1a1c200c84ba144e11b7bc5fe9362ba6390af3；G4原定义与实际G2事前绑定。实际运行状态读runs/active_stage4_pilot.json和该run/status/progress。全部三arm必须freshBase01/freshAdamW，禁止诊断权重复用。

## Pilot训练完成与完整评价

三arm训练已结束，服务退出0、41.15分钟、6144连续有限科学更新、0诊断，全部freshBase01/AdamW/原G4事前绑定和12个checkpoint链已核对。独立A2及三arm总审计见pilot_A2_completion_audit.json和pilot_completion_independent_review.json。

当前唯一run为pilot_evaluation_local_20261107T114256Z，协议eb1c9e2a758e5e364c380fb4ebeba42ae84bf79a881d433ac88e3e5a165dab3b，服务vsr-pilot-evaluation-local-20261107t114256z.service，目标1152次恢复、0训练。3arm均为同一训练seed20261102；不把sampler42/43当训练重复。G4定义/全数据/模型/native/runtime/量化/指标/mask-pairs均沿用冻结协议；完成全矩阵及verify后才给出G4/收益结论。当前状态读runs/active_stage4_evaluation.json。

## 最终G4结果与当前停止状态

全1152、38016PNG/source/hash及独立point/547CI重算完成。受损帧LPIPS增益.009195653>.004334114通过，但clean A2−C1 PSNR−.133037647dB、SSIM−.005554447均低于零，原G4 AND门槛未过。程序正常退出0，服务MainPID0，GPU已释放。没有改容差/规则或启动4096、sweep、Stage5。

本轮后台automation vsr已删除，避免对停止状态反复空查；它不代表整个研究目标已实现。数据、源码和全部记录保留。下一步可审阅方案见next_protocol_options_not_executed.md（仅规划，未执行），正式三seed方案仍未启动。
