# VSR：带教师特征监督的 Latent Diffusion 视频超分辨率

本项目研究：在 Wan2.1-T2V-1.3B 视频超分辨率训练中，使用 EMA teacher 的中间特征监督，能否改善严重退化帧的恢复和时间一致性，并保持推理开销不变。

**当前状态（2026-11-06）：3个 Base 已训练并审计完成；Base评价在228/1152处按用户要求停止；自蒸馏正式训练尚未开始。实验停止，自动推进暂停。** 新分工为 RTX5080 本地开发验证、远程 H10080GB 主实验。当前按10个核心模块规划代码收拢，源码尚未完成合并，H100尚未部署；下列目标结构不代表新接口已经可运行。

## 模型做什么

```text
LRV → 上采样 → 冻结VAE编码 → LR latent条件 ─┐
                                          ↓
初始噪声 ─────────────────────────→ Latent Diffusion → HR latent → VAE解码 → HRV
                                          ↑
                                Teacher特征监督（仅训练）
```

LR latent通过学习的残差分支注入Wan；主干冻结，训练LoRA和LR分支。真实HR视频经VAE编码后提供flow-matching恢复目标。

主方法A2使用学生参数的EMA版本作为teacher：teacher看到完整LR条件，学生的部分LR latent条件被破坏。二者使用相同的加噪HR latent和timestep，在受损位置对齐内部特征。推理仅使用学生；teacher、训练投影头和优化器不参与推理。

## 10个核心Python文件：各自负责什么

下表是**确定的目标职责**。测试、第三方代码和历史实现不计入10个业务模块。新文件尚未完成整合；最后一栏给出现有实现，可直接查看。

| 目标文件 | 用来做什么 | 输入 → 输出 | 当前实现位置 |
| --- | --- | --- | --- |
| `data.py` | 数据读取、固定划分与窗口、×4退化、训练缓存、mask采样和评价集读取 | 视频/清单/配置 → LR/HR、缓存、mask、元数据 | [data](vsr/data.py)、[cache](vsr/cache.py)、[preparation](vsr/stage2_preparation.py)、[framewise](scripts/stage2_framewise.py)、[mask/破坏](vsr/self_distillation.py) |
| `vae.py` | 冻结VAE加载、latent归一化、LR/HR编码及HR解码 | 视频或latent → latent或视频 | [preflight](vsr/preflight.py)、[preparation](vsr/stage2_preparation.py)、[inference](vsr/inference.py)；尚无独立vae模块 |
| `model.py` | 构造Wan、LoRA和LR残差分支，注入条件与检查架构 | 官方权重/配置、加噪HR latent/LR条件 → velocity和特征 | [model](vsr/model.py)、[train.make_model](vsr/train.py)、[架构检查](vsr/restoration_architecture.py) |
| `teacher.py` | trainable-only EMA更新、teacher前向与特征捕获 | 学生/EMA参数、完整LR条件 → 无梯度teacher特征 | [EMA](vsr/self_distillation.py)、[teacher pass](vsr/self_distillation_step.py)；尚无独立teacher模块 |
| `losses.py` | flow-matching、学生投影头、受损token余弦特征损失 | 预测/目标、学生/teacher特征、mask → 损失与诊断 | [alignment](vsr/self_distillation.py)、[step](vsr/self_distillation_step.py)；尚无独立losses模块 |
| `train.py` | 统一Base/C1/A2训练、优化器/RNG、日志、资源验证与pilot模式 | 配置/训练资产/可选matched Base → checkpoint和训练日志 | [Base](vsr/train.py)、[方法step](vsr/self_distillation_step.py)、[G3](scripts/stage3_resource_gate.py)、[pilot](scripts/stage4_pilot.py) |
| `infer.py` | 仅LR输入的超分推理、UniPC、tiling与视频输出 | LR视频/学生checkpoint/采样配置 → HR视频 | [推理核心](vsr/inference.py)、[视频CLI](vsr/infer_video.py)；尚未合并为infer模块 |
| `evaluate.py` | 统一Base/方法矩阵，调用推理与指标，记录完整观测及门槛汇总 | checkpoint列表/固定资产/协议 → 观测与报告 | [Base评价](scripts/stage2_evaluate_bases.py)、[方法评价](scripts/stage4_evaluate_pilot.py) |
| `metrics.py` | RGB PSNR/SSIM、LPIPS、TecoGAN、帧/时间对池及种子统计 | 保存的预测/目标像素与mask → 指标、统计与区间 | [RGB](vsr/evaluation.py)、[感知/时序](vsr/perceptual_metrics.py)、[池](vsr/metric_pools.py)、[G2统计](scripts/stage2_metrics.py) |
| `checkpoints.py` | 学生权重加载、原子保存、训练状态与推理状态分离 | 官方模型/任务checkpoint → 恢复模型或新checkpoint | [加载](vsr/checkpoints.py)、[方法保存](vsr/method_checkpoint.py) |

整理完成后的业务结构：

```text
vsr/
  __init__.py
  data.py          vae.py           model.py
  teacher.py       losses.py        train.py
  infer.py         evaluate.py      metrics.py
  checkpoints.py
configs/           # 科学设置和两台机器各自的路径配置
tests/            # 组件、兼容性和回归测试
docs/             # 计划、协议、实验记录、历史索引
runs/             # 本机输出；大文件不提交Git
```

日常围绕数据准备、训练、推理、评价四类入口操作。Base、C1、A2和资源验证通过配置选择，共享核心函数；不再为每个新阶段建立一套相互导入的入口。具体新CLI在源码重构与验收后发布，目前以表中的现有实现为准。

详见[代码整理计划](docs/CODE_ORGANIZATION.md)；需要查找任意现有Python文件时，见[全部151个文件用途索引](docs/PYTHON_FILE_INDEX.md)。

## 实验主线

| 阶段 | 要回答的问题 | 设置 / 状态 |
| --- | --- | --- |
| Base | 普通超分的水平与受损帧弱点是什么？ | 已完成3训练seed×8192步；完整评价待H100新run |
| Base完整评价G2 | 受损帧的恢复缺口是否超过Base种子波动？ | H100新目录完整1152行，重算G2；通过后才进入G3 |
| 资源与语义检查G3 | EMA、mask、梯度和开销是否正确？ | 相关CPU检查通过；真实H100检查未运行 |
| Pilot | Teacher监督是否有额外收益？ | 同一Base分别续训flow-only、C1、A2，各2048步；未运行 |
| 正式比较 | 收益是否跨训练seed稳定？ | Pilot通过后先冻结统计方案，再做3seed比较 |
| 消融与最终评价 | 哪些设计起作用，在哪些场景有效？ | 依赖前述结果；外评、扩展与研究图表后续开展 |

**核心对照为A2与C1。** C1使用相同条件破坏但无特征监督，flow-only衡量普通续训收益。目前尚无新方法有效的结论。

首轮固定33连续帧/stride1、128×224 HR、×4、train1641/val32；H100显存增加不自动改变输入、有效batch或预算。3个已有Base继续使用。旧5080的228条评价保留为历史，不拼接进H100完整1152行矩阵。

## 本地5080开发 → GitHub → H100实验

1. **本地开发。** 独立开发分支维护代码，先跑受影响CPU测试，再按批准的小规模协议检查真实模型闭环。当前GPU工作仍暂停。
2. **发布固定版本。** 把通过检查的代码、配置和文本记录提交GitHub，记录完整commit SHA；保留正常提交历史。
3. **H100取指定提交。** 服务器clone/fetch后checkout明确SHA，使用自己的路径、环境和私有资产。正式run期间固定代码与依赖；本地后续开发使用其他提交。
4. **服务器验收。** 核对环境与资产hash，通过小样本兼容性检查后再运行完整实验。每个run记录commit、有效配置、环境、资产hash、预算与状态。
5. **回收记录。** 文本日志、统计和协议可同步GitHub；权重、原始视频、tensor缓存与大体积输出保存在获授权的私有存储。

代码分发方式：

```bash
git clone https://github.com/HaoGuo-Cyper/VSR.git
cd VSR
git fetch origin
git checkout --detach <已通过本地检查的完整commit-SHA>
```

这些命令只取得代码。**当前仓库尚不能在H100上clone后直接启动旧实验脚本**：runner仍有本机路径、固定gjs检查与设备假设，需要适配并生成新协议。`requirements.portable.txt`尚未通过全新安装验证，不能当作已验收的H100环境说明。

| 验证层级 | 验证什么 | 当前状态 |
| --- | --- | --- |
| CPU / tinyWan | 数据、mask、EMA、损失、checkpoint、指标 | 现有相关检查通过；10模块重构尚未验收 |
| 5080小规模GPU | 真实加载、少量更新、保存/载入、推理闭环 | 旧Base训练完成；新结构验证待后续启动指令 |
| H100小规模GPU | 新环境、精度、路径、有限值及资源兼容性 | 待迁移与执行；规则事前固定 |
| H100完整实验 | 完整评价、G2/G3、pilot及后续门槛 | 未运行；局部成功不代替完整实验 |

两台机器无需共享显存或同步训练。G3时间比在同一H100环境内测量。失败时保留记录、停止依赖任务并报告，不自动重试或调整强度。

## 计划与记录入口

- [H100新任务计划](docs/H100_MAIN_EXPERIMENT_PLAN_20261106.md)：已完成、下一步、私有迁移清单及实验顺序。
- [代码整理计划](docs/CODE_ORGANIZATION.md)：10模块边界、重构顺序与验收。
- [TASKS.md](TASKS.md)：执行状态；[DECISIONS.md](DECISIONS.md)：科学设置；[RESULTS.md](RESULTS.md)：真实执行记录。
- [AGENTS.md](AGENTS.md)：代理工作约束；[RESEARCH_PLAN.md](RESEARCH_PLAN.md)：技术参考，最新决策优先。
- [原README存档](docs/history/README_before_ten_module_plan_20261106.md)：旧阶段与命令，仅供追溯，不作为当前启动依据。

本仓库是研究工程，目前验证的是具体实现与协议，尚未证明自蒸馏带来稳定超分收益。
