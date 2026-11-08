# VSR：带教师特征监督的 Latent Diffusion 视频超分辨率

**最新完成：VideoMAEv2 / Wan 视频超分适配（2026-11-08 UTC）。** 两组各2048步、26次诊断、完整val448/test128及独立复算和固定视频逐帧复核完成。本配置未显示明确LPIPS增益或训练加速，并有PSNR/SSIM保真度代价。见[最终报告](docs/videorepa_videomaev2_20261108/final_report.md)、[逐视频指标](docs/videorepa_videomaev2_20261108/analysis/test/per_clip_metrics.csv)、[复现与8检查点索引](docs/videorepa_videomaev2_20261108/reproduce_and_checkpoints.md)。[事前方案与必要适配](docs/videorepa_videomaev2_20261108/plan.md)保留原样；图像/视频/权重只保留本机，全部训练/逐行评价文本已公开，完成状态见[closure记录](docs/videorepa_videomaev2_20261108/completion.json)。下方保留各历史轮记录。

**已完成的 Wan TRD 第二轮（2026-11-08）：教师目标对照。** 保留现有Wan框架，比较原始基线、冻结Base教师、EMA教师，固定VideoREPA代码式关系损失与共同训练条件。预算3×2048步、41次独立预飞诊断、val640/test192，先检查目标与梯度可用性。方案见[教师目标计划](docs/ema_teacher_targets_20261108/plan.md)和[本轮执行记录](docs/ema_teacher_targets_20261108/execution.md)。上一轮完整结果保留如下。

第一轮研究问题：在相同训练条件下，用真实 HR 视频的冻结 **VAE latent** 或 **DINOv2 patch 特征**监督学生中间层，能否改善 ×4 视频超分质量或训练效率。

**已完成第一轮：基线、HR-VAE、HR-DINOv2 三组探索实验。** 一个训练种子，基线一条轨迹，两种对齐各最多两个权重，共最多五条轨迹。先测真实梯度尺度与开销，再锁定每条2048步的统一预算；所有候选保留，验证集选权重，再打开最终测试评价。方案见[执行计划](docs/fixed_teacher_20261108/plan.md)、[数据与指标定义](docs/fixed_teacher_20261108/data_and_metric_protocol.md)和[原论文与适配](docs/fixed_teacher_20261108/method_references.md)。本轮已完成：5×2048步、完整val1024和冻结选择后的test192，均通过来源/全帧/hash及独立统计核验。见[完整报告](docs/fixed_teacher_20261108/final_report.md)、[全部候选与逐视频CSV](docs/fixed_teacher_20261108/analysis/val/report_text_public.md)、[测试指标与配对区间](docs/fixed_teacher_20261108/analysis/test/report_text_public.md)及[准确复现/恢复命令和20个checkpoint索引](docs/fixed_teacher_20261108/reproduce_and_checkpoints.md)。

**本轮结果（单训练seed，探索性）：** 验证选定VAE λ0.11、DINO λ0.094。test LPIPS Baseline **0.114820**、VAE **0.113553**、DINO **0.114659**；VAE配对收益0.001267，95%描述区间[0.000222,0.002403]，同时SSIM下降。DINO的LPIPS收益区间跨0。两者未显示训练加速；不能据本轮宣称多seed普遍收益或综合质量全面提升。完整两种训练曲线、6份固定片段对比视频保存在本机outputs；本仓库只同步文本与CSV，不上传权重/媒体/cache。

本地 RTX5080 负责本轮开发与实验；H100迁移延期。旧 EMA pilot 的[完整结果](docs/local_continuation_20261107/G4_report.md)保留，其 clean 保真门槛未通过。旧EMA运行保持封存；新第二轮按上述独立协议执行。

## 模型路径

```text
LR视频 → 双三次上采样 → 冻结VAE编码 → LR latent条件 ─┐
                                                     ↓
初始噪声 ─────────────────────────────→ Latent Diffusion → HR latent → VAE解码 → HR视频
                                            ↑
真实HR → 冻结VAE 或 冻结DINOv2 → 中间层辅助监督（仅训练）
```

学生主干冻结，训练 rank8 LoRA 和 LR 残差分支，原 flow-matching 损失保持不变。两种对齐都从第8个 block（index7，共30层）取特征，使用单层 FP32 Linear：VAE输出64维，以元素平均 Smooth L1 对齐同一次干净HR latent patch；DINO输出768维，以通道归一化后的平均余弦损失对齐局部patch表示。DINO逐帧编码后按首帧独立、后续每4帧均值映射到9个latent时间位置。

第一轮不使用EMA、LR条件破坏或额外噪声门槛。各轮推理均只加载学生；HR、教师、辅助头和优化器均不进入推理。

## 每个Python文件做什么

`vsr/`包含以下10个业务模块，另有一个不启动实验的`__init__.py`。

| 文件 | 职责 | 主要输入 → 输出 |
| --- | --- | --- |
| [data.py](vsr/data.py) | 数据清单、固定划分/窗口、×4退化、流式缓存准备和校验、时序mask/条件破坏、固定val32读取 | 视频或封存资产/配置 → LR/HR、latent、mask、元数据 |
| [vae.py](vsr/vae.py) | 冻结FP32 VAE加载、latent几何与归一化、posterior-mode编码/解码、缓存策略 | 视频或latent → latent或视频 |
| [model.py](vsr/model.py) | Wan模型工厂、LoRA、LR残差注入、架构与精度设置、参数hash | 官方权重/配置及LR条件 → 模型、velocity和特征 |
| [teacher.py](vsr/teacher.py) | 冻结VideoMAEv2在线提取与位置/时间映射、DINOv2提取、中间层捕获及历史EMA | 学生/EMA参数、完整LR条件 → 无梯度teacher特征 |
| [losses.py](vsr/losses.py) | flow-matching、VideoMAEv2三层投影/fullGram、VAE Smooth L1、DINO cosine及历史损失 | 预测/目标、学生/teacher特征/mask → 损失与诊断 |
| [train.py](vsr/train.py) | 固定教师五轨迹训练/梯度校准/真实小测/完整状态续跑；保留Base及历史smoke/resource/pilot入口，管理优化器/RNG和日志 | 缓存/配置/可选matched Base与前置结果 → checkpoint、开发/训练或资源报告 |
| [infer.py](vsr/infer.py) | 仅LR推理、原生UniPC、空间tiling、视频CLI；使用固定文本缓存 | LR视频/学生checkpoint/采样设置 → HR视频 |
| [evaluate.py](vsr/evaluate.py) | 固定教师完整val/test、中间checkpoint评价与验证选参receipt、协议/来源/PNG/MP4核验及同协议行恢复；保留历史评价 | checkpoint列表/固定评价资产 → 完整观测、门槛与统计 |
| [metrics.py](vsr/metrics.py) | RGB PSNR/SSIM、LPIPS、TecoGAN、固定帧/时间对池、seed/clip统计与特征诊断 | 保存像素或封存观测 → 指标、种子统计、配对区间 |
| [checkpoints.py](vsr/checkpoints.py) | 学生权重加载、HR训练head/AdamW/RNG完整保存恢复、训练态与推理分离、原子保存及共享GPU租约 | 官方底座/任务权重/显式执行身份 → 恢复模型、新checkpoint或租约记录 |

```text
vsr/       # 10个核心模块 + __init__.py
configs/   # 科学配置、评价模板；机器路径显式指定
tests/     # 当前主线CPU回归
legacy/    # 旧业务代码、阶段脚本、测试及逐文件hash清单
vendor/    # 固定来源的第三方指标代码
docs/      # 计划、协议、验证报告和历史记录
runs/      # 本机输出，大文件不提交GitHub
```

新主线直接包含实现，不动态加载`legacy/`。训练不再承担模型工厂，推理不导入训练或评价，各阶段复用统一入口。详细边界见[代码结构](docs/CODE_ORGANIZATION.md)。

## 先从这四个入口看起

```bash
python -m vsr.data --help
python -m vsr.train --help
python -m vsr.infer --help
python -m vsr.evaluate --help
```

| 日常任务 | 入口 |
| --- | --- |
| 当前VideoMAEv2预飞/训练 | `vsr.train vmae-prepare / vmae-preflight / vmae-execute` |
| 当前VideoMAEv2评价 | `vsr.evaluate vmae-build-config / vmae-prepare / vmae-execute / vmae-verify / vmae-lock-finals` |
| 准备/检查数据 | `vsr.data manifest / prepare-cache / check-cache` |
| 第一轮校准与训练 | `vsr.train ft-calibration-prepare / ft-calibrate / ft-prepare / ft-execute` |
| 历史Base/开发入口 | `vsr.train prepare / execute`，模式`base / smoke / resource / pilot` |
| 预览权重配置/视频超分 | `vsr.infer preview / run`；run要求显式`--execute` |
| 第一轮评价、复核和选参 | `vsr.evaluate ft-prepare / ft-execute / ft-verify / ft-select` |

`prepare`只生成新协议和来源绑定，不加载GPU模型；`execute`要求明确的协议SHA、新run和GPU准入。本轮可恢复工程故障允许同协议完整状态续跑；历史停止run不混入新实验。CLI与参数示例见[使用说明](docs/CLI_GUIDE.md)。

## 本地CPU检查

使用项目依赖已准备好的Python环境：

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

测试包括tinyWan前反向、EMA、mask、数据/缓存、checkpoint、推理分块、完整1152行合成矩阵与指标回归。测试不启动GPU实验。真实LPIPS权重和3Base私有资产存在时还会运行对应CPU兼容检查，缺少时明确skip。依赖记录见`requirements.lock.txt`；`requirements.portable.txt`仍不是已验收的H100全新安装指南。

## 本轮实验顺序

| 阶段 | 工作及完成标准 |
| --- | --- |
| 数据与来源 | 固定×4双三次、33连续帧、128×224 HR；检查sourceID、文件重复和跨集合视觉近重复，封存修复后的val32/test32 |
| 正确性与短测 | 零权重基线一致、教师冻结、学生梯度、VAE重建、DINO在线缓存一致、完整保存恢复、真实显存与吞吐 |
| 统一训练 | 同Base01/fresh AdamW/数据与噪声序列，baseline1＋VAE2＋DINO2，各2048步，最多1.25RE |
| 验证与选参 | 共享起点及每条512/1024/2048步，1024次完整恢复；每方法按最终LPIPS选权重 |
| 最终测试 | 选参receipt固定后才构建正式test RGB资产；baseline/VAE/DINO最终模型共192次恢复 |
| 分析与交付 | LPIPS主指标及PSNR/SSIM/时间指标、逐视频差异、质量随步数/实际训练秒曲线、成本拆分和固定视频 |

完整恢复指已声明33帧裁剪片段，不表示原始长视频全长或全分辨率。测试源与训练隔离，但历史模型曾使用原test集合，不能称项目级盲测。自动数据完整性缩略图筛查早于选参，正式测试RGB资产及模型分数晚于选参。单训练种子结果仅作探索性证据。

旧三Base、G2/G3和EMA pilot/G4保留原始协议与结果；本轮使用同一Base01起点，不继承旧EMA质量门槛作为成功标准。

## 本地5080开发与实验 → GitHub同步

1. **5080开发维护**：写代码，运行相关CPU检查，保留10个核心模块和旧实验记录。
2. **固定执行版本**：通过检查后记录完整commit及源码hash；运行期间固定代码与依赖。
3. **本地独立run**：使用显式模型/数据/输出路径、共享GPU锁和新协议，不续跑旧停止目录。
4. **完成当前比较**：正确性/短测 → 五条统一训练 → 完整验证和选参 → 独立来源测试 → 分析交付；可恢复工程故障保留记录并完整状态续跑。
5. **同步结果**：代码、配置、协议、统计和文本日志可回GitHub；权重、原始视频、tensor缓存和凭证通过私有通道管理。

H100迁移方案保留为历史计划，当前延期，不是本地主线的前置条件。

旧checkpoint中的原配置保留，新的`--model-dir`/`--model-directory`只覆盖实际资产位置并另记审计。GPU执行共用租约，必须显式指定device/UUID；桌面进程需完整身份声明。新准入逻辑已通过本机5080桌面身份与共享租约实测；H100驱动、依赖、GPU数值和现场准入尚未验收。

## 历史与研究记录

- [旧代码归档](legacy/README.md)：原`vsr/`、`scripts/`、`tests/`共144个文件，逐字节保留，主线运行不依赖归档。
- [Python文件索引](docs/PYTHON_FILE_INDEX.md)：当前主线与历史代码分别列出。
- [H100历史计划](docs/H100_MAIN_EXPERIMENT_PLAN_20261106.md)：迁移清单和原两机分工方案，当前延期；主线以本地继续计划为准。
- [TASKS](TASKS.md)、[DECISIONS](DECISIONS.md)、[RESULTS](RESULTS.md)：任务、决策及真实实验记录。
- [旧README](docs/history/README_before_ten_module_plan_20261106.md)：历史说明，不作当前启动依据。

旧EMA结果存在恢复与保真的权衡；本轮固定教师比较尚未出分，不据辅助损失推断收益。所有历史记录保留。


Previous completed study (2026-11-08): [Wan TRD final report](docs/ema_teacher_targets_20261108/final_report.md), [reproduction and checkpoints](docs/ema_teacher_targets_20261108/reproduce_and_checkpoints.md). Full evaluation completed; EMA improvement over the original baseline remains insufficient, with additional training cost. The separately authorized [VideoMAEv2/Wan prospective plan](docs/videorepa_videomaev2_20261108/plan.md) retains the original fixed video teacher and starts only after this delivery and its own correctness/resource checks.


Latest completed study: [VideoMAEv2/Wan final report](docs/videorepa_videomaev2_20261108/final_report.md), [completion](docs/videorepa_videomaev2_20261108/completion.json). Full planned budget and delivery are complete; LPIPS gain evidence remains insufficient and no training acceleration was observed. No further experiment is running.
