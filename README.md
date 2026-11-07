# VSR：带教师特征监督的 Latent Diffusion 视频超分辨率

本项目研究：在Wan2.1-T2V-1.3B视频超分训练中，加入EMA teacher的特征监督，能否改善受损帧恢复和时间一致性，同时保持推理开销不变。

**当前状态（2026-11-07）：本地5080完整Base评价G2及真实资源检查G3均已通过；三组pilot已启动。** 3个Base各8192步已完成，旧228/1152评价保留为历史，不拼接新结果。10模块主线此前通过[130项CPU回归](docs/code_refactor_20261106/cpu_validation.json)、[19项开发入口检查](docs/development_smoke_20261107/cpu_preflight_review.json)及[真实5080保存/重载/推理闭环](docs/development_smoke_20261107/completion_audit.json)。本次严格桌面兼容修正另通过[40项相关CPU检查](docs/local_continuation_20261107/desktop_cpu_validation.json)和独立审查。当前进度见[本地继续计划与状态](docs/local_continuation_20261107/plan.md)；G2完成1152次恢复和38,016张PNG/来源复核；G3完成70诊断更新，A2/flow平均耗时比1.406≤1.5。G4和方法质量结论仍待pilot及完整评价，H100未连接或迁移。

## 模型路径

```text
LRV → 上采样 → 冻结VAE编码 → LR latent条件 ─┐
                                          ↓
初始噪声 ─────────────────────────→ Latent Diffusion → HR latent → VAE解码 → HRV
                                          ↑
                               EMA teacher特征监督（仅训练）
```

主干冻结，训练LoRA和LR残差分支。真实HR视频的VAE latent提供flow-matching恢复目标。A2的teacher看到完整LR条件，学生看到部分被破坏的LR latent条件；两者使用相同的加噪HR latent和timestep，学生在受损位置学习teacher特征。推理只使用学生。

## 每个Python文件做什么

`vsr/`包含以下10个业务模块，另有一个不启动实验的`__init__.py`。

| 文件 | 职责 | 主要输入 → 输出 |
| --- | --- | --- |
| [data.py](vsr/data.py) | 数据清单、固定划分/窗口、×4退化、流式缓存准备和校验、时序mask/条件破坏、固定val32读取 | 视频或封存资产/配置 → LR/HR、latent、mask、元数据 |
| [vae.py](vsr/vae.py) | 冻结FP32 VAE加载、latent几何与归一化、posterior-mode编码/解码、缓存策略 | 视频或latent → latent或视频 |
| [model.py](vsr/model.py) | Wan模型工厂、LoRA、LR残差注入、架构与精度设置、参数hash | 官方权重/配置及LR条件 → 模型、velocity和特征 |
| [teacher.py](vsr/teacher.py) | trainable-only EMA、安全权重交换、teacher前向和特征捕获 | 学生/EMA参数、完整LR条件 → 无梯度teacher特征 |
| [losses.py](vsr/losses.py) | flow-matching、学生投影头、masked cosine特征损失 | 预测/目标、学生/teacher特征/mask → 损失与诊断 |
| [train.py](vsr/train.py) | 统一Base训练、固定预算开发smoke、G3资源验证、flow-only/C1/A2 pilot；优化器/RNG、日志及门槛检查 | 缓存/配置/可选matched Base与前置结果 → checkpoint、开发/训练或资源报告 |
| [infer.py](vsr/infer.py) | 仅LR推理、原生UniPC、空间tiling、视频CLI；使用固定文本缓存 | LR视频/学生checkpoint/采样设置 → HR视频 |
| [evaluate.py](vsr/evaluate.py) | 统一3Base/3arm完整矩阵、协议与来源绑定、保存预测/行记录、G2/G4重算 | checkpoint列表/固定评价资产 → 完整观测、门槛与统计 |
| [metrics.py](vsr/metrics.py) | RGB PSNR/SSIM、LPIPS、TecoGAN、固定帧/时间对池、seed/clip统计与特征诊断 | 保存像素或封存观测 → 指标、种子统计、配对区间 |
| [checkpoints.py](vsr/checkpoints.py) | 学生权重加载、训练态分离、原子保存、共享GPU租约与进程准入 | 官方底座/任务权重/显式执行身份 → 恢复模型、新checkpoint或租约记录 |

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
| 准备/检查数据 | `vsr.data manifest / prepare-cache / check-cache` |
| 训练或开发/资源验证 | `vsr.train prepare / execute`，模式`base / smoke / resource / pilot` |
| 预览权重配置/视频超分 | `vsr.infer preview / run`；run要求显式`--execute` |
| 完整评价/结果复核 | `vsr.evaluate prepare / execute / verify` |

`prepare`只生成新协议和来源绑定，不加载GPU模型；`execute`要求明确的协议SHA、新run和GPU准入。旧失败/停止run不支持原地resume。CLI与参数示例见[使用说明](docs/CLI_GUIDE.md)。

## 本地CPU检查

使用项目依赖已准备好的Python环境：

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

测试包括tinyWan前反向、EMA、mask、数据/缓存、checkpoint、推理分块、完整1152行合成矩阵与指标回归。测试不启动GPU实验。真实LPIPS权重和3Base私有资产存在时还会运行对应CPU兼容检查，缺少时明确skip。依赖记录见`requirements.lock.txt`；`requirements.portable.txt`仍不是已验收的H100全新安装指南。

## 实验顺序

| 阶段 | 设置与当前状态 |
| --- | --- |
| Base | 3个训练seed各8192步，已完成并审计；继续使用 |
| 5080开发验证 | 已通过3诊断updates＋保存/学生重载＋1次UniPC20恢复；0正式RE，不代替G2/G3/G4 |
| Base评价G2 | 已完成1152 restores及完整输出复核，G2通过；旧228条单独保留 |
| 资源与语义G3 | 已完成70诊断updates，精确语义/显存通过；A2/flow=1.406≤1.5 |
| Pilot | 同Base01分别续训flow-only、C1、A2，各2048步，共0.75RE；正在本地训练 |
| Pilot评价G4 | 3arm完整1152 restores，按事前定义判断收益，待做 |
| 正式比较与扩展 | 前述门槛通过后再做多seed、消融、最终外评 |

最重要的是**A2超过C1**：C1采用同样条件破坏但没有teacher监督，flow-only衡量普通续训收益。首轮保持33连续帧/stride1、128×224 HR、×4、train1641/val32。显存增加不自动改变科学设置。

## 本地5080开发与实验 → GitHub同步

1. **5080开发维护**：写代码，运行相关CPU检查，保留10个核心模块和旧实验记录。
2. **固定执行版本**：通过检查后记录完整commit及源码hash；运行期间固定代码与依赖。
3. **本地独立run**：使用显式模型/数据/输出路径、共享GPU锁和新协议，不续跑旧停止目录。
4. **按门槛推进**：完整Base评价G2 → 真实资源与语义G3 → flow-only/C1/A2 pilot → 完整评价G4；正常通过后继续，失败保留记录并停止依赖任务。
5. **同步结果**：代码、配置、协议、统计和文本日志可回GitHub；权重、原始视频、tensor缓存和凭证通过私有通道管理。

H100迁移方案保留为历史计划，当前延期，不是本地主线的前置条件。

旧checkpoint中的原配置保留，新的`--model-dir`/`--model-directory`只覆盖实际资产位置并另记审计。GPU执行共用租约，必须显式指定device/UUID；桌面进程需完整身份声明。新准入逻辑已通过本机5080桌面身份与共享租约实测；H100驱动、依赖、GPU数值和现场准入尚未验收。

## 历史与研究记录

- [旧代码归档](legacy/README.md)：原`vsr/`、`scripts/`、`tests/`共144个文件，逐字节保留，主线运行不依赖归档。
- [Python文件索引](docs/PYTHON_FILE_INDEX.md)：当前主线与历史代码分别列出。
- [H100历史计划](docs/H100_MAIN_EXPERIMENT_PLAN_20261106.md)：迁移清单和原两机分工方案，当前延期；主线以本地继续计划为准。
- [TASKS](TASKS.md)、[DECISIONS](DECISIONS.md)、[RESULTS](RESULTS.md)：任务、决策及真实实验记录。
- [旧README](docs/history/README_before_ten_module_plan_20261106.md)：历史说明，不作当前启动依据。

尚未获得自蒸馏的真实质量结论。已有三个Base继续使用；完整评价及后续实验按本地继续计划推进，旧停止记录保留。
