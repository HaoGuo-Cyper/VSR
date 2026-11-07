# VSR 新任务计划：H100 主实验 / RTX 5080 开发验证

> 2026-11-07最新人工决定：继续在本地5080完成既定工程，H100迁移暂缓。新的完整Base评价已启动；当前入口改为 [本地继续计划](local_continuation_20261107/plan.md)。下文保留此前H100规划历史，不代表已迁移。

更新：2026-11-07（北京时间）。**10模块源码整理及本地5080开发smoke已通过；完整GPU实验仍停止，H100未连接/迁移，自动推进PAUSED。**

用户确认新增设备为另一台机器上的 H100 80GB，并要求重新整理任务，先不启动实验。本文是当前路线入口；原任务编号、历史记录和科学门槛继续保留。H100访问方式、环境和资产均尚未验收。

**最新结构决定（同日）：** 用户随后明确授权实际整理源码和CPU测试、保留历史、更新README并同步GitHub，禁止GPU实验/H100迁移。R1–R4已完成CPU范围：主线10个真实业务模块，旧144文件字节保留，130CPU测试通过且CUDA未初始化。职责见[README](../README.md)，接口见[CLI说明](CLI_GUIDE.md)，验收见[报告](code_refactor_20261106/cpu_validation.json)。M1服务器信息尚待确认；R5本地5080已通过3诊断updates＋1次推理闭环，H100小验证及完整实验尚未运行。

## 1. 研究目标和方法

目标：提升视频超分辨率，重点是局部严重退化帧的恢复和时间一致性，最终推理不增加teacher开销。

推理路径：

`LRV → 上采样 → 冻结VAE编码 → LR latent条件`

`初始噪声 + LR latent条件 → Wan latent diffusion → HR latent → 冻结VAE解码 → HRV`

训练时，真实HR视频的VAE latent提供原有flow-matching恢复目标。A2另加EMA teacher的中间特征监督：teacher看到完整LR条件，学生的部分LR latent条件被破坏；两者使用相同的加噪HR latent和timestep。Teacher仅训练使用，HR VAE latent不是本方法的teacher。

关键问题：**A2能否超过只做相同退化增强的C1，同时保持正常输入的画质？** 三个必要pilot对照为普通flow-only续训、C1、A2，均从同一个Base开始，采用相同预算。

## 2. 已完成与当前位置

| 工作 | 状态 | 可沿用的成果 / 限制 |
| --- | --- | --- |
| 代码、数据和历史结果梳理 | 已完成 | Stage0按接受的内部证据范围关闭；旧外评积压未执行、已移出范围 |
| VAE裁剪审计、LR/HR latent差异检查 | 已完成（限定范围） | 58个支持案例；旧native720p untiled OOM保留，不重试 |
| D1–D4 / G1 | 已确定 / 通过 | 33连续帧、stride1、128×224 HR、×4、NoRef、matched Base续训、EMA0.999 |
| 3个新Base | 已完成并审计通过 | seeds20261102/03/04各8192步；24576步=3RE；不因换设备重训 |
| 数据与缓存审计 | 已完成 | train1641 / val32；5024缓存文件，40,963,007,497字节，完整hash及复用记录 |
| 固定逐帧退化验证集 | 已完成 | val32、clean及5种退化、固定mask/邻帧/pairs；224个NPY资产 |
| Base正式评价 / G2 | 已暂停、未完成 | 5080完成228/1152；不能形成G2结论；目录fresh-only，不原地续跑 |
| 自蒸馏与G3资源检查代码 | CPU检查及本地5080开发smoke通过 | EMA/特征损失真实GPU3步、保存重载及1restore通过；正式70步G3及H100均未运行 |
| 三组pilot训练/评价控制器 | CPU检查及审查已完成 | 无真实方法训练；G4定义已事前记录，分析脚本仍是未部署scratch候选 |
| 正式多种子比较、消融与最终评价 | 尚未开始 | 依赖各前置门槛 |

**当前位置：Stage2 / T2.4未完成；下一项是迁移准备。** 已通过的CPU测试不机械重跑；源码改动涉及的检查及H100兼容性另行验证。

## 3. 两台机器分工

| 设备 | 职责 | 实验记录要求 |
| --- | --- | --- |
| H100 80GB（远程机器） | 主实验：统一Base评价、G3真实检查、pilot、通过门槛后的正式训练和评价 | 独立checkout、环境、私有数据/权重、run目录与GPU锁；正式比较使用一致环境 |
| RTX 5080 16GB（现有机器） | 开发验证：代码检查、独立推理、辅助验证、可视化、文本整理及GitHub同步 | 开发结果标注设备与用途；保留历史资产；GPU开发验证也需后续启动指令 |
| GitHub | 保存代码、配置、计划、统计、文本日志及hash清单 | 不存权重/checkpoint、原始媒体、tensor缓存和凭证 |

两台机器显存独立；首轮采用独立任务分工。H100本地管理学生与teacher，不依赖5080在线提供teacher。先沿用已审查的EMA执行方式，是否改成双模型同时驻留属于另一个工程变更，需要检查数值和开销。

## 4. 需要迁移到H100的内容（全部尚未迁移）

目标根目录暂记 `<H100_PROJECT_ROOT>`。服务器地址、账户、实际目录、存储配额、访问方式及调度方式待确认，不假设已经能连接。

| 内容 | 来源 / 清单 | 验收标准 |
| --- | --- | --- |
| 代码与研究记录 | staging/GitHub选定提交；AGENTS、DECISIONS、TASKS、RESULTS及本计划 | 记录提交/tree/工作区状态；旧训练runtime41407b0与后续开发源码分别记录 |
| Wan官方模型和冻结VAE | `/home/hugo/Projects/ReprVSR-pretrained/models/Wan2.1-T2V-1.3B-Diffusers/`中所需模型、VAE、scheduler配置 | 私有传输或按相同revision获取，逐文件hash；训练不驻留T5编码器 |
| 3个Base checkpoint | actual `runs/stage2_protocol33_base_seed.../Base/checkpoint_008192.pt`及配置、初始化、summary、完整审计 | 3个文件SHA与下表一致；保持原文件，不重训、不覆盖 |
| 数据与划分 | HQ-VSR源、manifest、train1641/val32选择及固定顺序记录 | 内容hash/样本ID/帧索引一致，明确原路径到新路径映射 |
| 固定评价资产 | `runs/stage2_val_assets_20261105T132924Z/dataset.json`、224个NPY及`docs/stage2_eval_20261105/cohort_seal.json` | 原seal可追溯；路径变化单独记录，32clip、退化、mask与pairs不变 |
| 训练缓存和固定文本 | 已审计5024项缓存约40.96GB、seal、固定text embedding及配置 | 优先原样复制并复核hash；如需重建，另建来源链、先验证，不静默替换 |
| 指标依赖及权重 | 冻结native/metric源码、LPIPS Alex0.1、TecoGAN相关flow/metric资产和依赖清单 | 固定算法/权重/量化/池定义；核对H100兼容环境，不能仅复制本机Conda路径 |
| 环境和存储信息 | Python/PyTorch/CUDA/驱动、精度、attention backend、确定性设置、GPU和可用磁盘 | 记录实际兼容性与存储预算，不承诺跨架构逐位一致 |
| runner与路径绑定 | 新root、allowed checkpoint root、GPU选择、GPU锁、日志和启动/停止方式 | 本机硬编码路径及固定gjs guard不可直接套用；新host admission必须可审查并拒绝未声明GPU竞争 |
| 历史停止记录 | 两个5080 T2 run的协议、失败/停止记录、228条完整文本观测 | 保留历史；不把228行拼接进H100正式矩阵 |

Base最终checkpoint SHA256：

| 训练seed | SHA256 |
| --- | --- |
| 20261102 | `28356e22e4e718d3c378588793f60732dbc93825ec5d78b8a85c8150060da63f` |
| 20261103 | `87f60ce812797d73b27b9d2d0bcb41abb45ea4659ca2b6eb6092d02f3b58061a` |
| 20261104 | `c6f0aedcefac965334851e081d72916dccfa9559b469dd1ad819d7139df65edd` |

私有资产通过获授权的机器间通道传输，凭证不写入仓库。GitHub克隆本身不能取得私有数据和权重。本次不连接H100、不复制资产、不安装环境、不加载GPU模型。

## 5. 下一步任务表（按依赖顺序）

下列待做项均未执行；未测H100前不估算完成时间。1RE=8192正式训练updates。

| 顺序 / 对应原任务 | 任务 | 位置 | 状态与完成标准 |
| --- | --- | --- | --- |
| P0 / 本次 | 整理角色、状态、迁移清单和新任务顺序 | 开发工作区（纯文档） | 已完成；计划、项目入口、决策/日志及公开文本同步 |
| R0 | 按10模块规划，重写README与现有文件用途索引 | 开发工作区（纯文档） | 已完成；尚未合并源码 |
| R1–R4 | 收拢VAE/模型工厂/teacher/losses，统一train/infer/evaluate并做相关CPU回归 | 5080开发机（仅CPU） | 已完成；130项通过，旧3Base CPU payload/缓存/val32元数据兼容，旧代码已归档 |
| R5本地 | 5080固定预算开发smoke | 5080 | 已通过3诊断updates、学生保存/重载、1次33帧UniPC20；0正式RE |
| M1 | 确定H100访问/目录/存储/调度和所选提交 | 开发机规划 + H100元数据 | 待做；形成明确迁移清单，无实验启动 |
| M2 | 迁移代码、模型、3Base、数据、缓存、固定评价及指标资产 | H100 | 待做；文件数/大小/hash/权限/路径映射验收，新输出独立目录 |
| M3 | 审查H100环境、runner及兼容性协议 | 开发机编写 / H100检查 | 待做；root/device/host admission/锁/依赖绑定、相关CPU检查；GPU小样本验收规则事前固定，执行待后续启动指令 |
| E2 / T2.4、G2 | 新目录完整评价3Base | H100 | 待做；3seed×32clip×6condition×2samplingseed=1152 restores，复核全行/source/hash链，重算G2 |
| E3 / Stage3、G3 | 真实Wan语义、显存与开销检查 | H100 | 待做；70诊断updates另行计账，所有检查通过；诊断权重不复用 |
| E4 / T4.1 | 单训练seed三组pilot：flow-only/C1/A2 | H100 | 待做；各从fresh matched Base01/fresh AdamW开始，每组2048步，共6144步=0.75RE |
| E4评 / T4.3、G4 | 统一评价并做对照分析 | H100实验 / 5080整理 | 待做；3arm×32clip×6condition×2samplingseed=1152 restores，按事前G4定义判断，报告完整类型及不确定性 |
| E5 / Stage5 | 正式三训练seed比较 | H100 | 待做；先冻结T5.0统计/指标/预算，再沿原Stage5安排flow-only/C1/A2及DINO、VAE辅助基线 |
| E6 / Stage6–9 | 消融、机制、最终外评、可选扩展、写作 | H100实验 / 5080开发整理 | 待做；依赖正式收益；旧外评积压、旧native untiled重试仍不执行 |

迁移和CPU检查不等于G2/G3通过；小样本GPU兼容性检查不代替完整1152行评价。案例、精度和验收规则须在运行前固定；出现跨平台数值异常时保留记录并调查，不改阈值求通过。

## 6. 第一轮保持的科学设置与门槛

- 保持33帧/stride1、128×224 HR、×4、train1641/val32、原有效batch与优化器设置。更长视频、更高分辨率、扩大batch另作后续实验。
- A2保留LR latent层破坏且排除latent0；nominal rho0.25/span2；student9/teacher21；gamma0.8；sigma>0.3；trainable-only EMA0.999。整数mask的实际比例照实记录。
- E2保留原native restore、UniPC20/CFG1、采样seed42/43、PNG等效量化、LPIPS/TecoGAN及冻结mask/pairs。新host、环境、源码和路径形成**新协议SHA、新run**，不冒称仍是旧协议字节hash。
- G2按既定paired excess gap与三Base clean-gap seed spread规则重算；不能根据部分分数挑退化类型或改强度。
- G3保持70诊断updates：4个语义更新+66个benchmark更新；flow/C1/A2各2warmup+20timed，所有role在同一H100上fresh matched Base01/fresh AdamW、foreach=False。主门槛mean-all20(A2)/mean-all20(flow)<=1.5，完整报告慢值、active/C1/SD/teacher频次与峰值显存。5080时间不能作H100分母。
- G4原定义文件为`configs/stage4_g4_definitions_20261106.json`，SHA256 `d552dbb136223d246e05af202f25b072a2c822ad4eeaecead0aa0d6360b1ca58`。真实H100 E2/G2通过后绑定新协议与原定义，不看method分数改门槛；scratch分析脚本审查后才部署。
- G4要求A2相对C1的受损帧LPIPS收益超过既定Base种子波动，且clean RGB PSNR/SSIM不降。单seed pilot不作为正式多seed结论；T5.0另行冻结正式统计规则。
- rho0.1/span2可能归零，相关sweep保持挂起；G4失败不自动续到4096。新失败、非有限、OOM或需改科学方案时停止并报告。

## 7. 本次交付与暂停边界

本次已实际整理主线源码、完成相关CPU回归并更新文档/公开文本。旧冻结协议与停止记录保留，未使用新源码续跑旧run。随后用户授权的本地5080固定预算smoke已完成：3诊断updates＋1restore，记录见[验证审计](development_smoke_20261107/completion_audit.json)。未访问/迁移H100，未启动完整训练/评价，自动推进保持PAUSED。

后续如决定执行，从M1开始；实验启动仍待用户后续指令。原5080 actual checkout及全部历史输出保留。全模型训练、跨机teacher、HRI或更大输入不因新增H100自动进入第一轮。
