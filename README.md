# VSR：Wan2.1 条件视频超分独立实验

本工程已完成真实预检、两片段恢复闭环、64训练/8来源隔离验证的四组4096步pilot及有限权重搜索；正在进行1641训练/32验证的四组8192步正式轮，之后继续可选HRI和冻结协议的最终评价。pilot只提示Both可能改善部分时间指标，尚未确认模块的稳定收益。只读取旧目录中的原始 HQ-VSR 视频与经过官方逐文件校验的预训练模型，不加载旧微调、LoRA、缓存、优化器或旧生成结果。

## 运行

```bash
cd /home/hugo/Projects/VSR
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python -m unittest discover -s tests -v
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python scripts/make_manifest.py
/home/hugo/miniconda3/envs/reprvsr-pretrained/bin/python -m vsr.preflight --config configs/stage1.json
```

CUDA 运行必须能够访问宿主机 GPU；Codex 沙箱内 `torch.cuda.is_available()` 可能为 false，需要使用其宿主机执行授权。没有安装、修改驱动或旧环境。环境精确版本见 `requirements.lock.txt`、每次运行的 `environment.json`（含依赖源码 SHA256）。

每次预检新建 `runs/stage1_<UTC时间>/`，不接受 resume 参数。`runs/latest.json` 只指向最近一次尝试，失败记录保留。Base 与 Align 各自从相同官方 Wan 重新加载、建立新的优化器，以相同数据顺序、噪声、时间和3次更新检查机制。它们的 checkpoint 仅供第一阶段审计，不用于正式比较。

`scripts/make_manifest.py` 会重算原视频摘要并验证相同清单；已有清单不一致时拒绝覆盖。预检每次在新run生成文本/VAE缓存，绑定数据、模型、代码与配置；没有跨run自动复用。原始方案保存在 `docs/experiment-spec.txt`。

## 固定预检设置

- 官方 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a534b6463e45facd96134c9f345acfa5b`，保留原生16通道输入、30层 DiT、原文本通路、VAE归一化和 UniPC flow scheduler。
- ×4；9帧；HR 128×224 / LR 32×56；保留来源 FPS 与连续帧次序。固定 aspect resize + center crop；共享 bicubic antialias 空间退化，无每帧随机参数。
- VAE / UMT5 / DINOv2 冻结；LRV 上采样后 VAE encode；训练 128通道 stem 与层0、5、9的零初始化残差。层编号从0开始。
- LoRA rank=8、alpha=8，全部30层 self/cross attention 的 q/k/v/out.0；新增参数 FP32，原生 BF16及其 FP32例外保持；非重入梯度检查点。
- Align 仅取第9层输出，MLP 1536→1024→768；DINOv2-B/14 固定教师；权重0.1仅用于梯度调试。
- 真实配置推导 VAE stride=(4,8,8)，latent=(3,16,28)，tokens=(3,8,14)，HR锚点=0/4/8。锚点定义监督目标，不表示 latent 的独占感受野。完整图像缩放到224×392，DINO patch网格16×28，空间双线性映射到8×14。当前无padding，mask全有效；组件测试覆盖无效位置隔离。
- 原生 flow 方向为 noise−clean；`z_sigma=(1−sigma)*clean+sigma*noise`，保留官方scheduler配置与时间×1000约定，均匀抽取其1000格时间索引。VAE固定 posterior mode。
- 采样：标准高斯起点、官方UniPC、4步、CFG=1、seed42，只接收LRV与调试caption。3次更新与4步采样只验证闭环，不判断恢复性能。

## 数据协议

`manifests/hq_vsr.json` 是本轮新建并封存摘要的来源分组清单。source ID 按固定11字符解析，同源片段不跨集合，跨来源的相同字节文件隔离。调试只从 train 选两个固定片段；预检阶段未解码或评价val/test；后续pilot和正式训练只使用固定val子集作验证，内部test仍未用于模型选择。[DOVE作者仓库](https://github.com/zhengchen1999/DOVE#-datasets)将HQ-VSR列为训练语料，共2055段；本地2053段，完整性及差额尚未核实。本地 train/val/test 为新的内部划分，不能称为官方划分。

HQ-VSR 目前没有真实逐视频caption。统一调试提示词 `A high quality video.` 被明确标注为占位，不能据此报告文本语义遵循。用户于2026-11-02明确授权暂不考虑caption：后续训练继续使用同一固定generic文本并保留UMT5接口，不据此报告文本语义收益。

后续首轮正式训练只用 HQ-VSR，在固定数据、退化、初始化与预算下独立运行 Base/Align/Relation/Both。HQ-VSR val用于调参，internal test用于最终内部评价。REDS已由用户指定标准REDS4（000/011/015/020），YouHQ40作补充外部配对测试，均不调参；VideoLQ用于MUSIQ与播放盲评；OpenVidHD仅在模块收益验证后扩训。

Relation/Both的光流及关系损失训练入口已接入并通过组件检查；HRI适配器、0.5缺失训练和非参考帧评价入口已实现，实际训练完成情况以runs各组status/summary为准。LPIPS0.1.4与TecoGAN固定版本tLP/tOF已用官方权重与作者源码作CPU数值核对，协议在perceptual_metrics.json中单独记录。REDS4已由用户明确选定000/011/015/020，只用于外测。

## 记录与代码

- `vsr/data.py`：来源分组、确定性配对退化、解码、输出；数据缓存键绑定split、原视频hash、窗口、裁剪、退化，运行时另绑模型/代码/配置。
- `vsr/model.py`：LRV stem、残差hook、LoRA、单层投影；LRV关闭后不计算条件残差。
- `vsr/alignment.py`：实际配置几何、锚点、教师完整图像映射与valid-only loss。
- `vsr/audit.py`：官方Wan固定版本API清单验证；8个safetensors逐个SHA256；配置/tokenizer采用官方Git blob hash；DINO源码与权重hash固定。
- `vsr/preflight.py`：新run、UMT5、VAE重建、各组独立初始化、少量更新、alignment-only梯度、条件敏感度、无HR监督输入的采样。

成功run应有 `official_artifacts.json`、`correspondence.json`、`teacher_audit.json`、每clip元数据/HR/LR/LR_up/VAE视频、Base/Align的初始化、training.jsonl、新adapter、梯度、采样与status。所有视频指标需基于无损张量，MP4仅供播放。

## 来源

[Wan官方仓库](https://github.com/Wan-Video/Wan2.1)、[固定官方权重](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/0fad780a534b6463e45facd96134c9f345acfa5b)、[Diffusers Wan实现](https://github.com/huggingface/diffusers)、[DINOv2官方源码](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)。投影方案参考[REPA](https://github.com/sihyun-yu/REPA)，未直接复制其训练代码。

本工程不声称“严格同胚”或通用可解码共享空间。恢复、时间关系与参考传播是否有效均是后续待检验假设。

## 持续实验入口

```bash
python -m vsr.train --config configs/stage2_base_pilot64.json
python -m vsr.train --config configs/stage3_align_selected_pilot64.json
python -m vsr.train --config configs/stage4_relation_both_pilot64.json
```

上述命令每次从官方权重新建run；调参试验与4096步确认分开记录，不继承搜索checkpoint。`runs/active_training.json`记录最近训练，完整事实与负结果见`docs/`和每run的summary。指标按视频与所有预设种子平均；只在内部val选择权重，internal test及外测不参与选择。

## 正式轮v2与持续执行

当前正式配置为`configs/formal_hq_v2_noref.json`：1641训练片段、固定32来源隔离验证片段；每组8192更新，统一lambda_A=.01、lambda_R=1.0，从同一官方版本独立初始化。9帧来自每4原始帧采一次，跨33原始帧；配对LR/HR的FPS均为原FPS/4。该变化在正式轮前确定，没有用internal test或外部测试选择。

使用strict math-SDPA、关闭TF32及低精度归约，关系采样反向在CPU执行。`scripts/probe_determinism.py`已在本机GPU验证短路径重复；这不保证不同硬件或任意长训练位级相同。正式run的`environment.json`及逐文件摘要是启动版本依据，后续新增评价脚本另行记版本。

```bash
python -m vsr.train --config configs/formal_hq_v2_noref.json
# 全部四组完成后，封存本项目的新数据缓存；HRI仍全新初始化任务参数
python scripts/seal_cache.py /absolute/path/to/completed/formal/run
# 填写reference模板的data.cache_from后，再运行
python -m vsr.reference_training --config configs/formal_hq_v2_reference.json
# 在各组生成最终无损RGB后，CPU测感知/时间指标
python -m vsr.score_saved --run /absolute/path/to/run --variant Base --step 8192 --device cpu
python -m vsr.summarize_saved --result Base=/absolute/path/to/run/Base/step_008192 --result Align=/absolute/path/to/run/Align/step_008192 --reference Base --output reports/formal_matrix
```

不要在已有GPU训练进程运行时重复启动命令。`runs/active_training.json`及`docs/continuation-state.json`记录当前进程与后续顺序。完整四组后才生成缓存封印；可复用的仅是本工程绑定且校验通过的数据/官方文本与VAE编码，不继承此前任务权重或优化器。

`summarize_saved`严格核对来源、裁剪/退化缓存键、指标版本与采样种子；先对一个视频的全部采样种子平均，再报告视频均值。差异区间按原始来源分组bootstrap，并额外报告来源均值；这些区间不包含训练随机种子不确定性，也不作为稳定收益的单独证明。每个负结果、简单bicubic及关闭/打乱条件的结果都保留。

活动配置里早期文字字段“外部序列待指定”“LPIPS/tLP/tOF尚未实现”已过时；为保持运行配置摘要不变，原配置不追改。实际REDS选择与指标实现状态以`docs/protocol-supplement.json`和各次指标文件为准。

## 冻结评价入口

`vsr.score_saved`现在生成`perceptual_metrics_v2.json`，保留原来的评分文件。v2封存RGB评分文件、目标RGB/裁剪元数据、每个输出无损张量的SHA及实际noise_seed；`summarize_saved`只接受这些关联通过的完整结果。四组pilot已经按v2重新实测，所有指标与原评分逐项一致。

```bash
# GPU空闲后先做真实HRI图预检，不用此诊断模型初始化正式训练
python scripts/smoke_reference.py --cache-run /absolute/path/to/completed/formal/run --updates 3
# HRI正式组最终RGB产生后，参考首帧/中间帧分别测量，不合并参考选择
python -m vsr.score_reference_saved --run /absolute/path/to/reference/run --variant Both --step 8192 --device cpu
# 模型仅根据内部验证选定，native operator先在HQ-val验证；之后冻结外测
python -m vsr.evaluate_external --checkpoint Base=/absolute/path/to/run/Base/checkpoint_008192.pt --checkpoint Candidate=/absolute/path/to/run/Both/checkpoint_008192.pt --selection-note internal_validation_only --preview-max-side 960
```

外测默认REDS4+YouHQ40+VideoLQ，每条序列明确评估前17帧、原生空间尺寸×4、seed42/43、20步CFG1、HR tile256×448/overlap64。每个checkpoint从官方底座加载本工程新任务参数，仅推理，不恢复优化器。输出原尺寸无损PNG；所有外测指标从保存的uint8/255像素计算，与训练期间float32无损张量测量协议分开报告。前17帧是明确限定的外测窗口，不称为完整源视频评测；用户推理CLI默认保留实际输入的全部帧和FPS，不自动截短。原始PNG未知FPS时预览FPS单独声明。

参考评价先在完整序列计算时间指标，再剔除涉及已给参考帧的原始相邻对；不会删帧后拼出新的“相邻”关系。首/中参考分案，on/off使用相同非参考帧池，提供帧单列；无有效时间对记null，不记0。VideoLQ只报告MUSIQ补充和待收集的真实盲评，不把图像无参考分数当作运动/内容保持的充分证据。

`vsr.robustness`提供预先固定的LR压力条件：Gaussian sigma1/kernel7、噪声5/255、整片H264 CRF28/medium/yuv420p，以及按该顺序组合；帧数、尺寸、FPS必须保持。该候选协议仅用于未见退化检查，不声称已经拟合真实相机退化。CPU合成视频验证通过；尚未对内部test或外部数据评分。HQ内部评价reader将对自建test全部211段、或预先固定val32段按同样窗口/退化读取，禁止train和结果导向的test子集选择。

## 持久训练与中断恢复

2026-11-02正式Base已完成；同run的Align在3849更新时收到SIGTERM，退出码143，未完成部分不用于最终比较。准确终止来源尚未确认。Base的完整记录保留，Align/Relation/Both重新从官方权重各训练8192步，不恢复中断任务参数或优化器。额外消耗0.171064更新GPU小时单列。

`scripts/seal_prepared_cache.py`只在来源run为interrupted且预处理完整时可用；逐个重算原视频SHA，验证所有RGB、归一化latent两份一致性、文字编码、来源划分、官方配置和strict精度、静态与并发写入检查，最后原子发布数据资产seal。它不读取任务checkpoint，不把中断实验标成完成。本次1673片段/6701预处理资产已通过真实全量CPU校验。恢复配置是`configs/formal_hq_v2_remaining_noref.json`；数据缓存来自原V2正式run。

长训练通过独立systemd用户服务`vsr-training.service`执行，RuntimeMaxSec=infinity，进程组属于用户服务，日志写入`runs/console_formal_hq_v2_remaining_noref.log`。不要直接为同一任务再启动第二个GPU进程；先查`runs/active_training_service.json`、服务真实PID/状态及GPU进程，不能只信active_training.json的running文字。

```bash
systemctl --user status vsr-training.service --no-pager
tail -n 12 runs/console_formal_hq_v2_remaining_noref.log
# 配对/未见退化正式评价，checkpoint只由内部val选择
python -m vsr.evaluate_hq --help
# 完成VideoLQ机器评价后，用已封存PNG准备匿名人评包
python scripts/build_blind_review.py --evaluation-run /absolute/path/to/completed/evaluation --labels Base Candidate --output /absolute/path/to/new/blind_package
```

盲评participant.zip只有匿名视频、离线中文viewer和空评分表，真实方法映射单独保存在private/，不放入参与者ZIP。每视频包含全部固定采样种子，预选只按固定SHA顺序，不看分数。统一CPU编码、相同尺寸/FPS/质量；源PNG与刺激均留SHA。Viewer的内联JS已通过Node语法解析。尚未生成真实VideoLQ盲评材料或收集人工评分。

## 完整机制诊断

`vsr.noref_feature_diagnostics`在GPU空闲后，对本工程最终无参考Align/Relation/Both checkpoint读取封存的固定val32，重新计算官方DINO/RAFT目标后释放教师，再恢复任务模型；不读取旧teacher/flow payload或含它们的top-level cache。固定sigma=.5、t=500及原全局clip_index噪声，零更新，记录对齐/空间与时间关系误差、有效边、方差/有效秩、对应retrieval，以及纯输入描述性的光流位移、可见率、Laplacian纹理量。Base无投影头明确不适用。

`vsr.reference_diagnostics`针对本工程最终HRI checkpoint，分别在on/off与已知参考索引下测A-only、R-only梯度，不把FM总梯度当成辅助监督证据；固定val32单独记录特征/关系诊断。该流程要求在HRI预处理完成且任何variant训练目录创建之前，封存teacher/flow/单帧参考缓存payload的不可覆盖SHA seal。未来参考runner已在该边界接入`seal_reference_supervision`并单独计时；不能在训练结束后补造此seal。

```bash
python -m vsr.noref_feature_diagnostics --checkpoint runs/<no-reference-run>/Both/checkpoint_008192.pt
python -m vsr.reference_diagnostics --checkpoint runs/<reference-run>/Both/checkpoint_008192.pt
```

上述均为oracle-noised-HR机制诊断，不能替代仅LRV输入的恢复评价或证明泛化收益；不得与恢复指标混表。两个入口都检查训练标记、实际systemd状态和GPU进程，拒绝与训练争用。HRI共享预处理、T1编码、加载、完整variant与完整实验墙钟已计时，突然OS终止则仍需服务日志核对。当前166项CPU检查通过；真实CUDA机制诊断和HRI smoke尚待运行。
