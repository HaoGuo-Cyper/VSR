# Python文件用途：主线与历史分开

2026-11-06实际源码整理后，日常业务代码是`vsr/`的10个模块，加一个包入口。测试、第三方和保留的旧源码另列，不要求阅读全部历史文件才能运行主线。本机5080已完成一次开发smoke；完整GPU实验仍暂停。

本索引覆盖171个Python文件：10个业务模块、1个包入口、9个当前测试、138个归档Python、10个第三方文件、3个历史审计快照。归档总计144个受Git管理文件，另6个是支持文本。

## 当前10个核心模块

| 文件 | 用途 |
| --- | --- |
| [vsr/data.py](../vsr/data.py) | 配对数据、固定窗口/退化、流式缓存、mask和val32 reader。 |
| [vsr/vae.py](../vsr/vae.py) | 冻结VAE、latent归一化/几何、编码解码及缓存策略。 |
| [vsr/model.py](../vsr/model.py) | Wan/LoRA/LR分支构造、架构/精度设置和参数hash。 |
| [vsr/teacher.py](../vsr/teacher.py) | trainable-only EMA、权重交换和teacher特征捕获。 |
| [vsr/losses.py](../vsr/losses.py) | flow-matching、学生投影头和受损token余弦损失。 |
| [vsr/train.py](../vsr/train.py) | 统一Base/smoke/resource/pilot、优化器/RNG、日志、预算和门槛。 |
| [vsr/infer.py](../vsr/infer.py) | 仅LR视频CLI、原生UniPC采样和空间tiling。 |
| [vsr/evaluate.py](../vsr/evaluate.py) | 完整Base/pilot矩阵、来源/行文件链验证及G2/G4。 |
| [vsr/metrics.py](../vsr/metrics.py) | RGB/LPIPS/TecoGAN、指标池、统计和特征诊断。 |
| [vsr/checkpoints.py](../vsr/checkpoints.py) | 学生/训练态权重加载保存、原子写入和共享GPU租约。 |

## 包入口

| 文件 | 用途 |
| --- | --- |
| [vsr/__init__.py](../vsr/__init__.py) | 包说明；导入不启动实验。 |

## 当前CPU测试

| 文件 | 用途 |
| --- | --- |
| [tests/test_core_integration.py](../tests/test_core_integration.py) | 10模块依赖边界、完整归档hash、无CUDA导入、真实3Base CPU payload与原缓存/val32元数据。 |
| [tests/test_data_vae.py](../tests/test_data_vae.py) | 数据退化/流式缓存/路径映射/mask、VAE归一化/cache policy及CUDA准备准入模拟。 |
| [tests/test_evaluate_metrics.py](../tests/test_evaluate_metrics.py) | 合成完整1152行/G2/G4、缺行篡改拒绝、摘要复核及租约内失败记录。 |
| [tests/test_metrics_features.py](../tests/test_metrics_features.py) | 特征方差、有效秩、mask与对应检索。 |
| [tests/test_metrics_perceptual.py](../tests/test_metrics_perceptual.py) | 官方LPIPS/TecoGAN作者实现的CPU数值对照。 |
| [tests/test_metrics_pools.py](../tests/test_metrics_pools.py) | 原始帧/时间对池、空池与索引。 |
| [tests/test_metrics_statistics.py](../tests/test_metrics_statistics.py) | 配对clip/source bootstrap与种子不确定性边界。 |
| [tests/test_model_checkpoint_infer.py](../tests/test_model_checkpoint_infer.py) | tinyWan/旧Base兼容、原子checkpoint、tiling、模型目录覆盖与GPU租约模拟。 |
| [tests/test_teacher_losses_train.py](../tests/test_teacher_losses_train.py) | EMA/mask/损失/tinyWan step、G3计数/计时与统一训练runner拒绝路径。 |

## 归档业务模块

| 文件 | 用途 |
| --- | --- |
| [legacy/pre_refactor_20261106/vsr/__init__.py](../legacy/pre_refactor_20261106/vsr/__init__.py) | 包说明与模块入口标识。 |
| [legacy/pre_refactor_20261106/vsr/alignment.py](../legacy/pre_refactor_20261106/vsr/alignment.py) | 推导VAE/token对应关系，提取冻结DINOv2特征并计算点对点对齐。 |
| [legacy/pre_refactor_20261106/vsr/audit.py](../legacy/pre_refactor_20261106/vsr/audit.py) | 核对官方模型文件、版本及摘要，生成来源审计记录。 |
| [legacy/pre_refactor_20261106/vsr/blind_review.py](../legacy/pre_refactor_20261106/vsr/blind_review.py) | 从封存PNG生成匿名VideoLQ播放包，保留方法映射的私有记录。 |
| [legacy/pre_refactor_20261106/vsr/cache.py](../legacy/pre_refactor_20261106/vsr/cache.py) | 按需读取配对RGB缓存，校验绑定关系并限制内存占用。 |
| [legacy/pre_refactor_20261106/vsr/calibrate_relation.py](../legacy/pre_refactor_20261106/vsr/calibrate_relation.py) | 在零参数更新下运行关系损失权重校准。 |
| [legacy/pre_refactor_20261106/vsr/calibration.py](../legacy/pre_refactor_20261106/vsr/calibration.py) | 分别测量各目标的生成器梯度尺度，提出后续权重参考。 |
| [legacy/pre_refactor_20261106/vsr/checkpoints.py](../legacy/pre_refactor_20261106/vsr/checkpoints.py) | 将任务LoRA/LR权重加载到新建官方底座，验证参数和来源。 |
| [legacy/pre_refactor_20261106/vsr/data.py](../legacy/pre_refactor_20261106/vsr/data.py) | 建立HQ-VSR清单、来源隔离划分、确定性裁剪与×4配对退化。 |
| [legacy/pre_refactor_20261106/vsr/evaluate_external.py](../legacy/pre_refactor_20261106/vsr/evaluate_external.py) | 按冻结协议运行外部配对/无参考数据评价。 |
| [legacy/pre_refactor_20261106/vsr/evaluate_hq.py](../legacy/pre_refactor_20261106/vsr/evaluate_hq.py) | 按固定val32或完整内部test211及预设退化模式运行评价。 |
| [legacy/pre_refactor_20261106/vsr/evaluation.py](../legacy/pre_refactor_20261106/vsr/evaluation.py) | 实现有界诊断片段的条件采样、RGB指标和逐视频汇总。 |
| [legacy/pre_refactor_20261106/vsr/external_data.py](../legacy/pre_refactor_20261106/vsr/external_data.py) | 读取外部测试元数据和原生PNG序列，校验文件与配对关系。 |
| [legacy/pre_refactor_20261106/vsr/feature_diagnostics.py](../legacy/pre_refactor_20261106/vsr/feature_diagnostics.py) | 计算脱离梯度图的特征方差、有效秩和对应检索诊断。 |
| [legacy/pre_refactor_20261106/vsr/flow.py](../legacy/pre_refactor_20261106/vsr/flow.py) | 加载本地冻结RAFT-large，计算HR监督锚点间光流。 |
| [legacy/pre_refactor_20261106/vsr/frozen_evaluation.py](../legacy/pre_refactor_20261106/vsr/frozen_evaluation.py) | 共享冻结评价执行流程，保存无损PNG及输入/指标绑定记录。 |
| [legacy/pre_refactor_20261106/vsr/hq_evaluation_data.py](../legacy/pre_refactor_20261106/vsr/hq_evaluation_data.py) | 读取固定HQ内部划分和窗口，构造预声明的LR压力条件。 |
| [legacy/pre_refactor_20261106/vsr/infer_video.py](../legacy/pre_refactor_20261106/vsr/infer_video.py) | 用户视频推理CLI，读取LR视频及可选HRI并保存完整超分视频。 |
| [legacy/pre_refactor_20261106/vsr/inference.py](../legacy/pre_refactor_20261106/vsr/inference.py) | 实现原生×4分块恢复、全局噪声、一致重叠融合及完整帧返回。 |
| [legacy/pre_refactor_20261106/vsr/method_checkpoint.py](../legacy/pre_refactor_20261106/vsr/method_checkpoint.py) | 原子保存新方法学生checkpoint，分离EMA/优化器等训练专用状态。 |
| [legacy/pre_refactor_20261106/vsr/method_diagnostics.py](../legacy/pre_refactor_20261106/vsr/method_diagnostics.py) | 对学生特征CPU快照计算低频方差和熵有效秩统计。 |
| [legacy/pre_refactor_20261106/vsr/metric_pools.py](../legacy/pre_refactor_20261106/vsr/metric_pools.py) | 保留原始时间邻接关系，筛选参考/非参考帧及时间指标池。 |
| [legacy/pre_refactor_20261106/vsr/model.py](../legacy/pre_refactor_20261106/vsr/model.py) | 为官方Wan配置LoRA、LR stem、残差注入和特征捕获。 |
| [legacy/pre_refactor_20261106/vsr/native_operator_qa.py](../legacy/pre_refactor_20261106/vsr/native_operator_qa.py) | 编排预声明原生像素算子检查，封存结构观察与评价记录。 |
| [legacy/pre_refactor_20261106/vsr/native_qa_reader.py](../legacy/pre_refactor_20261106/vsr/native_qa_reader.py) | 读取原生像素验证输入及显式窗口，保持原始HR几何。 |
| [legacy/pre_refactor_20261106/vsr/native_qa_restore.py](../legacy/pre_refactor_20261106/vsr/native_qa_restore.py) | 包装原生恢复算子，记录tile、噪声和VAE编码结构检查。 |
| [legacy/pre_refactor_20261106/vsr/native_resource_probe.py](../legacy/pre_refactor_20261106/vsr/native_resource_probe.py) | 在固定最大几何下做短合成资源检查，不读取数据像素或评分。 |
| [legacy/pre_refactor_20261106/vsr/no_reference.py](../legacy/pre_refactor_20261106/vsr/no_reference.py) | 使用本地固定MUSIQ-KonIQ实现计算逐帧无参考质量补充指标。 |
| [legacy/pre_refactor_20261106/vsr/noref_feature_diagnostics.py](../legacy/pre_refactor_20261106/vsr/noref_feature_diagnostics.py) | 对NoRef模型运行零更新、带oracle加噪HR的特征机制诊断。 |
| [legacy/pre_refactor_20261106/vsr/paired_statistics.py](../legacy/pre_refactor_20261106/vsr/paired_statistics.py) | 汇总配对视频结果，按来源聚类估计条件不确定性。 |
| [legacy/pre_refactor_20261106/vsr/perceptual_metrics.py](../legacy/pre_refactor_20261106/vsr/perceptual_metrics.py) | 计算固定LPIPS-Alex0.1及明确裁剪/帧池的TecoGAN时间指标。 |
| [legacy/pre_refactor_20261106/vsr/preflight.py](../legacy/pre_refactor_20261106/vsr/preflight.py) | 运行早期真实模型闭环预检，包含文本/VAE准备和少量诊断更新。 |
| [legacy/pre_refactor_20261106/vsr/prepared_cache.py](../legacy/pre_refactor_20261106/vsr/prepared_cache.py) | 校验中断训练已完成的预处理资产并创建不可覆盖的缓存封印。 |
| [legacy/pre_refactor_20261106/vsr/probes.py](../legacy/pre_refactor_20261106/vsr/probes.py) | 以oracle加噪HR运行去噪和LR重投影诊断，定位训练问题。 |
| [legacy/pre_refactor_20261106/vsr/reference.py](../legacy/pre_refactor_20261106/vsr/reference.py) | 实现可选单帧HRI交叉注意力及缺失参考行为。 |
| [legacy/pre_refactor_20261106/vsr/reference_diagnostics.py](../legacy/pre_refactor_20261106/vsr/reference_diagnostics.py) | 对HRI模型分别测量辅助目标梯度及特征机制，保持零更新。 |
| [legacy/pre_refactor_20261106/vsr/reference_evaluation.py](../legacy/pre_refactor_20261106/vsr/reference_evaluation.py) | 按相同噪声比较有/无HRI输入，并排除提供参考帧的主指标贡献。 |
| [legacy/pre_refactor_20261106/vsr/reference_model.py](../legacy/pre_refactor_20261106/vsr/reference_model.py) | 将可选HRI分支接入ConditionalWan，同时保留原LR条件通路。 |
| [legacy/pre_refactor_20261106/vsr/reference_training.py](../legacy/pre_refactor_20261106/vsr/reference_training.py) | 从官方模型独立初始化并训练历史HRI各变体。 |
| [legacy/pre_refactor_20261106/vsr/relation.py](../legacy/pre_refactor_20261106/vsr/relation.py) | 计算空间及光流对应的时间关系监督。 |
| [legacy/pre_refactor_20261106/vsr/reproducibility.py](../legacy/pre_refactor_20261106/vsr/reproducibility.py) | 设置并记录随机性/严格精度策略，计算可训练参数初始化摘要。 |
| [legacy/pre_refactor_20261106/vsr/restoration_architecture.py](../legacy/pre_refactor_20261106/vsr/restoration_architecture.py) | 验证新NoRef续训架构标记及方法组名，保持历史架构兼容。 |
| [legacy/pre_refactor_20261106/vsr/robustness.py](../legacy/pre_refactor_20261106/vsr/robustness.py) | 实现预声明的模糊、噪声、codec等LR压力退化。 |
| [legacy/pre_refactor_20261106/vsr/score_reference_saved.py](../legacy/pre_refactor_20261106/vsr/score_reference_saved.py) | 对已保存HRI无损输出评分，保留原始时间对和参考排除规则。 |
| [legacy/pre_refactor_20261106/vsr/score_saved.py](../legacy/pre_refactor_20261106/vsr/score_saved.py) | 对已保存NoRef无损输出计算固定感知和时间指标。 |
| [legacy/pre_refactor_20261106/vsr/self_distillation.py](../legacy/pre_refactor_20261106/vsr/self_distillation.py) | 实现EMA、连续帧mask、latent条件破坏及masked特征对齐原语。 |
| [legacy/pre_refactor_20261106/vsr/self_distillation_step.py](../legacy/pre_refactor_20261106/vsr/self_distillation_step.py) | 实现flow-only/C1/A2统一更新步、独立RNG和matched Base续训装载。 |
| [legacy/pre_refactor_20261106/vsr/stage2_base_queue.py](../legacy/pre_refactor_20261106/vsr/stage2_base_queue.py) | 顺序执行33帧Base队列及优化器诊断，失败停止并保持独立初始化。 |
| [legacy/pre_refactor_20261106/vsr/stage2_preparation.py](../legacy/pre_refactor_20261106/vsr/stage2_preparation.py) | 流式生成33帧Base预处理缓存，复用明确绑定的固定文本编码。 |
| [legacy/pre_refactor_20261106/vsr/summarize_reference_saved.py](../legacy/pre_refactor_20261106/vsr/summarize_reference_saved.py) | 配对汇总HRI已保存结果，分别保留不同参考选择。 |
| [legacy/pre_refactor_20261106/vsr/summarize_saved.py](../legacy/pre_refactor_20261106/vsr/summarize_saved.py) | 从绑定通过的已保存评分生成配对比较及来源聚类区间。 |
| [legacy/pre_refactor_20261106/vsr/summarize_training_repeats.py](../legacy/pre_refactor_20261106/vsr/summarize_training_repeats.py) | 先汇总采样重复，再报告独立训练种子的配对比较与波动。 |
| [legacy/pre_refactor_20261106/vsr/train.py](../legacy/pre_refactor_20261106/vsr/train.py) | 从官方底座构建模型并执行历史Base/Align/Relation/Both训练和诊断。 |

## 归档阶段脚本

| 文件 | 用途 |
| --- | --- |
| [legacy/pre_refactor_20261106/scripts/analyze_existing_losses.py](../legacy/pre_refactor_20261106/scripts/analyze_existing_losses.py) | 读取历史JSONL损失，生成固定窗口的训练诊断汇总。 |
| [legacy/pre_refactor_20261106/scripts/audit_hq_inventory.py](../legacy/pre_refactor_20261106/scripts/audit_hq_inventory.py) | 仅检查视频头和封存目录，统计几何与帧数符合条件的样本。 |
| [legacy/pre_refactor_20261106/scripts/audit_vae_full_resolution.py](../legacy/pre_refactor_20261106/scripts/audit_vae_full_resolution.py) | 准备/执行冻结全分辨率VAE编码对照；保留原untiled OOM历史尝试。 |
| [legacy/pre_refactor_20261106/scripts/audit_vae_reconstruction.py](../legacy/pre_refactor_20261106/scripts/audit_vae_reconstruction.py) | 准备/执行VAE重建及LR–HR latent条件差异审计。 |
| [legacy/pre_refactor_20261106/scripts/bootstrap_training_seed_uncertainty.py](../legacy/pre_refactor_20261106/scripts/bootstrap_training_seed_uncertainty.py) | 对已有val32分数做训练种子与来源交叉bootstrap补充分析。 |
| [legacy/pre_refactor_20261106/scripts/build_blind_review.py](../legacy/pre_refactor_20261106/scripts/build_blind_review.py) | 将已完成的封存PNG评价结果交给匿名播放包生成器。 |
| [legacy/pre_refactor_20261106/scripts/consolidate_existing_statistics.py](../legacy/pre_refactor_20261106/scripts/consolidate_existing_statistics.py) | 汇总既有HQ评分JSON，区分验证重复与内部test协议。 |
| [legacy/pre_refactor_20261106/scripts/frozen_external/work/external_startup_gpu_guard_v2.py](../legacy/pre_refactor_20261106/scripts/frozen_external/work/external_startup_gpu_guard_v2.py) | 历史外评启动守卫：分类完整GPU进程清单并检查运行标记。 |
| [legacy/pre_refactor_20261106/scripts/frozen_external/work/gpu_process_classification.py](../legacy/pre_refactor_20261106/scripts/frozen_external/work/gpu_process_classification.py) | 仅据OS元数据严格分类GPU进程，不执行GPU或服务命令。 |
| [legacy/pre_refactor_20261106/scripts/frozen_external/work/probe_global_HRI_encode_allocator.py](../legacy/pre_refactor_20261106/scripts/frozen_external/work/probe_global_HRI_encode_allocator.py) | 历史单次全HRI编码内存分配检查，不采样或评价超分质量。 |
| [legacy/pre_refactor_20261106/scripts/frozen_external/work/run_frozen_HQ_test_T1_policy.py](../legacy/pre_refactor_20261106/scripts/frozen_external/work/run_frozen_HQ_test_T1_policy.py) | 在原数值评价流程外加入T1 VAE缓存策略，运行历史内部test。 |
| [legacy/pre_refactor_20261106/scripts/frozen_external/work/run_frozen_external_T1_policy_desktop_v2.py](../legacy/pre_refactor_20261106/scripts/frozen_external/work/run_frozen_external_T1_policy_desktop_v2.py) | 在冻结数值协议外包装T1缓存生命周期及历史外评阶段。 |
| [legacy/pre_refactor_20261106/scripts/frozen_external/work/run_native_QA_T1_policy.py](../legacy/pre_refactor_20261106/scripts/frozen_external/work/run_native_QA_T1_policy.py) | 以T1缓存策略运行固定HQ原生算子检查并关联事件记录。 |
| [legacy/pre_refactor_20261106/scripts/frozen_external/work/t1_vae_cache_policy.py](../legacy/pre_refactor_20261106/scripts/frozen_external/work/t1_vae_cache_policy.py) | 为指定Wan VAE实例安装/移除仅推理使用的T1缓存保留策略。 |
| [legacy/pre_refactor_20261106/scripts/link_local_assets.py](../legacy/pre_refactor_20261106/scripts/link_local_assets.py) | 建立指向本地已有资产的链接，不复制或修改原内容。 |
| [legacy/pre_refactor_20261106/scripts/make_manifest.py](../legacy/pre_refactor_20261106/scripts/make_manifest.py) | 从HQ-VSR原始文件重新生成并核对来源划分清单。 |
| [legacy/pre_refactor_20261106/scripts/plan_external_completion.py](../legacy/pre_refactor_20261106/scripts/plan_external_completion.py) | 只读统计旧外评覆盖及预计工作量，不启动旧积压任务。 |
| [legacy/pre_refactor_20261106/scripts/prepare_external_data.py](../legacy/pre_refactor_20261106/scripts/prepare_external_data.py) | 检查/准备外部测试归档和元数据，不解码像素或运行模型。 |
| [legacy/pre_refactor_20261106/scripts/probe_determinism.py](../legacy/pre_refactor_20261106/scripts/probe_determinism.py) | 用真实模型短反向及三更新重复检查严格确定性。 |
| [legacy/pre_refactor_20261106/scripts/profile_protocol_resources.py](../legacy/pre_refactor_20261106/scripts/profile_protocol_resources.py) | 准备或测量零优化器更新的合成Base前后向资源消耗。 |
| [legacy/pre_refactor_20261106/scripts/report_learning_curves.py](../legacy/pre_refactor_20261106/scripts/report_learning_curves.py) | 从绑定通过的既有验证分数绘制固定种子学习曲线。 |
| [legacy/pre_refactor_20261106/scripts/reproduce_external_row.py](../legacy/pre_refactor_20261106/scripts/reproduce_external_row.py) | 为单条历史外评建立新协议并复现固定恢复结果。 |
| [legacy/pre_refactor_20261106/scripts/seal_cache.py](../legacy/pre_refactor_20261106/scripts/seal_cache.py) | 封存已完成实验的数据/text/VAE缓存以供受控复用。 |
| [legacy/pre_refactor_20261106/scripts/seal_prepared_cache.py](../legacy/pre_refactor_20261106/scripts/seal_prepared_cache.py) | 校验中断run的完整预处理并创建复用seal。 |
| [legacy/pre_refactor_20261106/scripts/smoke_reference.py](../legacy/pre_refactor_20261106/scripts/smoke_reference.py) | 对可选HRI分支运行隔离的真实模型小规模图检查。 |
| [legacy/pre_refactor_20261106/scripts/stage2_evaluate_bases.py](../legacy/pre_refactor_20261106/scripts/stage2_evaluate_bases.py) | 准备/执行三Base完整val32评价矩阵，并保存协议、观测和G2记录。 |
| [legacy/pre_refactor_20261106/scripts/stage2_framewise.py](../legacy/pre_refactor_20261106/scripts/stage2_framewise.py) | 封存mask/退化协议并用CPU生成固定逐帧退化val32资产。 |
| [legacy/pre_refactor_20261106/scripts/stage2_metrics.py](../legacy/pre_refactor_20261106/scripts/stage2_metrics.py) | 按固定帧/pair池聚合受损帧指标，重算三Base的G2门槛。 |
| [legacy/pre_refactor_20261106/scripts/stage3_resource_gate.py](../legacy/pre_refactor_20261106/scripts/stage3_resource_gate.py) | 准备/执行真实G3语义及资源诊断，区分70诊断更新与正式训练。 |
| [legacy/pre_refactor_20261106/scripts/stage4_evaluate_pilot.py](../legacy/pre_refactor_20261106/scripts/stage4_evaluate_pilot.py) | 准备/执行三组pilot完整评价，输出配对描述统计。 |
| [legacy/pre_refactor_20261106/scripts/stage4_pilot.py](../legacy/pre_refactor_20261106/scripts/stage4_pilot.py) | 在真实G2/G3通过后准备/执行三组各2048步的matched Base pilot。 |

## 归档测试

| 文件 | 用途 |
| --- | --- |
| [legacy/pre_refactor_20261106/tests/test_alignment.py](../legacy/pre_refactor_20261106/tests/test_alignment.py) | 验证VAE/token几何、DINO特征映射、mask及teacher脱梯度。 |
| [legacy/pre_refactor_20261106/tests/test_auxiliary_preparation_variants.py](../legacy/pre_refactor_20261106/tests/test_auxiliary_preparation_variants.py) | 验证各历史变体的辅助teacher/flow目标准备范围与兼容性。 |
| [legacy/pre_refactor_20261106/tests/test_blind_review.py](../legacy/pre_refactor_20261106/tests/test_blind_review.py) | 验证匿名包完整性、身份隔离、固定选样和离线viewer语法。 |
| [legacy/pre_refactor_20261106/tests/test_cache.py](../legacy/pre_refactor_20261106/tests/test_cache.py) | 验证惰性缓存读取、来源绑定、路径限制、形状及有限性。 |
| [legacy/pre_refactor_20261106/tests/test_calibration.py](../legacy/pre_refactor_20261106/tests/test_calibration.py) | 验证分目标梯度校准使用一致输入且不产生模型更新。 |
| [legacy/pre_refactor_20261106/tests/test_checkpoints.py](../legacy/pre_refactor_20261106/tests/test_checkpoints.py) | 验证checkpoint来源、键/shape/dtype及仅任务权重恢复。 |
| [legacy/pre_refactor_20261106/tests/test_data.py](../legacy/pre_refactor_20261106/tests/test_data.py) | 验证来源隔离、重复字节隔离、确定性退化和解码窗口。 |
| [legacy/pre_refactor_20261106/tests/test_evaluate_external.py](../legacy/pre_refactor_20261106/tests/test_evaluate_external.py) | 用替身组件验证外评矩阵、PNG量化、聚合和原始时间对。 |
| [legacy/pre_refactor_20261106/tests/test_evaluate_hq.py](../legacy/pre_refactor_20261106/tests/test_evaluate_hq.py) | 验证HQ固定val/test协议、量化、评分池与状态兼容检查。 |
| [legacy/pre_refactor_20261106/tests/test_evaluation.py](../legacy/pre_refactor_20261106/tests/test_evaluation.py) | 验证RGB指标已知值、逐视频权重、采样控制及状态恢复。 |
| [legacy/pre_refactor_20261106/tests/test_external_data.py](../legacy/pre_refactor_20261106/tests/test_external_data.py) | 验证外部序列元数据、归档/PNG校验和配对读取。 |
| [legacy/pre_refactor_20261106/tests/test_feature_diagnostics.py](../legacy/pre_refactor_20261106/tests/test_feature_diagnostics.py) | 验证特征方差、有效秩、有效mask和对应检索诊断。 |
| [legacy/pre_refactor_20261106/tests/test_flow.py](../legacy/pre_refactor_20261106/tests/test_flow.py) | 验证RAFT输入几何、锚点顺序、冻结行为和本地权重校验。 |
| [legacy/pre_refactor_20261106/tests/test_hq_evaluation_data.py](../legacy/pre_refactor_20261106/tests/test_hq_evaluation_data.py) | 验证完整内部test、固定val顺序、窗口和退化读取约定。 |
| [legacy/pre_refactor_20261106/tests/test_infer_video.py](../legacy/pre_refactor_20261106/tests/test_infer_video.py) | 验证用户视频CLI、完整帧/FPS、HRI几何及输出检查。 |
| [legacy/pre_refactor_20261106/tests/test_inference.py](../legacy/pre_refactor_20261106/tests/test_inference.py) | 验证×4分块、全局噪声重叠一致性、padding和完整帧返回。 |
| [legacy/pre_refactor_20261106/tests/test_method_checkpoint.py](../legacy/pre_refactor_20261106/tests/test_method_checkpoint.py) | 验证方法学生checkpoint回读、训练状态隔离和不可覆盖保存。 |
| [legacy/pre_refactor_20261106/tests/test_method_diagnostics.py](../legacy/pre_refactor_20261106/tests/test_method_diagnostics.py) | 验证CPU特征快照的所有权、std/有效秩及RNG/训练轨迹不变。 |
| [legacy/pre_refactor_20261106/tests/test_metric_pools.py](../legacy/pre_refactor_20261106/tests/test_metric_pools.py) | 验证参考排除后的帧池、原始时间对及无有效pair时的空值。 |
| [legacy/pre_refactor_20261106/tests/test_model.py](../legacy/pre_refactor_20261106/tests/test_model.py) | 验证LoRA/LR零初始化、条件分支、梯度通路和checkpoint回放。 |
| [legacy/pre_refactor_20261106/tests/test_native_operator_qa.py](../legacy/pre_refactor_20261106/tests/test_native_operator_qa.py) | 用合成输入验证原生算子检查协议、绑定和失败保留行为。 |
| [legacy/pre_refactor_20261106/tests/test_native_qa_reader.py](../legacy/pre_refactor_20261106/tests/test_native_qa_reader.py) | 验证原生像素读取的metadata-only构造、几何与输入摘要。 |
| [legacy/pre_refactor_20261106/tests/test_native_qa_restore.py](../legacy/pre_refactor_20261106/tests/test_native_qa_restore.py) | 验证原生tile、实际噪声、HRI T1编码和异常清理。 |
| [legacy/pre_refactor_20261106/tests/test_native_resource_probe.py](../legacy/pre_refactor_20261106/tests/test_native_resource_probe.py) | 验证短资源检查范围、模型加载前守卫与失败记录。 |
| [legacy/pre_refactor_20261106/tests/test_no_reference.py](../legacy/pre_refactor_20261106/tests/test_no_reference.py) | 验证本地MUSIQ预处理、权重选择和逐帧汇总。 |
| [legacy/pre_refactor_20261106/tests/test_noref_feature_diagnostics.py](../legacy/pre_refactor_20261106/tests/test_noref_feature_diagnostics.py) | 验证NoRef零更新oracle诊断、固定噪声及状态清理。 |
| [legacy/pre_refactor_20261106/tests/test_paired_statistics.py](../legacy/pre_refactor_20261106/tests/test_paired_statistics.py) | 验证采样重复与来源聚类的正确统计单位和完整配对。 |
| [legacy/pre_refactor_20261106/tests/test_perceptual_metrics.py](../legacy/pre_refactor_20261106/tests/test_perceptual_metrics.py) | 验证LPIPS/TecoGAN作者协议、裁剪、时间循环及已知结果。 |
| [legacy/pre_refactor_20261106/tests/test_prepared_cache.py](../legacy/pre_refactor_20261106/tests/test_prepared_cache.py) | 验证中断run预处理seal、允许资产及缺失/篡改拒绝。 |
| [legacy/pre_refactor_20261106/tests/test_probes.py](../legacy/pre_refactor_20261106/tests/test_probes.py) | 验证oracle去噪诊断、局部噪声RNG和共享LR重投影。 |
| [legacy/pre_refactor_20261106/tests/test_reference.py](../legacy/pre_refactor_20261106/tests/test_reference.py) | 验证HRI模块零初始化、缺失参考、帧索引和梯度。 |
| [legacy/pre_refactor_20261106/tests/test_reference_diagnostics.py](../legacy/pre_refactor_20261106/tests/test_reference_diagnostics.py) | 验证HRI分目标零更新诊断、T1缓存绑定和状态恢复。 |
| [legacy/pre_refactor_20261106/tests/test_reference_evaluation.py](../legacy/pre_refactor_20261106/tests/test_reference_evaluation.py) | 验证HRI on/off共享噪声及非参考评价池。 |
| [legacy/pre_refactor_20261106/tests/test_reference_model.py](../legacy/pre_refactor_20261106/tests/test_reference_model.py) | 验证HRI包装器与普通模型的初始等价性、RNG和梯度。 |
| [legacy/pre_refactor_20261106/tests/test_reference_timing.py](../legacy/pre_refactor_20261106/tests/test_reference_timing.py) | 验证HRI各阶段计时、异常成本记录及不改变实验绑定。 |
| [legacy/pre_refactor_20261106/tests/test_reference_training.py](../legacy/pre_refactor_20261106/tests/test_reference_training.py) | 验证逐帧参考编码、缺失参考RNG和HRI训练checkpoint状态。 |
| [legacy/pre_refactor_20261106/tests/test_relation.py](../legacy/pre_refactor_20261106/tests/test_relation.py) | 验证空间/光流对应关系损失、坐标缩放和无效边处理。 |
| [legacy/pre_refactor_20261106/tests/test_reproducibility.py](../legacy/pre_refactor_20261106/tests/test_reproducibility.py) | 验证精度/随机性配置、初始化摘要及不意外推进RNG。 |
| [legacy/pre_refactor_20261106/tests/test_restoration_architecture.py](../legacy/pre_refactor_20261106/tests/test_restoration_architecture.py) | 验证新方法架构标记、真组名及历史Base/HRI兼容性。 |
| [legacy/pre_refactor_20261106/tests/test_robustness.py](../legacy/pre_refactor_20261106/tests/test_robustness.py) | 验证固定模糊/噪声/codec压力退化及输入几何。 |
| [legacy/pre_refactor_20261106/tests/test_score_reference_saved.py](../legacy/pre_refactor_20261106/tests/test_score_reference_saved.py) | 验证HRI保存输出评分时的参考排除与采样完整性。 |
| [legacy/pre_refactor_20261106/tests/test_self_distillation.py](../legacy/pre_refactor_20261106/tests/test_self_distillation.py) | 验证EMA数学、权重交换、latent mask/破坏及masked特征损失。 |
| [legacy/pre_refactor_20261106/tests/test_self_distillation_step.py](../legacy/pre_refactor_20261106/tests/test_self_distillation_step.py) | 验证tinyWan统一更新步、Base/C1等价路径、teacher信息差和RNG。 |
| [legacy/pre_refactor_20261106/tests/test_smoke_reference.py](../legacy/pre_refactor_20261106/tests/test_smoke_reference.py) | 验证HRI小检查的缓存来源限制和计算图/缺失参考行为。 |
| [legacy/pre_refactor_20261106/tests/test_stage2_evaluate_bases.py](../legacy/pre_refactor_20261106/tests/test_stage2_evaluate_bases.py) | 验证三Base评价准备、依赖绑定、GPU守卫、恢复生命周期和完整矩阵。 |
| [legacy/pre_refactor_20261106/tests/test_stage2_preparation.py](../legacy/pre_refactor_20261106/tests/test_stage2_preparation.py) | 验证流式33帧缓存准备、封存复用和篡改预先拒绝。 |
| [legacy/pre_refactor_20261106/tests/test_stage2_queue.py](../legacy/pre_refactor_20261106/tests/test_stage2_queue.py) | 验证正式/诊断配置约束和队列/worker GPU租约生命周期。 |
| [legacy/pre_refactor_20261106/tests/test_stage3_resource_gate.py](../legacy/pre_refactor_20261106/tests/test_stage3_resource_gate.py) | 验证G3元数据、完整G2前置、70更新预算及开销聚合规则。 |
| [legacy/pre_refactor_20261106/tests/test_stage4_evaluate_pilot.py](../legacy/pre_refactor_20261106/tests/test_stage4_evaluate_pilot.py) | 验证事前评价criteria、完整三组配对及元数据/失败处理。 |
| [legacy/pre_refactor_20261106/tests/test_stage4_pilot.py](../legacy/pre_refactor_20261106/tests/test_stage4_pilot.py) | 验证pilot预算、G3前置、lazy latent、生命周期和完成记录。 |
| [legacy/pre_refactor_20261106/tests/test_summarize_reference_saved.py](../legacy/pre_refactor_20261106/tests/test_summarize_reference_saved.py) | 验证HRI配对汇总、参考分案、来源聚类及seal检查。 |
| [legacy/pre_refactor_20261106/tests/test_summarize_saved.py](../legacy/pre_refactor_20261106/tests/test_summarize_saved.py) | 验证已保存结果的完整种子、协议一致性、像素和来源绑定。 |
| [legacy/pre_refactor_20261106/tests/test_summarize_training_repeats.py](../legacy/pre_refactor_20261106/tests/test_summarize_training_repeats.py) | 验证训练种子等权、采样重复先汇总及跨种子来源匹配。 |
| [legacy/pre_refactor_20261106/tests/test_temporal_stride.py](../legacy/pre_refactor_20261106/tests/test_temporal_stride.py) | 验证stride1/4时间索引、FPS、配对退化及缓存时间约定。 |

## 第三方固定源码

| 文件 | 用途 |
| --- | --- |
| [vendor/metric_sources/tecogan_metrics.py](../vendor/metric_sources/tecogan_metrics.py) | 固定TecoGAN作者指标参考源码，保留时间指标定义。 |
| [vendor/metric_sources/tecogan_networks_basic.py](../vendor/metric_sources/tecogan_networks_basic.py) | 固定TecoGAN/感知距离基础网络参考实现。 |
| [vendor/metric_sources/tecogan_pretrained_networks.py](../vendor/metric_sources/tecogan_pretrained_networks.py) | 固定TecoGAN感知距离所用预训练网络封装参考实现。 |
| [vendor/metric_sources/tecogan_util.py](../vendor/metric_sources/tecogan_util.py) | 固定TecoGAN指标参考源码的张量/图像转换辅助函数。 |
| [vendor/musiq_pytorch/__init__.py](../vendor/musiq_pytorch/__init__.py) | 固定IQA-PyTorch MUSIQ子集的包标识及来源说明。 |
| [vendor/musiq_pytorch/arch_util.py](../vendor/musiq_pytorch/arch_util.py) | MUSIQ模型结构辅助函数；禁用联网加载。 |
| [vendor/musiq_pytorch/multiscale_trans_util.py](../vendor/musiq_pytorch/multiscale_trans_util.py) | MUSIQ多尺度Transformer图像预处理。 |
| [vendor/musiq_pytorch/musiq_arch.py](../vendor/musiq_pytorch/musiq_arch.py) | MUSIQ多尺度图像质量Transformer网络。 |
| [vendor/musiq_pytorch/padding.py](../vendor/musiq_pytorch/padding.py) | MUSIQ所需padding/边界扩展算子。 |
| [vendor/musiq_pytorch/registry.py](../vendor/musiq_pytorch/registry.py) | MUSIQ子集的模型/组件注册工具。 |

## 历史审计源码快照

| 文件 | 用途 |
| --- | --- |
| [docs/stage2_20261106/desktop_guard_proposal/candidate_cpu_fixtures.py](../docs/stage2_20261106/desktop_guard_proposal/candidate_cpu_fixtures.py) | 保存桌面守卫候选的CPU假数据检查快照，不查询实时GPU或/proc。 |
| [docs/stage2_20261106/desktop_guard_proposal/candidate_source.py](../docs/stage2_20261106/desktop_guard_proposal/candidate_source.py) | 保存已审查桌面兼容候选源码快照，实际采用情况另见采用记录。 |
| [docs/stage2_20261106/three_base_audit_source.py](../docs/stage2_20261106/three_base_audit_source.py) | 保存三Base完成审计源码快照，校验日志/缓存/checkpoint字节而不加载模型。 |
