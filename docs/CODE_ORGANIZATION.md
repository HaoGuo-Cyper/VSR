# 10模块主线与实验开发流程

日期：2026-11-06。用户接受按10个业务Python模块规划整个实验，README说明各文件职责。**本次完成职责/实验/重构计划与README，尚未实际合并源码，没有新增Python入口。** 本地5080开发验证，GitHub分发，H100固定提交做主实验。实验和自动任务仍暂停。

## 模块边界与现有来源

| 模块 | 收拢来源 | 接口边界 |
| --- | --- | --- |
| data.py | data/cache/prepared_cache/stage2_preparation数据部分、stage2_framewise、mask采样/条件破坏 | 固定划分、顺序、窗口、mask/pairs不变；VAE调用单独接口 |
| vae.py | preflight/preparation/inference/evaluation中的加载、归一化、encode/decode | 冻结VAE、posterior mode和归一化保持；不导入train；文本准备走明确资产接口 |
| model.py | ConditionalWan、train.make_model、restoration_architecture | 模型工厂从train移出；LoRA/残差参数名兼容3Base；不管理优化器 |
| teacher.py | TrainableEMA、teacher特征hook/forward | teacher no_grad；仅EMA LoRA/LR；交换在学生图建立前结束；不持有学生投影头 |
| losses.py | flow loss、MaskedTemporalAlignment、学生projector、token mask展开 | 不更新EMA/优化器；接收特征/目标/mask，输出损失与诊断 |
| train.py | Base与方法共享循环、NoRefContinuationStep、queue、G3/pilot模式、训练诊断 | 配置选择模式；诊断不作pilot初始化；预算分账；日志/锁放内部，不再添加阶段入口 |
| infer.py | inference、infer_video、evaluation中重复采样 | 单一UniPC实现；仅LR推理；不读HR/teacher/训练投影头/优化器；不导入train/evaluate |
| evaluate.py | T2/T4矩阵、来源绑定、行记录、G2/G4汇总接口 | 3seed/3arm轴明确；复用infer/metrics；不以一个阶段入口导入另一个入口 |
| metrics.py | RGB、perceptual_metrics、metric_pools、paired_statistics、stage2_metrics、feature统计 | 固定uint8/255、mask/pairs、seed平均规则；CPU统计与GPU计时分开 |
| checkpoints.py | checkpoints、method_checkpoint、旧train保存函数 | 学生/训练态分离；原子新文件写入；调用model工厂，不反向导入train |

__init__.py、配置、测试、第三方代码不计入10个业务模块。整理时合并重复流程、统一函数接口；旧DINO/Relation/HRI/外评工具不混入新主线，按历史索引追溯，不删除历史结果。

## 依赖方向

数据准备调用data/vae；训练调用data、model、teacher、losses、metrics和checkpoints；推理调用vae、model、checkpoints；评价调用data、infer、metrics。底层模块不反向导入train/evaluate。

设备、模型目录、数据根、run根、allowed checkpoint root、精度/backend显式传递。本机路径不作为H100隐含前提。科学设置与机器路径配置分开，最终有效配置及hash写入run。

## 重构顺序

| 任务 | 状态 | 完成标准 |
| --- | --- | --- |
| R0 职责与README | 已完成 | 本文、README、全部Python用途索引、旧README存档及项目入口；未声称源码已合并 |
| R1 公共基础 | 待做 | 收拢VAE、make_model移入model、明确data/cache/checkpoint接口，消除train/checkpoint循环 |
| R2 方法收拢 | 待做 | teacher/losses/model解耦，EMA/mask/projector/feature统计职责清晰；受影响CPU数值/RNG/梯度检查通过 |
| R3 统一入口 | 待做 | train统一Base/C1/A2/G3/pilot，infer统一采样，evaluate统一矩阵；CLI/import不自动加载模型或启动GPU |
| R4 回归与历史整理 | 待做 | 数据/指标/旧3Base兼容，相关CPU回归通过；CLI/配置说明准确，历史实现映射完整后切换主入口 |
| R5 两端小验证 | 待做，待启动指令 | 5080小闭环、H100兼容性案例事前固定；新run/源码/环境/配置可追溯，诊断与正式预算分账 |

M1服务器信息与清单规划可独立推进，部署源码在R4后固定。M2私有迁移、M3服务器适配见H100计划。R4未通过不能写“10模块已可用”；完整G2通过后才能真实G3，G3通过后才能pilot。

## 回归验收

- 数据ID/顺序、33stride1窗口、退化、cache键、固定val32/mask/原始时间pairs不变。
- 3Base参数名/shape/恢复值保持，推理只加载学生，不误用训练态EMA/projector/optimizer。
- 小模型验证关闭辅助项等价、mask局部性、teacher stop-gradient、冻结参数不变和独立RNG轨迹；EMA更新时机不变。
- 相同保存像素上的量化、PSNR/SSIM/LPIPS/TecoGAN、池和G2/G4公式与旧实现一致；保留异常值/空池语义。
- 各host明确GPU/device/admission/完整生命周期锁。H100不要求本地gjs；实际物理显存和5080开发容量目标分别记录，旧16GiB设备假设需审查。
- 新源码生成新协议；旧失败/停止run和源码由原提交追溯，不复用旧fresh-only目录，不拼接旧228行。

只重跑受影响检查。整理代码不重训3Base，不增加计划外科学实验。CPU回归不代替真实GPU门槛。

## 两机与GitHub协作

1. 5080用开发分支写代码、检查和小规模测试；H100独立checkout。
2. 发布记录完整commit与测试范围；H100运行期间固定源码和依赖，不git pull覆盖活动实验。
3. 服务器取指定版本后解析自己的路径、核对资产hash。旧固定路径/gjs/设备假设先适配，不绕过并发检查。
4. CPU→5080小GPU→H100小GPU→完整矩阵，逐层记录结果。当前GPU执行仍暂停，未来案例/预算/规则先固定。
5. 代码/配置/统计/文本日志经GitHub正常提交，不force push。同名公共文件先读取最新提交再修改，避免两端直接覆盖同一分支。
6. 权重、原始媒体、tensor缓存和凭证走私有通道。run先写本机，再导出允许公开的文本，不整体git add runs目录。

H100环境、完整矩阵、G3、pilot与正式多seed实验均未运行。完整科学任务见[H100计划](H100_MAIN_EXPERIMENT_PLAN_20261106.md)。
