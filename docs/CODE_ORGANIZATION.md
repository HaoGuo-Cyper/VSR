# 10模块主线：实际整理结果

2026-11-06，用户授权实际源码整理与相关CPU测试，明确禁止GPU实验和H100迁移。

## 已完成的结构

`vsr/`实际保留10个业务模块：data、vae、model、teacher、losses、train、infer、evaluate、metrics、checkpoints，另有__init__.py。模块职责和输入输出见[README](../README.md)。它们直接包含实现，没有导入legacy或动态载入归档包。

- 模型工厂和精度设置归model；checkpoint不再反向导入train。
- VAE与归一化归vae；infer仅依赖数据、模型、VAE和checkpoint，不导入train/evaluate。
- teacher负责EMA和特征捕获，学生投影头及监督损失在losses。
- data统一数据、流式准备、缓存和mask；metrics统一像素/时序/特征/统计定义。
- train统一Base、resource和pilot；evaluate统一3Base/3arm，保留轴区别和全部1152观测要求。
- checkpoints统一原子产物写入及GPU租约；CUDA预处理、训练、推理、评价共享显式准入。

## 状态

| 任务 | 结果 |
| --- | --- |
| R0职责规划 | 已完成，旧规划由本次实际实现更新 |
| R1公共基础 | 已完成，公共VAE/模型/数据/checkpoint接口已迁移 |
| R2方法收拢 | 已完成，EMA/损失/step及诊断已迁移 |
| R3统一入口 | 已完成，四类CLI的help和拒绝路径经过CPU检查 |
| R4回归/历史保留 | 已完成CPU范围；相关结果见[报告](code_refactor_20261106/cpu_validation.json) |
| R5两端GPU验证 | 尚未运行，等待后续启动指令；H100尚未连接/迁移 |

旧实现按字节移至[legacy](../legacy/README.md)，144文件hash清单来源为旧提交60e2b4b；新包10个业务模块与旧归档分别计数，测试和第三方不计入10个业务模块。

## CPU验收范围

- old/new tinyWan参数名与输出、Base checkpoint模式、原子保存和学生/训练态分离。
- EMA、teacher stop-gradient、冻结参数、mask局部性、独立RNG、关闭辅助项等价及非有限拒绝。
- 数据/VAE流式准备与cache reader、不同CWD下显式路径映射。
- 原RGB/LPIPS/TecoGAN和G2定义，冻结G4点门槛及10000次PCG64配对clip区间。
- 合成1152行的完整指标/marker/预测文件hash链；缺行、篡改、误配pilot与部分run拒绝。
- GPU lease使用CPU假进程清单/锁测试；失败记录及cleanup在租约内，保留原始异常。
- 本机私有资产可用时检查真实3Base任务参数的CPU恢复，以及原训练缓存和val32元数据；不加载完整Wan或解码评价像素。

CPU通过不代表真实GPU数值/容量/速度通过。G2/G3/G4科学结论均未新增，原228行评价仍停止。

## 后续开发约定

在现有10模块内维护功能；阶段差异优先使用配置，不再添加相互导入的阶段入口。5080开发分支通过相关检查后发布完整commit，H100独立checkout固定版本。科学设置与机器路径分开记录；每次新实现使用新协议和新run，不复用旧fresh-only目录。

下一步是按[H100计划](H100_MAIN_EXPERIMENT_PLAN_20261106.md)确认服务器信息及准备迁移。本次不执行远程连接、安装、资产复制、小GPU测试或完整实验。
