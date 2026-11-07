# 当前主线入口说明

2026-11-07。下列是已实现接口；本机5080开发smoke与新完整Base评价G2均已通过，用户授权在本地按门槛继续，H100迁移延期。当前入口见[本地继续计划](local_continuation_20261107/plan.md)。示例中的`/private/...`、SHA与UUID需按实际机器和新协议填写，不能直接复制为启动命令，也不能重复启动已有评价。

## 1. 开发时先看帮助和CPU测试

```bash
python -m vsr.data --help
python -m vsr.train --help
python -m vsr.infer --help
python -m vsr.evaluate --help
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
```

CLI help/import不加载GPU模型，不下载权重。普通CPU测试无需私有数据；有本机官方指标权重及已有Base资产时执行额外CPU兼容检查，否则明确skip。

## 2. 数据：data.py

- `manifest --raw-root ... --output ... [--seed ...]`：原视频清单与来源隔离，CPU工作。
- `prepare-cache --config ... --output ... [--project-root ...] [--device cpu]`：固定文本缓存、流式LR/HR准备与VAE编码，新目录、新seal；不构建T5/DINO/任务模型。已有已审计数据优先复用。
- `check-cache --cache-root ... [--seal-sha256 ...] [--verify-all]`：检查旧protocol33缓存。verify-all重算全部资产，普通reader只核对实际使用的条目。

CUDA预处理额外要求`--gpu-lock`与`--gpu-admission`，使用和训练/推理/评价相同的租约。CPU默认不查询GPU。若固定数据/窗口改变，应另立科学协议，不覆盖已有缓存。

## 3. 训练：train.py

`prepare`只绑定元数据/文件hash，不加载GPU模型。`execute`需要已准备的新run及明确SHA。

```text
python -m vsr.train prepare
  --mode base|smoke|resource|pilot
  --config /private/matched-Base-source-config.json
  --cache-root /private/sealed-training-cache
  --model-directory /private/official-Wan-directory
  --allowed-checkpoint-root /private/runs
  --device cuda:0
  --gpu-lock /private/runs/stage2_gpu.lock
  --gpu-admission /private/host-admission.json
  --memory-cap-gib <事前声明的上限>
  --run-dir /private/runs/NEW-RUN
  --run-root /private/runs
```

- `base`：fresh official Base8192入口；现有三个Base已完成，本次及下一轮无需重训。
- `smoke`：另加`--base-checkpoint`与`--expected-base-sha`，仅用Base01和首3训练clip；固定3A2诊断updates+保存/学生重载+1次UniPC20恢复，0正式RE。无需G2/G3，不能用于声称科学门槛通过；本次已完成，不自动重跑。
- `resource`：另加`--base-checkpoint`、`--expected-base-sha`及`--g2-run`；必须重算完整G2通过，执行70诊断updates。
- `pilot`：resource的参数外再加`--g3-run`和`--g4-definitions`；同Base01、fresh optimizer、flow-only/C1/A2各2048步。G4原定义在任何方法训练/评分之前绑定。

实际执行命令形态为`python -m vsr.train execute --run-dir ... --run-root ... --expected-protocol-sha ...`；按前置门槛启用新run：完整Base评价G2真实通过后才进入resource，G3通过后才进入pilot。重构前的old run不能交给新入口resume。传入matched Base的原config，模型目录迁移使用独立`--model-directory`，不改原checkpoint配置。

## 4. 推理：infer.py

- `preview --checkpoint ... --allowed-checkpoint-root ... [--model-dir ...]`：只预览来源配置。
- `run`额外需要`--input`、`--output`、`--text-embedding`和显式`--execute`；device默认cpu。CUDA要求`--gpu-lock`与`--gpu-admission`。
- 采样选项：`--seed`、`--steps`、`--tile-height`、`--tile-width`、`--overlap`；默认使用完整输入帧与原FPS。只有显式`--frames`才截取前缀。
- 固定文本缓存直接提供，不加载T5。主线NoRef，不接受HRI参考图像。仅学生恢复权重参与推理。

## 5. 评价：evaluate.py

1. 以[配置模板](../configs/evaluation_bases.template.json)填入明确绝对路径、3个已审计Base SHA、固定text及LPIPS权重、host准入。
2. `python -m vsr.evaluate prepare --config ... --run-dir ...`创建新协议。
3. 审核新协议及前置检查后用`execute --run-dir ... --expected-protocol-sha ...`执行；既定本地计划正常通过门槛后继续，无需重复许可，已有run不重复启动。
4. `verify --run-dir ...`重查全1152行/预测文件/完成marker/来源hash并重算门槛；部分run不能解锁后续阶段。

`kind=bases`要求3训练seed×32clips×6条件×2采样seed；`kind=pilot`要求flow-only/C1/A2的3arm矩阵，另需`base_evaluation_run`、`pilot_training_run`和`G4_definitions`，核对真实6144步训练及对应最终2048 checkpoint。

旧val32 seal和224资产内容不改。`path_map`只决定到哪里寻找同一批字节；旧源码绑定应指向`legacy/pre_refactor_20261106/`的相同字节。不同host上的FFmpeg/CPU依赖差异必须显式验收并事前记录，不关闭校验来求通过。H100迁移当前延期；未来启用时仍需现场环境与GPU验证。

## 6. 设备身份与公开同步

所有CUDA执行共用checkpoints.gpu_lease。admission声明`expected_gpu_uuid`，可选`desktop_processes`完整pin；空清单要求目标GPU无compute进程，不能仅按进程名字放行。CPU检查使用mock inventory；本机5080已验证真实桌面准入/共享锁，H100准入尚未现场验证。

GitHub只同步代码、配置、协议、统计和文本日志；模型、checkpoint、原视频、tensor缓存及凭证通过私有通道提供。实验期间固定协议绑定的数值源码与依赖，不git pull覆盖运行源码；文档和文本日志可单独同步。
