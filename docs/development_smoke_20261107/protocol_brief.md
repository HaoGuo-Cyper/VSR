# 5080本地开发验证：事前协议

2026-11-07。用户明确授权一次本地5080小规模GPU验证模型加载、少量前反向、保存及推理闭环，通过后更新记录和GitHub；不启动完整实验，不迁移H100，不恢复自动推进。

- 使用已审计Base01 seed20261102 /8192，fresh official backbone+恢复学生LoRA/LR，fresh AdamW/EMA/projector。
- 只用原sealed cache前3个train clip，保持33连续帧、128×224 HR、×4、原1000格UniPC、bf16/strict math及冻结FP32 VAE驻留。
- 恰好3个A2诊断updates：rho.25/span2、排除latent0、gamma.8、EMA.999、student9/teacher21。自然时间步索引23/954/808，sigma0.9922218323/0.1287695169/0.4174489975，teacher true/false/true；不挑时间步、不自动重试。
- 在全新run/A2/checkpoint_000003.pt保存；明确development_only/0正式RE。释放engine/optimizer/EMA/projector后重新加载学生，逐参数hash一致；不恢复teacher/projector/optimizer用于推理。
- 用第1个train clip的原sealed LR与固定文本做恰好1次原native UniPC20/CFG1/seed42、33帧、单crop推理；不读val/test指标。
- 通过标准：3步loss/梯度有限、冻结参数不变、EMA更新3/teacher激活2；保存重载学生完全一致；推理有限/非恒定/维度[1,3,33,128,224]、33PNG读回像素一致、MP4帧数/尺寸/FPS一致，未超16GiB及实际物理显存。
- 共用actual runs/stage2_gpu.lock，fresh host完整桌面身份准入；共享租约覆盖模型载入至失败记录/cleanup。启动前发现原锁0664，记录并在同inode收紧0600，不换锁、不放宽guard。
- 任何新运行错误/非有限/OOM即保留记录并停止；旧失败/停止run不复用。本次不通过G2/G3/G4、不做方法质量结论，诊断权重不得用于正式训练。

新增模式在原train.py内部，不增加第11个业务模块。smoke相关CPU保存/重载与预算拒绝检查、源码独立审查先于GPU启动；完整metadata protocol的源码/资产/环境SHA在prepare后冻结。
