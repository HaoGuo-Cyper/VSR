# HR-VAE clean latent 新一轮：数据与协议审计

2026-11-08，只读CPU审计；未加载VAE/任务模型或运行GPU。机器可读结果见同目录 `hr_vae_data_audit_20261108.json`。

## 可直接使用的目标

实际producer为 `/home/hugo/Projects/VSR_/runs/stage2_protocol33_base_seed20261102_20261106T025845.129456Z`。原冻结编码源码 `vsr/stage2_preparation.py:121–141` 与 `vsr/preflight.py:46–49` 证明：配套官方Wan VAE冻结FP32，输入HR RGB×2−1，取 `latent_dist.mode()`，再按逐通道 `(z−mean)/std` 归一化。原编码器3个文件hash与producer记录一致。

33连续帧/stride1、HR128×224对应clean latent `[1,16,9,16,28]`。已CPU读取train首例及val首例确认FP32、有限、无梯度、binding/hash正确；未重新编码或反序列化全部1673个latent。全部1673份metadata的seal、ID/source/split、窗口、退化与原编码源码binding已核对。缓存clean **已经归一化，不能再归一化一次**。

当前等价实现见 `work/VSR-sync/vsr/vae.py:75–85,143–159`；reader按seal与形状校验，见 `vsr/data.py:355–417`。VAE是跨时序causal编码，9个latent位置不能解释成33个独立逐帧target。

DiT patch `[1,2,2]`，按T/H/W排列后目标为 `[1,1008,64]`；官方hidden维为1536。建议固定第8层输出（zero-based7，30层约1/4深度），独立轻量投影头1536→64，element-mean SmoothL1、beta1，目标detach。不要沿用旧EMA/cosine/corruption机制。本轮冻结目标允许直接复用cache，不必每步在线加载VAE。

## 数据划分与clean验证

manifest按11位原YouTube source ID分组，所有片段随source分配，跨split重复视频hash也被拒绝（`vsr/data.py:74–153`）。实际数据为：

| split | 原source | clips |
| --- | ---: | ---: |
| train | 1442 | 1641 |
| val | 180 | 201 |
| test | 180 | 211 |

三组source交集及manifest视频hash交集均为0。当前cache训练1641、验证32（32不同source），`test_accessed=false`；选择顺序事前固定（`vsr/data.py:698–713`）。

逐一CPU核对全部32份val：封存dataset的HR.npy/LR_clean.npy与cache对应paired_rgb中的HR/LR **float32逐元素完全相同**，source、33帧窗口一致。可用于新一轮clean bicubic×4验证；不需要旧5种额外corruption矩阵或旧G4标准。

## 本轮固定预算与报告范围

按根代理已选方案：新baseline及HR权重0.01、0.1各2048步，fresh同初始化/AdamW/数据与噪声顺序，共6144更新=.75RE。先绑定共同初始student hash；新baseline不是复用旧C1/gamma控制。

共同step0评价64次（val32×sampler42/43）；每arm在512/1024/2048各评价64次，共 **640次clean恢复**。主指标用完整33帧LPIPS；RGB PSNR/SSIM、时间指标、完整33帧片段与按更新数/累计GPU训练时间曲线一起报告。事前冻结权重/step选择规则，无增益或证据不足都是可接受结果。公共缓存预处理成本与更新GPU计时分开声明。

工程故障可从本轮完整提交的checkpoint续跑；必须恢复student/head/AdamW/4流RNG/steps/数据顺序/累计时间，并验证protocol/config。不能把原Base优化器或半写入状态当成本轮resume，不能出现重复/跳过的更新记录。

## 必须披露的边界

- 本轮只用val32筛选，**不新访问test像素或指标**。虽然211个test raw文件都存在且与train/val原source隔离，历史 `docs/stage0_analysis_20261105/statistics.md:9` 明确8个旧模型曾在test211上报告过，因此不能称其为整个工程从未看过的pristine test。将来报告集应在本轮选择前另行冻结，并披露历史暴露。
- 本轮交付“完整33帧固定验证片段/crop”，不冒称原始视频全长。现cache只保存一个33帧窗口；原视频全长需要另外的时间覆盖、分窗/重叠/补帧及资源协议，640次验证预算不能当作全源视频评价预算。
- 旧EMA pilot的失败G4保留为历史；本轮是HR-VAE clean latent辅助监督，不沿用旧恢复/clean双门槛，也不把两种教师方法的证据混在一起。
