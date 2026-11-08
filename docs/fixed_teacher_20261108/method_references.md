# 原始方法核对与本项目适配

核对日期2026-11-08，以下均为一手论文/作者代码。方法编号2601.17830当前v2标题为SRA²；用户所称VAE-REPA按该编号理解。

| 来源 | 核对结果 | 本轮适配差异 |
| --- | --- | --- |
| [SRA² / VAE-REPA，v2 §3.2、表1](https://arxiv.org/html/2601.17830v2) | 使用SD-VAE输出的紧凑latent作目标，SiT中间层经投影对齐；原文Smooth L1 β=.05、求和式表述及多层MLP | 本项目复用同一次Wan视频VAE干净latent，原channel缩放；β=1、按元素mean和单Linear为本轮明确简化，不称逐项论文复现 |
| [REPA v4，§3与附录G](https://arxiv.org/html/2410.06940v4)、[作者loss代码](https://github.com/sihyun-yu/REPA/blob/main/loss.py)、[投影头代码](https://github.com/sihyun-yu/REPA/blob/main/models/sit.py) | 对齐干净外部视觉表示与噪声输入hidden；官方实现两端channel L2后mean负cos，投影头为三Linear/两SiLU | 本轮使用等深单Linear，1−cos与负cos仅相差常数但梯度相同；student为Wan block7，两方法一致；权重用训练样本梯度量级校准，不复制论文λ |
| [DINOv2 v2](https://arxiv.org/html/2304.07193v2)、[官方代码](https://github.com/facebookresearch/dinov2) | 冻结预训练视觉表示包含局部patch tokens | 固定本地官方B/14无register权重及准确代码/文件SHA；取最终x_norm_patchtokens，不含CLS。HR128×224等比缩至112×196得到8×14网格；33逐帧表示按首帧0及后续每4帧均值映射到9latent位置 |

共同差异：原论文主要是ImageNet图像生成与全模型训练，本项目是预训练NoRef Base的LoRA/LR分支视频超分续训、33连续帧、native UniPC20及配对LPIPS/PSNR/SSIM/时间指标。不能直接移植论文的4% GFLOPs或加速倍数。时间分组按VAE名义压缩率定义，不声称是其卷积完整感受野的精确逆映射。DINO与VAE输出维度不同，头参数分别1,180,416与98,368；报告解释为完整监督配置的效果与成本比较。
