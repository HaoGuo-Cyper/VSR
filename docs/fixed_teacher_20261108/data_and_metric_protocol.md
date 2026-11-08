# 本轮数据、编码边界与指标定义

实现与数值源码在校准协议中已冻结；本说明展开现有代码，不修改运行协议。

## 配对×4与原视频隔离

从原HQ-VSR视频顺序解码到封存33帧窗口，OpenCV BGR转RGB，保留原FPS、stride1，不做补帧/循环。窗口起点由SHA256(window,seed,recordID)确定；同一窗口各帧共享空间操作，无随机翻转或其他增强。先按max(128/H,224/W)等比缩放（四舍五入尺寸、OpenCV INTER_AREA），再中心裁出128×224。小于目标尺寸的原视频拒绝，不上采样伪造HR，无crop padding；因此“HR”是固定缩小/裁剪后的监督图像，不是原720p完整输出。

RGB转FP32/255得到[1,3,33,128,224]、范围[0,1]。torch.nn.functional.interpolate bicubic / align_corners=False / antialias=True，显式size32×56，并clamp[0,1]生成LR；同一函数上采样回128×224给VAE条件路径。边缘使用固定PyTorch实现，不另加padding。没有额外模糊/噪声/codec/drop-frame损坏；所有帧参数相同。torch版本/实现绑定随protocol保存。

原sourceID按视频整体分组，train1442sources/1641clips，原val180/201、test180/211；原sourceID与文件SHA跨split无交集。自动视觉完整性又发现跨ID近重复：排除1val/3test来源，仅在原split内固定hash顺序补齐；训练集/Base/历史结果不变。新val保留31原窗口与NPY字节、新增1源；test保留29预选窗口、新增3源。全部1705待比较train/evalclip在声明的多点缩略图+真实窗口筛查下无flag；这不是穷尽所有offset/crop/transcode/语义重复的证明。详见cohorts/completion.json、各source/window清单和integrity audit。

选参只用新val32。正式test32的RGB资产和模型分数在完整val选择receipt后才开启；此前只允许绑定的自动完整性缩略图例外，无人工质量阅图或模型分数。历史模型已评过原test211，本轮不称项目级盲测。

## VAE、token与DINO对应

官方Wan配套VAE冻结eval/FP32。每个独立33帧clip开始重置causal缓存；HR×2−1编码，posterior.mode()确定性取值，按官方逐channel(z−mean)/std归一化，得到[1,16,9,16,28]。原FM clean target直接复用，不重采样、不再次归一化。encode/decode结束清缓存，解码逆归一化后sample/2+.5并clamp[0,1]，不改官方因果/空间边界。输入尺寸整除native步长、33=4×8+1，不额外补时空边缘。

学生native patch[1,2,2]产生9×8×14=1008个token，按T/H/W排列；VAE每patch向量按pt/ph/pw/C（channel最快）展平为64。两方法都取block7（第8/30层）输出，已接收LR注入0/5，后续注入9仍受FM训练。VAE用单Linear1536→64、elementmean SmoothL1 β1，target停止梯度。

DINOv2-B/14固定官方无register版本、权重/source commit与所有文件SHA已封存。HR crop全区域等比torch bicubic/AA缩至112×196、clamp[0,1]、ImageNet mean/std；14patch无补边，输出8×14的x_norm_patchtokens（排除CLS/register，保留encoder自身finalLayerNorm）。33帧独立图像编码；局部frame0单独，1–4、5–8…29–32各均值，得到9×8×14，随后在通道维L2归一化算mean(1−cos)。学生同深单Linear1536→768。网格按同一HR区域的名义16×16cell对应，不声称是VAE/DINO全感受野的精确逆映射；两种encoder本身均有更广上下文。

DINO所有1673缓存的来源/起点/裁剪/RGB文件SHA/预处理/教师版本严格绑定，未沿用旧stride4特征。GPU train0/1在线重提与缓存逐元素精确相等；VAE同两例也与原FM target精确相等。旧cache32验证例不用于本轮优化或选参。

## 完整正常推理与分数

每次从CPU采样种子42/43的新噪声和LR条件正常生成；native UniPC20、CFG1、FP32 VAE、tile256×448/overlap64。输入crop128×224小于tile时为单tile；33帧全部生成PNG和MP4，不拼接旧run行，不挑帧或类型。推理不使用HR、教师或辅助头；MP4仅播放，评分来自同一round(clamp(pixel)*255)/255无损PNG等价像素。

主LPIPS：固定Alex0.1的两个本地weightSHA，全部33帧、全RGB、无border shave，逐帧平均。PSNR：每帧RGB全像素FP64 MSE再算−10log10(max(MSE,1e−12))、33帧平均（上限120dB与exact-frame数记录）。SSIM：skimage默认7×7 uniform/sample-covariance窗口，channel_axis=-1、data_range1，全部33帧无Y变换或border shave。

时序沿用固定TecoGAN revision bc747a4651c1e08bd0a73bed08c4cb9fac99b4d5定义：cutfr2保留帧2…30，28相邻对；作者crop_8x8几何在128×224上取中心96×192（top/left16）。tLP100=100×mean(abs(LPIPS(HR前,HR后)−LPIPS(SR前,SR后)))；tOF先在全帧uint8灰度计算固定Farneback flow，再裁剪并平均flow差的空间L2，越低越好。它们不是另一种tLPIPS或E*warp。

每clip先平均采样42/43，再等权32个不同source；表中clipSD不是训练seedSD。本轮只有一个trainingseed，无法估计训练seedSD。10000 pairedclip PCG64/seed20261102/95%linear-percentile是条件于本训练seed的描述区间，无多重选择校正或多seed显著性宣称。val对最终2048的LPIPS选每family权重，精确tie选较小权重；test不重选，不继承旧EMA G4门槛。所有候选与中间512/1024/2048点均报告。
