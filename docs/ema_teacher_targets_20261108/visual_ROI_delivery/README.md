# Wan TRD 逐帧局部视觉检查

6张PNG覆盖固定3片段×sampler42/43，每张按HR／Baseline／Frozen／EMA四方法竖排，每方法11列×3行覆盖frame0–32。输入保持25fps／33帧，全部解码，逐格标帧号；未重新推理、计算或改变评分。

图像从已有有损comparison.mp4解码，属于压缩后的展示检查，不是原始评分PNG，不能用此图重新计算指标。ROI坐标相对原128×224画面，已加24行header偏移并排除footer。车ROI y36+h96超出原画面4行，按根确认用RGB(96,96,96)灰填充，起点、大小和缩放不变。

manifest.json记录原视频SHA、全部帧与源索引、ROI／padding、图像尺寸和每格原生像素hash；6PNG保存后已逐格无损读回核验。原视频字节未变，媒体不上传GitHub。
