# Wan TRD 教师目标视频对比

预声明的三个test clip，每clip采样42/43各一份，全33帧，原FPS。顺序为HR / LR bicubic x4 / baseline / frozen Base / EMA。

单panel原生128×224；可选2倍最近邻预览只放大像素，不增加细节。contact sheet固定帧0/8/16/24/32，标记相对时间和源帧位置。

MP4用于显示，有损编码不能替代原保存PNG/NPY或改变科学指标。展示固定crop，不是完整原始视频时长或原分辨率。

本工具不重新推理、不训练、不修改评分。三个final2048在完整val后全部锁定，不按分数选择；test资产与结果已有历史使用，不宣称独立盲测。

**公开仓库仅可同步文本manifest/说明；原始PNG、contact sheet、MP4、权重及tensor不得上传。**
