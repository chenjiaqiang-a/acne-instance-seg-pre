# ACNE原始数据集
包含276张标注了实例分割标签的人脸图像，剩余1838张人脸图像未标注标签，共2114张人脸图像。用于痤疮检测分割研究。
## 数据集参数
> 仅包含标注数据的参数
#### 图像参数
- rgb_mean = [0.55908102 0.41113535 0.35330288]
- rgb_std = [0.30621628 0.24620538 0.22664634]
#### 标注参数
运行`summarize.py`获取标注信息
## 拆分数据集
运行`split_data.py`将数据集拆分为训练集、验证集、测试集。

运行后将在`ACNE`内生成`train_list.txt`、`valid_list.txt`、`test_list.txt`三个文件。

|split|count|
|-|-|
|train|177|
|valid|44|
|test|55|
## 构造COCO格式的annotation
运行`generate_coco_ann`生成COCO格式的annotation（*在构造COCO格式的annotation前，请先拆分数据集*）

运行后将在`annotations`文件夹内生成`acne_train.json`、`acne_valid.json`、`acne_test.json`三个文件
## ACNE_seg数据集
提供经过清理后的ACNE_seg数据集、仅包含经过标注的数据，且标签为COCO格式