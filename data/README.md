# ACNE原始数据集
包含276张标注了实例分割标签的人脸图像，剩余1838张人脸图像未标注标签，共2114张人脸图像。用于痤疮检测分割研究。
## 数据集参数
> 仅包含标注数据的参数
#### 图像参数
- rgb_mean = [0.55908102 0.41113535 0.35330288]
- rgb_std = [0.30621628 0.24620538 0.22664634]
- RGB_MEAN = [142.566, 104.840, 90.092]
- RGB_STD = [78.085, 62.782, 57.795]
#### 标注参数
运行`summarize.py`获取标注信息

|id|0|1|2|3|4|5|6|7|8|9|10|
|-|-|-|-|-|-|-|-|-|-|-|-|
|name|BG|paule|nevus|nodule|open_comedo|closed_comedo|atrophic_scar|hypertrophic_scar|melasma|pustule|other|
||背景|丘疹|痣|节结|开口粉刺|闭口粉刺|萎缩性瘢痕|肥厚性瘢痕|黄褐斑|脓疱|其它|
|count|-|5550|1461|209|3297|5877|8958|774|3955|1268|428|

- min_area = 51.25
- min_width = 5.9
- min_height = 6.7
## 拆分数据集
运行`split_data.py`将数据集拆分为训练集、验证集、测试集

|split|count|patch|
|-|-|-|
|train|200|-|
|valid|16|328|
|test|60|1264|
## 构造COCO格式的annotation
运行`generate_coco_ann`生成COCO格式的annotation（*在构造COCO格式的annotation前，请先拆分数据集*）

运行后将在`annotations`文件夹内生成`acne_train.json`、`acne_valid.json`、`acne_test.json`三个文件
## ACNE_seg数据集
为方便实验，请拆分数据并构造COCO格式的annotation，并按照下面的文件结构，组织数据，形成ACNE_seg数据集。
```text
/ACNE_seg
+---annotations
|   +---acne_train.json
|   +---acne_valid.json
|   \---acne_test.json
+---images
+---valid_patch
+---test_patch
+---test_list.txt
+---train_list.txt
\---valid_list.txt
```
提供经过清理后的ACNE_seg数据集仅包含经过标注的数据，且标签为COCO格式