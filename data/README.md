# ACNE数据集
## ACNE原始数据集
ACNE数据集是一个用于研究痤疮严重程度分级和痤疮实例分割的医学影像数据集。该数据集总共收集了2216张VISIA设备拍摄的痤疮患者的高清人脸图像（经过了初步清洗），其中：
- 1187张图像标注了严重程度标签（共8个等级）
- 276张图像标注了实例分割标签（10个类别）
- 1010张图像没有标签

部分图像既标注了严重程度标签，也标注了实例分割标签。
### 数据集结构
```text
/ACNE
+---ACNE
|   +---annotations
|   |   +---CXM__刘某某_痤疮_20200703131528000_斑点.json
|   |   \---...
|   +---images
|   |   +---CXM__刘某某_痤疮_20200703131528000_斑点.jpg
|   |   \---...
|   +---other
|   |   +---J__杜某某_痤疮_20201027104007000_标准照片.jpg
|   |   \---...
|   +---HX_test_20220615_second_round_add_8class.txt
|   \---HX_train_20220615_second_round_add_8class.txt
+---.gitignore
+---generate_coco_ann.py
+---README.md
+---split_det_data.py
\---det_summarize.py
```
## ACNE_det数据集
要开展痤疮实例分割实验，还需对数据集进行进一步清理，拆分训练集、验证集和测试集，并生成COCO格式的annotations，并整理形成ACNE_det数据集。
### 数据集参数
> 所有参数都是基于标记数据计算。
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
### 拆分数据集
运行`split_det_data.py`将数据集拆分为训练集、验证集、测试集

|split|count|patch|
|-|-|-|
|train|200|-|
|valid|26|540|
|test|50|1052|
### 构造COCO格式的annotation
运行`generate_coco_ann`生成COCO格式的annotation（*在构造COCO格式的annotation前，请先拆分数据集*）

运行后将在`annotations`文件夹内生成`acne_train.json`、`acne_valid.json`、`acne_test.json`三个文件
### ACNE_det数据集结构
为方便实验，请拆分数据并构造COCO格式的annotation，并按照下面的文件结构，组织数据，形成ACNE_det数据集。
```text
/ACNE_det
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
## ACNE_cls数据集
要开展痤疮严重程度分级实验，可以直接使用ACNE数据集，视为8分类任务，也可以使用经过进一步优化的ACNE_cls数据集。
