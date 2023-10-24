# ACNE实例分割初步实验
基于Acne数据集，开展人脸痤疮检测和分割的初步实验。实验中需要解决的问题，本质上是一个实例分割问题，即检测并定位人脸图像上的痤疮，进一步分割出痤疮的轮廓。本次实验将从零实现一个Mask R-CNN模型，并从头训练这个模型，调节训练参数，获得一个可以接受效果。
开展实验的目的如下：
- 熟悉数据集的特点，归纳出给数据集存在的问题以及解决的方向。
- 通过从零实现Mask R-CNN，了解在目标检测和实例分割领域的基本概念，熟悉解决此类问题的流程
- 在Acne数据集上获取一个baseline结果，为此后的进一步研究提供参考
## 项目结构
```text
/acne-segmentation
+---data                  # 存放数据
+---acne_data.py          # 数据处理
+---acne_seg.py           # 训练&测试
+---config.py             # 配置
+---inspect_data.ipynb    # 数据处理可视化
+---inspect_model.ipynb   # 模型结果可视化
+---model.py              # 模型结构
+---README.md
+---utils.py              # 工具函数
\---visualize.py          # 可视化函数
```
## 项目依赖
```text
numpy
torch
torchvision
scikit-image
pycocotools
matplotlib
```