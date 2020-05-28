# PaddlePaddle-SSD
本开源是基于PaddlePaddle实现的SSD，参考了PaddlePaddle下的models的[ssd](https://github.com/PaddlePaddle/models/tree/release/1.6/PaddleCV/ssd) ，包括MobileNetSSD，MobileNetV2SSD，VGGSSD，ResNetSSD。使用的是VOC格式数据集，同时提供了预训练模型和VOC数据的预测模型。

# 快速使用
1. 将图像数据集存放在`dataset/images`目录下，将标注数据存放在`dataset/annotations`目录下。
2. 执行`create_data_list.py`程序生成数据列表。
3. 在下面的表格中下载预训练模型，解压到`pretrained`目录下。
4. 修改`config.py`参数，其中最重要的是`class_num`、`use_model`、`pretrained_model`。`class_num`是分类数量加上背景一类。`use_model`是指使用的模型，分别有resnet_ssd、mobilenet_v2_ssd、mobilenet_v1_ssd、vgg_ssd四种选择。`pretrained_model`是预训练模型的路径。
5. 执行`train.py`程序开始训练，每训练一轮都会更新保存的模型，训练过程中可以随时停止训练。
6. 执行`infer.py`预测图像，预测模型的路径在`config.py`配置文件中查找。

![]()


# 模型下载

| 模型名称 | 所用数据集 | 预训练模型 | 预测模型 |
| :---: | :---: | :---: | :---: |
| VGG_SSD网络的VOC预训练模型 | pascalvoc | [点击下载](https://resource.doiduoyi.com/#734q63k) | [点击下载](https://resource.doiduoyi.com/#w84qc89) |
| ResNet_SSD网络的VOC预训练模型 | pascalvoc | [点击下载](https://resource.doiduoyi.com/#cuyggu7) | [点击下载](https://resource.doiduoyi.com/#a0o1u4k) |
| MobileNet_V1_SSD网络的VOC预训练模型 | pascalvoc | [点击下载](https://resource.doiduoyi.com/#aum9kao) | [点击下载](https://resource.doiduoyi.com/#y86w98i) |
| MobileNet_V2_SSD网络的VOC预训练模型 | pascalvoc | [点击下载](https://resource.doiduoyi.com/#g1uqo28) | [点击下载](https://resource.doiduoyi.com/#6o5kiay) |


# SSD模型介绍
TODO


# 代码详解
TODO
