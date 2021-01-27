# orange
## 前言
* 这里是orange研究语义分割（Image Segmentation）进行整理总结，总结的同时也希望能够帮助更多的小伙伴。后期如果有学习到新的知识也会与大家一起分享。

------
## 教程目录（后期会根据学习内容增加）
* SegNet（已完成）
* MobileNet_SegNet（已完成）



### 目录
1. [实现的内容 Achievement](#实现的内容)
2. [所需环境 Environment](#所需环境)
3. [使用方法与数据集](#使用方法与数据集)

### 实现的内容
#### SegNet（已完成）
- 实现斑马线分割

#### MobileNet_SegNet（已完成）
- 实现斑马线分割（轻量化）

## 所需环境
* Anaconda3（建议使用）
* python3.6.6
* VScode 1.50.1 (IDE)
* pytorch 1.3 (pip package)
* torchvision 0.4.0 (pip package)

## 使用方法与数据集
你可以下载后进入你所想要训练的模型的文件夹，然后运行train.py进行训练。  
在训练之前，需要先下载数据集，并将其存储到dataset中。  
大家关心的多分类的代码在Muiti_Class_deeplab_Mobile里。  

斑马线数据集：  
链接：https://pan.baidu.com/s/1uzwqLaCXcWe06xEXk1ROWw 提取码：pp6w   

VOC数据集：  
链接: https://pan.baidu.com/s/1Urh9W7XPNMF8yR67SDjAAQ 提取码: cvy2
