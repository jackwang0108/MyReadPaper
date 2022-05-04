---
title: 'Deep Learning for 3D Point Clouds: A Survey阅读笔记'
date: 2021-2-12 14:23:55
img: https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220212223048005.png
summary: '本文是Deep Learning for 3D Point Clouds: A Survey的阅读笔记'
categories:
  - 论文阅读笔记
  - PointClouds
  - TPMAI
tags:
  - PointClouds
  - Review
  - 论文阅读笔记
  - TPAMI
  - IEEE Transactions on Pattern Analysis and Machine Intelligence
---

> 本文是Deep Learning for 3D Point Clouds: A Survey的阅读笔记

![Deep Learning for 3D Point Clouds: A Survey](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220212223048005.png)







# Deep Learning for 3D Point Clouds: A Survey阅读笔记

这篇文章我很早就读过了，但是时至今日，感觉遗忘了许多，因此特地写成文章记录下来。





## Paper Meta-Information

- 题目：Deep Learning for 3D Point Clouds: A Survey
- 类型：综述
- 年份：TPAMI 2020
- 期刊：TPAMI 2020, IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE



## Glossary









## 1. Abstract

现在的DL技术可以很好的解决2D的视觉任务，可是由于点云数据本身的特点，所以给DL技术的运用带来了新的挑战。因此目前是一个Thriving的研究领域。为了能够激发后来的研究，本文对最近深度学习在点云运用上的进步做出了综述。

**本文主要关注点云中的三个任务**：

- **3D Shape Classification**
- **3D Object Detection and Tracking**
- **3D PointCloud Segmentation**

最后，本文对一些在公开数据集上的模型进行了讨论

> 感觉这三个任务基本上就是2D视觉任务的对应版本嘛……
>
> 不过对于要入门一个field的话，最重要的还是后面这一部分，对现有的模型和数据集进行了讨论





## 2. Introduction

近年来因为多种3D传感器的发展所以3D格式的数据越来越容易获取。**而相比2D数据，3D数据能够提供更加丰富的信息，因此3D数据可以提供对周围环境更加丰富的语义信息**。所以我们要研究深度学习在三维数据上的运用

三维格式的数据可以各种格式的数据来表示，包括：

- 深度图，Depth Image

  ![深度图像](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/Real-image-and-its-depth-map-in-Ballet-sequence.png)

- 点云，Point Cloud

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/p6.png" alt="点云" style="zoom:67%;" />

- 多边形网络，Meshes

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/250px-Dolphin_triangle_mesh.png" alt="Mesh数据" style="zoom:150%;" />

- 体素，Volumetric Grids（类似于我的世界中的方块）

  ![Voxel Grids数据](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/FROST_GeoToVoxelGrid_PRTVGrid3_IndexMtlIndex.png)

  

而因为点云数据没有进行离散化等操作、保留了原始几何体的信息，因此要研究点云。而点云研究现在又有很多问题，因此本文要关注这些问题



目前有一些三维点云的数据集，有：

- ModelNet
- ScanObjectNN
- ShapeNet
- PartNet
- S3DIS
- ScanNet
- Semantic3D
- ApolloCar3D



三维点云发展起来之后，又可以帮助处理下游任务：

- 点云形状分类，3D shape classification
- 点云目标检测和追踪，3D object detection and tracking
- 点云分割，3D point cloud segmentation
- 点云配准，3D point cloud registration
- 3D姿态估计，6-DOF pose estimation
- 3D重建， 3D reconstruction



本文的贡献如下：

- 第一篇全面涵盖几个重要的点云理解任务的深度学习方法的综述文章，包括三维形状分类、三维目标检测与跟踪、三维点云分割。
- 与现有的文章不同，本文专注于3D点云的深度学习方法，而不是所有类型的3D数据。
- 本文介绍了点云深度学习的最新进展。因此，它为读者提供了最先进的SOTA方法。
- 对几个公开的数据集上的现有方法进行了全面的比较，并进行了简要的总结和深刻的讨论。



本文的结构：

- 第2节介绍了各个任务的数据集和评估指标，Datasets and Evaluation Metrics
- 第3节介绍了三维形状分类的方法，3D Shape Classification
- 第4节介绍了现有的三维目标检测和跟踪方法，3D object detection and tracking
- 第5节本文综述了点云分割（Point Cloud Segmentation）的方法，包括语义分割（Semantic Segmentation）、实例分割（Instance Segmentation）和部分分割（Part Segmentation）
- 最后，第6节对论文进行了总结



![A taxonomy of deep learning methods for 3D point clouds](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213000237771.png)







## 3. Background

Background主要介绍了点云学习中使用的数据集和Evaluation Metrics



### 1. Dataset

对于不同的任务，input都是点云，但是因为任务不同，因此label也不同，所以针对不同的任务，数据集也是不同的。



#### 1. 3D Shape Classification

对于3D Shape Classification任务而言，其数据集可以分为两类：

- **合成的数据集（Synthetic Datasets）**：合成数据集是通过3D绘图软件（例如Autodesk）的软件绘制的图形，例如花瓶、被子等等。这些软件绘制出来的图形都是Mesh形式的数据，然后通过随机采样等方法从平面上随机采样得到点。**因此合成数据集都是无遮挡的、没有来自背景的点的数据**

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/2-1.png" alt="合成点云的例子" style="zoom:67%;" />

- **现实世界的数据集（Real-World Datasets)**：现实世界的数据集是通过激光扫描器等设备扫描得到的，因此现实世界的数据集中的数据（形状）多少会有点遮挡，而且也会含有来自环境背景的噪声。

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/pc.jpg" alt="显示世界扫描得到的点云" style="zoom:67%;" />



#### 2. 3D Object Detection and Tracking

3D Object Detection and Tracking这两个任务数据集可以分成两种：

- **室内数据集（Indoor Datasets）**：室内数据集中的点云主要是通过深度图像或者采集到的3D mesh转换得到的。

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/blog-img-Our-Experience-While-Working-With-Point-Clouds-and-Scan-To-BIM.jpg" alt="室内点云" style="zoom: 50%;" />

- **室外数据集（Outdoor Datasets）**：室外数据集主要是为了自动驾驶任务收集的数据集。而由于激光的问题，因此通常只有一部分是物体是没有遮挡的，而且越远处，打到物体上的点就越少

  ![室外场景点云](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/pointcloud.png)





#### 3. 3D Point Cloud Segmentation

3D Point Cloud Segmentation任务的数据是按照获取点云的不同的设备来划分的：

- Mobile Laser Scanners (MLS)：搭载在无人车上的这类移动平台获取的点云数据，例如SemanticKitti

  ![Kitti数据集的MLS平台](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/passat_sensors.jpg)

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213151046309.png" alt="MLS的点云" style="zoom:150%;" />

- Aerial Laser Scanners (ALS)：搭载在无人机上的这类空中平台获取的点云数据

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/suitable_drone.png" alt="可以搭载Velodyne激光雷达的无人机" style="zoom:67%;" />

  <img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/laser_scanning29-1.jpg" alt="ALS获取的数据" style="zoom:150%;" />

- Static Terrestrial Laser Scanners (TLS)：固定在地面上的平台获取的点云数据

- RGB-D cameras：RGB-D相机采集到的点云数据

- 其他3D扫描设备采集到的点云数据



![不同任务数据集的总结表](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213022325608.png)





### 2 Metrics

同样，针对不同的任务，Evaluation Metrics也不同



#### 1. 3D Shape Classification

3D Shape Classification用Overall Accuracy（OA）和Mean Class Accuracy（mAcc）作为评判标准。OA就是对所有测试example的分类准确率，而mAcc则是对每一类进行计算之后平均得到的分类准确率



#### 2. 3D Object Detection and Tracking

3D Object Detection用Average Precision（AP）作为评判标准，其实就是Precision-Recall曲线下的面积

3D Object Tracking用Precision、Success、Average Multi-Object Tracking Accuracy（AMOTA）和Average Multi-Object Tracking Precision（AMOTP）作为评判标准。其中，AMOTA和AMOTP是最常用的。



#### 3. 3D PointCloud Segmentation

3D PointCloud Segmentation用Overall Accuracy（OA）、Mean Intersection of Union（mIoU）、Mean Accuracy（mAcc）作为评判标准。

此外，3D Instance Classification用Mean Accuracy Precision（mAP）作为评判标准







## 4. 3D Shape Classification

![3D Shape Classification深度学习算法编年史](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213022519517.png)



### 1. Multi-view based Methods

Multi-view based方法将3D形状投影到不同的视角，然后抽取对应的特征，最后把所有的特征聚合起来实现准确的形状分类。这类方法最重要的一步就是如何把多视角的特征聚合成一个全局可分辨的特征

<img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213152219963.png" alt="Multi-view based Methods的例子" style="zoom: 80%;" />





#### 1. MVCNN

MVCNN是利用多视角对3维形状进行分类的先驱工作，发表于ICCV 2015。其中对多个视角获得的二维投影图像进行卷积、提取特征。在最后进行View Pooling得到多个视角下的不同的特征，然后进行分类。

![MVCNN的示意图](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213160930393.png)



#### 2. MBNN

相比于MVCNN的最大池化，MBNN则是使用Harmonized Bilinear池化取代了最大池化

![MVCNN的示意图](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213163451100.png)



#### 3. Learning Relationships for Multi-View 3D Object Recognition

在Learning Relationships for Multi-View 3D Object Recognition文章中，Yang等人使用了一个relation network来让模型自己学习多个视角的图片（View to View）、不同的区域（Region to Reigon）之间的关系。

![Learning Relationships for Multi-View 3D Object Recognition的示意图](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213163952804.png)





### 2. Volumetric based Methods

Volumetric方法则是把点云转换为3D网格点，然后使用3D卷积网络等方法来学习特征，对点云进行分类。

总的来说，Voxel的方法分为直接对Voxel进行计算，基于八叉树的计算



#### 1. VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition

Maturana等人使用VoxNet直接对像素化的点云进行卷积

![VoxNet对像素化的点云进行](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213173844641.png)



#### 2. 3D ShapeNets: A Deep Representation for Volumetric Shapes

Wu等人则是使用了深度卷积信念网络来学习

![ShapeNets](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213174408097.png)





#### 3. OctNet: Learning Deep 3D Representations at High Resolutions

上面的两种方法的问题就在于只能用于计算小型的点云，无法计算大型点云数据，即放voxel grid中的点的数量一多之后，计算和内存的开销就会变得无法承受。因此就OctNet先对点云的数据进行预处理：使用八叉树结构先对点云进行分割，然后对分割后的点云进行学习

<img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213174926408.png" alt="OctNet对点云进行八叉树分割" style="zoom:67%;" />



#### 4. O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis

类似于OctNet，O-CNN中提出了对八叉树处理之后的点云进行学习的方法

![八叉树预处理](https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213175217317.png)





#### 5. PointGrid: A Deep Network for 3D Shape Understanding

PointGrid中则是对Voxel和Point同时进行学习

<img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213181201666.png" alt="PointGrid的二维展示" style="zoom:67%;" />





#### 6. 3D Point Cloud Classification and Segmentation using 3D Modified Fisher Vector Representation for Convolutional Neural Networks

Ben-Shabat等人则是在把点转换为Voxel数据的基础上，然后再转换为3D Modified Fisher Vector，然后再进行学习

<img src="https://jack-1307599355.cos.ap-shanghai.myqcloud.com/img/image-20220213182229135.png" alt="对3DmF向量及逆行学习的网络" style="zoom:80%;" />

