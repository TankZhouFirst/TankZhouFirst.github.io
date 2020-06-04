---
layout: post
title:  "MobileNet V1"
date:   2020-06-04 21:43:01 +0800
categories: 人工智能
tag: 图像分类
---


* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

-   [MobileNets : Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
-   [GitHub](https://github.com/jmjeon94/MobileNet-Pytorch/blob/master/MobileNetV1.py)

****

# 引入

## 要解决的问题

解决深度神经网络的**速度和精度**的相互矛盾，在**保证精度的同时，提升速度**。

<div style="text-align:center">
<img src="/images/MobileNet v1 的使用场景.PNG" width="90%">
<p>图 1：MobileNet 可以高效应用于手机，的用于各种检测任务</p>
</div><br>

## 主要思路

使用**深度可分离卷积（depthwise separable convolutions）**构建轻量级深度神经网络，通过两个全局**超参数**进行**速度和精度**（`latency and accuracy  `）的折中。

1.  **depthwise convolution** ：对输入特征图的**每个通道**分别卷积（使用单通道卷积核），进行特征提取
2.  **pointwise convolution** ： 对不同通道的特征进行融合（使用多通道的 $$ 1 \times 1$$ 卷积），生成新的特征

<div style="text-align:center">
<img src="/images/MobileNets V1 的思路图示.png" width="60%">
<p>图 2：MobileNets V1 的思路图示</p>
</div><br>

## 成就

详见实验结果，两种参数量级的模型下：

-   轻量级模型上，相比于 **SqueezeNet**，参数数量相当，但精度提升较多
-   标准模型上，优于 **GoogleNet**，与 **VGG** 相比，精度略逊，但是参数数量大幅度下降

# 模型详解

## depthwise separable convolutions

`MobileNets` 基于 **depthwise separable convolutions**，其将标准卷积分解为 **depthwise convolution** 和 **pointwise convolution**。两个字运算的作用和思路如上文所述，下面主要分析**参数数目**和**计算量**。

之所以叫可分离卷积，是因为，传统卷积运算同时进行**特征选择**和**特征融合**，而可分离卷积将其分为两步。

假设：

-   输入特征图尺寸：$$D_F \times D_F \times M$$
-   输出特征图尺寸：$$D_G \times D_G \times N$$

则：

-   **标准卷积**
    -   **参数数目**：$$D_K \times D_K \times M \times N$$
    -   **运算量**：$$D_K \times D_K \times M \times D_G \times D_G \times N$$
-   **可分离卷积**
    -   **depthwise convolution**
        -   参数数目：$$D_K \times D_K \times M$$
        -   运算量：$$D_K \times D_K \times M \times D_G \times D_G$$
    -   **pointwise convolution**
        -   参数数目：$$1 \times 1 \times M \times N$$
        -   运算量：$$1 \times 1 \times M \times D_G \times D_G \times N$$
    -   **SUM**
        -   **参数数目**：$$D_K \times D_K \times M + 1 \times 1 \times M \times N$$
        -   **运算量**：$$D_K \times D_K \times M \times D_G \times D_G + 1 \times 1 \times M \times D_G \times D_G \times N$$
    -   **占比**
        -   **参数比值**：$$\frac{1}{N}+\frac{1}{D_{K}^{2}}$$
        -   **运算量比值**：$$\frac{1}{N}+\frac{1}{D_{K}^{2}}$$

## 网络结构

### 标准 MobileNets

基本组件和网络详细结构如下所示：

<div style="text-align:center">
<img src="/images/可分离卷积 block.png" width="50%">
<p>图 3：标准卷积（左）和深度可分离卷积（右）</p>
</div><br>

<div style="text-align:center">
<img src="/images/详细网络结构.PNG" width="65%">
<p>图 4：MobileNets 详细网络结构</p>
</div><br>

在上面结构中：

1.  所有下采样通过 **stride=2** 的卷积实现，即 **s2**
2.  网络最后通过全局平均池化 **Avg Pool** 将特征尺寸变为 **1**，然后接上 **FC** 和 **Softmax**

### 可控超参

#### width multiplier

>   缩小每一层的尺寸，通过参数 $$\alpha$$ 控制每一层输入输出的通道数目

####  Resolution Multiplier

>   对输入图像进行缩放，从而每一层的特征图尺寸相应的缩放。

# 实验及结果

<div style="text-align:center">
<img src="/images/卷积与可分离卷积的效果比对.PNG" width="65%">
<p>表 1：卷积与可分离卷积的效果比对</p>
</div><br>

<div style="text-align:center">
<img src="/images/更瘦还是更浅.PNG" width="65%">
<p>表 2：更瘦还是更浅</p>
</div><br>

<div style="text-align:center">
<img src="/images/Width Multiplier.PNG" width="65%">
<p>表 3：Width Multiplier</p>
</div><br>

<div style="text-align:center">
<img src="/images/Resolution.PNG" width="65%">
<p>表 4：Resolution</p>
</div><br>

<div style="text-align:center">
<img src="/images/精度 VS 运算量.PNG" width="65%">
<p>图 5：精度 VS 运算量</p>
</div><br>

<div style="text-align:center">
<img src="/images/精度 VS 参数数目.PNG" width="65%">
<p>图 6：精度 VS 参数数目</p>
</div><br>

<div style="text-align:center">
<img src="/images/标准模型对比.PNG" width="65%">
<p>表 5：标准模型对比</p>
</div><br>

<div style="text-align:center">
<img src="/images/轻量级模型对比.PNG" width="65%">
<p>表 6：轻量级模型对比</p>
</div><br>