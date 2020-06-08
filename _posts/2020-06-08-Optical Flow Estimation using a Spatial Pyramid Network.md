---
layout: post
title:  "SPyNet"
date:   2020-06-08 07:41:01 +0800
categories: 人工智能
tag: 光流估计
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

- **Paper**：[Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850)
- **Code**：[GitHub](https://github.com/sniklaus/pytorch-spynet)

****

# 引入

## 背景

传统的公式法求解光流需要一系列的**强假设**，从图像亮度到空间平滑性。而真实场景下，常常不能很好地满足这些假设，因此将限制算法的性能。

而此前的基于深度学习的 **FlowNet** 模型，参数较多，运算较慢，且由于最大偏移的限制，只能计算较小运动的光流。这是因为比较大的 `motion`，可能不会包含在一个 `kernel ` 中，导致不能提取有效的响应信息。

因此，**FlowNet** 性能不够优秀，且不能实现实时且内存高效的运算。

## 本文贡献

本文结合深度学习与传统金字塔算法，采用 **coarse-to-fine** 的方式，进行光流估计，可以**很好地处理较大的光流**。

该模型直接在每一层学习一个 **CNN**，来进行光流更新。相较于 **FlowNet**，光流提取完全交由金字塔结构进行处理，具有如下优势：

1.  模型更简单，参数更少（减少 **96%** 的参数）
2.  由于每一金字塔层中，**warped images** 的光流较小，因此完全可以用简单的卷积运算来计算光流
3.  在标准 **benchmarks** 上，更加精确

>   利用金字塔结构，捕捉较大的运动；利用 **CNN**，学习局部像素级细节，例如边界等。

## 性能表现

与 **FlowNet** 相比：

1.  均使用  **Flying Chairs** 数据集进行训练
2.  在 `Flying Chairs` 和 `Sintel` 数据集上，与 `FlowNet` 表现相近
3.  在 `Middlebury ` 和 `KITTI `上，`finetune` 之后，性能显著提升
4.  模型尺寸减少 **96%**

但是与传统算法一样，**不足之处**在于：小目标的大运动很难捕捉。

# 网络细节

## 模型结构

我们采用传统的 **spatial pyramid**，以 **coarse-to-fine** 的方式，在每一层学习光流残差（**residual flow**），解决 **large motions** 的问题。**要求**：金字塔顶部（最小分辨率）的前后帧的差异，小于 **kernel** 尺寸，以便能够学习到有效信息。

在每一层，通过 `img1` 和当前层的 `flow` 进行 `warping`，可以近似另一张图片（光流学习的足够好的话）。同时在每一层，学习一个 `cnn` 来预测当前层 `flow` 相对于上一层 `flow` 的**增量**，类似于高斯金字塔。

**SPyNet** 结构如下：

<div style="text-align:center">
<img src="/images/Inference in a 3-Level Pyramid Network.png" width="98%">
</div>

如上图所示，$$G_0 $$ 在金字塔最高层（最小分辨率），通过 $$\left\{I_{0}^{1}, I_{0}^{2}\right\}$$ 计算光流差 $$v_0$$，初始光流 $$v = 0$$，其它层逐层计算相对与上一层的光流差。金字塔共 `5` 层。

## 空间采样

如上图所示，对原始图像进行多次下采样，得到每一层对应尺度的前后帧。然后，计算光流时，通过上采样获取每一层的基准光流。

如上图所示：

-   $$d(\cdot)$$​ 表示**下采样**函数，尺寸为 `0.5`
-   $$u(\cdot)$$ 为**上采样**函数，尺寸为 `2.0`
-   $$w(I, V)$$ 为 **warping** 操作，依据上一层光流上采样得到当前层的基准光流场，然后用基准光流场对 `img1` 进行 `warping`。（通过将图像按照光流移动，然后进行双线性插值，得到等效的另一帧）

## 前向推理

用 $$\{G_0, \cdots, G_k\}$$ 表示每一层的 `CNN`，用于计算残差 `flow`，`k` 表示金字塔的层：

$$
v_{k}=G_{k}\left(I_{k}^{1}, w\left(I_{k}^{2}, u\left(V_{k-1}\right)\right), u\left(V_{k-1}\right)\right)
$$


- $$v_k$$  光流残差
- $$V_{k-1}$$ 上一层 `flow` 的上采样
- $$\left\{I_{k}^{1}, I_{k}^{2}\right\}$$ 第 `k` 层的帧。其中，$$I^2_k$$ 使用 `flow` 进行 **warping** 得到等效的 $$I_k^1$$，

第 `k` 层的光流场为：

$$
V_{k}=u\left(V_{k-1}\right)+v_{k}
$$


## 训练

### 单层详解

从最高层（最低分辨率开始），逐层独立训练 $$\{G_0, \cdots, G_k\}$$ 中的每一个。其中，每一层的目标光流残差 $$\hat{v}_k$$ 通过如下方式获取：

$$
\hat{v}_{k}=\hat{V}_{k}-u\left(V_{k-1}\right)
$$

其中：

- $$\hat{V}_{k}$$ 为第 `k` 层的目标光流（通过对原始目标光流下采样得到对应层的目标光流）
- $$u(V_{k - 1})$$ 由上一层已训练好的 `cnn` 生成。

如下图所示，对上一层的目标光流进行上采样 $$u(V_{k - 1})$$ ，并以此对 $$I_k^2$$ 进行 **warping**，得到 $$I_k^1(warped)$$，并与 $$I_k^1$$，以及 $$u(V_{k - 1})$$ 一同，作为网络 $$G_k$$ 的输入，生成当前层的光流残差 $$v_k$$。训练每一层的 $$G$$，使得残差光流 $$v_k$$ 的平均 **End Point Error (EPE) loss** 最小。

<div style="text-align:center">
<img src="/images/获取每一层的 gt.png" width="75%">
</div>

> **这种结构下，每一层的网络只需要在现有上一层 flow 的基础上学习较小的偏差，从而简化任务。**
>
> 每一个 `G` 有 `5` 层，是在反复试验后，在速度和精度上的折中。

### **Flying Chairs**

模型共 `5` 层金字塔，最小分辨率 $$24 \times 32$$；最大分辨率 $$384 \times 512$$。在每一层中：

- 除最后一层，每一层接一个 **ReLU**
- 使用 $$7 \times 7$$ 的卷积核，因为这样实验效果比小尺寸卷积核更好
- 通道数依次为 `32`，`64`，`32`，`16`，`2`。输入 `img1` 和 `warp` 后的 `img2`，加上上采样得到的 `flow`，共 $$3 \times 2 + 2 = 8$$ 个通道；输出 `2` 个通道，分别表示 `x` 和 `y` 方向的向量。

逐层训练，后一层以前一层参数进行初始化：

- **Adam**，$$\beta_1 = 0.9, \beta_2 = 0.999$$
- $$bs = 32$$， 共 `4000` 次迭代每个 `epoch`

- 开始 `60` 个 `epoch`，$$lr = 1e-4$$，逐渐减小至 $$1e-5$$，直到收敛
- 用 **Flying Chairs** 和 **MPI Sintel** 进行训练

数据增强：

- randomly scale images by a factor of [1, 2]
- apply rotations at random within [−17, 17]
- random crop to match the resolution of the convnet
- additive white Gaussian noise sampled uniformly from $$\mathcal{N}(0,0.1)$$
- color jitter with additive brightness, contrast and saturation sampled from a Gaussian, $$\mathcal{N}(0,0.4)$$
- normalize the images using a mean and standard deviation computed from a large corpus of ImageNet

# 实验结果与分析

## 实验结果

### 性能

与 `FlowNet` 和 `Classic + NLP`（`traditional pyramid-based method`） 对比，使用 **average end point errors**：

<div style="text-align:center">
<img src="/images/Average end point errors (EPE).png" width="85%">
</div>

> **SPyNet  最精确，速度最快**

### Flying Chairs

<div style="text-align:center">
<img src="/images/Visualization of optical flow estimates.png" width="85%">
</div>

### MPI-Sintel

- 图像缩放到 $$448 \times 1024$$
- 使用 `6` 层金字塔，最后两层一样

两种方式测试：

1. 直接使用 **Flying Chairs** 训练
2. 分别使用 `Sintel Clean` 和 `Sintel Final` `finetune`，使用 **EPE** 评估，后缀加上 **+ft**

### 结果

1. **SPyNet** 相较于 `FlowNet` ，在各种速度上更精确（除了最大的速度 $$s > 40$$ `pixels / frame`）
2. **SPyNet** 的运动边界更精确，这对很多应用很重要

<div style="text-align:center">
<img src="/images/Visual comparison of optical flow estimates using our SPyNet model with FlowNet on the MPI Sintel dataset.png" width="85%">
</div>

<div style="text-align:center">
<img src="/images/Comparison of FlowNet and SpyNet on the Sintel benchmark for different velocities.png" width="95%">
</div>

> **上表中，$$s$$ 表示速度，$$d$$ 表示与边界的距离**。

## 分析

### Model Size

**SPyNet** 模型尺寸比 **FlowNet** 小 **96%**。

<div style="text-align:center">
<img src="/images/Model size of various methods.png" width="65%">
</div>

金字塔结构使得模型参数数目可以显著减少，而不用损失精度：

1. 直接使用 **warping** 函数，无需模型进行学习
2. 残差学习限制了输出空间光流场的变化

### 速度

<div style="text-align:center">
<img src="/images/速度与精度.png" width="75%">
</div>