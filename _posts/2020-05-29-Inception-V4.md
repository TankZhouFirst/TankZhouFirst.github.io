---
layout: post
title:  "Inception-V4"
date:   2020-05-29 08:07:01 +0800
categories: 人工智能
tag: 图像分类
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

-   [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)
-   [Github](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py)

****

# 引入

## 动机

近期 **ResNet** 的出现，使得 **CNN** 的性能 表现优异，达到了最新的 **Inception V3** 的性能。因此，可否将 **ResNet** 与 **Inception** 进行结合。

## 成就

1.  实验表明 ，**ResNet** 可以显著加速网络的收敛；但是其对模型性能的改善并不一定明显
2.  表明合理的特征值缩放可以稳定宽度残差 **Inception** 网络的训练
3.  详细提出三个网络结构
    1.  `Inception-ResNet-v1` : 计算复杂度与 `Inception-V3` 相近
    2.  `Inception-ResNet-v2` : 计算量稍大，但是显著提升了网络的性能
    3.  `Inception-v4 `: 不包含 残差连接的 `inception` 变体，性能与 `Inception-ResNet-v2` 相当

# 模型详解

各版本差别如下：

1.  每个 **ResBolck** 后都接一个不带 **ReLU** 的 $$1 \times 1$$ 卷积，用于拓展通道维数，便于加法匹配
2.  在残差版本中，我们只在标准层后使用 **BN** 层，而残差相加后不加 **BN** 层。

## Inception-V4

### 整体结构

<div style="text-align:center">
<img src="/images/Inception V4.PNG" width="40%">
<p>图 1：Inception V4</p>
</div><br>

### 子网络

#### Stem

>   **未标记 V 的为 same padding，剩余的为 valid padding。**

<div style="text-align:center">
<img src="/images/Stem.PNG" width="55%">
<p>图 2：Stem</p>
</div><br>

#### Inception-A

<div style="text-align:center">
<img src="/images/Inception A.PNG" width="85%">
<p>图 3：Inception-A</p>
</div><br>

#### Inception-B

<div style="text-align:center">
<img src="/images/Inception B.PNG" width="85%">
<p>图 4：Inception-B</p>
<p>注释：第三列应该为 7 * 1，属笔误</p>
</div><br>

#### Inception-C

<div style="text-align:center">
<img src="/images/Inception C.PNG" width="90%">
<p>图 5：Inception-C</p>
</div><br>

#### Reduction-A

<div style="text-align:center">
<img src="/images/Reduction-A.PNG" width="70%">
<p>图 6：Reduction-A</p>
</div><br>

#### Reduction-B

<div style="text-align:center">
<img src="/images/Reduction-B.PNG" width="70%">
<p>图 7：Reduction-B</p>
</div><br>

## Inception-ResNet-v1 / v2

### 整体结构

<div style="text-align:center">
<img src="/images/Inception-Res arch.PNG" width="45%">
<p>图 8：Inception-Res arch</p>
</div><br>

### 子网络

#### Stem

`Inception-ResNet v1` 的输入如下，`Inception-ResNet v2` 的输入，见图 2。

<div style="text-align:center">
<img src="/images/Stem for In-Res-v1.PNG" width="35%">
<p>图 9：Stem</p>
</div><br>

#### Inception-ResNet-A

<div style="text-align:center">
<img src="/images/In-Res-A-v1.PNG" width="70%">
<p>图 10：In-Res-A-v1</p>
</div><br>

<div style="text-align:center">
<img src="/images/In-Res-A-v2.PNG" width="70%">
<p>图 11：In-Res-A-v2</p>
</div><br>

#### Inception-ResNet-B

<div style="text-align:center">
<img src="/images/In-Res-B-v1.PNG" width="60%">
<p>图 12：In-Res-B-v1</p>
</div><br>

<div style="text-align:center">
<img src="/images/In-Res-B-v2.PNG" width="60%">
<p>图 13：In-Res-B-v2</p>
</div><br>

#### Inception-ResNet-C

<div style="text-align:center">
<img src="/images/In-Res-C-v1.PNG" width="60%">
<p>图 14：In-Res-C-v1</p>
</div><br>

<div style="text-align:center">
<img src="/images/In-Res-C-v2.PNG" width="60%">
<p>图 15：In-Res-C-v2</p>
</div><br>

#### Reduction-A

>   **同图 6**，参数如下：

<div style="text-align:center">
<img src="/images/parameter of Reduction-A.PNG" width="65%">
<p>表 1：parameter of Reduction-A</p>
</div><br>

#### Reduction-B

<div style="text-align:center">
<img src="/images/ReductionB-v1.PNG" width="80%">
<p>图 16：ReductionB-v1</p>
</div><br>

<div style="text-align:center">
<img src="/images/ReductionB-v2.PNG" width="80%">
<p>图 17：ReductionB-v2</p>
</div><br>

# 试验及结果

实验中发现，特征图数目接近 `1000` 时，网络可能假死，可通过降低残差连接来稳定训练：

<div style="text-align:center">
<img src="/images/general schema for scaling combined Inceptionresnet moduels.PNG" width="35%">
<p>图 18：general schema for scaling combined Inceptionresnet moduels</p>
</div><br>

<div style="text-align:center">
<img src="/images/Single crop - single model experimental results.PNG" width="65%">
<p>表 2：single crop-single model experimental results</p>
</div><br>

