---
layout: post
title:  "SqueezeNet"
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

-   [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/pdf/1602.07360.pdf)
-   [Github](https://github.com/forresti/SqueezeNet)

****

# 引入

## 要解决的问题

>   在**保障精度**的同时，简化模型结构，**减小模型体量**。

## 主要思路

主要策略如下：

1.  用 $$1 \times 1$$ 的卷积核替换 $$3 \times 3$$

    >   **存疑**，感受野过小，能否提取特征，还是只做特征融合？
    >
    >   有些单纯的为了减少参数而做，无实际意义。
    >
    >   当然，好在 **fire module** 中还有 $$3 \times 3$$ 的卷积核可做特征提取。

2.  减少 $$3 \times 3$$ 卷积核的输入特征图通道数（使用 **squeeze layer**）完成

3.  推迟网络的下采样，使得卷积层有足够大尺寸的特征图

    >   **前人的工作表明，推迟下采样，将使得分类的精度更高。**
    >
    >   可能是因为更大尺寸的特征图能提供更多的信息。

<div style="text-align:center">
<img src="//images/fire module.PNG" width="60%">
<p>图 1：fire module</p>
</div><br>

>   个人感觉思路略微牵强，强行操作后的解释，通用性不强，借鉴意义不是太大。
>
>   **但是模型简化道路上，不可否定其贡献。**

## 成就

参数数目较 **AlexNet** 少得多，但是精度相当。

# 模型详解

## fire module

**Fire module** 的结构如**图 1** 所示，使用如下三个参数进行表征：

1.  $$s_{1x1}$$ ：`squeeze` 部分的 $$1 \times 1$$ 的卷积核的数目
2.  $$e_{1 \times 1}$$ ：`expand` 部分的 $$1 \times 1$$ 的卷积核的数目
3.  $$e_{3 \times 3}$$ ：`expand` 部分的 $$ 3 \times 3$$ 的卷积核的数目

>   模型中限制：$$s_{1x1} < e_{1 \times 1} + e_{3 \times 3}$$

## 网络架构

如下图所示，为 **Squeeze** 模型的架构及其变体，从左至右依次为：标准、**simple bypass** 和 **complex bypass ** 三个版本。

<div style="text-align:center">
<img src="//images/squeeze 网络架构.PNG" width="98%">
<p>图 2：squeeze 网络架构</p>
</div><br>

此外，网络构造时，有如下细节：

1.  对 **expand** 中 $$3 \times 3$$ 的输出进行 **padding**，使其与 $$1 \times 1$$ 的输出尺寸一致
2.  在 **Sequeeze** 与 **expand** 之间使用 **ReLU**
3.  在 **fire9** 之后，使用 **dropout**，**p=0.5**
4.  设置初始 **lr = 0.04**

# 试验与结果

<div style="text-align:center">
<img src="//images/模型压缩相关.PNG" width="98%">
<p>表 1 ：模型压缩相关</p>
</div><br>

<div style="text-align:center">
<img src="//images/模型尺寸与精度.PNG" width="95%">
<p>图 3 ：模型尺寸与精度</p>
</div><br>

<div style="text-align:center">
<img src="//images/bypass 的影响.PNG" width="85%">
<p>表 2：bypass 的影响</p>
</div><br>