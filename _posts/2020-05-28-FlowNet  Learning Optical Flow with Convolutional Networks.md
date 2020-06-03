---
layout: post
title:  "FlowNet"
date:   2020-05-28 07:41:01 +0800
categories: 人工智能
tag: 光流估计
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

- **Paper**：[FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)
- **Code**：[GitHub](https://github.com/ClementPinard/FlowNetPytorch)
- **blog**：[基于 FlowNet 的光流估计](https://zhuanlan.zhihu.com/p/124400267)

****

# 引入

## 背景

`CNN` 在各类计算机视觉任务上取得了巨大的成功，但是在光流估计（**optical flow estimation**）方向上，基本空白。

光流估计需要精确的像素定位，并需要找寻输入图像之间的关联。二者就需要学习**图像的表征**，并对图像不同位置的**表征进行匹配**。

## 本文贡献

1. 首次使用 **CNN** 进行光流估计。网络结构如下图所示，先压缩，后展开。模型可进行端对端训练。借助于 **GPU**，即使全分辨率，速度也很快。
2. 由于光溜估计 需要较大的数据集，因此，创建人工 **Flying Chairs** 数据集。但是，该模型仍旧能够泛化到实物上。

<div style="text-align:center">
<img src="/images/FlowNet.png" width="55%">
</div><br>

## 性能表现

1. 速度很快，能达到 **10 fps**

# 网络解析

本文提出的模型直接推理整个图像的光流场，而不再是采用进行分片加上后处理的手段。

## 整体结构

本文提出了两种模型结构，如下所示：

<div style="text-align:center">
<img src="/images/The two network architectures: FlowNetSimple (top) and FlowNetCorr (bottom).png" width="95%">
</div><br>


### FlowNetSimple

在该结构的模型中，直接将输入图像，沿着颜色通道方向进行拼接，并经过后续网络，得到光流输出。这个版本简称为 **FlowNetS**。

但是，该方式下，用 **SGD** 等手段不易于学习。因此，有如下改进版本。

### FlowNetCorr

先分别将输入图片，送入**共享参数**的特征提取网络，得到同尺寸的特征图，然后对**特征图进行拼接**。

## Correlation Layer

在 **FlowNetCorr** 结构中，在得到通道数为 **256** 的特征图之后，需要进行 **correlation** 操作：搜索两个特征图之间，相互对应的部分。具体方法如下：

### 特征比对

将两个尺寸为 $$h \times w \times c$$ 的特征图 $$f_1$$ 和 $$f_2$$ 进行分割，每个特征图分割成若干个 **patch**，然后将两个特征图的 **patch** **两两**比对。

<div style="text-align:center">
<img src="/images/Correlation Layer.png" width="50%">
</div>

**patch** 对比的计算方式如下所示：

$$
c\left(\mathbf{x}_{1}, \mathbf{x}_{2}\right)=\sum_{\mathbf{o} \in[-k, k] \times[-k, k]}\left\langle\mathbf{f}_{1}\left(\mathbf{x}_{1}+\mathbf{o}\right), \mathbf{f}_{2}\left(\mathbf{x}_{2}+\mathbf{o}\right)\right\rangle  \quad(1)
$$

该计算与卷积运算类似，只不过直接使用特征图的 `patch` 进行，无需学习。其中，$$x_1$$ 和 $$x_2$$ 分别表示两个 **patch** 的**中心**所在原特征图的坐标，`patch` 的尺寸为 $$K = 2k + 1$$。

每次比对需要进行 $$cK^2$$ 次乘法，比较所有的 `patch`，需要重复 $$w^2h^2$$ 次（这里隐含了 **padding** 操作，$$f_1$$ 中的每个 `patch` 分别于 $$f_2$$ 中的每个 `patch` 进行对比）。**这么做运算量巨大！！！**

### 最大偏移和 stride

因此，针对这一问题，引入**最大偏移**，并引入 **stride** 机制。具体来说，细节如下所述。

先说 **最大偏移**。考虑到空间相邻性，$$f_1$$ 中的某一 `patch`，应该只与 $$f_2$$ 中的对应位置的邻域有关。即：设定最大偏移为 $$d$$，则 $$f_1$$ 中的某个 `patch` 中心为 $$(x_1, y_1)$$，则在 $$f_2$$ 中，只有中心在 $$[x1 - d:x1+d, y1-d:y1+d]$$ 之间的 `patch` 才需要参与比对。

接着说 **stride**。由于空间相关性，所以在 $$f_2$$ 中的 `patch` 进行比对时，可以间隔 `stride` 取一次 `patch`。

因此，假设最大偏移 $$d = 21$$，步长 $$stride=2$$，则所需要的计算量为：

$$
cK^2 * h * w * [(2 * d + 1) // 2]^2 = 441 \times (cK^2) \times h \times w
$$


### 图像特征引入

如上图所示，除了 **patch** 比对得到的 **441** 个通道之外，还有直接从输入图像特征图提取的特征（通过尺寸为 $$1 \times 1 \times 32$$ 的卷积核实现），其通道数为 **32**。因此，总的通道数为 **473**。

上述操作，总结如下图所示：

<div style="text-align:center">
<img src="/images/Correlation Layer 计算.jpg" width="90%">
</div>

## Refinement

**refine** 部分得到主要构成为上采样，即：**unpooling + convolution**。

<div style="text-align:center">
<img src="/images/Refinement of the coarse feature maps to the high resolution prediction.png" width="98%">
</div>

如上所示，对 **feature map** 进行 **upconvolutional**，然后将其与 **contractive** 部分对应尺度的特征图、以及一个上一级输出的粗光流的上采样（**upsampleed coarser flow**）进行 **concatenate**。

通过这种方式，可以同时提供由 **coarser feature maps** 提供的高抽象信息，以及 **lower layer feature maps** 提供的精细的局部信息。

每一步分辨率加倍，重复四次，得到最后的预测 **flow**。此时的尺寸仍为输入尺寸的 $$1/4$$。但是实验发现，相比于**双线性插值**，进一步增加层数，并不能改善结果。因此后续上采样采取双线性插值方式。

除了双线性插值，作者还尝试了 **variational refinement** 方式进行上采样，相应的模型加上后缀 **+v**。详细细节参考论文。其视觉效果如下所示：

<div style="text-align:center">
<img src="/images/The effect of variational refinement.png" width="98%">
</div>
# 实验与结果

> **从零开始训练模型**。

## 训练数据

现有的用于光流任务的数据集有：

|               | Frame pairs | Frames with ground truth | Fround truth density per frame |
| ------------- | ----------- | ------------------------ | ------------------------------ |
| Middlebury    | 72          | 8                        | 100%                           |
| KITTI         | 194         | 194                      | ～50%                          |
| Sintel        | 1041        | 1041                     | 100%                           |
| Flying Chairs | 22872       | 22872                    | 100%                           |

### Middleburry

仅包含 **8** 对图像用于训练。图像前后帧运动较小，通常小于 **10** 个像素点。

### KITTI

包含 **194** 对训练图像，图像之间运动较大，但是仅包含一种特定的运动类型。**groung truth** 从真实场景获取。

### MPI Sintel

从渲染的人工场景中获取 **ground truth**，这些场景重点关注真是图像特征性。

- 最终版本（**Final version**）包含运动模糊和大气效果，例如雾气等，而干净版本（**Clean version**）不包含这些效果
- **Sintel** 是当前能用的最大的数据集，包含 **1041** 对图像，包含较大运动和较小运动的**稠密光流场**

### Flying Chairs

**Flying Chairs** 是通过对收集自 **Flickr** 和 **3D chair** 的图像进行仿射变换（**affine transformations**）得到的。该数据集包含 **809** 种椅子类型，每种椅子包含 `62` 个视角。

<div style="text-align:center">
<img src="/images/Two examples from the Flying Chairs dataset.png" width="98%">
</div>

如上图所示，为 **Flying Chairs** 数据集的样例。

## 数据增强

我们在训练过程中，进行在线数据增强。数据增强的手段包含：**translation**、**rotation**、**scaling** 以及 **additive Gaussian noise**，另外，还加入了 **brightness**、**contrast**、**gamma** 和 **color** 的变化。

具体来说，操作如下：

> - sample translation from a the range [−20%; 20%] of the image width for x and y
> - rotation from [−17; 17]
> - scaling from [0:9; 2:0]
> - Gaussian noise has a sigma uniformly sampled from [0; 0:04]
> - contrast is sampled within [−0:8; 0:4]
> - multiplicative color changes to the RGB channels per image from [0:5; 2]
> - gamma values from [0:7; 1:5]
> - additive brightness changes using Gaussian with a sigma of 0:2

## 网络和试验细节

### 网络结构

- 两边均包含 **9** 层卷积层，其中有 **6** 个步长为 **2**（用于池化），每一层后面接一个 **ReLU** 非线性层
- 网络不包含任何全连接层，因此对输入图像的尺寸无要求
- **correlation layer**：$$k=0, d=20, s1=1, s2=2$$

### 训练细节

- **training loss**：使用 **endpoint error（EPE）**。预测光流向量和真实光流向量之间的欧氏距离，在每个像素点上的均值。
- **optimizer**：**Adam**，**β1=0.9**，**β2=0.999**
- **mini-batches**：8
- **lr**：初始为 **1e-4**，在 **300k** 次迭代后，之后每 **100k** 次迭代，减小一倍。
- 在训练 **FlowNetCorr** 时，使用 $$lr=1e-4$$ 时，会发生梯度爆炸。因此，先用 $$1e-6$$，在 **10k** 次迭代之后，缓慢增加到 $$lr=1e-4$$，之后变化如上所述。

### finetune

- 由于训练数据集差异较大，因此需要在目标数据集上进行 **finetune**
- 我们选择在 **Sintel** 上进行 **finetune**。同时使用 **Clean 和 Final** 的数据，并使用 $$lr=1e-6$$ `finetune` 几千次迭代
- **finetune** 的模型，后面加上 **+ft**

## 实验结果

### 数据表现

<div style="text-align:center">
<img src="/images/Average endpoint errors (in pixels) of our networks compared to several well-performing methods on different datasets.png" width="98%">
</div>

- 仅在 `non-realistic Flying Chairs` 上训练，就在  `real optical flow datasets` 上表现很好，优于 `LDOF`
- 在 `Sintel` 上 `finetune` 后，优于 `SOTA` 的 `EPPM`，且速度更快

### Sintel

<div style="text-align:center">
<img src="/images/Examples of optical flow prediction on the Sintel dataset.png" width="98%">
</div>

### Flying Chairs

<div style="text-align:center">
<img src="/images/Examples of optical flow prediction on the Flying Chairs dataset.png" width="98%">
</div>

## 分析

### Training data

- 由于使用了激进的数据增强手段，即使仅用 **Sintel**，也足够很好的学习到光流
- 不使用数据增强的话，仅用 **Flying Chairs** 训练的话，模型在 **Sintel** 上测试的时候，**EPE** 增加了将近两个像素

### Comparing the architectures

- **FlowNetS** 相较于 **FlowNetC**，在 **Sintel Final** 上泛化性能最好
- **FlowNetC** 相较于 **FlowNetS**，在 **Flying chairs 和 Sintel Clean** 上，表现更佳

**Flying Chairs** 不包含 **Sintel Final** 中的运动模糊和雾状。因此，上面的结果表明，即使 **FlowNetS** 和 **FlowNetC** 参数数目一致，**FlowNetC** 也更容易拟合训练数据。

> Though in our current setup this can be seen as a weakness, if better training data were available it could become an advantage.

- **FlowNetC** 对于较大的运动，表现不佳。原因是因为 **correlation** 的最大偏移量不支持预测更大的运动。可以通过增加最大偏移量进行缓解，但同时会增加运算量。
