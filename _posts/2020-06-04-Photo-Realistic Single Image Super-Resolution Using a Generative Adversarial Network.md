---
layout: post
title:  "SRResNet"
date:   2020-06-04 21:43:01 +0800
categories: 人工智能
tag: 超分辨
---


* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

- [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
    Network](https://arxiv.org/pdf/1609.04802.pdf)
- https://github.com/david-gpu/srez

****

# 引入

## 行业背景

### 超分辨基本定义

在超分辨（`super-resolution, SR`）指的是根据低分辨率的图像，生成对应的高分辨率图像。该任务中，存在一个核心问题：在较大的超分辨放大倍数上，如何复原精细的纹理细节？

本文只关注单图超分辨（`SISR`）。

### 相关工作

- 一开始是基于预测的（`Prediction-based`）算法，如 `bicubic` 等。速度快，但是容易生成模糊的图像
- 旨在建立低分辨与高分辨之间复杂映射的方法，其通常依赖于训练数据。许多算法要求成对的低分辨 `patch` 和高分辨图像
- 结合边缘导向的（`edge-directed`）`SR` 算法，其基于梯度剖面（`gradient profile prior`），可以有基于学习的细节综合。目标是重建真实的纹理细节，同时避免人工痕迹
- 多尺度字典，用于捕捉相似图像 `patches` 在不同尺度上的衰减（`redundancies`）
- 领域嵌入（`Neighborhood embedding`）方式对低分辨图像 `patch` 进行上采样，通过在低维空间（`low dimensional manifold`）上查找相似的低分辨训练 `patch`，并组合其相应的高分辨 `patches`，用于重建高分辨图像
- 基于 `CNN` 的 `SR` 算法，表现卓越，如：`LISTA`、`DRCN` 等等

### 行业痛点

#### 现存问题

目前主流的是基于优化的 `SR` 算法，其行为，通常由目标函数的选择所主导。

目前主要用的是最小化生成高分辨与目标图像之间的均方误差（`mean squared reconstruction error, MSE`）。

其结果通常具有较高的峰值信噪比（`peak signal-to-noise ratios, PSNR`），但是通常缺少高频细节，图像质量较为模糊（因为像素值平均），在高分辨下表现不尽如人意。`MSE` 损失趋向于生成平滑模糊的图像，感官感受的质量较差。

#### 图像信噪比

> 图像的信噪比计算：峰值信噪比，通常用于图像压缩等领域中信号重建的质量度量，其通过均方差 `MSE` 进行定义：


> $$
> \begin{equation}
> M S E=\frac{1}{m n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1}\|I(i, j)-K(i, j)\|^{2}
> \end{equation}
> $$


> 则峰值信噪比为：


> $$
> \begin{equation}
> P S N R=10 \cdot \log _{10}\left(\frac{M A X_{I}^{2}}{M S E}\right)=20 \cdot \log _{10}\left(\frac{M A X_{I}}{\sqrt{M S E}}\right)
> \end{equation}
> $$


> 其中，`MAX` 为图像点颜色的峰值，若采用 `8bit` ，则为 `255`。因此，均方差越小，`PSNR` 越大。

#### 图像高低频信息

> 只有当进行傅里叶变换后才有低频和高频之分，**低频**一般是大范围**大尺度**的信息，也就是背景，而**高频**反映的是**小范围细节**信息。
>
> 应用上对应高频滤波和低频滤波，如果你想得到局部信息，则相应要保留高频部分，滤掉低频部分，反之，若你想得到总体趋势变化，则相应要保留低频部分，滤掉高频部分。

## 本文主要贡献

本文提出 `SRGAN`，是第一个可以将真实自然图像进行 `4` 倍超分辨放大的框架。主要有以下特点：

1. 首次提出使用 `GAN` 的超深度残差结构，用于 `ISSR` 任务
2. 提出一种新颖的 `perceptual loss` ，其由两部分组成：用于指导生成多样性的自然图像的 `adversarial loss` 和用于计算图像内容相似度的 `content loss`。且使用由 `VGG` 生成的高抽象层次特征图定义 `content loss`
3. 在三个公共 `benchmark` 数据集上，使用一个可拓展的 `MOS`（``mean opinion score`） 测试进行确认。结果表明，`SRGAN` 当前最佳，且大幅度领先

# 本文工作

## 网络结构

深度网络可以映射复杂的函数，但是也较难训练。为了高效的训练深度网络，常用 `BN` 来解决 `ICS` 等问题。递归 `CNN` 网络能够取得不错的效果。此外，还可以使用近期流行的残差块（`residual blocks`）以及跳跃连接（`skip-connections`）。

本文所用的网络结构如下所示：

<div style="text-align:center">
<img src="/images/SRGAN 网络结构.png" width="99%">
<P>SRGAN 网络结构</P>
</div><br>

### 生成器

生成器网络 `G` 的核心部分，由 `B` 个完全相同的残差块构成。每个残差块中，使用两个 $$3 \times 3$$ 卷积核，`64` 特征图的卷积层，然后接一个 `BN` 层以及一个 `PReLU` 层作为激活函数。之后接两个训练好的亚像素卷积（`sub-pixel convolution layers`）来增加分辨率。

> **亚像素卷积**
>
> 参考论文：
>
> [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network](https://link.zhihu.com/?target=https%3A//www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
>
> 基本原理：
>
> <div style="text-align:center">
> <img src="/images/亚像素卷积原理图.png" width="99%">
> <P>亚像素卷积原理图</P>
> </div><br>
>
> 基本思路：将邻近的若干（$$r^2$$）尺寸为 $$[W,H,r^2C]$$ 的特征图，进行像素重排列，输出一个尺寸为 $$[rW,rH,C]$$ 特征图。
>
> 公式表示：


> $$
> \begin{equation}
> \left.\mathcal{P} \mathcal{S}(T)_{x, y, c}=T_{\lfloor x / r\rfloor},_{L^{y / r}}\right\rfloor, c \cdot r \cdot \bmod (y, r)+c \cdot \bmod (x, r)
> \end{equation}
> $$

### 判别器

为了区分真实 `HR` 图像和生成的 `SR` 图像，我们训练一个判别器网络我们同时定义一个判别器网络 $$D_{\theta_{D}}$$，用于解决如下问题：

$$
\begin{equation}
\begin{array}{rl}{\min _{\theta_{G}} \max _{\theta_{D}}} & {\mathbb{E}_{I^{H R} \sim p_{\text {train }}\left(I^{H R}\right)}\left[\log D_{\theta_{D}}\left(I^{H R}\right)\right]+} \\ {} & {\mathbb{E}_{I^{L R} \sim p_{G}\left(I^{L R}\right)}\left[\log \left(1-D_{\theta_{D}}\left(G_{\theta_{G}}\left(I^{L R}\right)\right)\right]\right.}\end{array}
\end{equation}
$$

在判别器网络中，使用 `LeakyReLU` 激活函数（$$\alpha = 0.2$$），且网络不使用最大池化。

判别器包含 `8` 个卷积层，卷积核 $$3 \times 3$$ 的数目逐层加倍，从 `64` 到 `512`。每次特征图加倍时，设定 `stride > 1` 来减少特征图尺寸。卷积层最后，接两层 `dense` 层，以及最后一个 `sigmoid` 激活函数来获取真伪样本分类的概率。

## 损失函数

### 训练目标及基本形式

`SISR` 的目标在于，根据低分辨输入图像 $$I^{LR}$$，生成超分辨图像 $$I^{SR}$$。在训练过程中，将高分辨图像 $$I^{HR}$$ 进行高斯滤波（Gaussian filter），然后以 $$r$$ 倍率进行下采样，得到 $$I^{LR}$$。因此，输入图像尺寸为：$$W \times H \times C$$，而 $$I^{SR}, I^{HR}$$ 的尺寸为：$$rW \times rH \times C$$。

我们的最后总目标是，训练一个网络 `G`，为低分辨图像 $$I^{LR}$$，生成的高分辨图像 $$I^{SR}$$，尽可能接近 其原始高分辨图像 $$I^{HR}$$。生成网络为前向传播的 `CNN` $$G_{\theta_{G}}$$，参数为 $$\theta_{G}= {\{W_{1:L};b_{1:L}\}}$$，对应的损失函数为：$$l^{SR}$$。

对于训练图像 $$I_n^{HR}$$，及其对应的 $$I_n^{LR}$$，我们求解：

$$
\begin{equation}
\hat{\theta}_{G}=\arg \min _{\theta_{G}} \frac{1}{N} \sum_{n=1}^{N} l^{S R}\left(G_{\theta_{G}}\left(I_{n}^{L R}\right), I_{n}^{H R}\right)
\end{equation}
$$

### Perceptual loss function

`Perceptual` 损失函数 $$l^{SR}$$ 的定义，对生成器网络的性能至关重要。我们不直接使用 `MSE loss`，而是定义 `Perceptual loss` 为 `content loss`（$$l_X^{SR}$$）和 `adversarial loss` 的加权和：

$$
\begin{equation}
l^{S R}=\underbrace{\underbrace{l_{\mathrm{X}}^{SR}}_{\text{content loss}}+\underbrace{10^{-3} l_{Gen}^{SR}}_{\text {adversarial loss }}}_{\text {percepual loss fror VGG based content losses }}
\end{equation}
$$

#### content loss

像素级的 `MSE loss` 计算方式如下：

$$
\begin{equation}
l_{M S E}^{S R}=\frac{1}{r^{2} W H} \sum_{x=1}^{r W} \sum_{y=1}^{r H}\left(I_{x, y}^{H R}-G_{\theta_{G}}\left(I^{L R}\right)_{x, y}\right)^{2}
\end{equation}
$$

我们定义 `VGG loss`，其基于预训练的 `19` 层 `VGG` 网络的 `ReLU` 激活层输出。其中，$$\phi_{i, j}$$ 表示在第 $$i$$ 层最大池化层之前，卷积层（激活层之后）的输出的第 $$j$$ 个特征图。那么，`VGG loss` 就定义为重建图像 $$G_{\theta_{G}}\left(I^{L R}\right)$$ 和参考图像 $$I^{HR}$$ 之间的欧拉距离：

$$
\begin{equation}
\begin{aligned} l_{V G G / i, j}^{S R}=\frac{1}{W_{i, j} H_{i, j}} & \sum_{x=1}^{W_{i, j}} \sum_{y=1}^{W_{i, j}}\left(\phi_{i, j}\left(I^{H R}\right)_{x, y}\right.\left.-\phi_{i, j}\left(G_{\theta_{G}}\left(I^{L R}\right)\right)_{x, y}\right)^{2} \end{aligned}
\end{equation}
$$

#### Adversarial loss

对抗损失基于判别器的输出概率定义：

$$
\begin{equation}
l_{G e n}^{S R}=\sum_{n=1}^{N}-\log D_{\theta_{D}}\left(G_{\theta_{G}}\left(I^{L R}\right)\right)
\end{equation}
$$

其中，$$D_{\theta_{D}}\left(G_{\theta_{G}}\left(I^{L R}\right)\right)$$ 表示判别器人为生成 `SR` 图像是真实 `HR` 图像的概率。我们不使用 $$\log \left[1-D_{\theta_{D}}\left(G_{\theta_{G}}\left(I^{L R}\right)\right)\right]$$ 是为了更好的梯度表现。

# 实验

## 数据和相似度度量

我们在三大基准数据集 `Set5`、`Set14` 和 `BSD100` 上进行实验，测试集为 `BSD300`。所有的实验的超分辨系数均为 $$r = 4$$。

## 训练细节和参数

我们在 `NVIDIA Tesla M40 GPU` 上训练源自 `ImageNet` 数据集的 `35` 万图片，其有别于测试集。

我们使用 `bicubic` 滤波器，对原始图像（`GBR, C=3`）进行 $$r=4$$ 的下采样。在每个 `mini-batch` 中，我们随机裁剪 `16` 张 $$96 \times 96$$ 的 `HR` 子图。

生成器网络是全卷积的，因此可以用于任意尺寸的图像。对于 `LR` 输入，变为 `[0,1]`；对于 `HR`，变为 `[-1, 1]`，并在该范围上计算 `MSE`。对于 `VGG loss`，需要变为原来的 `1/12.75 = 0.006` 倍，使之与 `MSE loss` 的范围相当。

我们使用 `Adam` 优化器，$$\beta_1 = 0.9$$。`SRResNet` 网络的学习速率为 $$10^{-4}$$，迭代 $$10^6$$ 次。在训练 `GAN` 时，使用训练好的基于 `MSE` （使用特征图）的 `SRResNet` 来初始化生成器，避免局部最优。

`SRGAN` 网络以 $$10^{-4}$$ 的学习速率，迭代 $$10^5$$，然后用 $$10^{-5}$$ 的学习速率，迭代相同次数。使用系数 `k=1` 来更新 `G` 和 `D`。生成器具有 `16` 个（`B=16`）残差块。

在测试期间，关闭 `BN`，使得输出仅依赖于输入。

## 试验结果

### Mean opinion score (MOS) 测试

`26` 个人，对不同 `SR` 模型生成的图像，进行人工投票，评分 `1～5`，越高越好。结果如下所示：

<div style="text-align:center">
<img src="/images/不同损失函数的性能.png" width="55%">
<P>不同损失函数的性能</P>
</div><br>

<div style="text-align:center">
<img src="/images/不同模型的性能表现.png" width="90%">
<P>不同模型的性能表现</P>
</div><br>

<div style="text-align:center">
<img src="/images/不同模型的 MOS 得分分布.png" width="60%">
<P>不同模型的 MOS 得分分布</P>
</div><br>

<div style="text-align:center">
<img src="/images/超分辨实例.png" width="95%">
<P>不同模型实例</P>
</div><br>

### content loss 的影响

我们尝试了不同的 `content loss` 选择带来的影响。也评估了不使用 $$l^{SR}_{MSE}$$ 和 $$l^{SR}_{VGG/2.2}$$ 时，生成器网络的性能。我们将 `SRResNet-MSE` 称为 `SRResNet`。

在训练 `SRResNet-VGG22` 时， $$l^{SR}_{VGG/2.2}$$ 缩放因子为 $$2 \times 10^{-8}$$。

在使用 `GAN` 时，结合对抗损失，`MSE` 方式提供最高的 `PSNR`，但是其图像较为模糊。同时我们发现，`VGG` 网络的更深层，相较于浅层的特征图，提供更精致的纹理特征。

具体数据表现，参考上一节的表格。

网络生成样例，如下图所示：

<div style="text-align:center">
<img src="/images/网络生成样例.png" width="95%">
</div><br>

### 其他结论

#### 性能（PSNR/time）与网络深度的关系

<div style="text-align:center">
<img src="/images/网络性能与网络深度的关系.png" width="90%">
<P>网络性能与网络深度的关系</P>
</div><br>

#### SRGAN 训练中的生成器评估

评估 `SRGAN` 的生成器网络的表现与迭代次数的关系。

<div style="text-align:center">
<img src="/images/SRGAN 生成器的表现与迭代次数的关系.png" width="95%">
<P>SRGAN 生成器的表现与迭代次数的关系</P>
</div><br>

## 生成样本实例

<div style="text-align:center">
<img src="/images/生成样本实例.png" width="95%">
</div><br>