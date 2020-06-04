---
layout: post
title:  "批量归一化 batch normalization"
date:   2019-07-01 21:13:01 +0800
categories: 人工智能
tag: 图像分类
---


* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**


****

**参考**

- **Paper** ：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- **Blogs** ：
  - [机器学习（七）白化 whitening](https://blog.csdn.net/hjimce/article/details/50864602)
  - [Batch Normalization 原理与实战](https://zhuanlan.zhihu.com/p/34879333)

****


# 引入

## 行业背景

在深度神经网络的训练过程中，每层输入的分布可能会随着之前层参数的变化而变化。随着层数加深，这种变化逐层累计。因此之前层的微小扰动可能会导致后续层较大变化。这将使得训练减缓，收敛困难。这种现象我们称之为 `internal covariate shift, ICS`。

当使用饱和非线性激活函数时，由于梯度几乎为 `0`，因此收敛更为缓慢。因此，需要小心的进行初始化，使用合适的激活函数，并设定合适的学习速率。

可以通过**固定每层的输入分布**进行改善上述问题，例如白化。此时每层的参数就无须适应上层的输出分布变化，因此各层之间进行了一定程度的脱耦，从而使得训练更加稳定。但是白化方式存在一定的不足。

## 本文主要贡献

- 本文主要贡献在于针对 `ICS` 问题，以及白化的不足，提出 `Batch Normalization`。`BN` 通过固定层输入的均值和方差来改善 `ICS`。**需要在 `activation` 层之前添加。**
- 通过 `BN`带来如下效果：
  - `BN` 方式来固定输入分布，对各层进行脱耦，增加网络的稳定性和训练速度
  - 网络对参数变化不敏感，从而减少梯度消失和梯度爆炸的情况，加速收敛，提高精度
  - 批量归一化后的输出将处于激活函数的非饱和区，将快速收敛，并可使用饱和激活函数
  - 起到一定的正则化效果，可以减少 `Dropout` 层的使用

# 白化 Whitening

## 什么是白化

白化（`Whitening`）是机器学习里面常用的一种规范化数据分布的方法，主要是 `PCA` 白化与 `ZCA` 白化。

白化的目的是去除输入数据的冗余信息。假设训练数据是图像，由于图像中相邻像素之间具有很强的相关性，所以用于训练时输入是冗余的。

白化的具体步骤及作用如下：

1. **去除特征之间的相关性**。通过 `PCA` 等手段，将数据（降维）映射到新的特征空间，从而一定程度上去除数据的相关性
2. **进行相同的归一化**。得到性的特征表示之后，使用相同的方差和均值进行归一化。对于 `PCA`，均值为 `0`，方差为 `1`；对于 `ZCA`，均值为 `0`，方差相同。

> 通过白化操作，我们可以减缓 `ICS` 的问题，进而固定了每一层网络输入分布，加速网络训练过程的收敛。

## 白化的不足

虽然白化可以固定每层的输入分布，减缓 `ICS` 问题。但是，其存在一定的缺陷：

1. **计算成本太高**：需要在每一层每一次迭代中进行白化操作，而白化中的 `PCA` 等操作计算量极高（涉及到矩阵的逆等等操作）。
2. **白化过程改变了每一层的输入特征表达**：由于白化中的 `PCA` 等降维操作，一定程度上改变了输入的原始表达，并逐层积累，可能会导致网络学习不到完整的信息。

# Batch Normalization

## 提出思路

针对以上问题，我们的改进思路为：在保持原有信息表达的同时，对每层输入进行归一化，并尽可能简化计算！

因此，主要从如下两点入手：

1. **单独对每个特征进行 `Normalization`**。针对白化的计算复杂度问题，我们在每一层中，对每个特征（也就是每个卷积核对应的输出）分别进行归一化，使其均值为 `0`，方差为 `1`。
2. **进行线性归一化变换**：因为白化削弱了每一层信息的表达，因此我们对每一层的输入进行恒等线性变换（`identity transform`），从而使得变换后的数据能够恢复原始表达。

基于以上思路，`Batch Normalization` 就诞生了。

> 需要注意的是，`BN` 说到底是一种归一化手段，因此可能会减小图像之间的绝对差异，突出相对差异，加快训练速度。同时，并非所有领域都可以使用 `BN`。

## 具体算法细节

### 归一化

通常归一化是在整个训练数据集上进行的，但是这样做效率较低，且不宜于使用随机梯度下降。因此，我们对其进行了改进，改为只在每个 `Batch size` 内进行归一化，这需要统计每个 `batch` 内所有样本在对应层对应通道的输入值，计算其均值 `mean` 和标准差 `std`。

### 训练中的 BN

在训练过程中，`BN` 的算法思路如下图所示，其中 $$ \epsilon $$ 用于防止分母为 `0`：

<div style="text-align:center">
<img src="/images/Batch Normalizing Transform.png" width="60%">
<p>前向传播中的 BN 算法</p>
</div><br>

在上面的算法中，对于每层的每个特征通道上，计算当前 `batch` 内的所有样本的对应输入，并统计其均值和方差。然后用该均值和方差，对每个样本对应的输入进行归一化。经过归一化后，所有的输入特征的均值为 `0`，标准差为 `1`。

同时，为了防止归一化导致特征信息的丢失，我们为每一个特征引入一组**可学习参数** $$\gamma^{(k)}, \beta^{(k)}$$，用于恢复原始输入特征，如 `scale and shift` 所示。特别的，当且仅当 $$\gamma^2=\sigma^2,\beta=\mu$$ 时，可以实现等价变换（`identity transform`）并且保留了原始输入特征的分布信息。

可以发现：

$$\begin{aligned}
E[y_i] &= E[\gamma \hat{x_i} + \beta] \\
&= \gamma E[\hat{x_i}] + \beta \\
&= \beta \\

D[y_i] &=  D[\gamma \hat{x_i} + \beta] \\ 
&= \gamma^2 D[\hat{x_i}] \\
&= \gamma^2
\end{aligned}$$

因此，变换之后，当前层的当前通道的输入服从**固定的分布**，即均值为 $$\beta$$，标准差为 $$\gamma$$ 的正态分布，而其具体的形态，由整个训练集决定。也就是说，每一层的每个通道的分布是需要学习的，并且可能是不同的。

### 反向传播中的 BN

在反向传播阶段，`BN` 层中的反向传播如下所示：

<div style="text-align:center">
<img src="/images/BN 中的链式法则.png" width="70%">
<p>反向传播中的 BN 层</p>
</div><br>

如上所示，即为 `BN` 层中 $$\gamma, \beta$$ 的更新公式。这种可为的变换方式，使得网络可以通过学习，自适应训练数据的分布。

### 推理过程的 BN

在推理阶段，少量的输入样本并没有准确的 `batch` 的均值和标准差的统计信息，那么该如何计算？

答案是利用历史训练样本数据的统计信息，训练过程中，会记录每个特征的均值和方差的无偏差估计：

$$
\begin{aligned}
\mu_{test} &=\mathbb{E} (\mu_{train}) \\

\sigma^2_{test} &=\frac{m}{m-1}\mathbb{E}(\sigma^2_{train})
\end{aligned}
$$

然后利用上面的数据，使用如下公式进行转换：

$$
BN(X_{test})=\gamma\cdot \frac{X_{test}-\mu_{test}}{\sqrt{\sigma^2_{test}+\epsilon}}+\beta
$$

其中，$$\gamma$$ 和 $$\beta$$ 为学习到的参数。

## BN 的优势

### 对网络各层进行脱耦

由于每一层的输入分布固定，因此各层不用去适应上一层的变化，即：各层之间进行了脱耦。这有利于各层独立学习，提高整体网络的稳定性和训练速度。

### 对参数范围不敏感

参数范围不合适时，配合上不适宜的学习速率，很可能会导致梯度爆炸或消失。而使用 `BN` 可以使得模型对参数范围不那么敏感。

假定参数由 $$w$$ 变为 $$aw$$，]略 `b`，则：

$$
\begin{aligned}
\hat{\mu} &= a\mu \\
\hat{\delta} &= a \delta \\

BN((aW)x) &= \gamma \frac{aWx - \hat{\mu}}{\hat{\delta}} + \beta = \gamma \frac{aWx - a\mu}{a\delta} + \beta = \gamma \frac{Wx - \mu}{\delta} + \beta = BN(Wx) \\

\frac{\partial BN((aW)x)}{\partial x} &= \gamma \frac{1}{\hat{\delta}}(aW) = \gamma \frac{W}{\delta} = \frac{\partial BN(Wx)}{\partial x} \\

\frac{\partial BN((aW)x)}{\partial (aW)} &= \gamma \frac{1}{\hat{\delta}}(x) = \frac{1}{a} \gamma \frac{x}{\delta} = \frac{1}{a}\frac{\partial BN(Wx)}{\partial W} \\

\end{aligned}
$$

从上面推导可以发现，使用 `BN` 后，即使上一层的参数进行了放大，但是这一层的输入并未发生变化。**因此，`BN` 可以消除各层之间的耦合，从而使得训练更加稳定**。

此外，参数的缩放并未影响对输入的梯度，因此输入的变化，不会影响参数的更新。

另外注意，`loss` 对参数的梯度的缩放倍数，与参数自身的缩放倍数成反比。根据参数更新公式，可以发现网络可以以相对恒定的速率进行参数更新，从而稳定训练。

>  **同时，由于对参数不敏感，所以可以使用较大的学习速率**。

###有一定的正则化效果

由于 `BN` 参数是在每个 `batch` 内进行更新的，因此在训练过程中，不同 `batch` 之间的分布可能会有些微差异，这就为网络的训练引入了一定的随机噪音，这在一定程度上引入了正则化效果。

另外，作者通过也证明了网络加入 `BN` 后，可以丢弃 `Dropout`，模型也同样具有很好的泛化效果。 

### 缓解梯度消失

> **`BN` 允许网络使用饱和性激活函数（例如 `sigmoid`，`tanh` 等），缓解梯度消失问题。**

在不使用 `BN` 层的时候，由于网络的深度与复杂性，很容易使得底层网络变化累积到上层网络中，导致模型的训练很容易进入到激活函数的梯度饱和区；**通过 `normalize` 操作可以让激活函数的输入数据落在梯度非饱和区，缓解梯度消失的问题**；另外通过自适应学习 $$\gamma$$ 与 $$\beta$$ 又让数据保留更多的原始信息。

# 实验结论

## ICS 的影响

为了验证训练中的 `ICS` 的影响，以及 `BN` 对该问题的有效性，我们在 `MNIST` 数据集上进行测试。

我们使用一个极其简单的网络，其中输入为 $$ 28 \times 28 $$ 的二进制图像，以及 `3` 层全连接隐藏层，每一层具有 `100` 个特征数。每一个隐藏层，计算 $$ y = g(Wu + b) $$，并使用 `sigmoid` 非线性激活函数，且权值参数 `W` 按照高斯分布进行随机初始化。

在隐藏层最后，接一个 `10` 个神经元的全连接层，以及一个 `cross-entropy loss`。网络训练 `5000` 步，`bs = 60`。每个隐藏层的激活层之后，添加 `BN`。我们旨在比较基准网络和使用 `BN` 的网络的表现，验证 `BN` 的影响。

<div style="text-align:center">
<img src="/images/验证 BN 的作用.png" width="85%">
<p>验证 BN 的作用</p>
</div><br>

可以发现，使用 `BN` 之后，收敛更快，且训练更稳定。

## ImageNet 数据集上的表现

在一个 `Inception` 网络的变体中，引入 `BN`，在 `ImageNet` 分类数据集上进行试验。卷积层使用 `ReLU` 激活函数，最后不使用全连接层。模型使用随机梯度下降进行训练，`bs = 32`。在这个基础上，我们在每一个非线性激活函数**之前**加上 `BN` 层。

### 加速 BN 网络

简单的添加 `BN` 到网络中，并不能很好的利用 `BN` 的优势，因此，我们还做了如下更改：

1. **Increase learning rate**：在 `BN` 的模型中，可以通过较大的学习速率来加速训练，同时不会产生不良的影响
2. **Remove Dropout**：`BN` 同时具有正则化的效果，因此可以移除 `Dropout` 层来加速训练，同时不会导致过拟合
3. **Reduce the L2 weight regularization**：尽管原始网络中使用 `L2 loss` 来控制过拟合，但是在 `BN` 版本的模型中，这部分正则项的 `loss` 减小了 `5` 倍。
4. **Accelerate the learning rate decay**：在训练过程中，学习速率指数衰减。因为我们的模型相较于原始 `Inception` 训练更快，因此我们对学习速率的衰减速度加快为 `6` 倍
5. **Remove Local Response Normalization**：尽管原始 `Inception` 和其他网络受益于该技术，但是我们发现使用 `BN` 之后，这一项并不必要。
6. **Shuffle training examples more thoroughly**：我们进行了训练数据内随机性 `shuffling`，这将避免相同的样本重复出现在同一 `mini-batch` 中。这将带来在验证集上 `1%` 的改善。
7. **Reduce the photometric distortions**：由于 `BN` 后的网络训练更快，因此处理每一个样本的时间更少，从而需要更多的关注图像的真实信息，而不是变形后的信息。

### 单网络的分类

我们评估了所有如下的网络，所有的网络均在 `LSVRC2012` 训练数据进行训练，并在验证集上测试：

- **Inception**：初始学习速率为 `0.0015`
- **BN-Baseline**：与 `Inception` 一样，只是在每一层非线性激活层之前加上 `BN`。
- **BN-x5**：在 `BN-Baseline` 的基础上，进行了上一节的更改。初始学习速率为 `0.0075`。
- **BN-x30**：与 `BN-x5` 相似，但是初始学习速率为 `0.045`
- **BN-x5-Sigmoid**：与 `BN-x5` 相似，但是使用 `sigmoid` 作为激活函数。

<div style="text-align:center">
<img src="/images/Single crop validation accuracy of some models.png" width="85%">
<p>各模型收敛曲线对比</p>
</div><br>

可以发现，使用 `BN` 之后，收敛更快，精度更高，且可以使用饱和激活函数。

<div style="text-align:center">
<img src="/images/达到最高精度所需的步数.png" width="70%">
<p>各模型收敛步数和精度</p>
</div><br>


### 集成模型

我们使用集成模型，在 `ImageNet 2015` 数据集上，验证集上 `top-5` 错误率为 `4.9%`，测试集上的错误率为 `4.82%`。我们的集成模型使用 `6` 个网络。每一个都是基于 `BN-30`。

<div style="text-align:center">
<img src="/images/Batch-Normalized Inception comparison with previous state of the art.png" width="95%">
<p>集成模型对比</p>
</div><br>