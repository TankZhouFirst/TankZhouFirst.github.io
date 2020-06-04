---
layout: post
title:  "生成对抗网络 GAN"
date:   2019-08-12 07:37:01 +0800
categories: 人工智能
tag: GANs
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

- [Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- [GANs 的训练技巧](https://github.com/soumith/ganhacks)

****

# 摘要

本文提出一种全新的框架来评估生成模型（`generative models`），其通过对抗（`adversarial process`）学习的过程进行实现。在该过程中，存在两个模型，分别为生成模型（`generative model`）`G` 和判别模型（`discriminative model`）`D`。

> `G` 用于获取数据集分布；而 `D` 用于评估输入的数据，是来自原始数据，而不是来自 `G` 生成的数据的概率。对于 `G` 而言，训练目标就是尽可能让 `D` 犯错；而对于 `D` 而言，其目标在于正确的判定输入数据的来源。因此，这是一个两个模型之间的博弈游戏。

对于任意的模型 `G` 和 `D`，总存在唯一的收敛结果，此时 `G` 生成数据的分布与原始数据集一致，而 `D` 的判别概率总为 `0.5` 左右，达到纳什均衡。

`GAN` 是一种训练的框架，一种理念，并非具体的模型。因此，`G` 和 `D` 均可为多层神经网络，且整个系统可以通过反向传播进行训练。训练过程无需任何的马尔科夫链（`Markov chains `）或非展开的近似推理网络（`unrolled approximate inference networks`）。

# 引入

## 当前现状

到目前为止，深度学习领域，最具成效的应用均与判别模型相关，因为其有着成熟的反向传播算法和 `dropout` 算法，具有表现较为良好的梯度表现。

而深度生成模型则影响较小，因为极大似然估计和相关策略中，存在大量复杂的统计运算，而这些计算的近似，则较为困难。同时，判别模型中广泛使用的分段式线性单元，在生成模型中较难应用。这对这些问题，我们提出了一种新的生成模型评估策略来进行避免。

## 基本原理

可以将生成模型和判别模型分别视作假币制造者和警察。假币制造者期望做出无法被辨认的假币，而警察则希望能够识别出所有的假币。各自对立的目的，将迫使它们相互改进，假币做的越来越好，警察的辨识能力也越来越高，直到最后，无法再辨认出假币为止。

该框架可以为多种模型和优化算法生成特定的训练算法。在本文中，`G` 和 `D` 均为多层感知机，其中，**`G` 以随机噪声作为输入，生成样本数据**，我们称之为对抗网络（`adversarial nets`）。在这种情况下，我们使用反向传播和 `dropout` 进行训练，且只在前向传播过程中进行采样。整个过程无任何近似推理或者马尔科夫链。

# 相关工作

## 随机生成网络

当前主流的深度生成模型，都聚焦于能够提供详尽的参数化概率分布函数的模型。随后用极大对数似然估计来进行训练。在这一类模型中，最成功的当属玻尔兹曼机（`Boltzmann machine `）。这类模型通常具有复杂的似然函数，因此需要似然梯度的近似。而这些运算往往是困难的。

因此，无需显示表征似然，且能够从期望的分布生成数据的生成模型呼之欲出。随机生成网络就是这样一个能够使用反向传播进行训练的网络。而本文对该思路进行拓展，移除了其中所用到的马尔科夫链。

我们通过如下观察来反向传播生成过程中的导数：

$$
\lim _{\sigma \rightarrow 0} \nabla_{\boldsymbol{x}} \mathbb{E}_{\epsilon \sim \mathcal{N}\left(0, \sigma^{2} \boldsymbol{I}\right)} f(\boldsymbol{x}+\epsilon)=\nabla_{\boldsymbol{x}} f(\boldsymbol{x})
$$

## VAE

`GAN` 与 `VAE` 有些类似，又有差别。两者均为一个可微的生成器网络，以及另一个神经网络。与 `GANs` 不同，在 `VAE` 中，另一个网络是一个识别模型，进行近似推理。`GANs` 需要对可见单元（数据）可微，因此无法对离散数据进行建模；而 `VAE` 则需要对隐含单元可微，因此不能有离散的隐含变量。

## 对抗样本

生成对抗网络有时候容易和另一个概念混淆 —— 对抗样本（`adversarial examples`）。对抗样本是通过基于梯度下降算法直接作用于分类网络的输入而发现的的样本，其目的是找到与数据类似但是被错误分类的样本。
这与本文的工作不同，因为对抗样本并非一个训练生成网络的机制。相反，对抗样本主要是一种分析工具，用来展示神经网络在训练过程中的表现，通常可以很轻易并确定的区分两张人类无法辨别的差异的图片。
对抗样本的存在表明，对抗生成网络的训练不那么容易，因为判别网络可能很容易的判定一张图片的类别，而不用评估人类可观测的属性。（**也就是说，可能网络本身训练的好，但是并不是人类需要关注的点**）

# 对抗网络

## 纳什均衡

先了解下纳什均衡，纳什均衡是指博弈中这样的局面，对于每个参与者来说，只要其他人不改变策略，他就无法改善自己的状况。对应于 `GAN`，情况就是生成模型 `G` 拟合了训练数据的分布，判别模型再也判别数据来源，准确率为 `50%`，约等于乱猜。这时双方网络都得到利益最大化，不再改变自己的策略，也就是不再更新自己的权重。

## 基本原理

### 目标函数

当 `G` 和 `D` 均为多层感知机时，可以直接应用对抗模型。

要在数据 $$x$$ 上学习生成器分布 $$p_g$$，我们对输入噪声定义一个先验分布 $$p_z(z)$$，然后通过函数 $$G(z;\theta_g)$$ 表征一个噪声分布到数据空间的映射，其中，`G` 为可微函数，其为由参数 $$\theta_g$$ 定义的多层感知机。同时，我们需要定义另一个多层感知机 $$D(x;\theta_d)$$ ，其输出一个标量，表示判定概率。**$$D(x)$$ 表示输入 $$x$$ 来自原始数据而非 $$p_g$$ 的概率**。

对于 `D`，我们的训练目标是最大化其正确分辨输入数据来源的概率。同时，我们训练 `G`，来最小化 $$log(1 - D(G(z)))$$，即：使得 `D` 尽可能将生成数据判定为原始数据。综上所述，该博弈游戏的目标函数为（公式一）：

$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

在下一节中，我们将展示对抗网络的理论分析，从本质上展示：若 `G` 和 `D` 的空间足够，该训练目标函数能够使得生成模型学习到原始数据分布。

### 训练图示

非正式的证明和演示，如**图一**所示。

<div style="text-align:center">
<img src="/images/对抗网络原理图示.png" width="95%"/>
<p>图 1 ：对抗网络原理图示</p>
</div><br>

如上图所示，生成对抗网络在训练的同时，更新**判别模型的分布** $$D(x)$$（**蓝色虚线**），从而使之能够区分来自**原始数据分布** $$p_x$$（**黑色点线**）和**生成数据分布** $$p_g(G)$$（**绿色实线**）。

在图形的下半部分为生成器输入噪声 $$z$$ 到输出 $$x$$ 的映射，这里对 $$z$$ 进行均匀采样。其中，向上的曲线表示映射 $$x=G(z)$$ 如何组成非均匀分布 $$p_g$$ 。其中，$$G$$ 在 $$p_g$$ 的高概率密度区域（绿线凸起的地方）进行收缩，而在 $$p_g$$ 的低密度区域进行发散。

- $$(a)$$：对抗网络中，两个网络接近收敛，此时 $$p_g$$ 与 $$p_d$$ 相似，此时，分类器 $$D$$ 较为精确。
- $$(b)$$：在算法内循环中，对 $$D$$ 进行训练，来对输入数据进行判定，并收敛至：$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x)+ p_{g}(x)}$$
- $$(c)$$：在更新 $$G$$ 时，$$D$$ 的梯度将指引 $$G(z)$$ （也就是生成分布）进行改变，使之更易于被判定为原始数据
- $$(d)$$：在进行若干次迭代后，若 $$G$$ 和 $$D$$ 有足够的容量，将会在某一时刻，无法进一步优化，此时 $$p_g = p_{data}$$。此时，$$D$$ 无法判别数据的来源，此时 $$D(x) = \frac{1}{2}$$

### 算法思路

事实上，我们必须以一种迭代的，数值形式的途径来实现该游戏。在训练的内循环中优化 `D` 是计算所不允许的，而且在数据集体量庞大的时候，将会导致过拟合。相反，我们选择**每 `k` 步 `D` 的优化后，进行一次 `G` 的优化**。这种策略将使得，只要 `G` 变化的足够缓慢， `D` 维持于其最优值附近。

> **内循环更新 D 的梯度，固定 G 的梯度，外循环更新 G 的梯度。**

<div style="text-align:center">
<img src="/images/算法 1.png" width="95%"/>
<p>算法 1：GANs 训练算法</p>
</div><br>

此外，公式一可能并不会提供足够的梯度，来使得 `G` 训练的足够好。因此，在训练的早期，此时 `G` 表现相当差，`D` 可以以较大概率拒绝 `G` 生成的数据作为输入。这种情况下，$$log(1 - D(G(z)))$$ 将趋于饱和。与其训练 `G` 来使之最小化 $$log(1 - D(G(z)))$$，我们可以训练 `G`，来使之最大化 $$D(G(z))$$，也就是最小化 $$log(D(G(z)))$$。该目标函数将导致 `G` 和 `D` 的变化具有相同的固定点，但是在训练早期，能够提供更大的梯度。

<div style="text-align:center">
<img src="/images/改善 GAN 训练初期.png" width=70%"/>
<p>改善 GAN 训练初期</p>
</div><br>


# 理论结果

当 $$z \sim p_z$$ 时，生成器 $$G$$ 显式的定义了一个概率分布 $$p_g$$ 作为生成样本分布，即 $$G(z)$$。因此，若容量和训练时间足够，则根据算法一，将会训练得到一个较好的 $$p_{data}$$ 的评估器。本节中，我们并未使用参数化的设置，而是一个无限容量的模型来研究概率密度函数空间上的收敛。

## D：全局最优性证明 $$p_g = p_{data}$$

首先考虑，对于任意给定的生成器 $$G$$，所对应的最优判别器 $$D$$。

### 结论 1

**内容**

若 $$G$$ 固定，则最优化 $$D$$ 为（公式二）：

$$
D_{G}^{*}(\boldsymbol{x})=\frac{p_{\text {data}}(\boldsymbol{x})}{p_{\text {data}}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}
$$

**证明**

对于任意给定的生成器 $$G$$，判别器 $$D$$ 所对应的训练目标是，最大化 $$V(G, D)$$（公式三）：

$$
\begin{aligned} V(G, D) &=\int_{\boldsymbol{x}} p_{\text { data }}(\boldsymbol{x}) \log (D(\boldsymbol{x})) d x+\int_{\boldsymbol{z}} p_{\boldsymbol{z}}(\boldsymbol{z}) \log (1-D(g(\boldsymbol{z}))) d z \\ &=\int_{\boldsymbol{x}} p_{\text { data }}(\boldsymbol{x}) \log (D(\boldsymbol{x}))+p_{g}(\boldsymbol{x}) \log (1-D(\boldsymbol{x})) d x \end{aligned}
$$

采用微元思想，让积分内的每一项最大，则其积分一定也最大。

对于任意的 $$(a, b) \in \mathbb{R}^{2} \backslash\{0,0\}$$，在 $$y = \frac{a}{a + b} \in [0, 1]$$ 时，函数 $$y \rightarrow a \log (y)+b \log (1-y)$$ 取得最大值。判别器只在 $$S u p p\left(p_{\text { data }}\right) \cup S u p p\left(p_{g}\right)$$ 上定义，因此结论得证。

注意，D 的训练目标可以解释为最大化用于评估条件概率 $$P(Y = y|x)$$ 的对数似然，其中，Y 表示 x 是来自 $$p_{data}$$ 还是 $$p_g$$，分别对应 y=1 和 y=0。此时，公式一可变换为（公式四）：

$$
\begin{aligned} C(G) &=\max _{D} V(G, D) \\ &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}\left[\log \left(1-D_{G}^{*}(G(\boldsymbol{z}))\right)\right] \\ &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right] \\ &=\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log \frac{p_{\text { data }}(\boldsymbol{x})}{P_{\text { data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \frac{p_{g}(\boldsymbol{x})}{p_{\text { data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right] \end{aligned}
$$

### 原理 1

**内容**

当且仅当 $$p_g = p_{data}$$ 时，实际的训练目标函数 $$C(G)$$ 取得全局最小值。此时，$$C(G) = -log4$$。

**证明**

当 $$p_g = p_{data}$$ 时，$$D^*_G(x) = \frac{1}{2}$$。 因此，代入公式四，可得：

$$
\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}[-\log 2]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}[-\log 2]=-\log 4
$$

用 $$C(G)=V\left(D_{G}^{*}, G\right)$$ 减去上式时，可以得到（公式五）：

$$
C(G)=-\log (4)+K L\left(p_{\mathrm{data}} \| \frac{p_{\mathrm{data}}+p_{g}}{2}\right)+K L\left(p_{g} \| \frac{p_{\mathrm{data}}+p_{g}}{2}\right)
$$

其中，$$KL$$ 指代 $$KL$$ 散度（`Kullback–Leibler divergence`）。	上式还可以用模型分布和数据生成过程之间的 [`JS` 散度](https://blog.csdn.net/VictoriaW/article/details/56494922)（`Jensen-Shannon divergence`）进行表示（公式六）：

$$
C(G)=-\log (4)+2 \cdot J S D\left(p_{\text { data }} \| p_{g}\right)
$$

推导过程如下所示：

<div style="text-align:center">
<img src="/images/JS 散度推导.jpeg" width="95%"/>
<p>JS 散度推导</p>
</div><br>

由于两个数据分布之间的 `JS` 散度总是非负，且两分布相同时，散度为 `0`，因此 $$C^{*}=-\log (4)$$ 是 $$C(G)$$ 的全局最小值，且仅当 $$p_g = p_{data}$$ 时，取等号。此时，生成模型完美的复现了数据分布。

## G：算法 1 收敛性证明(Convergence of Algorithm 1)

### 结论 2

**内容**

若 $$G$$ 和 $$D$$ 容量足够，且在算法一的每一步中，判别器可以在给定 $$G$$ 时，能够收敛至最优，且 $$p_g$$ 同时进行更新，以便最小化：

$$
\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right]
$$

则最后，$$p_g$$ 将收敛至 $$p_{data}$$。

**证明**

将 $$V(G,D)= U(p_g, D)$$ 作为 $$p_g$$ 的函数，此时，`D` 固定。

注意，在数据分布 $$p_g$$ 内，$$U(p_g, D)$$ 是凸函数。凸函数的上确界的次导数包含该函数取得最大值处的导数。换言之，若 $$f(x)=\sup _{\alpha \in \mathcal{A}} f_{\alpha}(x)$$ ，且 $$f_{\alpha}(x)$$ 对于所有的 $$\alpha$$，在 $$x$$ 上均为凸函数，则：若 $$\beta=\arg \sup _{\alpha \in \mathcal{A}} f_{\alpha}(x)$$，$$\partial f_{\beta}(x) \in \partial f$$。（不懂）

这相当于给定相应 $$G$$ 时，在最优 $$D$$ 处，对 $$p_g$$ 计算一个梯度下降更新。根据原理一，在 $$p_g$$ 上，$$\sup _{D} U\left(p_{q}, D\right)$$ 是凸函数，且只有唯一的全局最优解，因此只需要对 $$p_g$$ 进行足够小的更新，$$p_g$$ 就能收敛至 $$p_x$$，从而结论得证。

事实上，对抗网络只能通过函数 $$G(z;\theta_g)$$ 表示有限类型的 $$p_g$$ 分布，且我们优化 $$\theta_g$$ 而非 $$p_g$$ 自身，因此，不能应用该结论。然而，多层感知机的优异的性能事实上表明，他们是一个可行的模型，尽管其理论基础略微欠缺。

# 实验

我们在一系列数据集上，训练对抗网络，例如 `MNIST`、`Toronto Face Database`、`CIFAR-10` 等。**生成器网络使用的是 `ReLU` 和 `sigmoid` 的混合，而判别器网络只使用 `maxout` 激活函数。在训练判别器网络时，使用了 `Dropout` 层**。而我们的理论框架允许使用 `dropout`，且可以在生成器的中间层使用噪声，但在我们的实现中，我们只在初始层以噪声作为输入。

我们通过在生成器 `G` 的生成样本上，拟合一个高斯 `Parzen` 窗口，并在该分布下记录对数似然估计，来评估 $$p_g$$ 下，测试集数据的概率。其中，高斯分布的参数 $$\delta$$ 通过交叉验证集上的交叉验证进行获取。具体方式这里暂不讨论。结果如下表（表一）所示。

<div style="text-align:center">
<img src="/images/GANs 实验结果.png" width="80%"/>
<p>表 1 ：Parzen window-based log-likelihood estimates</p>
</div><br>

如上表所示，为 `MNIST` 数据集上，测试集上样本的平均对数似然值及其标准差。在 `FTD` 数据集上，我们计算数据集不同文件夹下的数据的标准差，但是其 $$\delta$$ 不同，其使用每个路径下的验证集进行选择。

这种评估对数似然的方法在一定程度上会存在高方差的问题，且在高维空间上的性能表现并不好。但是已经是我们已知的最好的方式了。

图二图三中，我们展示了一些从训练后的生成器采样而来的样本，质量还不错。

<div style="text-align:center">
<img src="/images/手写数字.png" width="90%"/>
<p>图 3：手写数字</p>
</div><br>

# 优缺点

## 缺点

- 可解释性差，生成模型分布并不存在显式的表示
- 训练不稳定。训练阶段， $$D$$ 必须与 $$G$$ 进行较好的同步，尤其是不应该训练多次 $$G$$ 之后，而不更新 $$D$$，否则容易出现 `the Helvetica scenario`。此时，$$G$$ 将会出现坍塌（collapses），也就是说，对于许多输入噪声 $$z$$，生成的数据是一样的，此时其数据多样性将明显不够。
- 它很难去学习生成离散的数据，比如说文本

## 优点

- 整个训练不再需要马尔科夫链，只需要反向传播更新梯度
- 在学习过程中，无需进行推理，且该框架通用性强，理论上，只要是可微分函数都可以用于构建 `D` 和 `G`

表 `2` 总结了生成对抗模型与其它生成模型之间的对比。

<div style="text-align:center">
<img src="/images/生成模型的对比.png" width="98%"/>
<p>表 2：生成模型的对比</p>
</div><br>

前面提到的优点都是计算上的。对抗网络同时还具有一些统计上的优点：

- 因为生成器网络不是直接使用数据样本进行更新的，而是通过判别器传递过来的梯度进行更新的。这意味着，输入成分并未直接复制到生成器参数。
- 对抗网络可以十分锐化（sharp），即使是对于衰减的分布而言，而基于马尔科夫链的方法则要求分布在一定程度上较为模糊（blurry），以便能够融合多种模式。

此外，与其他结构相比，`GANs` 具有以下优势：

- `GANs` 可以比完全明显的信念网络（`NADE, PixelRNN, WaveNet` 等）更快的产生样本,因为它不需要在采样序列生成不同的数据

- 相比于变分自编码器，`GANs` 没有引入任何决定性偏置（`deterministic bias`），变分方法引入决定性偏置,因为他们优化对数似然的下界,而不是似然度本身,这看起来导致了 `VAEs` 生成的实例比 `GANs` 更模糊.
- 相比非线性 `ICA`（`NICE, Real NVE` 等），`GANs` 不要求生成器输入的潜在变量有任何特定的维度或者要求生成器是可逆的.
- 相比玻尔兹曼机和 `GSNs`，`GANs` 生成实例的过程只需要模型运行一次,而不是以马尔科夫链的形式迭代很多次.

# GANs 训练技巧

## 归一化输入

- 归一化图像输入，使之处于 $$[-1, 1]$$ 之间
- 使用 `tanh` 作为生成器输出的最后一层

## 优化的损失函数

- 论文中使用 $$min(log(1-D))$$ 来优化生成器，但实际应用中，使用 $$max(logD)$$ 来优化它，因为原始的公式下，在训练早期，容易发生梯度消失

## 使用球形分布的输入噪声

- 不要使用均匀分布的输入噪声

    <div style="text-align:center">
    <img src="/images/cube.png" width="30%"/>
    <p>cube</p>
    </div><br>

- 从高斯分布进行采样

<div style="text-align:center">
<img src="/images/sphere.png" width="45%"/>
<p>sphere</p>
</div><br>

- 更多细节，参考[论文](https://arxiv.org/abs/1609.04468) 和[源码](https://github.com/dribnet/plat)

## BatchNorm

- 对 `real` 和 `fake` 分别构建 `mini-batch`，每个 `mini-batch` 只能包含一种类别的图像
- 若不能使用 `batchnorm`，则可以使用 `instance normalization`

## 避免稀疏梯度：ReLU、MaxPool

- 若训练中使用稀疏的梯度，则 `GANs` 的稳定性将不能得到保证
- 可以使用 `ReLU`、`MaxPool`，`leakyReLU` 也是不错的
- 对于下采样，使用 `AveragePooling` 或 `Conv2d + stride`
- 对于上采样，使用 [`PixelShuffle`](https://arxiv.org/abs/1609.05158) 或 `ConvTransposed2d + stride`

## 使用软标签或噪声标签

- Label 平滑技术。比方说，`real=1, fake=0`，那么对于每个样本，若为 `real`，则使用一个 `0.7~1.2` 之间的随机值作为 `label`；对于 `fake` 样本，则为 `0.0 ~ 0.3` 之间的值
- 是的判别器的输入样本的 `label` 附带噪声干扰，例如：偶尔翻转样本的 `label`

## DCGAN / Hybrid 模型

- 尽可能使用 `DCGAN`
- 否则，可以使用 `Hybrid` 模型：`KL + GAN or VAE + GAN`

## 借鉴强化学习的稳定性技巧

- 经验重复
    - 维护一个此前迭代中的复用的缓存，并偶尔进行展示
    - 保存过去时刻 `G` 和 `D` 的断点，并每几次迭代进行更换
- 所有的可用于 `deep deterministic policy gradients` 的稳定性技巧

## 使用 Adam 优化器

- 判别器使用 `SGD`，生成器使用 `Adam`

## 追踪训练早期的失败

- `D` 的 `loss` 趋于零：失败
- 检查梯度的范数：若大于 `100`，则表示可能出现问题
- 若一切正常，则 `D` 的 `loss` 应该具有较小的方差，且持续减小
- 若 `G` 的 `loss` 持续降低，则其可能正在生成垃圾样本愚弄生成器

## 不要使用统计数据平衡 `loss`

- 不要试图找寻一个训练规划（训练 `D` 或 `G` 多少次）来避免坍塌

- 若要尝试，请在具有足够的原则指导的情况下进行，而非仅靠直觉，例如：

    ```python
    while lossD > A:
      train D
    while lossG > B:
      train G
    ```

## 对输入添加噪声

- 对 `D` 的输入添加人工噪声
- 对 `G` 的每一层添加高斯噪声

## 在 G 中使用 dropout 层

- 使用 `dropout` 的形式，添加 `50%` 的噪声
- 在某些层使用,无论是训练还是推理阶段
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)

## 其它

- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)