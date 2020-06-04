---
layout: post
title:  "CycleGAN"
date:   2019-08-14 16:52:01 +0800
categories: 人工智能
tag: GANs
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)
- [Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [HomePage](https://junyanz.github.io/CycleGAN/)

****

# 引入

## 行业现状

图对图转换(`Image-to-image translation`) 是计算机视觉领域的一个新兴的任务，其目标是给输入样本加入另一个数据集或者样本的一些属性，即：学习输入输出之间的映射关系。常用的应用领域有**风格转移**等等。  

此前，较为成功的算法有 `pix2pix`，但问题是，其训练所需的数据集必须是一一对应(`paired`)，而很多时候，这样的数据集并不易获取。比如说将莫奈的画与真实自然风景相互转换的任务中，不可能得到实际的样本。

其中，`paired` 和 `unpaired` 的解释如下所示：

<div style="text-align:center">
<img src="/images/样本对的定义.PNG" width="70%"/>
<p>图 2：样本对的定义</p>
</div><br> 

## 本文工作

针对数据对稀少的问题，我们提出一种进行图像转换的方式，使得其可以在缺失匹配数据对的情况下，进行学习。我们的目标是，通过对抗损失，学习一种映射 $$G:X \to Y$$，从而使得 $$G(x)$$ 的分布，与 `Y` 的分布无法区分。

由于该映射是高度约束的（`highly under-constrained`），因此我们同时使用一个反向的映射 $$F:Y \to X$$，并引入循环一致性损失（`cycle consistency loss `）来使得 $$F(G(X)) \approx X$$（反之亦然）。

如下图所示，为 `CycleGAN` 的应用。给定两个无序的集合 `X` 和 `Y`，`CycleGAN` 可以学习其中一个数据集的一些特殊的特性，并将其转换到另一个图像数据集上，从而实现图像风格之间的转换。

<div style="text-align:center">
<img src="/images/CycleGAN 的应用.png" width="94%"/>
<p>图 1：CycleGAN 的应用</p>
</div><br>

值得注意的是，本文提出的方法有**使用限制**：假设两个数据集之间有内在的联系，比如说**同一物体或场景的不同演绎**，比方说不同时节天气下的同一物体等等，而不能是不相关的物体，或者涉及形状的变化。

## 相关工作

### **Generative Adversarial Networks (GANs)**

本文基于 `GAN` 的架构，从而实现从 `x` 到 `y` 的映射生成任务，`GAN` 的基本原理，详见其他笔记。

### **DCGAN**

先来看下 `DCGAN`，它的整体框架和最原始的那篇 `GAN` 是一模一样的，在这个框架下，输入是一个噪声 `z`，输出是一张图片（如下图所示）。因此，我们实际只能随机生成图片，没有办法控制输出图片的样子，更不用说像 `CycleGAN` 一样做图片变换了。

<div style="text-align:center">
<img src="/images/DGAN 结构.jpg" width="90%"/>
<p>DGAN 结构</p>
</div><br>

### **Image-to-Image Translation**

一开始，图对图翻译使用非参数纹理模型(`non-parametric texture model`)，并使用成对的输入输出图像对进行训练。  

之后发展成为使用 `CNNs` 在输入输出图像集上学习参数化转换函数。我们的实现建立在 `pix2pix` 框架的基础上，其使用条件对抗生成网络(`conditional generative adversarial network`) 来学习输入图像到输出图像的映射。与之不同的是，本文提出的方法不需要配对的输入输出样本。

### **Unpaired Image-to-Image Translation**

之前也有一些研究非配对样本学习的问题，其目标是关联两个图像域：`X` 和 `Y`。比如使用 `Bayesian` 架构引入先验概率分布，该分布由两部分构成，分别为由原图像计算得到的基于批量的马尔科夫随机域和多种风格图像计算得到的似然项。  

最近的有 `CoGAN` 和 `cross-modal scene networks` 使用权值共享(`weight-sharing`)策略来学习交叉领域的通用表示(`common representation across domains`) 等等。

与上面的方式不同，我们的研究不需要依赖任何目标导向的输入输出之间的预定义的近似函数，也不需要假设输入输出位于相同的低维嵌入空间。这使得我们的方式是一种适用于视觉和图像任务的通用的解决方案。

### **Cycle Consistency**

使用转移性(`transitivity`) 作为一种规则化结构化数据的 (`regularize structured data`) 方式有一段历史了。在视觉跟踪中，强制执行简单的**前向-后向一致性**(`forward-backward consistency`) 成为标准技巧已经数十年了。  

在语言领域，通过 `back translation and reconciliation` 验证和改善翻译是人工翻译者和机器翻译的技巧。

最近，高阶循环一致性 (`higher-order cycle consistency`)  被用于动作 (`motion`)， `3-D` 模型匹配(`3D shape matching`)、 `cosegmentation`、`dense semantic alignment` 和 `depth estimation` 等结构化数据。  

在本文中，我们引入 `cycle consistency loss` 来使得 `G` 和 `F` 彼此一致。

### **Neural Style Transfer**

`Neural Style Transfer` 是另一种进行图像转换的方式，其通过结合一张图像的内容以及另一张图像的风格来完成图对图转换，这是基于匹配预训练的深度特征的 `Gram` 矩阵的统计特性来实现的。

而我们的工作中，我们主要关注的是如何通过尝试捕获高层次外观结构之间的对应关系来学习两个图像集之间的映射，而非两张特定的图片。因此，我们的方式还可以拓展到其他应用场景。

# CycleGAN 构想

## 整体框架

我们的目标是通过给定的训练样本 $${\{x_i\}}_{i=1}^N$$ 和 $${\{y_j\}}_{j=1}^M$$，学习两个不同图像域 `X` 和 `Y` 之间的映射关系。其中，训练集的数据分布分别为：$$ x \sim p_{data}(x)$$ 和 $$ y \sim p_{data}(y)$$。

如下图所示，模型包含两个映射分别为 `G` 和 `F`。此外还引入了两个判别器 $$D_X$$ 和 $$D_Y$$，分别用于辨识 `x` 和 $$F(y)$$ 以及 `y` 和 $$G(x)$$。其中，目标函数包含两部分，分别为**对抗损失** (`adversarial losses`) 和**循环一致性损失**(`cycle consistency losses`)。对抗损失用于匹配生成分布与目标域训练数据分布；而循环一致性损失用于避免学习到 `G` 和 `F` 相冲突的情况（也就是说，必须可以相互还原）。

<div style="text-align:center">
<img src="/images/CycleGAN 整体框架.png" width="94%"/>
<p>图 3：CycleGAN 整体框架</p>
</div><br>

## 对抗损失

每一个映射对应一个对抗损失（`adversarial loss`），如下所示：

$$
\mathcal{L}_{GAN}(G, D_Y, X, Y) = \Bbb{E}_{y \sim p_{data}(y)}[logD_Y(y)] + \Bbb{E}_{x \sim p_{data}(x)}[log(1 - D_Y(G(x)))]
$$

我们的训练目标是：

$$
\min_G\max_{D_Y} \quad \mathcal{L}_{GAN}(G, D_Y, X, Y)
$$

其中，`G` 尝试使得生成的样本 $$G(x)$$ 与 `Y` 数据集的样本更为接近；而 $$D_Y$$ 则期望，能够正确区分两者。上面的是对于 $$X \to Y$$ 而言的；对于 $$Y \to X$$ 同理有：$$\mathcal{L}_{GAN}(F, D_X, Y, X)$$。

## 循环一致性损失

理论上，对抗损失可以保证学习到映射 `G` 和 `F`，分别输出与期望目标与数据集一致的分布。然而，在足够大的容量的情况下，网络可以将同一输入图像集映射到目标域中的任意随机样本，其中任何已学习的映射可以得出与目标域相匹配的输出分布。因此，仅仅依靠对抗损失并不能保证已学习的函数可以将单个输入 $$x_i$$ 转换为期望的输出 $$y_i$$。

为了进一步减小映射函数的可能性，我们认为应设函数应该保持循环一致性(`cycle-consistent`)：如图 `3` 所示，对于每个图像 `x`，经过一个循环后得到的 $$F(G(x))$$ 应该与之尽可能接近。 我们使用如下函数来表征这种误差。

$$
\mathcal{L}_{cyc}(G, F) = \Bbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1] + \Bbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1]
$$

如下图所示，为一个循环后还原的图像。

<div style="text-align:center">
<img src="/images/CycleGAN 循环输出实例.PNG" width="70%"/>
<p>图 4：CycleGAN 循环输出实例</p>
</div><br>

## 完整目标函数

完整的目标函数如下所示：

$$
\mathcal{L}(G,F,D_X,D_Y) = \mathcal{L}_{GAN}(G,D_Y,X,Y) + \mathcal{L}_{GAN}(F,D_X,Y,X) + \lambda \mathcal{L}_{cyc}(G,F)
$$

其中，$$\lambda$$ 用于平衡两者之间的重要性。因此，我们的总目标就是：

$$
G^*, F^* = arg \min_{G,F} \max_{D_x, D_Y} \mathcal{L}(G,F,D_X,D_Y)
$$

换个角度思考，我们的网络可以视作两个自编码器(`autoencoders`)，分别为 `X --> fake Y --> Rec X` 和 `Y --> fake X --> Rec Y`。

# 实现

## 网络结构

本文的网络借用的他人的设计。网络包含 `2` 个步长为 `2` 的卷积层，一些残差块，以及 `2` 个 `fractionally strided` 卷积层，步长为 `1 / 2` 。对于 `128 * 128` 的输入，使用 `6` 个 `blocks`，对于 `256 * 256` 及更高的输入，使用 `9` 个 `blocks`。  

此外，使用 `instance normalization`。对于判别器，使用 `70 * 70` 的 `PatchGAN`，旨在鉴别 `70 * 70` 的重叠图像块(`overlapping image patches`) 的真伪。这种贴片级鉴别器结构具有比全图像鉴别器更少的参数，并且可以以完全卷积方式处理任意大小的图像。

## 训练细节

为了稳定训练过程，我们使用了两个技巧。  

首先，对于 $${\mathscr {L}_{GAN}}$$ (公式一)，使用 `least-squares` 损失来代替负的对数似然目标。因为这种损失函数在训练期间更加稳健，易于生成高质量的结果。也就是说，对于 `GAN` 损失 $$\mathscr {L}_{GAN}{(G,D,X,Y)}$$ ，训练 `G` 的时候，最小化 $$\Bbb{E}_{x \sim p_{data}(x)}[(D(G(x)) - 1)^2]$$，训练 `D` 的时候，最小化 $$\Bbb{E}_{y \sim p_{data}(y)}[(D(y) - 1)^2] + \Bbb{E}_{x \sim p_{data}(x)}[(G(x))^2]$$。

其次，为了减少模型的波动性，我们使用生成图像的历史池而非最近批次的生成图。在这里，维护一个图形缓冲池，设定存储 `50` 张先前生成的个图像。

所有的实验中，公式 `3` 中设定 $$\lambda = 10$$，且使用 `Adam` 优化器，`batch size` 为 `1`。所有网络的初始学习速率为 `0.0002`。前 `100` 个 `epoch` 中，使用该学习速率，且后 `100` 个 `epoch` 中，不断线性下降到 `0`。  

在实际训练中，优化 `D` 的时候，目标函数值 `/ 2`，所以相对于生成器，`D` 的训练进度减慢。  权值初始化为高斯分布 $$\mathcal{N}(0, 0.02)$$。

## 源码细节

这部分从源码实现角度，对 `CycleGAN` 的原理进行解读（个人理解，不一定对）。

`CycleGAN` 的网络结构如下所示：

<div style="text-align:center">
<img src="/images/77.jpg" width="80%"/>
<p>CycleGAN 框架</p>
</div><br>

对应的 `CycleGAN` 的训练优化步骤如下所示：

<div style="text-align:center">
<img src="/images/CycleGAN 训练优化步骤.png" width="99%"/>
<p>CycleGAN 训练优化步骤</p>
</div><br>

如上面手稿所示：

1. 首先是前向传播，在 `A --> B` 的方向，依次得到 `fakeB`，用于训练 `DA`；同时作为 `GB` 的输入，用于返回 `rec_A`。对 `B --> A` 的方向，也是一样的。

2. 前向传播完成后，开始计算 `loss`，以便进行反向传播。`loss` 包含几部分。其中，`GA`、`GB`、`cycle_A`、`cycle_B` 的 `loss` 容易理解，不再进行讲解

3. 需要注意的是 `idt_A` 和 `idt_B` 这两个 `loss`。其目的如下（根据源码推理的，不一定对）：

   `B` 中的样本经过 `G_A`，我们期望，不会改变其内在本质信息，得到的还是 `B`（外在形式可能发生变化）；同理，包含 `A` 内在信息的 样本经过 `G_B`，也不会改变 `A` 的内在信息。

   原始样本 `A` 经过 `G_A`，变成 `fakeB`，而 `fakeB` 又经过 `G_B`，还原成 `rec_A`。根据循环一致性，`rec_A` 与 `real_A` 本质上，应该是没有区别的。也就是说，`GA + GB` 未改变 `A` 中样本的内在信息。而由上面的 `idt_loss` 的期望可知，`GB` 是不会改变输入数据中的 `A` 的信息的。因此可得，`GA` 也不会更改 `A` 中样本的信息的。

   综上所述，该 `loss` 下的 `CycleGAN` 只会更改 `A` 的外在表现，但是不会改变内在信息，从而实现风格迁移，而不会造成较大的偏差。如：白天的海滩，只会变成梵高风格的白天的海滩，且经过还原后，仍保留白天海滩的信息。

4. 首先，使用第一部分的 `loss` 更新 `G`；接着，使用第二部分的 `loss` 更新 `D` 和 `G`。

# 实验结果

## 验证

实验使用与 `pix2pix` 相同的验证集和方式，在质量和数量上，对该算法与一些基准(`baselines`)进行比较。其中任务包含：`Cityscapes` 数据集上的 `semantic labels <-> photo` 以及 `Google Maps` 上的 `map <-> aerial photo`。

### **AMT perceptual studies**

在 $$map <-> aerial photo$$ 任务中，我们在 `Amazon Mechanical Turk (AMT)` 上进行 `real vs fake` 认知研究(`perceptual studies`)，从而判定我们是研发的输出仿真性。

对于测试的每一个的算法，对应 `25` 名实验者，每名实验者观看多对实验图像对，每一对分别包含一张真假图片，可以是 `map` 或 `photo`，并且点击其认为是真实图像的图片。前 `10` 次是用于熟悉，其选择都会有对应的反馈，告知正确与否，然后以后 `40` 次的实验数据作为评分依据。

### **FCN score**

尽管认知研究(`perceptual studies`)可能是评估图像真实性的黄金准则，但我们也尝试着用自动数值计量的方式(不依赖于人的经验)来评估。

因此，我们使用 `FCN score` 来评估 `Cityscapes labels <-> photo` 任务。`FCN` 准则评估生成的图像的可辨别性，其用的是现成的语义分割算法 (`the fully-convolutional network`)。

对于每张生成的图片，`FCN` 将会预测一个 `label` 映射。随后将会使用标准的语义分割准则，对该标签和 `ground truth label` 进行比较。该准则的意思就是：如果从标签 `car on the road` 生成图像，然后用 `FCN` 检测到该生成的图像中有 `car on the road`，则表示成功。

### **Semantic segmentation metrics**

为了评估 `photo <-> labels`，我们使用 `Cityscapes benchmark` 的标准准则，包括 `per-pixel accuracy`、`per-class accuracy` 和 `mean class Intersection-Over-Union (Class IOU)`。

## **Baselines**

### CoGAN

该算法分别为 `X` 和 `Y` 学习一个生成器，且前几层共享隐含变量的表示。从 `X` 到 `Y` 的翻译可以表示为先寻找生成图像 `X` 的隐含表示，然后将该隐含表示转换为风格 `Y`。

### **SimGAN**

使用对抗损失来训练从 `X` 到 `Y` 的转换。其中用正则项 $${||x - G(x)||}_1$$ 用于惩罚像素级别的较大变化。

### **Feature loss + GAN**

`SimGAN` 的变种，不再在像素级别上进行 `L1` 损失计算，而是在的深度神经网络的深层表示层 (`VGG-16 relu4_2`)进行，因此也称为 `perceptual loss`。

### **BiGAN/ALI**

非条件 `GANs` 学习的是从随机噪声 `z` 到 `X` 的生成器，而 `BiGAN/ALI` 提出还需要学习如何从 `X` 到 `z`。尽管其设计初衷不是用于图对图翻译，但是功能类似。

### **pix2pix**

`pix2pix` 用成对的数据进行训练，其性能相当于图对图翻译的上限。本文的结构用的是非配对数据集，比较其性能可以接近 `pix2pix` 到什么程度。

##与 baseline 的比较 

<div style="text-align:center">
<img src="/images/不同方式的映射结果.PNG" width="98%"/>
<p>不同方式的映射结果</p>
</div><br>

上图为不同的 `label <-> photos` 方式训练结果。

<div style="text-align:center">
<img src="/images/不同方式下aerial photos 与 maps 的转换.PNG" width="95%"/>
<p>不同方式下aerial photos 与 maps 的转换</p>
</div><br>

上图为不同的方式，用 `Google Map` 训练，实现 `aerial photos <-> maps` 映射的结果。

如下表所示，我们的算法有几乎 `1/4` 的可能性会真假难辨。`aerial photos <-> maps`，输入 `256 * 256`。而其他的 `baseline` 则成功的概率几乎为 `0`。

下表为：`maps <--> aerial photos` 问题上的，`AMT "real vs fake"` 测试结果，分辨率为：$$256 \times 256$$。

<div style="text-align:center">
<img src="/images/AMT “real vs fake” test on maps↔aerial photos.PNG" width="90%"/>
<p>AMT “real vs fake” test on maps↔aerial photos</p>
</div><br>

如下面两个表格所示，分别为数据集 `Cityscapes` 上，`label --> photo` 和 `photo --> label` 的性能表现，可以看到，我们的算法性能较为出色。

<div style="text-align:center">
<img src="/images/FCN-scores for different methods.PNG" width="80%"/>
<p>FCN-scores for different methods</p>
</div><br>

上表为不同方法的 `FCN` 得分，在 `Cityscapes labels <--> photo` 上进行评估的。  

<div style="text-align:center">
<img src="/images/Classification performance of photo→labels for different methods.PNG" width="80%"/>
<p>Classification performance of photo→labels for different methods</p>
</div><br>

上表为在 `cityscapes` 数据集上，不同方法的 `photos <--> labels` 分类性能表现。

## 损失函数的分析

如下面两个表格所示，移除 `GAN loss` 后，会导致结果表现急剧下降，移除 `cycle-consistency loss` 亦是如此。因此可得，两者对于整体表现均很重要。

<div style="text-align:center">
<img src="/images/cityscapes 数据集上不同 loss 的 FCN 得分.PNG" width="82%"/>
<p>cityscapes 数据集上不同 loss 的 FCN 得分</p>
</div><br>

在 `cityscapes` `photos <--> labels` 数据集上，对不同 `cycle-GAN` 变种的 `FCN` 得分的评估。

<div style="text-align:center">
<img src="/images/cityscapes 数据集上不同 loss 的分类性能.PNG" width="82%"/>
<p>cityscapes 数据集上不同 loss 的分类性能</p>
</div><br>

上表为在 `cityscapes photos <--> labels` 数据集上，基于不同损失函数的分类性能。

除此之外，我们还评估了只包含一个方向的 `cycle loss` 的性能。`GAN + forward cycle loss` 即：$$\Bbb{E}_{x \sim p_{data}(x)}[||F(G(x)) - x||_1]$$， `GAN + backward cycle loss` 表示 $$\Bbb{E}_{y \sim p_{data}(y)}[||G(F(y)) - y||_1]$$。但是单向 `loss` 经常会导致训练的不稳定性以及 `mode collapse`。如下图所示：

<div style="text-align:center">
<img src="/images/单向 loss 对应的输出.PNG" width="95%"/>
</div><br>

上图为基于 `cityscapes` 数据集训练的 `labels <--> photos` 映射模型，不同变体的性能表现。

## **Image reconstruction quality**

在图 `4` 中可以看到一些重建的随机样本 $$F(G(x))$$。可以观察到，重建的样本与原始输入较为接近。

## **Additional results on paired datasets**

如下图所示，为 `CycleGAN` 生成的样本，其质量与完全监督学习的 `pix2pix` 接近，但是无需使用成对的样本。

<div style="text-align:center">
<img src="/images/CycleGAN 的生成样本.PNG" width="70%"/>
<p>CycleGAN 的生成样本</p>
</div><br>

如上图所示，为用 `pix2pix` 中使用的配对的数据集训练的 `CycleGANs` 的实验结果。

# 应用

## **Collection style transfer** 

如下列图片所示。

<div style="text-align:center">
<img src="/images/风格转移.png" width="90%"/>
<p>图集风格转移</p>
</div><br>

与风格迁移不同，我们的模型学习整个艺术品集的风格，而不是指示单张图片的。因此，模型可以生成 `Van Gogh` 的风格图片，而不仅仅是其单个作品 `Starry Night`。  

其中，`Cezanne`、`Monet`、`Van Gogh`、`and Ukiyo-e` 的作品尺寸分别为：`526`，`1073`，`400` 和 `563`。

## **Object transfiguration**

如下图所示，模型实现 `ImageNet` 数据集中，物体类别之间的转换。

<div style="text-align:center">
<img src="/images/物品转换.png" width="90%"/>
<p>物品转换</p>
</div><br>

## **Season transfer**

如上图所示，实现的是夏冬季节之间的转换，用了 `854 winter` 和 `1273 summer` 张图片。

## **Photo generation from paintings**

<div style="text-align:center">
<img src="/images/照片与绘图转换.png" width="90%"/>
<p>照片与绘图转换</p>
</div><br>

这类应用中，通常需要添加额外的 `loss` 来促使网络映射时，保留输入输出之间的颜色成分。具体来说，即：$$ \mathscr{L}_{identity}(G,F) = \Bbb{E}_{y \sim p_{data}(y)}[||G(y) - y||_1] + \Bbb{E}_{x \sim p_{data}(x)}[||F(x) - x||_1]$$。


如果没有这项损失，则生成器 `G` 和 `F` 就会随意更改输入图像。如下图所示，生成器通常将白天的图像变为黄昏是拍摄的图片。 

<div style="text-align:center">
<img src="/images/图像与绘画转换.png" width="80%"/>
<p>identity loss 的影响</p>
</div><br>

## **Photo enhancement**

<div style="text-align:center">
<img src="/images/图像增强.png" width="95%"/>
<p>图像增强</p>
</div><br>

## **Limitations and Discussion**

如下图所示，为一些失败的例子。

<div style="text-align:center">
<img src="/images/一些失败的例子.png" width="95%"/>
<p>一些失败的例子</p>
</div><br>

对于涉及到几何形状变化的例子，效果不甚理想。例如，`dog --> cat` 转换中，效果不好。可能是因为生成器更倾向于外观的变化。还有一些其他缺陷，如图所示。  尽管如此，在大多数时候，算法表现好不错。