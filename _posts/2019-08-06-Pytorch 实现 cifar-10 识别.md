---
layout: post
title:  "Pytorch 实现 cifar-10 分类"
date:   2019-08-06 21:54:01 +0800
categories: 人工智能
tag: Pytorch
---


* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**


****

# 概述

`cifar10` 数据集是一个常用的彩色图片数据集，共有 10 个类别，分别为：`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`。其中，每张图片的尺寸为 `32 * 32 * 3`。

用 `Pytorch` 完成分类任务时，一般分为如下几个步骤：

- 数据获取和预处理；
- 神经网络模型的建立；
- 神经网络的训练；
- 神经网络的评估；
- 神经网络的优化和超参数调整。

通常需要结合可视化工具来更加高效的查看网络状态以及优化网络。

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.autograd import Variable
from tensorboardX import SummaryWriter

batch_size = 256
class_num = 10
```

# 数据准备和预处理

## 创建 transform

```python
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
```

首先创建一个 `transforms.Compose` 对象，它类似于一个容器，将多种 `transforms` 进行打包，在创建数据集时将会用到。该 `transforms` 包含两个子变换。

其中 `transforms.ToTensor()` 表示将读取到的图像数据 `（numpy.ndarray 或 PIL 类）` 转换为 `Tensor`，数值从 `0~255` 转变到了 `0~1`；维度从 `[H,W,C]` 变为 `[C,H,W]`。

`transforms.Normalize()` 表示归一化，将数值范围转换为 `-1~1`。第一个 `tuple` 表示各个通道的均值，第二个参数表示各个通道的方差，其计算方式如下所示：
$$
Channel =（channel-mean）/ std
$$

## 创建 Dataset 和 Dataloader

在 `Pytorch` 中，并不提供用于数据读取的 `API`，这一点和 `TensorFlow` 不同。虽然 `Pytorch` 并未提供专门的 `API` 用于各类不同数据读取，但是 `Pytorch` 提供了专门的接口用于数据预处理。此外，`Pytorch` 还支持队列化输出 `mini-batch` 样本。

在这里，数据的读取和数据集的创建直接使用 `torchvision` 模块封装好的 `API` 来进行。`torchvision` 主要包含 `datasets/models/transforms/utils` 几个模块，分别表示数据集模块/模型模块/数据转换模块/以及一些工具套件。

### 创建 Dataset

定义好了 `transform` 后，就可以创建数据集了。上面代码中，直接使用官方 `API` `torchvision.datasets.CIFAR10` 来创建数据集，指定数据存放路径为 `root`；`train` 为 `True` 表示获取训练数据，否则为测试数据；`download` 表示下载数据集，若存在，则不下载；`transform` 指定对数据集进行何种预处理。

#### 训练集

```python
# 创建 Dataset
download = True
if os.path.exists('./data'):
    download = False
    
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
```

#### 测试集

```python
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
```

### 创建 Dataloader

创建完数据及后，使用 `torch.utils.data.DataLoader` 创建数据加载器。其中，`trainset` 表示以 `trainset` 数据集作为 `loader` 的输入；`batch_size` 表示 `loader` 单次输出的样本数；`shuffle=True` 表示随即输出样本，否则按顺序输出样本；`num_workers` 表示使用多少个子进程来读取数据。

到此， `dataloader` 就创建完毕。`Dataloader` 为可迭代对象，除了使用 `for` 读取，还可以直接将其转换为迭代器，并使用 `next` 进行访问，如下所示：

```python
dataiter = iter(trainloader)
images, labels = dataiter.next()
```

# 创建模型

定义一个神经网络时，需要创建一个新的 `class`，它继承自 `torch.nn.Module` 类。对于新创建的类，需要实现其中的两个函数，分别为 `__init__` 和 `forward`。其中， `__init__` 用于实例化用到的网络层；`forward` 用于定义前向传播的过程。如下面代码所示：

```python
# 创建模型
# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 48, 3)
        self.bn1   = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU()               # 30

        self.conv2 = nn.Conv2d(48, 48, 3, stride=2)
        self.bn2   = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU()               # 14

        self.conv3 = nn.Conv2d(48, 96, 3)
        self.bn3   = nn.BatchNorm2d(96)
        self.relu3 = nn.ReLU()               # 12

        self.conv4 = nn.Conv2d(96, 96, 3, stride=2)
        self.bn4   = nn.BatchNorm2d(96)
        self.relu4 = nn.ReLU()               # 5

        self.conv5 = nn.Conv2d(96, 192, 3)
        self.bn5   = nn.BatchNorm2d(192)
        self.relu5 = nn.ReLU()               # 3

        self.conv6 = nn.Conv2d(192, 384, 3)
        self.bn6   = nn.BatchNorm2d(384)
        self.relu6 = nn.ReLU()               # 1

        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, class_num)
 
    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.bn4(self.relu4(self.conv4(x)))
        x = self.bn5(self.relu5(self.conv5(x)))
        x = self.bn6(self.relu6(self.conv6(x)))
        x = x.view(-1, 384)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

net = Net()

if torch.cuda.is_available():
    net = net.cuda()

print(net)
```

网络构建完毕后，可以通过 `net.named_parameters()` 来获取所有可训练的网络参数。如下所示：

```python
for name, parm in net.named_parameters():
    print(name, ':', parm.size())
```

# 模型训练与测试

## 定义损失函数和优化器

网络的训练都是基于特定的损失函数进行的。以最小化损失为目标，利用梯度下降算法或其变种，进行参数优化。具体代码如下所示：

```python
# 可视化，讲 graph 写入 tensorboard
dummy_input = torch.rand(100, 3, 32, 32).cuda()
with SummaryWriter(comment='cifar10') as w:
    w.add_graph(net, dummy_input)
 
criterion = nn.CrossEntropyLoss()

def lr_decay(step):
    if step < 50:
        return 0.1
    else:
        decay_rate = step // 5
        return 0.1 * (0.8**decay_rate)

optimizer = optim.SGD(net.parameters(), lr=lr_decay(0), momentum=0.9, weight_decay=0.002)

# TensorBoard writer
writer = SummaryWriter()
```

`Pytorch` 将深度学习中常用的优化方法全部封装在 `torch.optim` 之中，所有的优化方法都是继承基类 `optim.Optimizier`。

在上面的代码中，使用到了交叉熵作为损失函数，并选用 `Adam` 作为优化算法，其中，学习速率为 `0.001`，每部迭代更新 `net.parameters()`。

## 模型的训练

一般来说，神经网络的训练流程为：输入 `batch` 数据 --> 梯度清零 --> 前向传播 --> 计算 `loss` --> 反向传播 --> 更新参数。

```python
def step_train():
    print('...... Training ......')
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # 清零梯度
        optimizer.zero_grad()
        # 获取输出
        outputs = net(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

        if i % 10 == 9:
            writer.add_scalar('data/train loss', running_loss, epoch * 60 + i)
            print('[%d, %5d] training loss: %.3f' % (epoch + 1, i + 1, running_loss))
            running_loss = 0.0

    acc = 100.0 * (correct / total)
    writer.add_scalar('data/train acc', acc, epoch)
    print('Accuracy of the network on the 60000 train images: %d %%' % acc)
```

在上面的代码中，有几点需要注意的。首先是使用使用 `for` 循环读取 `trainloader` 数据。此时，再使用 `cuda()` 方法将其转移至 `GPU` 执行，加快运行。在每次迭代之前，需要对 `grad` 进行清零。随后，计算网络的输出，损失函数，并进行反向传播，更新参数。

## 测试

可以在训练过程中进行测试。每迭代完一次，进行一次测试。代码如下所示：

```python
def step_test():
    print('...... Testing ......')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)  # 更新测试图片的数量
            correct += (predicted == labels).sum().item()  # 更新正确分类的图片的数量

    acc = 100.0 * (correct / total)
    writer.add_scalar('data/test acc', acc, epoch)
    print('Accuracy of the network on the 10000 test images: %d %%' % acc)
```

上面的代码中，`torch.max(outputs.data, 1)` 表示返回一个列，列元素是输入的 `outputs.data` 的每行最大值。

官方给出的准确率为 `53%`，太低。我调整了神经网络，层数不变，每层的神经元数目相应增加。

除了总分类精度，还可以测试单独的各分类的精度，如下所示：

```python
def class_acc():
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            c = (predicted == labels).squeeze()
            
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100.0 * class_correct[i] / class_total[i]))
```

## 迭代过程

```python
for epoch in range(100):
    optimizer = optim.SGD(net.parameters(), lr=lr_decay(epoch), momentum=0.9, weight_decay=0.002)

    step_train()
    step_test()
    class_acc()
```

## 可视化

在训练过程中，引入可视化，观察 `loss` 以及训练集/测试集精度有助于观察模型的状况。在这里，使用 `tensorboard` 记录了模型的流图以及训练过程中的损失值，还有训练精度，测试集精度，得到的曲线如下所示（输入 `tensorboard --logdir='runs'`）：

<div style="text-align:center">
<img src="/images/cifar-10-tensorboard.png" width="95%">
</div><br>
实际上，上面的曲线中，精度的骤变，是由于学习速率的改变导致的。在训练过程中，当 `loss` 不再持续走低时，可以通过调节学习速率，来进一步提高模型的效果。

最后的训练结果如下所示：

<div style="text-align:center">
<img src="/images/cifar-10-精度.png" width="80%">
</div>


# 总结

深度学习的模式基本上是固定的，关键在于模型的选择，优化算法的选择，以及超参数的调试，这需要深厚的理论基础和大量的经验。

可以看到，分阶段降低学习速率，可以提高训练的精度。

