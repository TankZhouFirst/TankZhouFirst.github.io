---
layout: post
title:  "Pytorch optimizer 与 learning rate"
date:   2019-08-20 10:37:01 +0800
categories: 人工智能
tag: Pytorch
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**：

- [torch.optim](<https://ptorch.com/docs/1/optim#how-to-adjust-learning-rate>)

****

# optimizer

`Pytorch` 提供了大量的 `optimizer` 接口。关于 `optimizer` ，查看笔记《常用优化器》。

## 一般形式

一般来讲，`optimizer` 都有如下形式（基类）：

```python
class torch.optim.Optimizer(params, defaults)
```

每个 `optimizer` 都有对应的状态，可以通过如下方式获取，保存，与加载：

```python
# 获取状态
state_dict = optimizer.state_dict()

# 保存状态
torch.save(state_dict, 'optimizer.pth')

# 加载状态
optimizer.load_state_dict(state_dict)
```

## 常用的 optimizer

要使用优化器，首先要定义一个对应的实例，如下所示：

### SGD

**定义方式**

```python
optimizer = torch.optim.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

**参数说明**

```python
momentum      :  动量因子（默认：0）
weight_decay  :  权重衰减（L2 范数）（默认：0）
dampening     :  动量的抑制因子（默认：0）
nesterov      :  使用 Nesterov 动量（默认：False）
```



### Adadelta

**定义方式**

```python
optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

**参数说明**

```python
rho          : 用于计算平方梯度的运行平均值的系数（默认值：0.9）
weight_decay : 权重衰减 (L2范数)（默认值: 0）
```

### Adagrad

**定义方式**

```python
optimizer = torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
```

**参数说明**

```python
lr_decay     : 学习率衰减（默认: 0）
weight_decay : 权重衰减 (L2范数)（默认值: 0）
```

### Adam

**定义方式**

```python
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```

**参数说明**

```python
betas        : 计算梯度运行平均值及其平方的系数（默认：0.9，0.999）
weight_decay : 权重衰减 (L2范数)（默认值: 0）
```

### Adamax

**定义方式**

```python
optimizer = torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```

**参数说明**

```python
betas        : 计算梯度运行平均值及其平方的系数（默认：0.9，0.999）
weight_decay : 权重衰减 (L2范数)（默认值: 0）
```

### ASGD

**定义方式**

```python
optimizer = torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
```

**参数说明**

```python
lambd        : 衰减期（默认：1e-4）
alpha        : eta更新的指数（默认：0.75）
t0           : 指明在哪一次开始平均化（默认：1e6）
weight_decay : 权重衰减（L2范数）（默认: 0）
```

### RMSprop

**定义方式**

```python
optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

**参数说明**

```python
momentum      :  动量因子（默认：0）
alpha         :  平滑常数（默认：0.99）
centered      :  如果为 True，则计算中心化的 RMSProp，
                 通过其方差的估计来对梯度进行归一化
weight_decay  :  权重衰减（L2范数）（默认: 0）
```

### Rprop

**定义方式**

```python
optimizer = torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
```

**参数说明**

```python
etas        :  对（etaminus，etaplis）, 
               它们是乘数增加和减少因子（默认：0.5，1.2）
step_sizes  :  允许的一对最小和最大的步长（默认：1e-6，50）
```

## 优化步骤

所有的 `optimizer` 都会实现 **step()** 更新参数的方法。它能按两种方式来使用。

### optimizer.step()

这是大多数 `optimizer` 所支持的简化版本。一旦梯度被如 `backward()` 之类的函数计算好后，我们就可以调用该函数。

```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

### optimizer.step(closure)

一些优化算法例如 `Conjugate Gradient` 和 `LBFGS` 需要重复多次计算函数，因此你需要传入一个闭包去允许它们重新计算你的模型。这个闭包会清空梯度， 计算损失，然后返回。

```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```

# learning rate

## 一般形式

一般情况下，使用固定的学习速率，如下所示：

```python
optimizer = optim.Adam(model.parameters(), lr= lr)
```

设置固定的学习速率，并使用 `Adam` 对 `model` 的所有参数进行优化。

## 自动调整学习速率

**torch.optim.lr_scheduler** 提供了几种方法来根据 `epoches` 的数量调整学习率。

### LambdaLR

可以使用 `torch.optim.lr_scheduler.LambdaLR` 来基于指定函数返回值，设定学习速率衰减因子。

```python
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

其中，`lr_lambda` 表示 **lambda** 函数，用于根据当前 `epoch` 计算 `lr` 的乘子。如果有多组参数，应该设置多个 **lambda** 函数。

```python
# Assuming optimizer has two groups.
lambda1 = lambda epoch: epoch // 30
lambda2 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)
```

### StepLR

可以使用如下代码，根据当前 **epoch** 设置学习速率。下面代码中，每经过 **step_size** 个 `epoch`，学习速率就乘以 **gamma**。其中，**last_epoch** 表示到哪个 `epoch` 停止更新学习速率。

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)

# 每次 epoch 之后，需要更新 lr
scheduler.step()
```

### MultiStepLR

下面代码，将在指定的 `epoch` 使用指定的 **learning rate**。

```python
# 首先定义一个 optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9)

# 设置 30 和 80 epoch 时，学习速率乘以 0.1
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1, last_epoch=-1)

# 更新学习速率
scheduler.step()
```

### ExponentialLR

将每个参数组的学习速率设置为每一个时代的初始 `lr` 衰减。当 `last_epoch=-1` 时，将初始 `lr` 设置为 `lr` 。

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)
```

### ReduceLROnPlateau

当指标停止改善时，降低学习率。当学习停滞不前时，模型往往会使学习速度降低`2-10` 倍。

这个调度程序读取一个指标量，如果没有提高 `epochs` 的数量，学习率就会降低。

```python
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    mode='min', factor=0.1, patience=10,
    verbose=False, threshold=0.0001, threshold_mode='rel', 
    cooldown=0, min_lr=0, eps=1e-08)
```

其参数含义如下所示：

```python
1. mode
   str, 为 min, max 中的一个. 
   在最小模式下，当监测量停止下降时，lr 将减少; 
   在最大模式下，当监控量停止增加时，lr 会减少。默认值：'min'。

2. factor
    float, 使学习率降低的因素。 new_lr=lr * factor. 默认: 0.1
    
3. patience
   int, epochs 没有改善后，学习率将降低。 默认: 10.
    
4. verbose
    bool, 若为 True，则会向每个更新的 stdout 打印一条消息
    默认: False.
      
5. threshold
    float， 测量新的最优值的阈值，只关注显着变化。 默认: 1e-4

6. threshold_mode
    str， 为 rel, abs 中的一个. 
    在 rel 模型, (默认)
        max mode : dynamic_threshold=best(1+threshold)
        min mode : best(1-threshold)
    在 abs 模型中, dynamic_threshold = best + threshold
    
7. cooldown
    int， 在 lr 减少后恢复正常运行之前等待的时期数。默认的: 0.
    
8. min_lr
    float or list，标量或标量的列表。
    对所有的组群或每组的学习速率的一个较低的限制。 默认: 0.
```

实例如下所示：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')
for epoch in range(10):
    train(...)
    val_loss = validate(...)
    # Note that step should be called after validate()
    scheduler.step(val_loss)
```

## 设置分组 lr

上面的方法，都是对同一网络的所有参数设置一个 `learning rate`。但是在 **finetune** 中经常会需要对网络的不同部分设置不同的学习速率。如下所示：

```python
optimizer = optim.SGD([
        {'params' : net.features.parameters()},   # 使用默认学习速率
        {'params' : net.classifier.parameters(), 'lr' : 1e-2}
    ], lr=1e-5)
```

上面的代码表示，对于 `net.classifier.parameters()` 部分，设置学习速率为 `1e-2`，网络的其余部分的学习速率均为 `1e-5`。

## 查看学习速率

学习速率可以通过 `optimizer` 的 **param_groups** 数组中的元素的 **lr** 来查看，如下所示：

```python
# 直接查看第 0 个参数数组的 lr
optimizer.param_groups[0]['lr']

# 查看所有的 lr
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr
```

在上面的代码中，提到了参数数组，详见 `optimizer` 部分，其中提到可以为不同分组的参数组设置学习速率。