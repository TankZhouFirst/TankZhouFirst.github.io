---
layout: post
title:  "Pytorch 之 Tensor 属性及操作"
date:   2019-08-20 08:58:01 +0800
categories: 人工智能
tag: Pytorch
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

# Tensor 的属性

## 类型属性

### 类型查看

可以通过如下代码查看 `tensor` 的类型：

```python
gpu_tensor.type()     # 类型
```

### tensor 支持的的类型

`pytorch` 中的 `tensor` 支持如下类型：

| 数据类型          | CPU tensor         | GPU Tensor              |
| ----------------- | ------------------ | ----------------------- |
| 32 bit 浮点       | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64 bit 浮点       | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16 bit 半精度浮点 | N/A                | torch.cuda.HalfTensor   |
| 8 bit 无符号整型  | Torch.ByteTensor   | Torch.cuda.ByteTensor   |
| 8 bit 有符号整型  | Torch.CharTensor   | Torch.cuda.CharTensor   |
| 16 bit 有符号整型 | Torch.ShortTensor  | Torch.cuda.ShortTensor  |
| 32 bit 有符号整型 | Torch.IntTensor    | Torch.cuda.IntTensor    |
| 64 bit 有符号整型 | Torch.LongTensor   | Torch.cuda.LongTensor   |

其中，`HalfTensor` 是专门为 `GPU` 设计的，其占用的内存空间只有 `CPU` 下 `FloatTensor` 的一半。

## 尺度属性

### 尺寸查看

可以通过如下代码，查看 `tensor` 的相关尺寸。

```python
gpu_tensor.shape      # 尺寸，tuple

gpu_tensor.size()     # 尺寸，tuple 的 size 对象 
gpu_tensor.dim()      # 维度
gpu_tensor.numel()    # 所有元素个数
```

## 查看值

```python
# 对于标量元素的 tensor
x.item()
```

# Tensor 的操作

## 操作的类型

### API 角度

从 `Pytorch API` 角度而言，对 `tensor` 的操作，可分为两类：

1. **torch.function**：如 `touch.save()`
2. **tensor.function**：如 `tensor.save()`

为方便使用，对 `tensor` 而言，大部分操作同时支持这两类接口。

### 存储角度

从存储角度讲，对 `tensor` 的操作又分为两类：

1. 不修改自身，而是返回新的 `tensor`：如 `a.add(b)`
2. 对自身进行修改，即 **inplace** 操作，如：`a.add_(b)`

## 创建 Tensor

### 一般创建方式

```python
t.Tensor(2,3)       # 创建 2 * 3 的 tensor
t.Tensor([1,2,3])   # 创建 tensor，值为 [1,2,3]

# 从 numpy 创建 tensor
torch.Tensor(numpy_array)
torch.from_numpy(numpy_array)

# 将 tensor 转换为 numpy
numpy_array = pytensor2.numpy()  # 在 cpu 上
numpy_array = pytensor2.cpu().numpy()  # 在 gpu 上

# 在制定 GPU 上创建与 data 一样的类型
torch.tensor(data, dtype=torch.float64, device=torch.device('cuda:0'))
```

> **Tensor 与 numpy 对象共享内存**，所以他们之间切换很快，几乎不消耗资源。但是，这意味着如果其中一个变化了，则另一个也会跟着改变。

### 拷贝创建

通过 **clone** 的方式进行创建，如下所示：

```python
b = a.clone()
```

### 创建特殊矩阵

```python
x = torch.empty(5, 3)    # 创建空的 Tensor
x = torch.ones(3,2)      # 创建 1 矩阵
x = torch.zeros(2,3)     # 创建 0 矩阵
x = torch.eye(2,3)       # 创建单位矩阵 

x = torch.arange(1,6,2)  # 创建 [1, 6)，间隔为 2
x = torch.linspace(1, 10, 3)  # [1, 10]  等间距取 3 个数

x = torch.randn(2,3)     # 随机矩阵
x = torch.randperm(5)    # 长度为 5 的随机排列
```

## 常用操作

`tensor` 的 `API` 与 `Numpy` 类似。

### 索引操作

> 索引出来的结果与原 tensor 共存，同时更改。

```python
a = t.randn(3,4)
b= a[:, 1]
```

### 类型转换

各种类型之间可以转换，**type(new_type)** 是通用做法。而 **CPU** 和 **GPU** 之间通过 **tensor.cuda** 和 **tensor.cpu** 进行转换。

```python
import torch

# 设置默认类型
t.set_default_tensor_type('torch.IntTensor')

x = torch.ones(2,2)

b = x.float()
b = x.type(t.floatTensor)  # 两者等效
```

### 沿指定维度取最值

```python
import torch
 
x = torch.randn(3,4)
print(x)
 
# 沿着行取最大值。返回 value 和 index
# torch.min 同理
max_value, max_idx = torch.max(x, dim=1)
print(max_value)
print(max_idx)
```

### 指定维度求和

```python
import torch
 
x = torch.randn(3,4)
print(x)
 
# 沿着行对x求和
sum_x = torch.sum(x, dim=1)
print(sum_x)
```

### 修改维数

```python
import torch
 
x = torch.randn(3,4)
print(x.shape)

# 在指定维度增加
x = x.unsqueeze(0)
print(x.shape)
x = x.unsqueeze(1) # 在第二维增加
print(x.shape)

# 在指定维度减少
x = x.squeeze(0) # 减少第一维
print(x.shape)
x = x.squeeze()
print(x.shape)
```

### 维度转置

```python
import torch
 
x = torch.randn(3,4,5)
print(x.shape)

# 使用permute和transpose进行维度交换
x = x.permute(1,0,2)
print(x.shape)

# transpose交换tensor中的两个维度
x = x.transpose(0,2)
print(x.shape)
```

### 修改尺寸

常用的修改尺寸的方式有 **tensor.view**，**tensor.unsqueeze()**，以及 **tensor.squeeze()** 等，详见笔记 《`max()、view()、 squeeze()、 unsqueeze()`》。

```python
import torch
 
# 使用 view 对 tensor 进行 reshape
x = torch.randn(3,4,5)
print(x.shape)

x = x.view(-1, 5)
# -1 表示任意的大小，5 表示第二维变成 5
print(x.shape)

# 重新 reshape 成 (3, 20) 的大小
x = x.view(3,20)
print(x.shape)
```

除此之外，还有另一种方式，即 **resize**。与 **view** 不同，它可以修改 `tensor` 的尺寸。如果新尺寸总尺寸超过原尺寸，则会自动分配新空间；如果小于，则之前的数据依旧会保留。

```python
a = torch.arange(0, 6)
b = a.view(-1, 3)  # [[0,1,2], [3,4,5]]

b.resize_(1, 3)  # [0,1,2]  仍会保留截断的数据

b.resize_(3,3)   # [[0, 1, 2]， [3, 4, 5]， [0, 0, 0]]
```

### 转换为列表

通过 **tolist()** 可以将 `tensor` 转换为 **list**，如下所示：

```python
a.tolist()
```

## 其他操作

### 元素选择

| 函数                              | 功能                                                         |
| --------------------------------- | ------------------------------------------------------------ |
| `index_select(input, dim, index)` | 在指定维度 `dim` 上选取                                      |
| `masked_select(input, mask)`      | 如 ： `a[a>0]`，使用 `ByteTensor` 进行选取                   |
| `non_zero(input)`                 | 非 `0` 元素下标                                              |
| `gather(input, dim, index)`       | 根据 `index`，在 `dim` 维度上选取数据，输出 `size` 与 `index` 一致 |

如：

```python
a.masked_select(a > 1)  # 等价于 a[a>1]
```

### element-wise 操作

| 函数                                           | 功能                                  |
| ---------------------------------------------- | ------------------------------------- |
| `abs / sqrt / div / exp / fmod / log / pow...` | 绝对值/平方根/除法/指数/求余/幂...    |
| `cos / sin / asin / atan2 / cosh`              | 三角函数                              |
| `ceil / round / floor / trunc`                 | 上取整 / 四舍五入 / 下取整 / 保留整数 |
| `clamp(input, min, max)`                       | 截断为指定区间的值                    |
| `sigmoid / tanh ...`                           | 激活函数                              |

###归并操作 

| 函数                         | 功能                      |
| ---------------------------- | ------------------------- |
| `mean / sum / median / mode` | 均值 / 和 / 中位数 / 众数 |
| `norm / dist`                | 范数 / 距离               |
| `std / var`                  | 标准差 / 方差             |
| `cumsum / cumprod`           | 累加 / 累乘               |

以上大多数函数都有一个参数 **dim**，表示对指定维度进行归并运算。

### 比较运算

| 函数                          | 功能                                        |
| ----------------------------- | ------------------------------------------- |
| `gt / lt / ge / le / eq / ne` | 大于 / 小于 / 不小于 / 不大于 / 等于 / 不等 |
| `topk`                        | 最大的 k 个数                               |
| `sort`                        | 排序                                        |
| `max / min`                   | 比较两个 tensor 的最值                      |
