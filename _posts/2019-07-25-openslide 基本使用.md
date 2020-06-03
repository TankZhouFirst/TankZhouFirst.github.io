---
layout: post
title:  "openslide 基本使用"
date:   2019-07-25 23:03:01 +0800
categories: Python
tag: Python 第三方库
---


* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**


****

`openslide` 主要用于处理病理图像，支持多种格式，如 `svs`、`tiff` 等。

首先需要进行安装，使用如下命令：

```shell
$ pip install openslide-python
```

在使用之前，需要进行导入：

```python
import openslide
# 导入一个分层工具
from openslide.deepzoom import DeepZoomGenerator
```

### openslide

#### 读取文件

```python
slide = openslide.open_slide('dataset/HobI16-053768896760.svs')
```

其返回一个 `openslide` 对象：

```pyhton
OpenSlide('dataset/HobI16-053768896760.svs')
```

当处理完毕后，需要关闭文件：

```python
slide.close()
```

#### 查看属性

通过该 `OpenSlide` 对象，可以查看该病理图的各种信息。

```python
# 查看扫描仪器制造商
slide.detect_format('dataset/HobI16-053768896760.svs')

# 查看所有属性
slide.properties
```

#### 尺寸和层级

```python
# 查看病理图（实际）尺寸
# (33864, 34490)
slide.dimensions

# 下采样因子，对应缩放倍率(边长)
# (1.0, 4.00011598237068, 16.004210544783092)
slide.level_downsamples

# 可显示的级别数，等于上面 level_downsamples 的长度
# 3
slide.level_count

# 在指定层级上的对应尺寸
# 0 表示最大，最高不超过 slide.level_count
# (2116, 2155)
slide.level_dimensions[2]
```

#### 获取图像

```python
# 获取指定尺寸的 RGB 缩略图
slide.get_thumbnail((w, h))

# 读取部分区域
# (x, y) 表示最大分辨率下的左上角初始坐标
# level 表示图形的级别数
# (w, h) 表示要切割的尺寸(在 level 层级下对的尺寸)
slide.read_region((x,y), level , (1528,3432))
```

### DeepZoomGenerator

#### 开始切图

```python
# slide 读取的 opslide 对象，将对其进行切图
# tile_size 设置切图尺寸
# overlap 两次切图之间的重叠尺寸
# limit_bounds  若最后剩余尺寸不足切图尺寸，False 表示舍弃，True 表示保留
data_gen = DeepZoomGenerator(slide, tile_size=224, overlap=0,limit_bounds=False)
```

其返回一个 `DeepZoomGenerator` 对象：

```shell
DeepZoomGenerator(OpenSlide('dataset/HobI16-053768896760.svs'), tile_size=224, overlap=0, limit_bounds=False)
```

#### 相关属性

```python
# 切图总数
data_gen.tile_count

# 可以缩放的层次(以 2 倍为单位)
data_devide.level_count

# 不同层级下，对应的切图数目
data_devide.level_tiles

# 不同层级下，图片的尺寸
data_devide.level_dimensions
```

#### 获取切片

```python
# 将原图缩放到 level 层级对应的尺寸
# 按照前面指定的 tile_size 进取切片，并取 (i, j) 位置对应的 patch，作为返回值
data_devide.get_tile(level, (i,j))
```