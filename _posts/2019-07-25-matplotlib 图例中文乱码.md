---
layout: post
title:  "matplotlib 图例中文乱码"
date:   2019-07-25 23:39:01 +0800
categories: 数据分析
tag: matplot & seaborn
---


* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****


## 问题描述

使用 `matplotlib` 绘制图形时，中文显示不正常。

## 解决办法

1. 下载中文字体（[黑体](https://www.fontpalace.com/font-details/SimHei/)，看准系统版本）
2. Mac 下解压缩后双击安装
3. 找到 `matplotlib` 字体文件夹，例如：`matplotlib/mpl-data/fonts/ttf`，将 `SimHei.ttf` 拷贝到 `ttf` 文件夹下面
4. 修改配置文件 `matplotlib/mpl-data/matplotlibrc`

> - font.family         : sans-serif   
> - font.sans-serif     : SimHei, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif   
> - axes.unicode_minus:False，#作用就是解决负号'-'显示为方块的问题

5. 配置之后并不会生效，需要重新加载字体

```Python
from matplotlib.font_manager import _rebuild
_rebuild() #reload一下
```