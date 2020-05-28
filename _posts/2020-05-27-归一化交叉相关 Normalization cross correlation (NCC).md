---
layout: post
title:  "归一化交叉相关 NCC"
date:   2020-05-27 23:16:01 +0800
categories: OpenCV
tag: Opencv 集锦
---

* content
{:toc}


****

> **未经许可，严禁任何形式的转载！**

****

**参考**

- [归一化交叉相关 Normalization cross correlation (NCC)](https://www.cnblogs.com/YiXiaoZhou/p/5998153.html)

****

### 简介

**NCC** 用于描述两个图像的相似程度。一般 **NCC** 也会被用来进行图像匹配，即在一个图像中搜索与一小块已知区域的 **NCC** 最高的区域作为对应匹配，然后对准整幅图像。

假设两张图像 $$I_1 \in R^{m \times n}, I_2 \in R^{m \times n}$$，则其 **NCC** 计算如下：
$$
NCC = \frac{\sum I_1 I_2}{\sqrt {(\sum I_1 I_1) \times (\sum I_2 I_2)}}
$$

### 实例

假设如下两张图，需要进行匹配对齐。

<div style="text-align:center">
<img src="/images/left_img.png" width="80%"/>
<img src="/images/right_img.png" width="80%"/>
</div>

具体步骤如下：

1. 从第一幅图中，截取一小块区域 **patch**（这里手工截取，实际需要通过算法实现），如下所示：

    <div style="text-align:center">
    <img src="/images/tp.png" width="30%"/>
    </div>

2. 通过 **NCC** 算法，匹配该区域在另一张图中的位置。由于遍历全图计算量巨大，所以可以先缩放到合适尺寸，匹配出大致区域，然后在该区域附近，进行精确匹配，得到最终的位置。

### 代码实现

```python
import cv2
import numpy as np

def match_roi_region_base(patch_img, dist_img):
    '''
    在 dist_img 中搜索与 patch_img 最匹配的区域，并返回对应区域的左上角坐标

    patch_img：待匹配的图像 patch_img，灰度图
    dist_img：待搜索的图像，灰度图
    '''
    # 这里需要将输入图像转换成 float 类型，不然结果会有问题
    patch_img = patch_img.copy().astype(np.float32)
    dist_img = dist_img.copy().astype(np.float32)

    # 获取尺寸
    h, w = dist_img.shape
    hp, wp = patch_img.shape

    xm, ym = -1, -1   # 匹配区域左上角坐标
    v = -1            # 匹配相关性

    for y in range(h - hp):
        for x in range(w - wp):
            _v = np.sum(dist_img[y:y+hp, x:x+wp] * patch_img)
            _v /= (np.sqrt(np.sum(dist_img[y:y+hp, x:x+wp] ** 2)) * \
                   np.sqrt(np.sum(patch_img ** 2)))
            if _v > v:
                print(_v)
                v = _v
                xm, ym = x, y
    return xm, ym



def match_roi_region(patch_img, dist_img, sfactor):
    '''
    在 dist_img 中搜索与 patch_img 最匹配的区域，并返回对应区域的左上角坐标

    patch_img：待匹配的图像 patch_img，灰度图
    dist_img：待搜索的图像，灰度图
    sfactor：为加速匹配时所用到的缩放因子
    '''
    # 这里需要将输入图像转换成 float 类型，不然结果会有问题
    patch_img = patch_img.copy().astype(np.float32)
    dist_img = dist_img.copy().astype(np.float32)

    # 获取尺寸
    h, w = dist_img.shape
    hp, wp = patch_img.shape

    # 粗匹配，先进行缩小，匹配大致区域
    hp_s, wp_s = hp // sfactor, wp // sfactor   # 尺寸
    h_s, w_s = h // sfactor, w // sfactor
    patch_img_s = cv2.resize(patch_img, (wp_s, hp_s))   # 缩小
    dist_img_s = cv2.resize(dist_img, (w_s, h_s))

    xm_s, ym_s = match_roi_region_base(patch_img_s, dist_img_s)


    # refine 匹配
    xm_start, ym_start = (xm_s - 2) * sfactor, (ym_s - 2) * sfactor
    xm_end, ym_end = (xm_s + 2) * sfactor + patch_img.shape[1], (ym_s + 2) * \
                      sfactor + patch_img.shape[0]
    patch_img_rf = patch_img.copy()
    dist_img_rf = dist_img[ym_start:ym_end, xm_start:xm_end]
    xm, ym = match_roi_region_base(patch_img_rf, dist_img_rf)

    xm, ym = xm + xm_start, ym + ym_start

    out = dist_img.copy()
    cv2.rectangle(out, pt1=(xm, ym), pt2=(xm + wp, ym + hp), color=(0,0,255), \
                  thickness=1)
    out = out.astype(np.uint8)

    # Save result
    cv2.imwrite("out.jpg", out)
    cv2.imshow("result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return xm, ym
    

if __name__ == "__main__":
    patch_img = '/home/em/Desktop/DeepinScreenshot_select-area_20200515095142.png'
    cv2_gray_img = '/home/em/Desktop/NCC/data/camera/066.jpg'

    match_roi_region(cv2.imread(patch_img, 0), cv2.imread(cv2_gray_img, 0), 8)
```