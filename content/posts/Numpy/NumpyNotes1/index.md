---
title: Pandas Notes 1
subtitle:
date: 2023-07-15T19:08:57+08:00
draft: true
author:
  name: Jian YE
  link:
  email: 18817571704@163.com
  avatar:
description:
keywords:
license:
comment: false
weight: 0
tags:
  - draft
categories:
  - Pandas
hiddenFromHomePage: false
hiddenFromSearch: false
summary:
resources:
  - name: featured-image
    src: featured-image.jpg
  - name: featured-image-preview
    src: featured-image-preview.jpg
toc: true
math: false
lightgallery: false
---


## numpy function

### 1. **```np.stack()```**;**```np.vstack()```**;**```np.hstack()```**;**```np.concatenate()```** 区别
   - ```np.concatenate()```函数根据指定的维度，对一个元组、列表中的list或者ndarray进行连接
        ```python
        # np.concatenate()
        numpy.concatenate((a1, a2, ...), axis=0)#在0维进行拼接
        numpy.concatenate((a1, a2, ...), axis=1)#在1维进行拼接
        ```

    - ```np.stack()```函数的原型是numpy.stack(arrays, axis=0)，即将一堆数组的数据按照指定的维度进行堆叠。
        ```python
        # np.stack()
        numpy.stack([a1, a2], axis=0)#在0维进行拼接
        numpy.stack([a1, a2], axis=1)#在1维进行拼接
        ```
        > 注意:进行stack的两个数组必须有相同的形状，同时，输出的结果的维度是比输入的数组都要多一维。

    - ```np.vstack()```的函数原型：vstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组。它是**垂直（按照行顺序）的把数组给堆叠起来**。

    - ```np.hstack()```的函数原型：hstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组。它其实就是**水平(按列顺序)把数组给堆叠起来**，与vstack()函数正好相反。

    > ref: https://cloud.tencent.com/developer/article/1378491

### 2. np.sort()
### 3. np.unique()
### 4. np.argsort()