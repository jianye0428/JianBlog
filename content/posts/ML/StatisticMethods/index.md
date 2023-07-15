---
title: 统计学习方法
subtitle:
date: 2023-07-15T17:45:35+08:00
draft: true
author:
  name:
  link:
  email:
  avatar:
description:
keywords:
license:
comment: false
weight: 0
tags:
  - draft
categories:
  - ML
hiddenFromHomePage: false
hiddenFromSearch: false
summary:
resources:
  - name: featured-image
    src: featured-image.jpg
  - name: featured-image-preview
    src: featured-image-preview.jpg
toc: true
math: true
lightgallery: false
---


# CH01 --
# CH02 --
# CH03 --
# CH04 --
# CH05 -- Decision Tree 决策树
# CH06 -- Logistic Regression and Maximum Entropy Model 逻辑斯谛回归和最大熵模型
# CH07 -- Support Vector Machine 支持向量机
- **模型**:支持向量机 (SVM) 是一种二类分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器。支持向量机还包括核技巧，使它成为实质上的非线性分类器。
分离超平面:$\omega ^ * \cdot{x} + b^* = 0$
分类决策函数:$f(x) = sign (\omega ^ * \cdot{x} + b ^ *)$。

- **间隔最大化**,可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。

- 当训练数据线性可分时，通过硬间隔最大化，学习出线性<u>**可分支持向量机**</u>。当训练数据近似线性可分时，通过软间隔最大化，学习出<u>**线性支持向量机**</u>。当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习<u>**非线性支持向量机**</u>。

- **核技巧**：当输入空间为欧式空间或离散集合，特征空间为希尔伯特空间时，核函数表示将输入从输入空间映射到特征空间得到的特征向量之间的内积。通过核函数学习非线性支持向量机等价于在高维的特征空间中学习线性支持向量机。这样的方法称为核技巧。

- 考虑一个二类分类问题，假设输入空间与特征空间为两个不同的空间，输入空间为欧氏空间或离散集合，特征空间为欧氏空间或希尔伯特空间。支持向量机都将输入映射为特征向量，所以支持向量机的学习是在特征空间进行的。

- 支持向量机的最优化问题一般通过对偶问题化为凸二次规划问题求解，具体步骤是将等式约束条件代入优化目标,通过求偏导求得优化目标在不等式约束条件下的极值。

## 线性可分支持向量机：

- 当训练数据集线性可分时，存在无穷个分离超平面可将两类数据正确分开。利用间隔最大化得到唯一最优分离超平面$\omega ^ * \cdot{x} + b^* = 0$和相应的分类决策函数$f(x) = sign (\omega ^ * \cdot{x} + b ^ *)$称为线性可分支持向量机。

- 函数间隔：一般来说，一个点距离分离超平面的远近可以表示分类预测的确信程度。在超平面$\omega ^ * \cdot{x} + b^* = 0$确定的情况下，$|\omega \cdot{x} + b|$能够相对地表示点x距离超平面的远近，而$|\omega \cdot{x} + b|$与 y 的符号是否一致能够表示分类是否正确。所以可用 $\hat{\gamma}$ 来表示分类的正确性及确信度，这就是**函数间隔**。注意到即使超平面不变，函数间隔仍会受 $\omega$ 和 $b$ 的绝对大小影响。

# CH08 --
# CH09 --
# CH10 --
# CH11 --
