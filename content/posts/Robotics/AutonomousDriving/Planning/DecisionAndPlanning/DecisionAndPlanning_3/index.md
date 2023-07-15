---
title: Decision and Planning [3]
subtitle:
date: 2023-07-15T10:23:57+08:00
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
  - planning
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

{{< admonition quote "quote" false >}}
note abstract info tip success question warning failure danger bug example quote
{{< /admonition >}}

<!--more-->

## 决策规划（三）行为决策常用算法

### 基于规则的方法

在基于规则的方法中，通过对自动驾驶车辆的驾驶行为进行划分，并基于感知环境、交通规则等信息建立驾驶行为规则库。自动驾驶车辆在行驶过程中，实时获取交通环境、交通规则等信息，并与驾驶行为规则库中的经验知识进行匹配，进而推理决策出下一时刻的合理自动驾驶行为。
正如全局路径规划的前提是地图一样，自动驾驶行为分析也成为基于规则的行为决策的前提。不同应用场景下的自动驾驶行为不完全相同，以高速主干路上的L4自动驾驶卡车为例，其自动驾驶行为可简单分解为单车道巡航、自主变道、自主避障三个典型行为。
单车道巡航是卡车L4自动驾驶系统激活后的默认状态，车道保持的同时进行自适应巡航。此驾驶行为还可以细分定速巡航、跟车巡航等子行为，而跟车巡航子行为还可以细分为加速、加速等子子行为，真是子子孙孙无穷尽也。
自主变道是在变道场景（避障变道场景、主干路变窄变道场景等）发生及变道空间（与前车和后车的距离、时间）满足后进行左/右变道。自主避障是在前方出现紧急危险情况且不具备自主变道条件时，采取的紧急制动行为，避免与前方障碍物或车辆发生碰撞。其均可以继续细分，此处不再展开。
上面列举的驾驶行为之间不是独立的，而是相互关联的，在一定条件满足后可以进行实时切换，从而支撑起L4自动驾驶卡车在高速主干路上的自由自在。现将例子中的三种驾驶行为之间的切换条件简单汇总如表2，真实情况比这严谨、复杂的多，此处仅为后文解释基于规则的算法所用。

### 基于学习的方法
