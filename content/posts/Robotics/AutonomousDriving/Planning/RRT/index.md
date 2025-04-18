---
title: RRT (Rapidly-Exploring Random Tree) 算法详解
subtitle:
date: 2024-05-09T16:18:48+08:00
draft: false
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
  - RRT
categories:
  - AV
  - Robotics
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
pagestyle: wide
repost:
  enable: false
  url:
---

# 0. 基于采样的运动规划算法-RRT(Rapidly-exploring Random Trees)

RRT是Steven M. LaValle和James J. Kuffner Jr.提出的一种通过随机构建Space Filling Tree实现对非凸高维空间快速搜索的算法。该算法可以很容易的处理包含障碍物和差分运动约束的场景，因而广泛的被应用在各种机器人的运动规划场景中。

<br>
<center>
  <img src="images/0_1.webp" width="480" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

## 1、 Basic RRT算法

原始的RRT算法中将搜索的起点位置作为根节点，然后通过随机采样增加叶子节点的方式，生成一个随机扩展树，当随机树的叶子节点进入目标区域，就得到了从起点位置到目标位置的路径。

伪代码如下：

<br>
<center>
  <img src="images/1_1.webp" width="480" height="380" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

上述伪代码中，M是地图环境，$x_{init}$是起始位置，$x_{goal}$是目标位置。路径空间搜索的过程从起点开始，先随机撒点$x_rand$;然后查找距离 $x_rand$ 最近的节点 $x_{near}$;然后沿着 $x_{near}$ 到 $x_{rand}$方向前进stepsize的距离得到$x_{new}$; CollisionFree(M, $E_i$) 方法检测Edge $(x_{new},x_{near})$ 是否与地图环境中的障碍物有碰撞，如果没有碰撞，则将成功完成一次空间搜索拓展。重复上述过程，直至达到目标位置。

<br>
<center>
  <img src="images/1_2.webp" width="640" height="200" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源:https://www.researchgate.net/profile/Burak_Boyacioglu/publication/306404973/figure/fig1/AS:398553223581697@1472033901892/Basic-RRT-algorithm.png

## 2、基于概率的RRT算法

为了加快随机树收敛到目标位置的速度，基于概率的RRT算法在随机树的扩展的步骤中引入一个概率 $p$，根据概率 $p$ 的值来选择树的生长方向是随机生长($x_{rand}$) 还是朝向目标位置 $x_{goal}$ 生长。引入向目标生长的机制可以加速路径搜索的收敛速度。

<br>
<center>
  <img src="images/2_1.webp" width="480" height="560" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">基于概率的RRT算法</div>
</center>
<br>

## 3、RRT Connect算法

RRT Connect算法从初始状态点和目标状态点同时扩展随机树从而实现对状态空间的快速搜索。

<br>
<center>
  <img src="images/3_1.webp" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源:https://www.cs.cmu.edu/~motionplanning/lecture/lec20.pdf

## 4、RRT*算法

RRT*算法的目标在于解决RRT算法难以求解最优的可行路径的问题，它在路径查找的过程中持续的优化路径，随着迭代次数和采样点的增加，得到的路径越来越优化。迭代的时间越久，就越可以得到相对满意的规划路径。

<br>
<center>
  <img src="images/4_1.webp" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源：https://blog.csdn.net/gophae/article/details/103231053

RRT*算法与RRT算法的区别主要在于两点：

1. rewrite的过程。即为 $x_{new}$ 重新选择父节点的过程；
2. 随机树重布线的过程；

## **4.1 Rewrite**

下面我们看看Rewrite是如何实现的。RRT*在找到距离 $x_{rand}$ 最近的节点 $x_{nearest}$ 并通过CollisionFree检测之后，并不立即将 Edge(x_{nearest},x_{rand}) 加入扩展树中。

<br>
<center>
  <img src="images/4_2.webp" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>
图片来源：https://blog.csdn.net/weixin_43795921/article/details/88557317

而是以 $x_{rand}$ 为中心，$r$ 为半径，找到所有潜在的父节点集合，并与 $x_{nearest}$ 父节点的Cost对比，看是否存在更优Cost的父节点。

<br>
<center>
  <img src="images/4_3.webp" width="440" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源：https://blog.csdn.net/weixin_43795921/article/details/88557317

如下图所示，我们会计算路径 $x_{init} \rightarrow x_{parent} \rightarrow x_{child}$ 的Cost=$cost1$，再计算 $x_{init} \rightarrow x_{potential_parent} \rightarrow x_{child}$ 的Cost=$cost2$，比较 $cost1$和 $cost2$ 的大小。此处由于 $x_{potential_parent}$ 与 $ {x_child} $ 之间存在障碍物导致二者的直接连线不可达，所以 $cost > cost1$ ，不需改变 $ x_{child} $ 的父节点。

<br>
<center>
  <img src="images/4_4.webp" width="440" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源：https://blog.csdn.net/weixin_43795921/article/details/88557317

如下图所示，当路径 $\{x_{init} \rightarrow x_{parent}->x_{child}\} $的Cost大于 $\{x_{init} \rightarrow x_{potential\_parent} \rightarrow x_{child}\}$的Cost时，RRT^*算法会将Edge$\{ x_{parent} \rightarrow x_{child}\}$剔 除 , 并 新 增 Edge$\{ x_{potential\_parent} \rightarrow x_{child}\} $。

<br>
<center>
  <img src="images/4_5.webp" width="440" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源：https://blog.csdn.net/weixin_43795921/article/details/88557317

至此我们就完成了一次Rewrite的过程，新生成的随机树如下。

<br>
<center>
  <img src="images/4_6.webp" width="440" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源：https://blog.csdn.net/weixin_43795921/article/details/88557317

## 4.2 随机树重布线的过程

<br>
<center>
  <img src="images/4_7.webp" width="440" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

图片来源：https://blog.csdn.net/weixin_43795921/article/details/88557317

在为 $x_{new}$ 重新选择父节点之后，重布线使得生成新节点后的随机树减少冗余通路，减小路径代价。

如上图所示，$x_{new}$ 为新生成的节点，4、6、8是 $x_{new}$ 的近邻节点，0、 4、 5分别为近邻节点的父节点。

路径{0->4}的Cost为: 10

路径{0->4->6}的Cost为： 10 + 5 = 15

路径{0->1->5->8}的Cost为: 3 + 5 + 1 = 9

先尝试将节点4的父节点改为 $x_{new}$，到达节点4的路径变为{0->1->5->9->4}，新路径的Cost=3+5+3+4=15，新路径的Cost大于原路径Cost，所以不改变节点4的父节点。

再尝试改变节点8的父节点为 $x_{new}$，到达节点8的路径变为{0->1->5->9->8},新路径的Cost=3+5+3+3=14，新路径的Cost大于原路径Cost，随意不改变节点8的父节点。

再尝试改变节点6的父节点为 $x_{new}$，到达路径6的路径变为{0->1->5->9->6},新的Cost=3+5+3+1=12,新路径的Cost小于原路径Cost，因此将节点6的父节点更新为节点9。

重布线后的随机树如上右图所示。

## 4.3 RRT*算法效果

从RRT与RRT*的效果可以看出，RRT*的路径规划的结果优于RRT算法。

<br>
<center>
  <img src="images/4_8.webp" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

RRT^* VS RRT。图片来源：https://www.cc.gatech.edu/~dellaert/11S-AI/Topics/Entries/2011/2/21_PRM,_RRT,_and_RRT__files/06-RRT.pdf

<br>
<center>
  <img src="images/4_9.webp" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

RRT^* VS RRT With Obstacles。图片来源：https://www.cc.gatech.edu/~dellaert/11S-AI/Topics/Entries/2011/2/21_PRM,_RRT,_and_RRT__files/06-RRT.pdf


RRT^*算法+赛车动力学实现车辆180度转弯。图片来源：https://www.youtube.com/watch?v=KSB_9KE6fWI

## **参考链接**

1、基于RRT的运动规划算法综述([https://wenku.baidu.com/view/8de40fafbdeb19e8b8f67c1cfad6195f312be80a.html](https://link.zhihu.com/?target=https%3A//wenku.baidu.com/view/8de40fafbdeb19e8b8f67c1cfad6195f312be80a.html))

2、RRT维基百科([https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Rapidly-exploring_random_tree))

3、PRM, RRT, and RRT*，Frank Dellaer ([https://www.cc.gatech.edu/~dellaert/11S-AI/Topics/Entries/2011/2/21_PRM,_RRT,_and_RRT__files/06-RRT.pdf](https://link.zhihu.com/?target=https%3A//www.cc.gatech.edu/~dellaert/11S-AI/Topics/Entries/2011/2/21_PRM%2C_RRT%2C_and_RRT__files/06-RRT.pdf))

4、路径规划——改进RRT算法(https://zhuanlan.zhihu.com/p/51087819)

5、运动规划RRT*算法图解([https://blog.csdn.net/weixin_43795921/article/details/88557317](https://link.zhihu.com/?target=https%3A//blog.csdn.net/weixin_43795921/article/details/88557317))

6、全局路径规划：图搜索算法介绍4(RRT/RRT*)([https://blog.csdn.net/gophae/article/details/103231053](https://link.zhihu.com/?target=https%3A//blog.csdn.net/gophae/article/details/103231053))

ref:
[1]. https://zhuanlan.zhihu.com/p/133224593
[2]. https://blog.csdn.net/gophae/article/details/103231053
[3]. https://xwlu.github.io/wiki/path-planning/rrt/
[4]. ※ https://dlonng.com/posts/rrt