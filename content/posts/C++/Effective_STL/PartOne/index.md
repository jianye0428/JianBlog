---
title: Effective STL 精读总结 [1] | 容器
subtitle:
date: 2023-09-01T08:38:45+08:00
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
  - Effective STL
categories:
  - C++
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

# 前言

> Effective-STL总结系列分为七部分，本文为第一部分，涉及原书第一章，内容范围Rule01~12。为方便书写，Rule12简写为R12。

{{< admonition Note "Effective-STL系列List" false >}}
本博客站点系列内容如下：</br>
💡 [Effective STL(第3版)精读总结(一)](https://jianye0428.github.io/posts/partone/)</br>
💡 [Effective STL(第3版)精读总结(二)](https://jianye0428.github.io/posts/parttwo/)</br>
💡 [Effective STL(第3版)精读总结(三)](https://jianye0428.github.io/posts/partthree/)</br>
💡 [Effective STL(第3版)精读总结(四)](https://jianye0428.github.io/posts/partfour/)</br>
{{< /admonition >}}

## R01 慎重选择容器类型

STL 容器不是简单的好，而是确实很好。

容器类型如下：
  - 标准 STL 序列容器：`vector`、`string`、`deque`、`list`。
  - 标准 STL 关联容器：`set`、`multiset`、`map`、`multimap`。
  - 非标准序列容器 `slist` 和 `rope`。`slist` 是一个单向链表，`rope` 本质上是一个 "重型" `string`。
  - 非标准的关联容器：`hash_set`、`hash_multiset`、`hash_map`、`hash_multimap`。
  - `vector` 作为 `string` 的替代。
  - `vector` 作为标准关联容器的替代：有时 `vector` 在运行时间和空间上都要优于标准关联容器。
  - 几种标准的非 STL 容器：`array`、`bitset`、`valarray`、`stack`、`queue`、`priority_queue`。

容器选择标准:
  1. vector、list和deque有着不同的复杂度，vector是默认使用的序列类型。当需要频繁在序列中间做插入和删除操作时，应使用list。当大多数插入和删除操作发生在序列的头部和尾部时，应使用deque。
  2. 可以将容器分为连续内存容器和基于节点的容器两类。连续内存容器把元素存放在一块或多块(动态分配的)内存中，当有新元素插入或已有的元素被删除时，同一块内存中的其他元素要向前或向后移动，以便为新元素让出空间，或者是填充被删除元素所留下的空隙。这种移动会影响到效率和异常安全性。标准的连续内存容器有vector、string和deque，非标准的有rope。
  3. 基于节点的容器在每一个(动态分配的)内存块中只存放一个元素。容器中元素的插入或删除只影响指向节点的指针，而不影响节点本身，所以插入或删除操作时元素是不需要移动的。链表实现的容器list和slist是基于节点的，标准关联容器也是(通常的实现方式是平衡树)，非标准的哈希容器使用不同的基于节点的实现。
  4. 是否需要在容器的任意位置插入新元素？需要则选择序列容器，关联容器是不行的。
  5. 是否关心容器中的元素是如何排序的？如果不关心，可以选择哈希容器，否则不能选哈希容器(unordered)。
  6. 需要哪种类型的迭代器？如果是随机访问迭代器，那就只能选择vector、deque和string。如果使用双向迭代器，那么就不能选slist和哈希容器。
  7. 是否希望在插入或删除元素时避免移动元素？如果是，则不能选择连续内存的容器。
  8. 容器的数据布局是否需要和C兼容？如果需要只能选`vector`。
  9. 元素的查找速度是否是关键的考虑因素？如果是就考虑哈希容器、排序的`vector`和标准关联容器。
  10. 是否介意容器内部使用引用计数技术，如果是则避免使用`string`，因为`string`的实现大多使用了引用计数，可以考虑用`vector<char>`替代。
  11. 对插入和删除操作需要提供事务语义吗？就是说在插入和删除操作失败时，需要回滚的能力吗？如果需要则使用基于节点的容器。如果是对多个元素的插入操作(针对区间)需要事务语义，则需要选择`list`，因为在标准容器中，只有`list`对多个元素的插入操作提供了事务语义。对希望编写异常安全代码的程序员，事务语义尤为重要。使用连续内存的容器也可以获得事务语义，但是要付出性能上的代价，而且代码也不那么直截了当。
  12. 需要使迭代器、指针和引用变为无效的次数最少吗？如果是则选择基于节点的容器，因为对这类容器的插入和删除操作从来不会使迭代器、指针和引用变成无效(除非它们指向一个正在删除的元素)。而对连续内存容器的插入和删除一般会使得指向该容器的迭代器、指针和引用变为无效。

## R02 不要试图编写独立于容器类型的代码

1. 容器是以类型作为参数的，而试图把容器本身作为参数，写出独立于容器类型的代码是不可能实现的。因为不同的容器支持的操作是不同的，即使操作是相同的，但是实现又是不同的，比如带有一个迭代器参数的erase作用于序列容器时，会返回一个新的迭代器，但作用于关联容器则没有返回值。这些限制的根源在于，对于不同类型的序列容器，使迭代器、指针和引用无效的规则是不同的。

2. 有时候不可避免要从一个容器类型转到另一种，可以使用封装技术来实现。最简单的方式是对容器类型和其迭代器类型使用typedef，如`typedef vector<Widget> widgetContainer`; `typedef widgetContainer::iterator WCIterator`; 如果想减少在替换容器类型时所需要修改的代码，可以把容器隐藏到一个类中，并尽量减少那些通过类接口可见的、与容器有关的信息。

**一种容器类型转换为另一种容器类型：typedef**
```c++
class Widget{...};
typedef vector<Widget> WidgetContainer;
WidgetContainer cw;
Widget bestWidget;
...
WidgetContainer::iterator i = find(cw.begin(), cw.end(), bestWidget);
```
这样就使得改变容器类型要容易得多，尤其当这种改变仅仅是增加一个自定义得分配子时，就显得更为方便（这一改变不影响使迭代器/指针/引用无效的规则）。

## R03 确保容器中的对象拷贝正确而高效
- 存在继承关系的情况下，拷贝动作会导致**剥离**（slicing）: 如果创建了一个存放基类对象的容器，却向其中插入派生类对象，那么在派生类对象（通过基类的拷贝构造函数）被拷贝进容器时，它所特有的部分（即派生类中的信息）将会丢失。
  ```c++
  vector<Widget> vw;
  class SpecialWidget:			// SpecialWidget 继承于上面的 Widget
    public Widget{...};
  SpecialWidget sw;
  vw.push_back();					// sw 作为基类对象被拷贝进 vw 中
                  // 它的派生类特有部分在拷贝时被丢掉了
  ```
- **剥离意味着向基类对象中的容器中插入派生类对象几乎总是错误的**。
- 解决剥离问题的简单方法：<u>使容器包含指针而不是对象</u>。

1. 容器中保存的对象，并不是你提供给容器的那些对象。从容器中取出对象时，也不是容器中保存的那份对象。当通过如insert或push_back之类的操作向容器中加入对象时，存入容器的是该对象的拷贝。当通过如front或back之类的操作从容器中 取出对象时，所得到的是容器中对象的拷贝。进去会拷贝，出来也是拷贝，这就是STL的工作方式。
2. 当对象被保存到容器中，它经常会进一步被拷贝。当对vector、string或deque进行元素的插入或删除时，现有元素的位置通常会被移动（拷贝）。如果使用下列任何算法，next_permutation或previous_permutation，remove、unique，rotate或reverse等等，那么对象将会被移动（拷贝），这就是STL的工作方式。
3. 如果向容器中填充的对象拷贝操作很费时，那么向容器中填充对象这一简单操作将会成为程序的性能瓶颈。而且如果这些对象的拷贝有特殊含义，那么把它们放入容器还将不可避免地会产生错误。
4. 当存在继承关系时，拷贝动作会导致剥离。也就是说，如果创建了一个存放基类对象的容器，却向其中插入派生类的对象，那么派生类对象（通过基类的拷贝构造函数）被拷贝进容器时，它派生类的部分将会丢失。
5. 使拷贝动作高效、正确，并防止剥离问题发生的一个简单办法就是使容器包含对象指针，而不是对象本身。拷贝指针的速度非常快，而且总是会按你期望的方式进行。如果考虑资源的释放，智能指针是一个很好的选择。

## R04 调用 empty 而不是检查 size()是否为0

- `empty` 通常被实现为内联函数（inline function），并且它做的仅仅是返回 size() 是否为 0.
- `empty` 对所有标准容器都是常数时间操作，而对于一些 list 实现，size 耗费线性时间



ref:
[1]. https://www.cnblogs.com/Sherry4869/p/15128250.html</br>
[2]. https://blog.csdn.net/zhuikefeng/article/details/108164117#t42