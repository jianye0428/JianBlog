---
title: Effective STL [13] | 尽量使用vector和string来代替动态分配的数组
subtitle:
date: 2023-08-03T09:11:16+08:00
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
  - Effective
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

## 使用new动态分配内存时的注意事项

1. 必须确保以后会delete这个分配。如果后面没有delete，「**new就会产生一个资源泄漏**」。

2. 你须确保使用了**delete**的正确形式:
   如果使用了delete的错误形式，结果会未定义。在一些平台上，程序在运行期会当掉。另一方面，有时候会造成资源泄漏，一些内存也随之而去。
   - 对于分配一个单独的对象，必须使用“delete”。
   - 对于分配一个数组，必须使用“delete []”。

3. 必须确保只**delete**一次。如果一个分配被删除了不止一次，结果也会未定义。


## vector和string

1. vector和string消除了上面的负担，因为它们管理自己的内存。
   当元素添加到那些容器中时它们的内存会增长，而且当一个vector或string销毁时，它的析构函数会自动销毁容器中的元素，回收存放那些元素的内存。

2. vector和string是羽翼丰满的序列容器。
   虽然数组也可以用于STL算法，但没有提供像`begin`、`end`和`size`这样的成员函数，也没有内嵌像`iterator`、`reverse_iterator`或`value_type`那样的`typedef`。而且`char*`指针当然不能和提供了专用成员函数的`string`竞争。STL用的越多，越会歧视内建的数组。

## string 计数问题

很多`string`实现在后台使用了引用计数，「一个消除了不必要的内存分配和字符拷贝的策略，而且在很多应用中可以提高性能」。

事实上，一般认为**通过引用计数优化字符串很重要**，所以C++标准委员会特别设法保证了那是一个合法的实现。

**多线程使用**

如果你在多线程环境中使用了引用计数的字符串，你可能发现<font color=red>「避免分配和拷贝所节省下的时间都花费在后台并发控制上」</font>了，会因为线程安全性导致的性能下降。

如果用到的string实现是引用计数的，而且已经确定string的引用计数在多线程环境中运行，那么至少有3个合理的选择，而且没有一个放弃了STL：

  1. 「**看看库实现是否可以关闭引用计数，通常是通过改变预处理变量的值**」；
  2. 寻找或开发一个不使用引用计数的string实现（或部分实现）替代品；
  3. 「**考虑使用vector<char>来代替string，vector实现不允许使用引用计数，所以隐藏的多线程性能问题不会出现了**」。

当然，使用了vector<char>，就相当于放弃了string的专用成员函数，但大部分功能仍然可以通过STL算法得到，所以从一种语法切换到另一种不会失去很多功能。

## 结论

如果你在使用动态分配数组，你可能比需要的做更多的工作。
要减轻你的负担，就使用vector或string来代替。