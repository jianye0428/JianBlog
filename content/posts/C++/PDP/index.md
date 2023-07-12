---
title: Index
subtitle:
date: 2023-07-12T09:27:30+08:00
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
  - draft
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

https://blog.csdn.net/weixin_46396176/article/details/122449232


https://www.nowcoder.com/discuss/697725
https://blog.csdn.net/qq_41593516/article/details/126198291?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7EPayColumn-1-126198291-blog-126696227.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7EPayColumn-1-126198291-blog-126696227.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1

https://blog.csdn.net/qq_40145095/article/details/126696227

https://blog.csdn.net/Sophia_11/article/details/90284559?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-90284559-blog-90115599.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-90284559-blog-90115599.pc_relevant_vip_default&utm_relevant_index=4


https://blog.csdn.net/weixin_44690935/article/details/119788687?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-119788687-blog-90115599.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-119788687-blog-90115599.pc_relevant_vip_default&utm_relevant_index=5


[自动驾驶 决策](https://blog.csdn.net/qq_40145095/article/details/126696227)
## 1. C++语言考察

1. 引用和指针有什么区别，指针作为函数参数注意问题，引⽤作为函数参数以及返回值的好处？
[引用和指针的区别](https://blog.csdn.net/xsydalao/article/details/93623647)

用指针变量作函数参数可以将函数外部的地址传递到函数内部，使得在函数内部可以操作函数外部的数据，并且这些数据不会随着函数的结束而被销毁。

引⽤作为函数参数以及返回值的好处？

  - 在函数内部会对此参数进行修改
  - 提高函数调用和运行效率, 没有生成副本
  - 在内存中不产生被返回值的副本
2. 函数的参数和返回值的传递方式有哪些？他们的区别是什么？

值传递
引用传递
指针传递

3. 什么是虚函数？简述 C++虚函数作用及底层实现原理？
虚函数 多态

4. 构造函数为什么不能定义为虚函数？ ⽽析构函数⼀般写成虚函数的原因 ？

构造函数不能声明为虚函数的原因是:
> 1 构造一个对象的时候，必须知道对象的实际类型，而虚函数行为是在运行期间确定实际类型的。而在构造一个对象时，由于对象还未构造成功。编译器无法知道对象 的实际类型，是该类本身，还是该类的一个派生类，或是更深层次的派生类。无法确定。。。

> 2 虚函数的执行依赖于虚函数表。而虚函数表在构造函数中进行初始化工作，即初始化vptr，让他指向正确的虚函数表。而在构造对象期间，虚函数表还没有被初始化，将无法进行。

虚函数的意思就是开启动态绑定，程序会根据对象的动态类型来选择要调用的方法。然而在构造函数运行的时候，这个对象的动态类型还不完整，没有办法确定它到底是什么类型，故构造函数不能动态绑定。（动态绑定是根据对象的动态类型而不是函数名，在调用构造函数之前，这个对象根本就不存在，它怎么动态绑定？）
编译器在调用基类的构造函数的时候并不知道你要构造的是一个基类的对象还是一个派生类的对象。

为什么构造函数不能声明为虚函数，析构函数可以？

构造函数不能声明为虚函数，析构函数可以声明为虚函数，而且有时是必须声明为虚函数。
不建议在构造函数和析构函数里面调用虚函数。

构造函数不能声明为虚函数的原因是:
> 1 构造一个对象的时候，必须知道对象的实际类型，而虚函数行为是在运行期间确定实际类型的。而在构造一个对象时，由于对象还未构造成功。编译器无法知道对象 的实际类型，是该类本身，还是该类的一个派生类，或是更深层次的派生类。无法确定。。。

> 2 虚函数的执行依赖于虚函数表。而虚函数表在构造函数中进行初始化工作，即初始化vptr，让他指向正确的虚函数表。而在构造对象期间，虚函数表还没有被初 始化，将无法进行。

> 虚函数的意思就是开启动态绑定，程序会根据对象的动态类型来选择要调用的方法。然而在构造函数运行的时候，这个对象的动态类型还不完整，没有办法确定它到底是什么类型，故构造函数不能动态绑定。（动态绑定是根据对象的动态类型而不是函数名，在调用构造函数之前，这个对象根本就不存在，它怎么动态绑定？）
编译器在调用基类的构造函数的时候并不知道你要构造的是一个基类的对象还是一个派生类的对象。

> 析构函数设为虚函数的作用:
解释：在类的继承中，如果有基类指针指向派生类，那么用基类指针delete时，如果不定义成虚函数，派生类中派生的那部分无法析构。

> 如何定义一个只能在堆(栈)上生成对象的类?


1、只能在堆上生成对象：将析构函数设置为私有。
原因：C++是静态绑定语言，编译器管理栈上对象的生命周期，编译器在为类对象分配栈空间时，会先检查类的析构函数的访问性。若析构函数不可访问，则不能在栈上创建对象。

2、只能在栈上生成对象：将new 和 delete 重载为私有。
原因：在堆上生成对象，使用new关键词操作，其过程分为两阶段：第一阶段，使用new在堆上寻找可用内存，分配给对象；第二阶段，调用构造函数生成对象。
将new操作设置为私有，那么第一阶段就无法完成，就不能够再堆上生成对象。
(https://blog.csdn.net/huangyang1103/article/details/52119993)

线程间同步的方式有那些，项目中是怎么实现的？
