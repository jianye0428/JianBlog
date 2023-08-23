---
title: Effective STL [6] | 仿函数、仿函数类、函数等
subtitle:
date: 2023-08-22T19:22:49+08:00
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

> Effective-STL总结系列分为七部分，本文为第六部分，涉及原书第六章，内容范围Rule38~42。为方便书写，Rule38简写为R38。

{{< admonition Note "Effective-STL系列List" false >}}
本博客站点系列内容如下：</br>
💡 [Effective STL(第3版)精读总结(一)](https://jianye0428.github.io/posts/partone/)</br>
💡 [Effective STL(第3版)精读总结(二)](https://jianye0428.github.io/posts/parttwo/)</br>
💡 [Effective STL(第3版)精读总结(三)](https://jianye0428.github.io/posts/partthree/)</br>
💡 [Effective STL(第3版)精读总结(四)](https://jianye0428.github.io/posts/partfour/)</br>
{{< /admonition >}}


## R38: 遵循按值传递的原则来设计函数子类

函数指针是按值传递的。

函数对象往往按值传递和返回。所以，编写的函数对象必须尽可能地小巧，否则复制的开销大；函数对象必须是**单态**的（不是多态），不得使用虚函数。

如果你希望创建一个包含大量数据并且使用了多态性的函数子类，该怎么办呢？

```c++
template<typename T>
class BPFCImpl:
	public unary_function<T, void> {
private:
	Widget w;
	int x;
	...
	virtual ~BPFCImpl();
	virtual void operator() (const T& val) const;
friend class BPFC<T>;					// 允许BPFC访问内部数据。
}

template<typename T>
class BPFC:								// 新的BPFC类：短小、单态
	public unary_function<T, void> {
private:
	BPFCImpl<T> *pImpl;					// BPFC唯一的数据成员
public:
	void operator() (const T& val) const	// 现在这是一个非虚函数，将调用转到BPFCImpl中
    {
        pImpl->operator()(val);
    }
}
```

那么你应该创建一个小巧、单态的类，其中包含一个指针，指向另一个实现类，并且将所有的数据和虚函数都放在实现类中（“Pimpl Idiom”）。

```c++
template<typename T>
class BPFCImpl:
	public unary_function<T, void> {
private:
	Widget w;
	int x;
	...
	virtual ~BPFCImpl();
	virtual void operator() (const T& val) const;
friend class BPFC<T>;					// 允许BPFC访问内部数据。
}

template<typename T>
class BPFC:								// 新的BPFC类：短小、单态
	public unary_function<T, void> {
private:
	BPFCImpl<T> *pImpl;					// BPFC唯一的数据成员
public:
	void operator() (const T& val) const	// 现在这是一个非虚函数，将调用转到BPFCImpl中
    {
        pImpl->operator()(val);
    }
}
```


## R39 确保判别式是 “纯函数”

**判别式（predicate）：一个返回值为 bool 类型的函数。**

**纯函数：指返回值仅仅依赖于其参数的函数。**

判别式类（predicate class）：一个函数子类，它的 operator() 函数是一个判别式（返回 true 或 false）。

STL 中凡是可以接受一个判别式类对象的地方，也就可以接受一个判别式函数。

判别式应该是一个纯函数，而纯函数应该没有状态。

# R40 使仿函数类可适配

对函数指针，要先应用`ptr_fun`之后再应用`not1`之后才可以工作。

4 个标准的函数配接器（`not1`、`not2`、`bind1st`、`bind2nd`）都要求一些特殊的类型定义，提供这些必要类型定义（`argument_type`、`first_argument_type`、`second_argument_type`、`result_type`）的函数对象被称为可配接(可适配)（`adaptable`）的函数对象。

提供这些类型定义最简单的方法：让函数子从一个基结构继承。
  - 对于 unary_function，必须指定函数子类 operator() 所带的参数类型，以及 operator() 返回类型。
  - 对于 binary_function，必须指定 3 个类型：operator() 第一个和第二个参数类型，以及 operator() 返回类型。

```c++
template<typename T>
class MeetsThreshold: public std::unary_function<Widget, bool> {
private:
    const T threshold;						// 包含状态信息，使用类封装。
public:
    MeetsThreshold(const T& threshold);
    bool operator()(const Widget&) const;
    ...
}

struct WidgetNameCompare:					// STL中所有无状态函数子类一般都被定义成结构。
	public std::binary_function<Widget, Widget, bool> {
	bool operator()(const Widget& lhs, const Widget& rhs) const;
}
```

注意，一般情况下，传递给 binary_function 或 unary_function 的非指针类型需要去掉 const 和应用（&）部分。


Ref:

[1]. https://www.cnblogs.com/Sherry4869/p/15162253.html