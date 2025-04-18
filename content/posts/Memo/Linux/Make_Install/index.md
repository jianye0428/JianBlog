---
title: configure、make、make install 背后的原理
subtitle:
date: 2024-01-20T10:04:30+08:00
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
  - make_install
categories:
  - Linux
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
---

## 1、 简介

```shell
./configure
make
make install
```
以上三个命令是源码安装软件的通用步骤。其主要完成以下工作：

   - **`./configure`**: <font color=red>配置</font>，是用来检测你的安装平台的目标特征。比如它会检测你是不是有CC或GCC，并不是需要CC或GCC，它是个shell脚本。configure 脚本负责在使用的系统上准备好软件的构建环境。确保接下来的构建和安装过程所需要的依赖准备好，并且搞清楚使用这些依赖需要的东西。

   - **`make`**: <font color=red>构建</font>，用来编译，它从`Makefile`中读取指令，然后编译。下载的源码包一般没有一个最终的 `Makefile` 文件，一般是一个模版文件 http://Makefile.in 文件，然后 `configure` 根据系统的参数生成一个定制化的 `Makefile` 文件。这个过程会执行在 Makefile 文件中定义的一系列任务将软件源代码编译成可执行文件。

  - **`make install`**:<font color=red>安装</font>，它也从Makefile中读取指令，安装到指定的位置。make install 命令就是将可执行文件、第三方依赖包和文档复制到正确的路径。

{{<admonition quote "tips" false>}}
这些脚本是怎么产生的?
> 安装过程简单说就是 configure 脚本根据系统信息将 Makefile.in 模版文件转换为 Makefile文件，但是 configure 和 Makefile.in 文件是怎么产生的呢？

> 如果你曾经试着打开 configure 或者 Makefile.in 文件，你会发现超长而且复杂的 shell 脚本语言。有时候这些脚本代码比它们要安装的程序源代码还要长。

> 如果想手动创建一个这样的 configure 脚本文件是非常可怕的，好消息是这些脚本是**通过代码生成**的。

> 通过这种方式构建的软件通常是通过一个叫做 autotools 的工具集打包的。这个工具集包含 autoconf 、automake 等工具，所有的这些工具使得维护软件生命周期变得很容易。最终用户不需要了解这些工具，但却可以让软件在不同的 Unix 系统上的安装步骤变得简单。
{{</admonition>}}

## 2、 详细说明

### 2.1 configure命令

这一步一般用来生成 `Makefile`，为下一步的编译做准备，你可以通过在 configure 后加上参数来对安装进行控制，具体参数可以通过`configure --help` 察看，下面举几个例子:

```shell
./configure --prefix=/usr ...
```
  - `--prefix=/usr`: 意思是将该软件安装在 `/usr` 下面，执行文件就会安装在 /usr/bin (而不是默认的 /usr/local/bin), 资源文件就会安装在 /usr/share(而不是默认的/usr/local/share)。选项的另一个好处是卸载软件或移植软件。当某个安装的软件不再需要时，只须简单的删除该安装目录，就可以把软件卸载得干干净净；移植软件只需拷贝整个目录到另外一个机器即可（相同的操作系统）
  - `--bindir=`: 指定二进制文件的安装位置.这里的二进制文件定义为可以被用户直接执行的程序
  - `--enable-static与--enable-shared:`
    - `--enable-static`: 生成静态链接库
    - `--enable-shared`: 生成动态链接库
  - `--with-`: 用于启用或禁用特定功能或模块。例如:
    - `--with-ssl`表示启用SSL支持
    - `--without-gui`表示禁用图形界面。
  - `--with-package=dir`
  - `--with-apxs` 是指定 apache 的配置程序路径，php编译程序会通过这个程序查找apache的相关路径
  - `--with-libxml-dir`: 指向的是 libxml 的库路径
  - `--with-gd`: 指静态编译gd库
  - `--with-png-dir`: 指定 libpng 的路径
  - `--enable-`: 用于启用或禁用特定功能或模块。与--with-选项类似，但更常用于启用或禁用编译选项。
  - `--disable-`: 用于禁用特定功能或模块。与--enable-选项相反，用于禁用编译选项。
  - `–sys-config=`: 指定软件的配置文件。有一些软件还可以加上 `–with`、`–enable`、`–without`、`–disable` 等等参数对编译加以控制

### 2.2 make 命令

这一步就是编译，大多数的源代码包都经过这一步进行编译（当然有些perl或python编写的软件需要调用perl或python来进行编译）。如果 在 make 过程中出现 error ，你就要记下错误代码（注意不仅仅是最后一行），然后你可以向开发者提交 bugreport（一般在 INSTALL 里有提交地址），或者你的系统少了一些依赖库等，这些需要自己仔细研究错误代码。

可能遇到的错误：`make ***` 没有指明目标并且找不到 makefile。 没有Makefile，先`./configure` 一下，再`make`。`make uninstall` 是卸载，不加参数就是默认的进行源代码编译。

`make`工具，它是一个自动化编译工具，你可以使用一条命令实现完全编译。但是你需要编写一个规则文件，`make`依据它来批处理编译，这个文件就是`makefile`。

### 2.3 make install 命令

这条命令来进行安装（当然有些软件需要先运行 `make check` 或 `make test` 来进行一些测试），这一步一般需要你有 root 权限（因为要向系统写入文件）。

`make install` 和`make install prefix=/usr/local/` 等价。

`make install prefix=/usr/local/ sysconfdir=/etc DESTDIR=/tmp/build`支持`DESTDIR`的意义就是，保证所有要安装的文件，都会被安装在DESTDIR目录下，不会污染系统的package的目录。install也 是linux系统命令。

### 2.4 扩展说明

Linux的用户可能知道，在Linux下安装一个应用程序时，一般先运行脚本configure，然后用make来编译源程序，在运行make install，最后运行make clean删除一些临时文件。

configure是一个shell脚本，它可以自动设定源程序以符合各种不同平台上Unix系统的特性，并且根据系统叁数及环境产生合适的Makefile文件或是C的头文件(header file)，让源程序可以很方便地在这些不同的平台上被编译连接。

利用configure所产生的Makefile文件有几个预设的目标可供使用，其中几个重要的简述如下：
  - `make all`: 产生我们设定的目标，即此范例中的可执行文件。只打make也可以，此时会开始编译原始码，然后连结，并且产生可执行文件。只打make 默认就是`make all`，只编译其中某个目标则在后面给目标名称：make ce-common。
  - `make clean`: 清除编译产生的可执行文件及目标文件(object file，*.o)。
  - `make distclean`: 除了清除可执行文件和目标文件外，把configure所产生的Makefile也清除掉。
  - `make install`: 将程序安装至系统中。如果原始码编译无误，且执行结果正确，便可以把程序安装至系统预设的可执行文件存放路径。
  - `make dist`: 将程序和相关的档案包装成一个压缩文件以供发布。执行完在目录下会产生一个以PACKAGE-VERSION.tar.gz为名称的文件。 PACKAGE和VERSION这两个变数是根据http://configure.in文件中AM_INIT_AUTOMAKE(PACKAGE，VERSION)的定义。在此范例中会产生test-1.0.tar.gz的档案。
  - `make distcheck`: 和make dist类似，但是加入检查包装后的压缩文件是否正常。这个目标除了把程序和相关文件包装成tar.gz文件外，还会自动把这个压缩文件解开，执行 configure，并且进行make all 的动作，确认编译无误后，会显示这个tar.gz文件可供发布了。这个检查非常有用，检查过关的包，基本上可以给任何一个具备GNU开发环境-的人去重新编译。

## 3. 总结

通过源码编译安装一个软件如下:

```shell
./configure --prefix=/usr/local/${program_name}
make
make install
make clean
```
<mark>注意: </mark> `--prefix`可以在configure或者make install时指定安装路径。

## 4. 参考

- [Linux命令详解：./configure、make、make install 命令](https://zhuanlan.zhihu.com/p/77813702)
- [configure、 make、 make install 背后的原理(翻译)](https://zhuanlan.zhihu.com/p/77813702)