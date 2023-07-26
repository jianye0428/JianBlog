---
title: Hugo_command
subtitle:
date: 2023-07-09T10:00:34+08:00
draft: false
author:
  name: Jian YE
  link:
  email: 18817571704@163.com
  avatar:
description:
keywords:
license:
comment: true
weight: 0
tags:
  - hugo
categories:
  - Memo
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
tips for hugo installation and new site creation.
{{< /admonition >}}

<!--more-->

# Commands for creating sites backend with hugo

1. create a new markdown file
    ```shell
    hugo new posts/tech/name-of-file.md
    hugo new content/posts/tech/name-of-file.md
    ```

2. create new site
    ```shell
    hugo new site name
    ```

3. Install hugo

- Linux
    ```shell
    wget https://github.com/gohugoio/hugo/releases/download/v0.83.1/hugo_0.83.1_Linux-64bit.tar.gz
    tar -xf hugo_0.83.1_Linux-64bit.tar.gz
    sudo mv hugo /usr/local/bin/
    hugo version
    ```