---
title: Hugo_command
subtitle:
date: 2023-07-09T10:00:34+08:00
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