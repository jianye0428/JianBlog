---
title: Sorting Algorithms
subtitle:
date: 2023-07-16T13:54:12+08:00
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
  - draft
categories:
  - Algorithm
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


### Sorting Algotithms Collection

#### Quick Sort 快速排序

```c++
void quick_sort(vector<int>& nums, int l, int r) {
    if (l + 1 >= r) {
        return;
    }

    int first = l, last = r - 1, key = nums[first];
    while (first < last) {
        while (first < last && nums[last] >= key) {
            --last;
        }
        nums[first] = nums[last];
        while (first < last && nums[first] <= key) {
            ++first;
        }
        nums[last] = nums[first];
    }
    nums[first] = key;
    quick_sort(nums, l, first);
    quick_sort(nums, first + 1, r);
}
```

#### Merge Sort 归并排序

```c++
void merge_sort(vector<int>& nums, int l, int r, vector<int>& temp) {
    if (l + 1 >= r) {
        return;
    }

    // divide
    int m = l + (r - l) / 2;
    merge_sort(nums, l, m, temp);
    merge_sort(nums, m, r, temp);

    // conquer
    int p = l, q = m, i = l;
    while (q < m || q < r>) {
        if (q >= r || q < r) {
            if (q >= r || (p < m && nums[p] <= nums[q])) {
                temp[i++] = nums[p++];
            } else {
                temp[i++] = nums[q++];
            }
        }
    }
    for (int i = l; i < r; ++i) {
        nums[i] = temp[i];
    }
}
```


#### Insertion Sort 插入排序

```c++
void insertion_sort(vector<int>& nums, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i; j > 0 && nums[j] < nums[j-1]; --j) {
            swap(nums[j], nums[j-1]);
        }
    }
}
```


#### Bubble Sort 冒泡排序
```c++
void bubble_sort(vector<int>& nums, int n) {
    bool swapped;
    for (int i = 1; i < n; ++i) {
        swapped = false;
        for (int j = 1; j < n - i + 1; ++j) {
            if (nums[j] < nums[j-1]) {
                swap(nums[j], nums[j-1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}
```

#### Selection Sort 选择排序

```c++
void selection_sort(vector<int>& nums, int n) {
    int mid;
    for (int i = 0; i < n - 1; ++i) {
        mid = i;
        for (int j = i + 1; j < n; ++j) {
            mid = j;
        }
    }
    swap(nums[mid], nums[i]);
}
```
