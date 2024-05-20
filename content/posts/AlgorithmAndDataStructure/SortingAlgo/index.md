---
title: 排序算法
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
  - 算法
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


## Sorting Algotithms Collection 排序算法合集

### 0. 排序算法

排序算法在所有计算机算法，乃至整个计算机领域中，都占据着非常重要的地位。基础算法是软件的核心，而查找算法和排序算法则是计算机基础算法的核心。

排序算法是计算机科学中用于对元素序列进行排序的一系列算法。排序算法在实际应用中非常广泛，比如数据库索引、文件排序、数据检索等。

#### 0.1 定义：

排序算法是一种将一组数据元素重新排列成有序序列的算法。这个“有序”可以是升序或降序。

#### 0.2 作用：

排序算法在许多领域都有广泛的应用，包括但不限于：

•数据分析：对数据进行排序可以更容易地识别数据中的模式和趋势。

•数据库管理：数据库查询经常需要对结果进行排序。

•搜索算法：排序算法可以用于优化搜索过程，如二分搜索依赖于排序好的列表。

•算法实现：许多算法的实现依赖于排序，如归并排序是归并算法的基础。

#### 0.3 分类：

- 按算法的时间复杂度进行分类：
  - O(n^2) 算法：冒泡排序、选择排序、插入排序
  - O(n log n) 算法：快速排序、归并排序、堆排序
  - O(n) 算法：计数排序、桶排序、基数排序

- 按照空间复杂度（内存使用量）进行分类：
  - 原地排序算法（空间复杂度为 O(1)）：冒泡排序、选择排序、插入排序、堆排序
  - 非原地排序算法（空间复杂度大于 O(1)）：归并排序、计数排序

- 按照实现排序的方法进行分类：
  - 插入类排序：直接插入排序、二分插入排序、Shell排序（希尔排序）
  - 交换类排序：冒泡排序、快速排序、随机快速排序
  - 选择类排序：简单选择排序、堆排序
  - 归并类排序：归并排序
  - 分配类排序：计数排序、桶排序、基数排序
  - 混合类排序：鸡尾酒排序（冒泡排序的变体，双向冒泡）
  - 其他排序：拓扑排序、循环排序

### 1. Quick Sort 快速排序

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

### 2. Merge Sort 归并排序

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

### 3. Insertion Sort 插入排序

```c++
void insertion_sort(vector<int>& nums, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = i; j > 0 && nums[j] < nums[j-1]; --j) {
            swap(nums[j], nums[j-1]);
        }
    }
}
```


### 4. Bubble Sort 冒泡排序

冒泡排序（Bubble Sort）是一种简单的排序算法，它重复地遍历要排序的数列，一次比较两个元素，如果它们的顺序错误就把它们交换过来。遍历数列的工作是重复进行直到没有再需要交换，也就是说该数列已经排序完成。

#### 4.1 算法步骤

•开始：从数列的第一个元素开始，比较相邻的两个元素。

•比较与交换：如果左边的元素大于右边的元素，就交换它们两个。

•移动：移动到下一个元素对，重复步骤2。

•重复：继续这个过程，直到最后一次交换发生，此时数列的最后一个元素是最大的，已经被“冒泡”到它应该在的位置。

•减少比较次数：由于最大的元素已经在它应在的位置，所以下一次遍历可以减少一个比较（即从第一个元素开始，不需要再和它比较）。

#### 4.2 算法图解
冒泡排序从头开始，依次比较数组中相邻的2个元素，如果后面的数比前面的数大，则交换2个数，否则不交换。每进行一轮比较，都会把数组中最大的元素放到最后面。

<br>
<center>
    <img src="images/4_1.webp" width="320" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

#### 4.3 算法特点
- 简单：冒泡排序的原理简单，容易实现。
- 稳定：冒泡排序是一种稳定的排序算法，因为它不会改变相同元素之间的顺序。
- 时间复杂度：平均和最坏时间复杂度均为O(n^2)，其中n是数列的长度。在最好的情况下（即数列已经是排序状态），时间复杂度为O(n)。
- 空间复杂度：O(1)，因为冒泡排序是原地排序，不需要额外的存储空间。


#### 4.4 代码实现

```python
def bubble_sort(arr):
  n = len(arr)
  swapped = False
  for i in range(n):
      # 由于每次最大的元素都会被放到它应在的位置，所以可以减少比较次数
    swapped = False
    for j in range(0, n-i-1):
      # 相邻元素两两比较
      if arr[j] > arr[j+1]:
        # 发现元素顺序错误，交换它们
        arr[j], arr[j+1] = arr[j+1], arr[j]
        swapped = True
    if not flag:
        break
  return arr

# 示例
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr)
print("Sorted array is:", sorted_arr)
```

```c++
void print_arr(vector<int>& nums) {
  for (auto num:nums) {
    std::cout << num << " ";
  }
  std::cout << std::endl;
}

void bubble_sort(vector<int>& nums) {
  int n = nums.size();
  bool swapped = false;
  for (int i = 0; i < n; ++i) {
    swapped = false;
    for (int j = 0; j < n - i - 1; ++j) {
      if (nums[j] > nums[j+1]) {
        swap(nums[j], nums[j+1]);
        swapped = true;
      }
    }
    if (!swapped) { // 一旦没有交换操作，说明已经完成排序，可以跳出循环
      break;
    }
  }
}

int main() {
  vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
  print_arr(arr);
  bubble_sort(arr);
  print_arr(arr); // 11 12 22 25 34 64 90
  return 0;
}
```

#### 4.5 使用场景

冒泡排序由于其性能原因，通常不适用于大型数据集的排序。然而，它在以下情况下可能很有用：
- 数据规模较小：当数据集很小或者几乎已经排序时，冒泡排序的性能是可接受的。
- 教学目的：由于其简单性，冒泡排序常被用作教学示例，帮助初学者理解算法和排序的基本概念。
- 需要稳定性：在某些特定情况下，保持元素的相对顺序很重要，冒泡排序可以满足这种稳定性需求。

### 5. 选择排序

选择排序是一种简单直观的排序算法，它的工作原理是每一次从待排序的数据元素中选出最小（或最大）的一个元素，存放在序列的起始位置，直到全部待排序的数据元素排完。

#### 5.1 算法步骤

•开始：在未排序序列中找到最小（大）元素；

•交换：将找到的最小（大）元素与序列的第0个元素交换；

•移动：从序列的第1个元素开始，继续寻找最小（大）元素，然后与序列的第1个元素交换；

•重复：重复步骤2和3，直到序列的第n-1个元素（其中n是序列的长度）。

#### 5.2 算法图解

选择排序通过重复扫描数组，找到最小的元素，然后将其与当前位置的元素交换。这个过程会一直进行，直到整个数组被排序。

<br>
<center>
  <img src="images/5_1.webp" width="300" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">BP Network</div>
</center>
<br>

#### 5.3 算法特点

•简单：选择排序的实现相对简单，容易理解和编程实现。

•不稳定：选择排序在交换过程中可能会改变相同元素的顺序，因此它不是稳定的排序算法。

•时间复杂度：无论最好、最差还是平均情况下，时间复杂度都是O(n^2)，其中n是数列的长度。

•空间复杂度：O(1)，选择排序是原地排序，不需要额外的存储空间。

#### 5.4 代码实现

```python
def selection_sort(arr):
  n = len(arr)
  for i in range(n-1):
    # 找到最小元素的索引
    min_idx = i
    for j in range(i+1, n):
      if arr[j] < arr[min_idx]:
          min_idx = j
    # 将找到的最小元素交换到序列的前面
    arr[i], arr[min_idx] = arr[min_idx], arr[i]
  return arr

# 示例
arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = selection_sort(arr)
print("Sorted array is:", sorted_arr)

```
```c++
void print_arr(vector<int>& nums) {
  for (auto num:nums) {
    std::cout << num << " ";
  }
  std::cout << std::endl;
}

void selection_sort(vector<int>& nums) {
  int mid_idx;
  int n = nums.size();
  for (int i = 0; i < n - 1; ++i) {
    mid_idx = i;
    for (int j = i + 1; j < n; ++j) {
      if (nums[j] < nums[mid_idx]) {
        mid_idx = j;
      }
    }
    swap(nums[mid_idx], nums[i]);
  }
}

int main() {
  vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
  print_arr(arr);
  selection_sort(arr);
  print_arr(arr);
  return 0;
}
```

#### 5.5 适用场景

选择排序的性能相对较差，因此它不适用于大型数据集的排序。然而，在以下情况下，选择排序可能比较适用：

•数据规模较小：当数据集较小时，选择排序的简单性可能使其成为一个合适的选择。

•教学目的：由于其实现简单，选择排序常被用作教学示例，帮助初学者理解算法和排序的基本概念。

•排序过程中的特定操作：在某些特定的应用场景中，如果排序过程中需要频繁地访问未排序部分的元素，选择排序可能比其他算法更合适。。


### 6. 计数排序

### 7. 桶排序


## Reference:
[1]. https://mp.weixin.qq.com/s/P8MmmMc4vB_I9tnK3towLQ