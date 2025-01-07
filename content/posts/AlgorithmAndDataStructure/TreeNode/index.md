---
title: TreeNode 二叉树
subtitle:
date: 2023-07-16T14:49:58+08:00
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
math: false
lightgallery: false
---

Runebook
www.doc4dev.com

## 深度优先搜索

<br>
<center>
  <img src="images/TreeNode.png" width="480" height="180" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">二叉树示意</div>
</center>
<br>

- 前序遍历：中左右 `5 4 1 2 6 7 8`
- 中序遍历：左中右 `1 4 2 5 7 6 8`
- 后序遍历：左右中 `1 2 4 7 8 6 5`

**二叉树的定义**

```c++
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
```

### 前序遍历
**递归法**
```c++
class Solution {
public:
    void traversal(TreeNode* cur, vector<int>& vec) {
        if (cur == nullptr) return;
        vec.push_back(cur->val);
        traversal(root->left, vec);
        traversal(root->right, vec);
    }

    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        traversal(root, vec);
        return res;
    }
};
```


**迭代法**

```c++
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        if (root == nullptr) return res;
        st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            st.pop();
            res.push_back(node->val);
            if (node->right)  st.push(node->right);
            if (node->left) st.push(node->left);
        }
        return res;
    }
};
```


### 中序遍历

**递归法**

```c++
class Solution {
public:
    void traversal(TreeNode* cur, vector<int>& vec) {
        if (cur == nullptr) return;
        traversal(root->left, vec);
        vec.push_back(cur->val);
        traversal(root->right, vec);
    }

    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        traversal(root, vec);
        return res;
    }
};

```

**迭代法**
```c++
// 中序遍历
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;
        vector<int> res;
        if (root == nullptr) return;
        TreeNode* cur = root;
        while (!cur || !st.empty()) {
            if (cur != nullptr) {
                st.push(cur);
                cur = cur->left; // 左
            } else {
                cur = st.top();
                st.pop();
                res.push_back(cur->val);
                cur = cur->right;
            }
        }
        return res;
    }
};

```

### 后序遍历

**递归法**

```c++
class Solution {
public:
    void traversal(TreeNode* cur, vector<int>& vec) {
        if (cur == nullptr) return;
        traversal(root->left, vec);
        traversal(root->right, vec);
        vec.push_back(cur->val);
    }

    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        traversal(root, vec);
        return res;
    }
};

```

**迭代法**

```c++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        if (!root) return res;
        stack<TreeNode*> st;
        st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            st.pop();
            res.push(node->val);
            if (node->left) st.push(node->left);
            if (node->right) st.push(node->right);
        }
        reverse(res.begin(), res.end());
        return res;
    }
};

```

## 二叉树的统一迭代法

**迭代法前序遍历**

```c++
// 中左右
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        if (root) st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            if (node) {
                st.pop();
                if (node->right) st.push(node->right);
                if (node->left) st.push(node->left);
                st.push(node);
                st.push(nullptr);
            } else {
                st.pop();
                node = st.top();
                st.pop();
                res.push_back(node->val);
            }
        }
        return res;
    }
};

```

**迭代法中序遍历**

```c++
//左中右
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        if (root != nullptr) st.push(root);
        while(!st.empty()) {
            TreeNode* node = st.top();
            if (node != nullptr) {
                st.pop(); //将该节点弹出，避免重复操作
                if (node->right) st.push(node->right);
                st.push(node);
                st.push(nullptr);
                if (node->left) st.push(node->left);
            } else {
                st.pop();
                node = st.top();
                st.pop();
                res.push_back(node->val);
            }
        }
        return res;
    }
};

```

**迭代法后序遍历**

```c++
// 左右中
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        if (root) st.push(root);
        while (!st.empty()) {
            TreeNode* node = st.top();
            if (node) {
                st.pop();
                st.push(node);
                st.push(nullptr);
                if (node->right) st.push(node->right);
                if (node->left) st.push(node->left);
            } else {
                st.pop();
                node = st.top();
                st.pop();
                res.push_back(node->val);
            }
        }
        return res;
    }
};

```


## 广度优先遍历 (层序遍历)

**递归法**
```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> que;
        if (root) que.push(root);
        vector<vector<int>> ans;
        while (!que.empty()) {
            int size = que.size();
            vector<int> vec;
            for (int i = 0; i < size; ++i) {
                TreeNode* node = que.front();
                que.pop();
                vec.push_back(node->val);
                if (node->left) que.push(node->left);
                if (node->right) que.push(node->right);
            }
            ans.push_back(vec);
        }
        return ans;
    }
};
```

**迭代法**

```c++
class Solution {
public:
    void order(TreeNode* cur, vector<vector<int>>& res, int depth) {
        if (cur == nullptr) return;
        if (res.size() == depth) res.push_back(vector<int>());
        res[depth].push_back(cur->val);
        order(cur->left, res, depth + 1);
        order(cur->right, res, depth + 1);
    }
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        int depth = 0;
        order(root, res, depth);
        return res;
    }
};
```
