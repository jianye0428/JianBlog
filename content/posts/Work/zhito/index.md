---
title: Zhito 工作总结
subtitle:
date: 2023-07-15T14:14:26+08:00
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
## LaneGCN模型移植和部署

### LaneGCN模型

#### 一、数据处理

(1) `read_argo_data()`:
  - data['city']
  - data['traj']: shape: 32 x 50 x 2
  - data['steps']: shape: 32 x 50
(2) `get_obj_feat()`:
  - data['feat']: 障碍物前20秒轨迹点的相对位置坐标（vector）(只有19个，当前时刻补零)
  - data['ctrs']: 各个障碍物20帧的位置信息
  - data['orig']: agent在20帧时刻的位置，作为局部坐标的原点
  - data['theta']: agent在20帧的偏转角
  - data['rot']: 旋转矩阵以及坐标转换
  - data['gt_preds']: M x 30 x 2 各个障碍物在后3秒(30帧)的真实轨迹运动信息
  - data['has_preds']: 标记在当前时刻真值是否被观测到
  > 坐标转换
  > $$\begin{bmatrix} x_g & y_g\end{bmatrix}\begin{bmatrix} cos\theta & -sin\theta \\ sin\theta & cos\theta \end{bmatrix} =\begin{bmatrix} x_l & y_l\end{bmatrix}$$

(3) `get_lane_graph()`:
  - 以agent第20帧坐标为圆心，取boundingbox范围内的所有道路
  - 取lane_ids
    - 针对lane_id获得lane，然后通过lane.centerline获取道路中心点的位置，lane.centerline代表lane的一系列中心点
    - 对中心点针对agent的坐标原点进行坐标转换
    - 取ctrs、feat、turn、control、intersect = [], [], [], [], []
      - lane.centerline有10个点的坐标，相当于取9个lane node
      - ctrs: 每条lane centerline的10个点的前后相加除以2，作为lane node的中心点
      - feat：lane.centerline的10个点的前后位置相减，相当于9个lane segment，feat指vector feat
      - turn: [a, b] a=1: 左转 b=1: 右转
      - control: 是否有交通标志
      - intersect: 是否处于路口

  - 获得lane node之间的拓扑关系
    - node_idcs: lane node index (0 ~ 9) (9~18) ...
    - num_nodes: lane node 的总数量
    - lane node 之间的拓扑关系
      - pre['u']: 1 ~ 9 ... v 是 u 的 pre
      - pre['v']: 0 ~ 8 ...
      - suc['u']: 0 ~ 8 ... v 是 u 的 suc
      - suc['v']: 1 ~ 9 ...
      > 注意： pre['u'] pre['v'] suc['u'] suc['v'] 指的是lane node之间的关系
    - pre_pairs、 suc_pairs、 left_pairs、 right_pairs: 指的是lane 与 lane 之间的关系
  - 总结:
    - graph['ctrs']: lane node 中心点的坐标
    - graph['num_nodes']: lane node 点的数量
    - graph['feat']: lane node 的前后相减 矢量特征
    - graph['turn']: 道路是否为左转或者右转
    - graph['control']: 道路是否有交通标志
    - graph['intersect']: 是否处于交通路口
    - graph['pre']: lane node 前后拓扑关系
    - graph['suc']: lane node 前后拓扑关系
    - graph['lane_idcs']: lane node index
    - graph['pre_pairs']: lane 与 lane 之间的前后关系
    - graph['suc_pairs']:
    - graph['left_pairs']:lane 与 lane 之间的左右关系
    - graph['right_pairs']:

#### 二、数据前处理

(1) `preprocess()`:
  - 主要针对lane graph的数据进行数据前处理工作
    - 第一步: 根据lane_idc获得lane node数量和lane的数量
    - 第二步: 计算lane node 两两之间的距离
    - 第三步: 根据pre_pair、 suc_pair构建pre、suc矩阵
             根据left_pair、 right_pair构建left、right矩阵
    - 第四步: 取出角度(偏转角)$\theta < \pi/4$ 的lane node 节点
      构建left['u'] v 是 u 的左边lane node 节点
         left['v']
         right['u'] v 是 u 的右边lane node 节点
         right['v']

#### 三、LaneGCN 具体的网络结构

(1)、 data输入结构(以batch_size = 2 为例)
    对于data['feat']类型为list，len(list) = 2, list[0]=>其中一个scenario
    对于data['graph']类型为list，len(list) = 2, list[0]=>其中一个dict，dict中存储lane node信息
(2)、 对于`actor_gather()`和`graph_gather()`两个函数
  - actor_gather():
    - 输入: list (data['feat']) 输出: actors (M x 3 x 20), actor_idcs
    - 作用： 在此处，把batch输入的障碍物特征进行concatenation整合到一起，并完成转置，将时序放到第一维，为后续的FPN网络做准备
  - graph_gather()
    - 输入：list (data['graph']) 输出: graph
    - 作用: 把batch中输入的lane graph特征进行叠加(concatenation)，用于后续训练
(3)、ActorNet(): 提取障碍物actor的特征
  - 输入: actors (M x 3 x 20)
  - 输出: actor net output (M x 128)

  ActorNet 网络结构：
  - groups:
      ```c++
      group: Res1d(3, 32)     Res1d(32, 32)
      group: Res1d(32, 64)    Res1d(64, 64)
      group: Res1d(64, 128)   Res1d(128, 128)
      ```
  - outputs [ groups[0], groups[1], groups[2]]
  - lateral [conv1d[32, 128], conv1d[64, 128], conv1d[128, 128]]
  - 整体结构:
    ```c++
                (31 x 128 x 5)             (31 x 128 x 5)
    groups[2] Res1d(128, 128)  => conv1d(128, 128)     ====>  interpolate (31 x 128 x 10)
              Res1d(64, 128)                                      ||
             /\                                                   ||
             || (31 x 64 x 10)             (31 x 128 x 10)        \/
    groups[1] Res1d(64, 64)  => conv1d(64, 128)     ====>    sum (31 x 128 x 10)
              Res1d(32, 64)                                      ||
             /\                                               interpolate (31 x 128 x 20)
             || (31 x 32 x 20)               (31 x 128 x 20)      ||
    groups[0] Res1d(32, 32)  => conv1d(32, 128)     ====>    sum (31 x 128 x 20)
              Res1d(3, 32)                                      ||
             /\                                                  res1d(128, 128)
             ||                                                   ||  [:,:, -1]
        input: 31 x 3 x 20                                    output: 31 x 128
    ```
(4)、MapNet(): 提取lane node的特征
  - 输入: graph['idcs', 'ctrs', 'feats', 'turn', 'control', 'intersect', 'pre', 'suc', 'left', 'right']
  - 输出: feat, graph['idcs'], graph['ctrs']
    - graph['idcs']: lane node 的 index
      - len(graph['ctrs'][0]): 1206
      - len(graph['ctrs'][1]): 954
    - graph['ctrs']: 2160 x 2
  - 网络结构:
    - self.input: `Linear(2, 128)` `Linear(128, 128)` ==> 输入graph['ctrs'](N x 2) 输出: N x 128
    - self.seg: `Linear(2, 128)` `Linear(128, 128)` ==> 输入: graph['feat'](N x 2) 输出: N x 128

    - self.fuse() => dict()
      - self.fuse['ctr']: Linear(128, 128)
      - self.fuse['norm']: nn.GroupNorm(gcd(1, 128),  128)
      - self.fuse['ctr2']: Linear(128, 128), norm = groupNorm, ng = 1
      - self.fuse['left']: Linear(128, 128)
      - self.fuse['right']: Linear(128, 128)
      - self.fuse['pre0'] ~ self.fuse['pre5']: Linear(128, 128)
      - self.fuse['suc0'] ~ self.fuse['suc5']: Linear(128, 128)

  - 流程:
    ```c++
    lane node ctrs: n x 2 = self.input => n x 128
    lane node feats: n x 2 = self.seg => n x 128
    || relu
    n x 128
    || => resblock  =============================>
    pre0 ~ pre5
    suc0 ~ suc5  temp.index_add_(0, graph[pre][i]['u'], self.feat(graph[k1][k2]['v']))
    解释: 把feat的第v行(value)加到temp的第u行上
    对pre0 ~ pre5 / suc0 ~ suc5 / left / right执行相同操作 (图注意力)
    ||
    然后经过self.fuse['norm'] 和 relu模块加上 resblock
    ||
    得到输出: feat: n x 128
            graph["idcs"]
            graph["ctrs"]
    ```
(5)、**A2M()**: lane node 和 agent node 交互
  - 在A2M模块中，agent node 是 lane node, context node 是 vehicle node (以lane node为中心， actor node为context)
  - 输入: feat(nodes), graph, actors, actors_idcs, actor_ctrs
  - 输出: feat (n x n_agts)
  - 网络结构:
    - meta = torch.cat(graph['turn'], graph['control'], graph['intersect'])
    - feat = self.meta(Linear(128, 128))
    - 针对feat(lane node feature), graph['idcs'], graph['ctrs'], actors, actor_idcsm, actor_ctrs
      循环指行两次graph attention 操作

  - Atten网络结构；
    - 输入: agts, agts_idcs, agts_ctrs, ctx, ctx_idcs, ctx_ctrs， dist_th
    - 流程:
      ```c++
      - agts ======================================> resblock
      ||
      agts_ctrs 和 ctx_ctrs 两两求distance
      mask = dist < dist_th 求距离小于threshold的mask(筛选出距离小于threshold的车和道路)
      ||
      idcs = torch.nonzero(mask)
      hi.append(idcs[:, 0]) => row_idcs for agts
      wi.append(idcs[:, 1]) => col_idcs for ctxs
      ||
      dist = agt_cts[hi] - ctx_ctrs[wi] 根据threshold筛选出来的agent node 和context node 求distance
      ||  self.dist((2, n_ctx) (n_ctx, n_ctx))
      dist = self.dist(dist): n x n_ctx
      || self.query(n_agts, n_ctx)
      query = self.query(agts[hi]): n x n_ctxs
      || ctx = ctx[wi]: n x n_ctx
      ctx = torch.cat((dist, query, ctx), 1)
      ||self.ctx(3 * n_ctx, n_agt) (n_agt, n_agt)
      cts = self.ctx(ctx): n x n_agt
      ||agts = self.agts(n_agt, n_agt)
      agts.index_add_(0, hi, ctx) 把context的特征根据hi (index) 加到agents上
      || 加上resblock
      输出： n x n_agt
    ```
(6)、M2M(): map node 和 map node 交互
  - 输入： node graph
  - 输出: N x 128
  - 此处执行的操作和MapNet()相同

(7)、M2A(): map node 和 vehicle node交互
  - 输入: actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs
  - 输出: actors (n x 128)
  - 以vehice node为agent，lane node为context nodes，执行Attention注意力机制
(8)、A2A(): vehicle node 和 vehicle node 交互
  - 输入: actor, actor_idcs, actor_ctrs
  - 输出: n_actos x 128
  - 以vehicle node为agent nodes，同时以vehicle node为context nodes为context nodes，执行注意力机制
(9)、PredNet() 预测网络
  - 输入：actor_feats, actor_idcs, actors_ctrs
  - 输出: out['cls'], out['reg']
  - 网络结构:
    - pred[0] [LinearRes(128, 128), Linear(128, 2 x 30)]
    - pred[1] [LinearRes(128, 128), Linear(128, 2 x 30)]
    - pred[2] [LinearRes(128, 128), Linear(128, 2 x 30)]
    - pred[3] [LinearRes(128, 128), Linear(128, 2 x 30)]
    - pred[4] [LinearRes(128, 128), Linear(128, 2 x 30)]
    - pred[5] [LinearRes(128, 128), Linear(128, 2 x 30)]
  - 流程:
    ```c++
     根据actor_feats分别输入到PredNet中
      `Preds[n* 120, n* 120, n* 120, n* 120, n* 120, n* 120]`
       ||
       [n * 1 * 120, n * 1 * 120, n * 1 * 120, n * 1 * 120, n * 1 * 120]
       || torch.cat()
       n x 6 x 120
       ||reg.resize()
       n x 6 x 30 x 2
       || `dest_ctrs = reg[:, :, -1].detach()` (n x 6 x 2)
       || `self.dest(actors, actors_ctrs, dest_ctrs)`： 相当于把每个障碍物不同模态的轨迹终点与原点（20帧时刻轨迹点）之间的距离叠加到输出特征上中(叠加到actors上)，用到后面去求每个模态的概率(评分)
       || `self.cls(Linear(128, 128) Linear(128, 1))`: 作为cls的模态的打分
       || `cls, sort_idcs = cls.sort(1, descend=true)` 按照1维降序排列(最后让预测结果中的每一个障碍物的6个模态按降序排列，即第一条的轨迹的打分最高)
    -   输出： out['cls']、 out['reg']
    ```
(10) PredLoss的计算
  - 取出真值轨迹中有轨迹点的最后一个index
  - 根据真值轨迹是否有观测轨迹点的index取出需要进行比较的reg和cls (根据真实观测值取出相应的预测轨迹)
  - dist指的是每一个真值最后一个可观测点位置与6个模态轨迹相应预测轨迹点的l2距离
  - 把6个模态的l2距离叠加到一起，取出其中距离最小的轨迹点idx
  - mgn = cls[row_idcs, min_idcs].unsqueeze() - cls 将cls最小的分值 减去 cls分值
  - mask0：筛选出每条距离fde最小的轨迹，fde < 2 的轨迹
    mask1：筛选出距离减去fde最小轨迹距离 < 0.2 的轨迹
  - 求出mgn[mask0 * mask1]: 取min_dist < 2 但是排除距离它比较近的轨迹点
  - mask = mgn < 0.2
  - 最后计算cls和reg的loss
    - cls_loss += self.config["mgn"] * mask.sum() - mgn[mask].sum
    $\text{cls}_\text{loss} = max(0, {c_k} + \epsilon - \hat{c_k})$
    - reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) # 预测值和真值的 huber loss

### 在移植以及部署过程中，碰到的问题以及解决方法

(1). 由于模型应用场景的要求，对模型进行以下修改:

  - 道路节点的upsample()和downsample()的处理: 由于高速场景下，直线路段比较多，且高精地图提供的道路节点比较稀疏，我们对lane node进行upsample处理，每间隔20m取一个lane node，增加道路的拓扑信息；另一方面，在路口场景，由于道路节点比较密集，造成模型计算消耗过大，我们使用downsample的方式，每间隔5个lane node 选取一个lane node，来减少算力消耗
  - 在原来的模型中，只关心focal agent的预测轨迹；在项目中，我们将ego vehicle的位置作为局部坐标原点，通过一次推理来获得周围车辆未来三秒的轨迹；
  - 在高速场景下，考虑到ego vehicle在行车过程中更倾向于考虑自车前方的道路拓扑结构和障碍物，那么我们在选取道路结构时，倾向于以自车前方40m， 半径为200m的区域来来构造lane graph。
  - 同时，在筛选道路过程中，碰到单个lane过长的情况，采用直接提取lane node的方式，将超出范围内的lane node直接截断，来减少算力消耗
  - 考虑在高速场景下，并不需要6个模态，倾向于想模型修改为2-3模态进行训练。
  - 在高速场景下，由于lanegcn的模型算力消耗比较大，只推理自车前方、左侧和右侧共5辆关心的车辆；场景中其他车辆用轻量化的模型进行推理或者直接用基于规则的方式进行轨迹预测。

(2). 基于TensorRT模型推理方面遇到的问题以及解决方案

  - `scatterElement`自定义算子的编写
    - 原因: torch.index_add_() 重复索引的实现
     - __global__ 函数
       ```c
       __global__ void scatter_elements_op(float *output, const int *index, const float *update, const int ncolumns) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int transformidx = index[idx] * ncolumns + idx % ncolumns;
        atomicAdd(&output[transformidx], update[idx]);
       }
       ```
    - atomicAdd() 实现原理

  - torch.nonzero() 操作无法实现，注意力机制的修改
  - 如何在tensorrt定义动态输入？
    - ```c++
      IBuilder* builder = createInferBuilder(sample::gLogger); // create TensorRT
	    int maxWorkSpaceSize = 1<<32;
      builder->setMaxBatchSize(maxBatchSize);
	    builder->setMaxWorkspaceSize(maxWorkSpaceSize);

      nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U<<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
      network->getInput(0)->setDimensions(Dims3(-1, 3, 20)); // onnx 定义动态输入


      IBuilderConfig* config = builder->createBuilderConfig();
      config->setMaxWorkspaceSize(1<<32);
      IOptimizationProfile* profile = builder->createOptimizationProfile();
      // 定义tensorrt 动态输入维度的最小值、最大值和最优值
      profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims3(1, 3, 20));
      profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims3(200,  3,  20));
	    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims3(2000, 3,  20));
      ```


### L3 基于规则的预测模型开发

#### 逻辑

1. 执行`prediction_actor.cc`
2. 执行`PredictionActor::PredictionEndToEndProc()`函数:
  - 获取定位信息、感知信息
  - 在`MessageProcess::OnPerception()`函数中执行以下两个函数:
    - `EvaluatorManager::Instance()->Run()`: 判断障碍物的运行意图，给相应的lane sequence赋值概率，确定障碍物将要运动的lane sequence
    - `PredictorManager::Instance()->Run()`：根据evaluator获得的lane sequence, 获得相应的轨迹点或者路径点

**Evaluator 逻辑**
3. `EvaluatorManager::Run()`: 根据obstacleContainer中的每一个障碍物执行`EvaluateObstacle()`
  - `EvaluateObstacle()`: 针对障碍物的类型选择相应的Evaluator
    - 如果障碍物在道路上(`OnLane() == true`)  ==> 执行Evaluate()
    - 如果障碍物不在道路上，则通过`ModifyPriorityForOffLane()`给障碍修改谨慎等级
      - 如果障碍不在道路上， 筛选障碍物BoundingBox最左边和最右边的点，如果其中有一个点在ego_sequence上，则认为障碍物侵占自车车道，将priority改为Caution </br>

    3.1. 如果障碍物在车道上，调用evaluator的纯虚函数，根据相应的Evaluator实例化相应的Evaluator(), 在高速场景下对vehicle类型障碍物采用HighwayCVEvaluator.
      - 通过`HighwayCVEvaluator()`执行`Evalute()`函数
       - 对障碍物设定相应的Evaluator type
       - 执行`checkEgoFailed()`: 判断跟ego vehicle相关的pointer是否为空。如果相关ego vehicle的参数为nullpointer， 则设置current lane 的sequence概率为1. 认为障碍物会沿着当前道路行驶，因为无法参照自车进行预测。
       - 执行`obs_on_ego_lane = ObstacleOnEgoLane()`: 通过obstacle sequence和ego sequence是否overlapped, 判断obstacle是否在ego vehicle的车道上
       - 判断谨慎等级为ignore的车:
         - 如果车在ego vehicle前方而谨慎等级为ignore，则判断
           - 如果obstacle在自车车道上
           - 如果obstacle在ego 左边车道上
           - 如果obstacle在ego 右边车道上
         - 则将障碍物的谨慎等级改为caution并且将障碍物当前道路sequence的概率设为1.
       - 执行`SetObstacleLateralLanePosition()`：标定障碍物相对于ego的侧向位置
         - 如果障碍物在ego lane 上，设lane type 为 `LaneAssignType_same`
         - 如果障碍物不在ego lane上
           - 把障碍物的pos投影到ego lane上，如果pro_l > 0
              - 如果谨慎等级为ignore，则设obstacle的lane相对ego vehicle的位置为: left left
              - 如果谨慎等级为caution，则设obstacle的lane相对ego vehicle的位置为: left
           - 把障碍物的pos投影到ego lane上，如果pro_l < 0
              - 如果谨慎等级为ignore，则设obstacle的lane相对ego vehicle的位置为: right right
              - 如果谨慎等级为caution，则设obstacle的lane相对ego vehicle的位置为: right
       - 当obstacle的current lane数量>1时，执行`SetSplitLaneObstacle()`:
         - 对障碍物的每一条current lane进行判断:
           - 判断当前的current lane是否corrected(纠偏而且是OneToTwo())
           - 如果当前道路是split类型而且不是OneToTwo纠偏， 则把当前道路改为split
       - 当obstacle的current lane数量<=1时, 执行`CheckSplitRampLatExceedDistance()`:
         - 针对障碍物当前(index = 0)的lane sequence中的每一条lane segment， 如果lane segment所处的lane 是ramp driving， 则修改`l_exceed_lane_distance = 1.5m`
       - 执行`GetLateralPointByLane()`函数:
         - 把障碍物boundingbox最左边的点和最右边的点投影到自车ego lane上
         - 如果障碍物的左、右点都不在ego sequence上，而且ego vehicle 和 obstacle 不在同一条 lane上，且障碍物的谨慎等级不是ignore， 那么把障碍物的谨慎等级改为normal
       - use_ramp_mode = true
         - `SetIsOnRamp(false)`
         - `SetMergeSequenceIndex(-1)`
         - `SetMergeRelativeDistance(Flag_param)`
         - `SetProbabilityForMergeLane(obs, ego, left_neighbor, right_neighbor)`:
           - 如果是merge车道而且不是OnToTwo纠偏，将障碍物的谨慎等级改为caution, 将当前lane sequence的概率设为1
           - `SetIsOnRamp(true)`
           - `ChangeIntent()`: 根据speed来判断障碍物的纵向加减速意图
           - `SetLaneType(MERGE)`
           - 如果障碍物的lane sequence和ego lane sequence相交， 或者障碍物lane sequence和ego vehicle的左侧或者右侧，则将当前的lane sequence和probability设为1
       - 最后执行`EvaluateProbabilityByLaneSequence()`:
         - 如果障碍物没有压左车道，也没有压右车道，或者障碍物和ego vehicle在同一车道,则将所有的current lane sequence的概率置1
         - 如果障碍物压左车道且obstacle lane sequence与ego的left neighbor相交， 则将该left lane sequence的概率置1
         - 如果障碍物压右车道且obstacle lane sequence与ego的left neighbor相交， 则将该right lane sequence的概率置1

**Predictor 逻辑**
3. `PredictorManager::Run()`: 根据obstacleContainer中的每一个障碍物执行`PredicteObstacle()`

  - 根据Obstacle的类型，调用相应的Predictor
    - 以vehicle为例: 执行`RunVehiclePredictor()`
    - 对于静止或者ignore的obstacle， 执行`RunEmptyPredictor()`
  - vehicle on lane => 执行 move sequence predictor
  - vehicle off lane => 执行 lane sequence predictor

  - 以 move sequence predictor 为例
    - 确定预测轨迹时间长度
      - 如果障碍物在匝道上，预测轨迹为12秒
      - 如果障碍物不在匝道上，预测轨迹为7秒
    - 执行`FilterLaneSequenceWithMerge()`:
      - 对于障碍物的lane graph中的每一条lane sequence
        - 如果lane sequence所在lane的类型为parking, 则把相应的enable_lane_sequence置为false, 相当于过滤该条lane sequence
        - 如果lane sequence的类型既不是左转，也不是右转，也不是onto，也过滤掉该条lane sequence
        - 执行`distance = GetLaneChangeDistWithADC()` => 纵向距离
          - 如果 distance 处于(-50, 200)之间
            - 如果当前判断的sequence就是汇入的sequence
              - 判断是否有碰撞风险
              - 判断是否有侧向变道意图
              - 如果有碰撞风险或者没有左右变道意图，则把当前的sequence过滤掉
        - 如果distance小于threshold
          - 如果车没有merge到ego sequence， 则过滤掉sequence
      - 对于每一条sequence
        - 如果enable_lane_sequence[i] 为true
          - LaneSequenceWithMaxProb(设置不变道sequence的最大概率)
          - LaneChangeWithMaxProb(设置变道sequence的最大概率)
    - 对于每一条lane sequence
      - 如果概率为0 或者 enable_lane_sequence[i] 为false， 则不生成轨迹
      - 如果车要停止，则执行`DrawConstantAccelerationTrajectory()`
      - 如果车不停止（保持巡航模式），则执行`DrawMoveSequenceTrajectory()`
    - 对于`DrawConstantAccelerationTrajectory()`：
      - 根据障碍物的position和相应的lane info得到障碍物的`lane_l`和`lane_s`
      - total_num = (total_time) / period 计算轨迹点的数量
      - 根据$lane_s = v \times t + \frac{1}{2} \times a \times t^2$ 和 $speed = at$ 求出相应的lane_s 和lane_l
      - trajectory 由速度speed、lane_id、lane_s、lane_l标记
    - 对于`DrawMoveSequenceTrajectory()`：
      - 首先获取障碍物当前帧的位置信息
      - 通过侧向速度计算得出到侧向终点的时间
      - 对于每条lane sequence中的每一块lane segment
        - 计算distance_to_merge, 计算距离交汇点的纵向距离
        - 如果有道路汇入，则更信time_to_lat_end_state
      - 限制车辆加速度:如果加速度acc处于最大值与最小值之间， 大大取大， 小小取小
      - GetLaneStartPoint(): 根据position取得障碍物的起始点，用lane_s、lane_l表示
      - 进行先横后纵规划:
        - GetLateralPolynomial(): 获得侧向多项式的系数 (四个参数/三次多项式)
        - GetLongitualPolynomial()：获得纵向的多项式系数 (五个参数/四次多项式)
      - 对于每一个轨迹点，计算lane_l和lane_s
        - 如果自车(ego vehicle) 比障碍物早到起始点，停止延伸轨迹
        - 如果lane_s超多当前车道lane 的total_s, 停止延伸轨迹
      - 定义lane_speed (四次多项式一阶导数) 和 lane_acc(四次多项式二阶导数)
      - 最后生成轨迹点：
        - 如果lane_s超过当前lane的total_s，lane_s截断并且lane segment index + 1

### 数据闭环项目

  - 数据前处理工具链开发
    - 数据提取: 通过cyber_rt跑包，提取数据
    - 异常数据的清洗: 主要关注轨迹点跳变的情况；偏转角过大
    - 数据增强: 偏转角的适度调整、轨迹点在合理范围内的平移或者旋转

  - 模型维护
    - LaneGCN(LaneGCN_34):
    - MFTF:
    - LaneAttentionNet:

  - 数据清洗与增强
    - 过滤异常值数据，因为感知原因在某些瞬时时刻的heading角跳变数据剔除
    - 数据增强，最heading角进行范围的偏转


[极简翻译模型Demo，彻底理解Transformer](https://zhuanlan.zhihu.com/p/360343417)</br>
[港中文大学+商汤：mmTransformer 解决行为预测问题](https://zhuanlan.zhihu.com/p/361006176)
