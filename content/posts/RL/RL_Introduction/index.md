---
title: RL | 强化学习 -- 简介
subtitle:
date: 2023-07-14T08:21:32+08:00
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
  - RL
categories:
  - RL
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

## 1. 强化学习

Reinforcement Learning (RL): 强化学习</br>
强化学习是人工智能（AI）和机器学习（ML）领域的一个重要子领域，不同于`监督学习`和`无监督学习`，强化学习通过智能体与环境的不断交互(即采取动作)，进而获得奖励，从而不断优化自身动作策略，以期待最大化其长期收益(奖励之和)。强化学习特别适合序贯决策问题(涉及一系列有序的决策问题)。

<br>
<center>
  <img src="images/1_01.png" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">ML Categories</div>
</center>
<br>

在实际应用中，针对某些任务，我们往往无法给每个数据或者状态贴上准确的标签，但是能够知道或评估当前情况或数据是好还是坏，可以采用强化学习来处理。例如，下围棋(Go)，星际争霸II(Starcraft II)等游戏。

#### 1.1 强化学习的定义
Agent interacts with its surroundings known as the environment. Agent will get a reward from the environemnt once it takes an action in the current enrivonment. Meanwhile, the environment evolves to the next state. The goal of the agent is to maximize its total reward (the Return) in the long run.

智能体与环境的不断交互(即在给定状态采取动作)，进而获得奖励，此时环境从一个状态转移到下一个状态。智能体通过不断优化自身动作策略，以期待最大化其长期回报或收益(奖励之和)。

<br>
<center>
  <img src="images/1_01.ppm" width="640" height="280" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">强化学习流程图</div>
</center>
<br>

### 1.2 强化学习的相关概念


(1) <font color=red>状态 State ($S$)</font>: agent’s observation of its environment;</br>

(2) <font color=red>动作 Action ($A$)</font>: the approaches that agent interacts with the environment;</br>

(3) <font color=red>奖励 Reward ($R\_t$)</font>: the bonus that agent get once it takes an action in the environment at the given time step t.回报(Return)为Agent所获得的奖励之和。</br>

(4) <font color=red>转移概率 Transistion Probability ($P$)</font>: the transition possibility that environment evolves from one state to another. 环境从一个状态转移到另一个状态，可以是确定性转移过程，例如，$S\_{t+1} = f(S\_t, A\_t)$, 也可以是随机性转移过程，例如 $S\_{t+1} \sim p\left( S\_{t+1}|S\_t, A\_t \right)$</br>

(5) <font color=red>折扣因子 Discount factor ( $\gamma$ )</font>: to measure the importance of future reward to agent at the current state.</br>

(6) <font color=red>轨迹(Trajectory)</font>:是一系列的状态、动作、和奖励，可以表述为：

$$\tau = (S\_0, A\_0, R\_0, S\_1, A\_1, R\_1, ... )$$

用轨迹$\tau$来记录Agent如何和环境交互。轨迹的初始状态是从起始状态分布中随机采样得到的。一条轨迹有时候也称为片段(Episode)或者回合，是一个从初始状态(Initial State，例如游戏的开局)到最终状态(Terminal State，如游戏中死亡或者胜利)的序列。</br>

(7) <font color=red>探索-利用的折中(Exploration-Exploitation Tradeoff)</font>:
这里，探索是指Agent通过与环境的交互来获取更多的信息，而利用是指使用当前已知信息来使得Agent的表现达到最佳，例如，贪心(greedy)策略。同一时间，只能二者选一。因此，如何平衡探索和利用二者，以实现长期回报(Long-term Return)最大，是强化学习中非常重要的问题。</br>

因此，可以用$ (S，A，P，R，\gamma) $来描述强化学习过程。

### 1.3 强化学习的数学建模

(1) 马尔可夫过程 (Markov Process，MP) 是一个具备马尔可夫性质的离散随机过程。

马尔可夫性质是指下一状态 $ S\_{t+1} $ 只取决于当前状态 $S\_t$.

$$p(S\_{t+1}|S\_{t}) = p(S\_{t+1} | S\_0, S\_1, S\_2, ..., S\_t)$$

可以用有限状态集合 $\mathcal{S}$ 和状态转移矩阵 $\mathbf{P}$ 表示MP过程为 $<\mathcal{S}, \mathbf{P}>$。

为了能够刻画环境对Agent的反馈奖励，马尔可夫奖励过程将上述MP从 $<\mathcal{S}, \mathbf{P}>$ 扩展到了$ <\mathcal{S}, \mathbf{P}, R, \gamma>$。这里，$R$表示奖励函数，而 $\gamma$ 表示奖励折扣因子。

$$R\_t = R(S\_t)$$

回报(Return)是Agent在一个轨迹上的累计奖励。折扣化回报定义如下：

$$G\_{t=0:T} = R(\tau) = \sum\_{t=0}^{T}\gamma^{t}R\_t$$

价值函数(Value Function) $V(s)$是Agent在状态$s$的期望回报(Expected Return)。

$$V^{\pi} (s) = \mathbb{E}[R(\tau) | S\_0 = s]$$


(3) 马尔可夫决策过程 (Markov Decision Process，MDP)</br>

MDP被广泛应用于经济、控制论、排队论、机器人、网络分析等诸多领域。
马尔可夫决策过程的立即奖励(Reward，$R$)与状态和动作有关。MDP可以用$<\mathcal{S},\mathcal{A}, \mathbf{P}, R, \gamma>$来刻画。
$\mathcal{A}$表示有限的动作集合，此时，立即奖励变为

$$R\_t = R(S\_t, A\_t)$$

策略(Policy)用来刻画Agent根据环境观测采取动作的方式。Policy是从一个状态 $s \in \mathcal{S}$ 到动作 $a \in \mathcal{A}$的概率分布$\pi(a|s)$ 的映射，$\pi(a|s)$ 表示在状态$s$下，采取动作 $a$ 的概率。

$$\pi (a|s) = p (A\_t = a | S\_t = s), \exist{t} $$

期望回报(Expected Return)是指在一个给定策略下所有可能轨迹的回报的期望值，可以表示为：

$$J(\pi) = \int\_{\tau} p(\tau | \pi) R(\tau) = \mathbb{E}\_{\tau \sim \pi}[R(\tau)]$$

这里, $p(\tau|\pi)$表示给定初始状态分布 $\rho\_0$ 和策略 $\pi$，马尔可夫决策过程中一个 $T$ 步长的轨迹 $\tau$ 的发生概率，如下：

$$p(\tau | \pi) = \rho\_0(s\_0)\prod \limits\_{t=0}^{T-1} p(S\_{t+1} | S\_t, A\_t) \pi (A\_t | S\_t)$$

强化学习优化问题通过优化方法来提升策略，以最大化期望回报。最优策略$\pi^*$ 可以表示为:

$$\pi ^ * = \argmax\_{\pi} J(\pi)$$

给定一个策略 $\pi$，价值函数$V(s)$，即给定状态下的期望回报，可以表示为:

$$V^{\pi}(s) = \mathbb{E}\_{\tau \sim \pi} [R(\tau) | S\_0 = s] = \mathbb{E}\_{A\_t \sim \pi(\cdot | S\_t)} [\sum\_{t=0}^{\infin}\gamma^t R(S\_t, A\_t) | S\_0 = s]$$

在MDP中，给定一个动作，就有动作价值函数(Action-Value Function)，是基于状态和动作的期望回报。其定义如下：

$$Q^{\pi}(s, a) = \mathbb{E}\_{\tau \sim \pi}[R(\tau) | S\_0 = s, A\_0 = a] = \mathbb{E}\_{A\_t \sim \pi(\cdot | S\_t)}[\sum\_{t=0}^{\infin}\gamma^t R(S\_t, A\_t)|S\_0 = s, A\_0 = a]$$

根据上述定义，可以得到：

$$V^{\pi}(s) = \mathbb{E}\_{a \sim \pi}[Q^{\pi}(s,a)]$$


## 2. 深度强化学习

Deep Learning + Reinforcement Learning = Deep Reinforcement Learning (DRL)
深度学习DL有很强的抽象和表示能力，特别适合建模RL中的值函数，例如: 动作价值函数 $Q^\pi \left(s, a \right)$。
二者结合，极大地拓展了RL的应用范围。

## 3. 常见深度强化学习算法

深度强化学习的算法比较多，常见的有：DQN，DDPG，PPO，TRPO，A3C，SAC 等等。

<br>
<center>
  <img src="images/3_01.png" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">常见深度强化学习算法</div>
</center>
<br>

### 3.1 Deep Q-Networks （DQN）

DQN网路将Q-Learning和深度学习结合起来，并引入了两种新颖的技术来解决以往采用神经网络等非线性函数逼近器表示动作价值函数 `Q(s,a)` 所产生的不稳定性问题：
  - 技术1: 经验回放缓存（Replay Buffer）：将Agent获得的经验存入缓存中，然后从该缓存中均匀采用（也可考虑基于优先级采样）小批量样本用于Q-Learning的更新；
  - 技术2: 目标网络（Target Network）：引入独立的网络，用来代替所需的Q网络来生成Q-Learning的目标，进一步提高神经网络稳定性。

其中, 技术1 能够提高样本使用效率，降低样本间相关性，平滑学习过程；技术2 能够是目标值不受最新参数的影响，大大较少发散和震荡。

DQN算法具体描述如下：
<br>
<center>
  <img src="images/dqn.png" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">DQN 伪代码</div>
</center>
<br>
</br>
注意：这里随机动作选择概率$\epsilon$一般是随着迭代Episode和Time Step的增加，而逐渐降低，目的是降低随机策略的影响，逐步提高Q网络对Agent动作选择的影响。

该算法中，Line 14 具体更新方式如下：

$$\theta^Q\leftarrow\theta^Q+\beta\sum\_{i\in\mathcal{N}}\frac{\partial Q(s,a|\theta^Q)}{\partial\theta^Q}\left[y\_i-Q(s,a|\theta^Q)\right]$$

其中，集合$N$中为`minibatch`的$N$个$(S\_t,A\_t,R\_t,S\_{t+1})$经验样本集合，$\beta$表示一次梯度迭代中的迭代步长。

{{<admonition quote "参考文献" false>}}
[1] V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, Feb. 2015.
{{</admonition>}}

### 3.2 Deep Deterministic Policy Gradient（DDPG）

DDPG算法可以看作Deterministic Policy Gradient（DPG）算法和深度神经网络的结合，是对上述深度Q网络（DQN）在连续动作空间的扩展。

DDPG同时建立Q值函数（Critic）和策略函数（Actor）。这里，Critic与DQN相同，采用TD方法进行更新；而Actor利用Critic的估计，通过策略梯度方法进行更新。

DDPG算法具体描述如下：

<br>
<center>
  <img src="images/ddpg.png" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">DDPG 伪代码</div>
</center>
<br>

原论文中采用Ornstein-Uhlenbeck过程（O-U过程）作为添加噪声项N \mathcal{N}N，也可以采用时间不相关的零均值高斯噪声（相关实践表明，其效果也很好）。

{{<admonition quote "参考文献" false>}}
[1] Lillicrap, Timothy P., et al. “Continuous control with deep reinforcement learning”，arXiv preprint, 2015, online: https://arxiv.org/pdf/1509.02971.pdf
{{</admonition>}}

### 3.3 Proximal Policy Optimization（PPO）

PPO算法是对信赖域策略优化算法(Trust Region Policy Optimization, TRPO) 的一个改进，用一个更简单有效的方法来强制策略$\pi\_\theta$与$\pi\_{\theta}^{\prime}$相似。

具体来说，TRPO中的优化问题如下：

$$\begin{gathered}\max\_{\pi\_{\theta}^{\prime}}\mathcal{L}\_{\pi\_{\theta}}(\pi\_{\theta}^{\prime})\\\\s.t.\mathbb{E}\_{s\sim\rho\_{\pi\_\theta}}[D\_{KL}\left(\pi\_\theta\left|\left|\pi\_\theta^{\prime}\right.\right)\right]\leq\delta \end{gathered}$$

而PPO算法直接优化上述问题的正则版本，即：

$$\max\_{\pi\_{\theta}^{\prime}}\mathcal{L}\_{\pi\_{\theta}}\left(\pi\_{\theta}^{\prime}\right)-\lambda\mathbb{E}\_{s\sim\rho\_{\pi\_{\theta}}}\quad[D\_{KL}\left(\pi\_{\theta}||\pi\_{\theta}^{\prime}\right)]$$

这里，入为正则化系数，对应TRPO优化问题中的每一个$\delta$,都存在一个相应的$\lambda$,使得上述两个优化问题有相同的解。然而，入的值依赖于$\pi\_\theta$,因此，在PPO中，需要使用一个可动态调整的$\lambda$。具体来说有两种方法：
  (1) 通过检验KL散度值来决定$\lambda$是增大还是减小，该版本的PPO算法称为PPO-Penalty;
  (2) 直接截断用于策略梯度的目标函数，从而得到更保守的更新，该方法称为PPO-Clip。

PPO-Clip算法具体描述如下：

<br>
<center>
  <img src="images/ppo.png" width="640" height="320" align=center style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);">
  <br>
  <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">PPO 伪代码</div>
</center>
<br>

$$f(\theta')=\min\left(\ell\_t\left(\theta'\right)A^{\pi\_{\theta\_{dd}}}(S\_t,A\_t),clip(\ell\_t\left(\theta'\right),1-\epsilon,1+\epsilon)A^{\pi\_{\theta\_{dd}}}(S\_t,A\_t)\right)$$

这里，$clip(x,1-\epsilon,1+\epsilon)$表示将$x$截断在$[1-\epsilon,1+\epsilon]$中。

{{<admonition quote "参考文献" false>}}
[1] Schulman, J. , et al. “Proximal Policy Optimization Algorithms”，arXiv preprint, 2017, online: https://arxiv.org/pdf/1707.06347.pdf
[2] Schulman J, Levine S, Abbeel P, et al. “Trust region policy optimization”, International conference on machine learning. PMLR, 2015: 1889-1897, online: http://proceedings.mlr.press/v37/schulman15.pdf
{{</admonition>}}

## 4. 深度强化学习算法分类

### 4.1 根据Agent训练与测试所采用的策略是否一致

#### 4.1.1 off-policy (离轨策略、离线策略)
Agent在训练(产生数据)时所使用的策略 $\pi\_1$与 agent测试(方法评估与实际使用--目标策略)时所用的策略 $\pi\_2$ 不一致。

例如，在DQN算法中，训练时，通常采用 $\epsilon-greedy$ 策略；而在测试性能或者实际使用时，采用 $ a^* = arg \max\limits\_{a} Q^{\pi}\left( s, a \right) $ 策略。

常见算法有：DDPG，TD3，Q-learning，DQN等。

#### 4.1.2 on-policy (同轨策略、在线策略)

Agent在训练时(产生数据)所使用的策略与其测试(方法评估与提升)时使用的策略为同一个策略 $\pi$。

常见算法有：Sarsa，Policy Gradient，TRPO，PPO，A3C等。

### 4.2 策略优化的方式不同

#### 4.2.1 Value-based algorithms(基于价值的算法)

基于价值的方法通常意味着对动作价值函数 $Q^{\pi}(s,a)$的优化，最优策略通过选取该函数 $Q^{\pi}(s,a)$ 最大值所对应的动作，即 $\pi^* \approx \arg \max\limits\_{\pi}Q^{\pi}(s,a)$，这里，$\approx$ 由函数近似误差导致。

基于价值的算法具有采样效率相对较高，值函数估计方差小，不易陷入局部最优等优点，缺点是通常不能处理连续动作空间问题，最终策略通常为确定性策略。

常见算法有 Q-learning，DQN，Double DQN，等，适用于 Discrete action space。其中，DQN算法是基于state-action function $Q(s,a)$ 来进行选择最优action的。

#### 4.2.2 Policy-based algorithms(基于策略的算法)

基于策略的方法直接对策略进行优化，通过对策略迭代更新，实现累计奖励(回报)最大化。其具有策略参数化简单、收敛速度快的优点，而且适用于连续或者高维动作空间。

**策略梯度方法(Policy Gradient Method，PGM)**是一类直接针对期望回报通过梯度下降(Gradient Descent，针对最小化问题)进行策略优化的强化学习方法。其不需要在动作空间中求解价值最大化的优化问题，从而比较适用于 continuous and high-Dimension action space，也可以自然地对随机策略进行建模。

PGM方法通过梯度上升的方法直接在神经网络的参数上优化Agent的策略。

根据相关理论，期望回报 $J(\pi\_{\theta})$ 关于参数 $\theta$ 的梯度可以表示为：

$$\nabla\_\theta J(\pi\_\theta)=\mathbb{E}\_{\tau\sim\pi\_\theta}\left[\sum\_{t=0}^TR\_t\nabla\_\theta\sum\_{t^{\prime}=0}^T\log\pi\_\theta(A\_{t^{\prime}}|S\_{t^{\prime}})\right]=\mathbb{E}\_{\tau\sim\pi\_\theta}\left[\sum\_{t^{\prime}=0}^T\nabla\_\theta\log\pi\_\theta\left(A\_{t^{\prime}}|S\_{t^{\prime}}\right)\sum\_{t=0}^TR\_t\right]$$

当$T \rightarrow \infin$ 时，上式可以表示为：

$$\nabla\_{\theta}J(\pi\_{\theta}) = \mathbb{E}\_{\tau \sim \pi\_{\theta}}[\sum\_{t'=0}^{\infin}\nabla\_{\theta} \log \pi\_{\theta}(A\_{t'} | S\_{t'}) \gamma^{t'}\sum\_{t=t'}^{\infin} \gamma^{t-t'}R\_t]$$


在实际中，经常去掉 $ \gamma^{t^{\prime}} $，从而避免过分强调轨迹早期状态的问题。

上述方法往往对梯度的估计有较大的方法(奖励 $R\_t$ 的随机性可能对轨迹长度L呈指数级增长)。为此，常用的方法是引进一个基准函数 $b(S\_i)$，仅是状态 $S\_i$ 的函数。可将上述梯度修改为：

$$\nabla\_{\theta}J(\pi\_{\theta}) = \mathbb{E}\_{\tau \sim \pi\_{\theta}}[\sum\_{t'=0}^{\infin}\nabla\_{\theta} \log \pi\_{\theta}(A\_{t'} | S\_{t'}) (\sum\_{t=t'}^{\infin} \gamma^{t-t'}R\_t - b(S\_{t'}))]$$

常见的PGM算法有REINFORCE，PG，PPO，TRPO 等。

#### 4.2.3 Actor-Critic algorithms (演员-评论家方法)
Actor-Critic方法结合了上述 <font color=red>基于价值</font> 的方法和 <font color=red>基于策略</font> 的方法，利用基于价值的方法学习Q值函数或状态价值函数V来提高采样效率(Critic)，并利用基于策略的方法学习策略函数(Actor)，从而适用于连续或高维动作空间。其缺点也继承了二者的缺点，例如，Critic存在过估计问题，而Actor存在探索不足的问题等。

常见算法有 DDPG, A3C，TD3，SAC等，适用于 continuous and high-Dimension action space

### 4.3 参数更新的方式不同
Parameters updating methods

#### 4.3.1 Monte Carlo method(蒙特卡罗方法)
蒙特卡罗方法：必须等待一条轨迹 $\tau\_k$ 生成(真实值)后才能更新。

常见算法有：Policy Gradient，TRPO，PPO等。

#### 4.3.2 Temporal Difference method(时间差分方法)
时间差分方法：在每一步动作执行都可以通过自举法(Bootstrapping)(估计值)及时更新。

常见算法有：DDPG，Q-learning，DQN等。


## 参考
[1]. https://blog.csdn.net/b_b1949/article/details/128997146</br>
[2]. https://blog.csdn.net/magicyangjay111/article/details/132645347

