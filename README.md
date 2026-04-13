# AtomWorld-Twins

Traditional atomistic simulation is bottlenecked not only by speed, but by resolution. Before a simulator reaches the sparse key states that govern long-term materials evolution, it often must replay enormous numbers of microscopic events. AtomWorld-Twins addresses this by learning a time-aware key-state world model: instead of replaying every micro event, it predicts physically reachable macro transitions and the physical duration they accumulate, allowing atomistic systems to advance along the evolution backbone that drives long-term materials behavior.

KMC sits at the center of this problem because it is not merely a step-by-step simulator. Event selection and time advance are governed by the same physical rates, so any useful acceleration method must preserve both structural evolution and the clock. AtomWorld-Twins therefore treats acceleration as macro world modeling under continuous-time constraints, rather than as simply biasing trajectories toward faster apparent progress.

AtomWorld-Twins is a paper-facing repository for a teacher-student Dreamer macro world model for atomic KMC.

传统原子模拟的瓶颈不只是速度，更在于分辨率。系统在真正到达决定材料长期演化的稀疏关键状态之前，往往必须先回放海量微观事件。AtomWorld-Twins 的核心思路是学习一个带时间语义的关键状态 world model：它不再逐个微观事件回放，而是直接预测物理上可达的宏步转移及其累计物理时间，使原子系统能够沿着主导长期材料行为的演化主干向前推进。

KMC 之所以是这个问题的核心，不只是因为它常用，更因为它本质上不是普通的逐步模拟器。事件发生什么与等待多久由同一套物理速率共同决定，因此任何真正有用的加速方案都必须同时保住结构演化和时间语义。AtomWorld-Twins 因而不是把“加速”理解成偏置轨迹去更快下降，而是把它重新表述为一个受连续时间约束的宏步 world modeling 问题。

AtomWorld-Twins 是一个面向论文叙事的 atomic KMC teacher-student Dreamer macro world model 仓库。



## English

### Motivation

Traditional KMC provides exact micro-event sampling because transition selection and time advance are governed by the same physical rates:

- event selection depends on local rates
- residence time depends on the total rate

This makes KMC a continuous-time Markov chain rather than an ordinary fixed-step simulator. The difficulty is not only that long-horizon atomistic rollouts are expensive, but also that most of the budget is spent at micro-event resolution before the simulation reaches the sparse states that dominate aging, diffusion, and defect evolution. If we want to move further along those physically decisive trajectories while keeping the time axis correct, we can no longer stay in a purely event-by-event view.

### Core Idea

AtomWorld-Twins shifts the modeling target from micro-event replay to key-state evolution. Rather than imitating every vacancy hop, it learns a time-aware macro world model that answers three coupled questions:

- which sparse lattice edits are physically reachable
- what macro state follows the current key state
- how much accumulated physical time the macro transition takes

This is naturally a reachability-constrained Semi-Markov view of atomistic evolution, but the main intuition is simpler: replace expensive micro-event replay with macro evolution along the important trajectory backbone, without severing state change from time advance.

### Method At A Glance

AtomWorld-Twins uses a teacher-student Dreamer macro world model.

Teacher:

- The teacher is the atomistic KMC simulator itself.
- Starting from state X_t, it rolls out a fixed-k micro-event segment.
- It provides the terminal state X_t+k, the accumulated expected time, the realized time, and a path summary extracted from the micro trajectory.

Student:

- The student is a Dreamer-style macro world model operating in latent space.
- It encodes the current local patch and global summary into a latent state.
- It uses posterior and prior path latents to separate training-time identifiability from test-time generation.
- It predicts the next macro latent state, sparse reachable lattice edits, and macro duration.
- The time branch keeps `tau_exp` as the primary supervision target and uses a separate lognormal auxiliary head for `tau_real`, so realized waiting time is treated as a conditional distribution rather than a deterministic endpoint.

### Physical Commitments

The model is designed around three hard constraints.

- Inventory conservation: atom and vacancy counts must remain valid.
- Local reachability: predicted edits must lie inside the k-step reachable candidate set.
- Continuous-time consistency: duration supervision uses path-conditioned accumulated expected time as the primary target; realized waiting time is modeled only as an auxiliary conditional distribution rather than arbitrary endpoint regression.

This is the reason the output is defined as reachability-constrained sparse lattice edits instead of unrestricted dense reconstruction.

### Repository Scope

> Public repository note. The public tree ships only a minimal paper-facing teacher backend subset under `kmcteacher_backend/`.



### Quick Start

Environment:

- Python 3.10+
- PyTorch 2.0+
- A working environment for the Dreamer and the public `kmcteacher_backend/` teacher backend subset already included in this repository setup

Train the macro world model:

```bash
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --save_dir results/atomworld_twins_v1 \
  --dataset_cache results/atomworld_twins_v1/segments.pt \
  --segment_k 4 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --realized_tau_weight 0.25 \
  --train_segments 2000 \
  --val_segments 400 \
  --max_candidate_sites 128 \
  --epochs 80 \
  --device cuda
```

Evaluate time alignment against teacher segments:

```bash
cd dreamer4-main
python eval_macro_time_alignment.py \
  --checkpoint results/atomworld_twins_v1/best_model.pt \
  --cache results/atomworld_twins_v1/segments.pt \
  --split val \
  --output results/atomworld_twins_v1/eval_time_alignment.json \
  --save_all_samples
```


## 中文

### 动机

传统 KMC 之所以精确，是因为事件选择和时间推进都由同一套物理速率控制：发生什么事件取决于局部速率，停留多久取决于总速率。因此 KMC 本质上是一个连续时间马尔可夫链，而不是普通的固定步长模拟器。

真正的难点不只是原子级长时程 rollout 成本很高，更在于模拟预算会被大量逐微事件推进所吞掉，导致系统很难在现实预算内真正到达决定长期材料演化的关键状态。如果我们希望沿着那些真正重要的演化路径继续往前推进，同时又不丢掉时间尺度的准确性，就不能继续停留在逐微事件的 CTMC 叙事里。

### 核心思路

AtomWorld-Twins 把建模目标从微观事件回放转成关键状态演化。它不去模仿每一次 vacancy hop，而是在宏步层面同时回答三个耦合问题：

- 哪些稀疏晶格编辑在物理上真实可达
- 当前关键状态之后会走向哪个后继关键状态
- 这次宏步跳跃会消耗多少累计物理时间

这可以被形式化为一个受 reachability 约束的 Semi-Markov 视角，但更直观的理解其实更简单：用带时间语义的宏步演化替代昂贵的微观事件回放，同时不把状态变化和时间推进拆开。

### 方法概览

AtomWorld-Twins 采用 teacher-student Dreamer macro world model。

Teacher：

- teacher 直接使用原子级 KMC 模拟器本身。
- 从当前状态 X_t 出发，teacher rollout 一个 fixed-k 的微事件片段。
- teacher 提供宏步终点状态 X_t+k、累计期望时间、实际累积时间，以及从微观路径中提取的 path summary。

Student：

- student 是一个在隐空间中运行的 Dreamer 风格宏步世界模型。
- 它把当前局部 patch 与全局摘要编码成 latent state。
- 它用 posterior 和 prior 两套路径 latent 区分训练期可识别性与测试期生成。
- 它预测下一个宏步 latent state、受可达性约束的稀疏晶格编辑，以及宏步持续时间。
- 时间分支保持 `tau_exp` 为主监督，同时新增一个面向 `tau_real` 的对数正态辅助头，因此 realized waiting time 被表述为条件分布学习，而不是单点端点回归。

### 物理约束

模型围绕三条硬约束展开：

- Inventory conservation：原子和 vacancy 的数量必须保持合法。
- Local reachability：预测编辑必须落在 fixed-k 可达候选集合之内。
- Continuous-time consistency：时间监督以路径条件化的累计期望时间为主，`tau_real` 只作为辅助条件分布建模对象，而不是任意端点回归。

这也是为什么模型输出被定义为受约束的稀疏 lattice edit，而不是无限制的 dense reconstruction。

### 仓库边界

> 公开仓库说明。当前公开树只保留 `kmcteacher_backend/` 里的最小 paper-facing teacher backend 子集。



### 快速开始

环境要求：

- Python 3.10+
- PyTorch 2.0+
- 使用当前仓库中 Dreamer 与公开 `kmcteacher_backend/` teacher backend 子集所对应的可运行环境

训练宏步世界模型：

```bash
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --save_dir results/atomworld_twins_v1 \
  --dataset_cache results/atomworld_twins_v1/segments.pt \
  --segment_k 4 \
  --teacher_path_summary_mode stepwise \
  --tau_supervision_mode prior_main \
  --realized_tau_weight 0.25 \
  --train_segments 2000 \
  --val_segments 400 \
  --max_candidate_sites 128 \
  --epochs 80 \
  --device cuda
```

执行 teacher 对齐的时间评估：

```bash
cd dreamer4-main
python eval_macro_time_alignment.py \
  --checkpoint results/atomworld_twins_v1/best_model.pt \
  --cache results/atomworld_twins_v1/segments.pt \
  --split val \
  --output results/atomworld_twins_v1/eval_time_alignment.json \
  --save_all_samples
```



## License

This repository is released under the MIT License. See LICENSE for details.
