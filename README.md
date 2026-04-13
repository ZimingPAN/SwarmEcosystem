# AtomWorld-Twins

AtomWorld-Twins is a paper-facing repository for a teacher-student Dreamer macro world model for atomic KMC. The core claim is simple: the bottleneck of traditional atomistic simulation is not only speed, but also its dependence on micro-event resolution. If we want to reach sparse key states that govern long-term materials evolution while preserving physical time, the problem should be reformulated as a time-aware macro world-modeling problem.

AtomWorld-Twins 面向一篇聚焦 teacher-student Dreamer macro world model 的论文。核心主张很明确：传统原子模拟的瓶颈不只是速度，还在于它长期被困在逐微观事件推进的分辨率里；如果希望在保持时间准确的同时到达决定材料长期演化的稀疏关键状态，就应该把问题重新表述为一个带时间语义的宏步 world model 问题。



## English

### Problem

Traditional KMC provides exact micro-event sampling because transition selection and time advance are governed by the same physical rates:

- event selection depends on local rates
- residence time depends on the total rate

This makes KMC a continuous-time Markov chain rather than an ordinary fixed-step simulator. The difficulty is not only that long-horizon atomistic rollouts are expensive, but also that most of the budget is consumed at micro-event resolution before the simulation reaches the sparse key states that govern long-term materials evolution. If we want to observe only those important states while still keeping the time axis correct, we can no longer stay in the original event-by-event CTMC view. The natural reformulation is a macro world model with explicit state jump and duration semantics.

### Why A World Model Is Needed

Once we stop simulating every vacancy-hop explicitly, we also lose the direct analytic guidance that standard KMC gives us at each micro step. The model must then learn the atomistic physical world at the macro-step level:

- which sparse lattice edits are physically reachable
- what macro state follows the current key state
- how much accumulated physical time the macro transition should take

This is why AtomWorld-Twins is framed as a world model rather than a plain predictor.

### Method

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

### Physical Constraints

The model is designed around three hard constraints.

- Inventory conservation: atom and vacancy counts must remain valid.
- Local reachability: predicted edits must lie inside the k-step reachable candidate set.
- Continuous-time consistency: duration supervision uses path-conditioned accumulated expected time as the primary target; realized waiting time is modeled only as an auxiliary conditional distribution rather than arbitrary endpoint regression.

This is the reason the output is defined as reachability-constrained sparse lattice edits instead of unrestricted dense reconstruction.



### Quick Start

Environment:

- Python 3.10+
- PyTorch 2.0+
- A working environment for the Dreamer and RLKMC dependencies already included in this repository setup

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

### 问题定义

传统 KMC 之所以精确，是因为事件选择和时间推进都由同一套物理速率控制：发生什么事件取决于局部速率，停留多久取决于总速率。因此 KMC 本质上是一个连续时间马尔可夫链，而不是普通的固定步长模拟器。

真正的难点不只是原子级长时程 rollout 成本很高，更在于模拟预算会被大量逐微事件推进所吞掉，导致系统很难在现实预算内真正到达决定长期材料演化的关键状态。如果我们希望只保留这些重要状态点，同时又不丢掉时间尺度的准确性，就不能继续停留在逐微事件的 CTMC 叙事里。更自然的重写方式是一个显式建模状态跳跃和持续时间的宏步 world model。

### 为什么必须引入 World Model

一旦不再逐个 vacancy-hop 显式模拟，我们也就失去了传统 KMC 在每一个微步上提供的直接解析指导。模型必须在宏步层面重新学习 atom 物理世界的规律，也就是同时学会：

- 哪些稀疏晶格编辑在物理上真实可达
- 当前关键状态之后会走向哪个后继关键状态
- 这次宏步跳跃会消耗多少累计物理时间

这也是 AtomWorld-Twins 必须被定义成 world model，而不是普通预测器的原因。

### 方法概述

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



### 快速开始

环境要求：

- Python 3.10+
- PyTorch 2.0+
- 使用当前仓库中 Dreamer 与 RLKMC 相关依赖所对应的可运行环境

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
