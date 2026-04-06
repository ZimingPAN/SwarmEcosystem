<div align="center">

# 🐝 SwarmEcosystem

**World Model + GNN for Physics-Aware Kinetic Monte Carlo**

将世界模型与图神经网络融合，实现物理感知的动力学蒙特卡洛模拟

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

[English](#english) | [中文](#中文)

</div>

---

<a name="english"></a>

## 🌍 English

### Overview

SwarmEcosystem integrates **Model-Based Reinforcement Learning** (Gumbel MuZero & DreamerV4) with **Graph Neural Networks** to solve atomic-scale Kinetic Monte Carlo (KMC) simulations. By learning physics dynamics in latent space, we eliminate the variance explosion problem inherent in importance sampling approaches.

### Motivation

Traditional RL-KMC approaches (e.g., PPO + importance sampling) suffer from a fundamental issue: cumulative importance weights $w(\tau)$ exhibit **exponential variance growth** with trajectory length. This makes physical time estimation unreliable.

Our approach replaces post-hoc importance corrections with **world models** that natively learn the Poisson process dynamics of KMC in latent space, unifying spatial configuration evolution and temporal progression.

### Architecture

```
┌──────────────────────────────────────────────────┐
│                 KMC Environment                   │
│   BCC Lattice (Fe-Cu alloy + Vacancies)          │
└──────────────┬───────────────────────────────────┘
               │ Graph Observation (16-shell defect graph)
               ▼
┌──────────────────────────────────────────────────┐
│            GNN Encoder (Message Passing)          │
│   Node: 3D offset to agent + defect type         │
│   Edge: offset between defects                   │
└──────────────┬───────────────────────────────────┘
               │ Latent State
               ▼
┌──────────────────────────────────────────────────┐
│            World Model (Latent Dynamics)          │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐                │
│  │ Gumbel MuZero│  │ DreamerV4   │                │
│  │  + MCTS      │  │ + Imagination│               │
│  └──────┬──────┘  └──────┬──────┘                │
│         │                │                        │
│  ┌──────┴────────────────┴──────┐                │
│  │     Physics-Aware Heads      │                │
│  │  • Policy  • Value  • Reward │                │
│  │  • Time (Δt)  • Energy (ΔE) │                │
│  └──────────────────────────────┘                │
└──────────────────────────────────────────────────┘
```

### Dreamer Macro Edit World Model

Beyond per-event world models, we introduce a **Reachability-Constrained Semi-Markov Lattice Edit World Model** built on DreamerV4. Instead of rolling out one KMC event at a time, this model uses the atomistic KMC simulator as a teacher to learn **macro-step predictions** over a fixed number of micro-events (fixed-k).

#### Core Idea

A single macro step directly predicts:
1. Which candidate lattice sites will change
2. The final atom/vacancy type at each changed site
3. The accumulated expected physical time $\tau_{\text{exp}}$ over the macro step
4. The terminal latent state in Dreamer's latent space

#### Why Reachability Matters

Conservation alone (atom counts unchanged) is necessary but not sufficient. The model must also ensure that predicted edits correspond to states **physically reachable** within k legal vacancy-hop events. We enforce this via a **reachable candidate set** $C_t^{(k)}$ that restricts the edit output space to sites actually accessible from the current configuration.

#### Student Architecture

The student world model retains Dreamer's core structure (latent state, prior/posterior, imagination rollout) while adding macro-edit capabilities:

```
┌─────────────────────────────────────────────────────────┐
│              Dreamer Macro Edit World Model              │
│                                                          │
│  ┌────────────────┐   ┌──────────────────────────────┐  │
│  │ State Encoder   │   │ Path Posterior (train only)  │  │
│  │ Active Patch +  │   │ q(c | z_t, path, X_{t+k})   │  │
│  │ Global Summary  │   └──────────────┬───────────────┘  │
│  └───────┬────────┘                   │                  │
│          │ z_t                 ┌──────┴───────┐          │
│          ▼                    │ Path Prior    │          │
│  ┌───────────────────┐        │ p(c | z_t, k) │          │
│  │ Macro Dynamics    │◄───────┘ (inference)   │          │
│  │ z_{t+k} = G(z,c,k)│        └──────────────┘          │
│  └───────┬───────────┘                                   │
│          ▼                                               │
│  ┌───────────────────┐  ┌────────────────────────────┐  │
│  │ Edit Decoder      │  │ Duration Head              │  │
│  │ Mask + Type logits│  │ LogNormal(μ_τ, σ_τ)       │  │
│  └───────┬───────────┘  └────────────────────────────┘  │
│          ▼                                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Reachability + Conservation Projection            │  │
│  │ • Only edit within C_t^(k)                        │  │
│  │ • Inventory conservation                          │  │
│  │ • Edit scale bounded by fixed-k                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### Physical Constraints

Three non-negotiable constraints are enforced simultaneously:
- **Inventory conservation**: total counts of each atom/vacancy type are preserved
- **Local reachability**: edits must lie within the reachable candidate set $C_t^{(k)}$ derived from legal vacancy-hop sequences
- **Continuous-time consistency**: time labels use path-conditioned accumulated expected time $\tau_{\text{exp}} = \sum_{j=0}^{k-1} 1/\Gamma_{\text{tot}}(s_{t+j})$, not arbitrary endpoint regression

#### Training Strategy

- **Phase 0**: Offline statistics — verify task learnability (sparsity, time distributions, candidate set sizes)
- **Phase 1**: Fixed-k macro probe with teacher path summary and candidate set constraints
- **Phase 2**: Projected training with conservation and reachability violation driven to zero
- **Phase 3**: Dreamer rollout validation — continuous macro-step imagination via prior
- **Phase 4**: Multi-k extension ($k \in \{2, 4, 8, 16\}$)
- **Phase 5**: Optional learned micro teacher distillation

### Key Innovations

- **Physics-Time Discount**: $\gamma = \exp(-\Delta t / \tau)$ replaces fixed step discount, giving agents native temporal awareness
- **Log-Space Time Prediction**: Time head predicts $\log(\Delta t)$ via regression, correctly modeling the Poisson process ($\Delta t \sim \text{Exp}(\Gamma_{\text{tot}})$)
- **Gradient Isolation**: Two-pass backward prevents time head's large gradients from starving policy learning
- **State-Only Time Prediction**: $\Delta t$ depends on $\Gamma_{\text{tot}}$ (state property), not on action choice — enforcing KMC physics
- **Dreamer Macro Edit World Model**: A reachability-constrained semi-Markov lattice edit world model that predicts sparse lattice edits and accumulated physical time over fixed-k macro steps, bypassing per-event rollout entirely

### Project Structure

```
SwarmEcosystem/
├── LightZero-main/              # Gumbel MuZero framework
│   ├── lzero/model/             # MuZero + GNN model
│   └── zoo/kmc/                 # KMC training scripts
├── dreamer4-main/               # DreamerV4 framework
│   ├── dreamer4/                # Core Dreamer modules
│   │   └── macro_edit.py        # Macro edit world model
│   ├── train_dreamer_standalone.py
│   └── train_dreamer_macro_edit.py  # Macro edit training
├── RLKMC-MASSIVE-main/          # KMC environment + PPO baseline
│   └── RL4KMC/                  # Environment & graph encoding
├── doc/                         # Reference papers & reports
│   └── idea.md                  # Macro edit design document
├── eval_all_models.py           # Model evaluation
├── eval_time_alignment.py       # Physics time alignment eval
└── results/                     # Training results
```

### Quick Start

#### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA 12.1)
- torch_geometric
- numpy, scipy

#### Training

```bash
# Train Gumbel MuZero + GNN
cd LightZero-main
python zoo/kmc/train_muzero_standalone.py \
  --lattice_size 40 40 40 --cu_density 0.0134 --v_density 0.0002 \
  --max_shells 16 --neighbor_order 2NN \
  --total_iterations 500 --use_physics_discount

# Train DreamerV4 + GNN
cd dreamer4-main
python train_dreamer_standalone.py \
  --lattice_size 40 40 40 --cu_density 0.0134 --v_density 0.0002 \
  --max_shells 16 --neighbor_order 2NN \
  --total_iterations 500

# Train Dreamer Macro Edit World Model (fixed-k offline)
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --macro_k 4 --use_teacher_path_summary \
  --use_reachability_constraint
```

#### Evaluation

```bash
# Compare all models (PPO, MuZero, Dreamer)
python eval_all_models.py

# Physics time alignment evaluation
python eval_time_alignment.py \
  --muzero_ckpt path/to/muzero.pt \
  --dreamer_ckpt path/to/dreamer.pt
```

### References

- Li et al. (2025). *SwarmThinkers: Swarm Intelligence-Enhanced Kinetic Monte Carlo Simulations*
- Schrittwieser et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero)
- Danihelka et al. (2022). *Policy Improvement by Planning with Gumbel* (Gumbel MuZero)
- Hafner et al. (2024). *Mastering Diverse Domains through World Models* (DreamerV3/V4)
  
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<a name="中文"></a>

## 🇨🇳 中文

### 概述

SwarmEcosystem 将**基于模型的强化学习**（Gumbel MuZero 和 DreamerV4）与**图神经网络**融合，用于求解原子尺度的动力学蒙特卡洛（KMC）模拟。通过在隐空间中学习物理动力学，我们消除了重要性采样方法中固有的方差爆炸问题。

### 研究动机

传统的 RL-KMC 方法（如 PPO + 重要性采样）存在根本性问题：累积重要性权重 $w(\tau)$ 随轨迹长度呈**指数级方差增长**，导致物理时间估计不可靠。

我们的方法用**世界模型**替代事后的重要性修正，让模型在隐空间中原生学习 KMC 的泊松过程动力学，将空间构型演变和时间推进统一建模。

### 架构

```
┌──────────────────────────────────────────────────┐
│                 KMC 环境                          │
│   BCC 晶格 (Fe-Cu 合金 + 空位)                    │
└──────────────┬───────────────────────────────────┘
               │ 图观测 (16-shell 缺陷图)
               ▼
┌──────────────────────────────────────────────────┐
│           GNN 编码器 (消息传递)                    │
│   节点: 相对于 agent 的 3D 偏移 + 缺陷类型         │
│   边: 缺陷间偏移                                  │
└──────────────┬───────────────────────────────────┘
               │ 隐状态
               ▼
┌──────────────────────────────────────────────────┐
│            世界模型 (隐空间动力学)                  │
│                                                   │
│  ┌─────────────┐  ┌─────────────┐                │
│  │ Gumbel MuZero│  │ DreamerV4   │                │
│  │  + MCTS      │  │ + Imagination│               │
│  └──────┬──────┘  └──────┬──────┘                │
│         │                │                        │
│  ┌──────┴────────────────┴──────┐                │
│  │     物理感知预测头            │                │
│  │  • 策略  • 价值  • 奖励      │                │
│  │  • 时间 (Δt)  • 能量 (ΔE)   │                │
│  └──────────────────────────────┘                │
└──────────────────────────────────────────────────┘
```

### 核心创新

- **物理时间折扣**：$\gamma = \exp(-\Delta t / \tau)$ 替代固定步数折扣，赋予智能体原生的时间感知能力
- **对数空间时间预测**：time_head 预测 $\log(\Delta t)$，正确建模泊松过程
- **梯度隔离**：两次独立反向传播，防止 time_head 的大梯度吞噬策略学习的梯度
- **状态级时间预测**：$\Delta t$ 取决于 $\Gamma_{\text{tot}}$（状态属性），与动作选择无关——严格遵循 KMC 物理
- **Dreamer 宏步编辑世界模型**：基于可达性约束的半马尔可夫晶格编辑世界模型，在 fixed-k 宏步上直接预测稀疏晶格编辑和累积物理时间，跳过逐事件 rollout

### Dreamer 宏步编辑世界模型

在逐事件世界模型之外，我们引入了基于 DreamerV4 的**可达性约束半马尔可夫晶格编辑世界模型**。该模型不再逐个 KMC 事件推演，而是以原子级 KMC 模拟器为教师，学习 **fixed-k 宏步预测**。

#### 核心思想

一个宏步直接预测：
1. 哪些候选晶格位点会发生变化
2. 变化位点的最终原子/vacancy 类型
3. 宏步累积的期望物理时间 $\tau_{\text{exp}}$
4. Dreamer 隐空间中的终态 latent

#### 为什么需要可达性约束

仅靠守恒（各类原子总数不变）是必要条件而非充分条件。模型输出的编辑还必须对应 k 个合法 vacancy-hop 微事件内**物理上可达**的终点。为此，我们通过**可达候选集合** $C_t^{(k)}$ 将编辑输出空间限制在当前构型下实际可到达的位点范围内。

#### Student 架构

Student 世界模型保留 Dreamer 的核心结构（隐状态、prior/posterior、想象 rollout），同时新增宏步编辑能力：

```
┌─────────────────────────────────────────────────────────┐
│              Dreamer 宏步编辑世界模型                      │
│                                                          │
│  ┌────────────────┐   ┌──────────────────────────────┐  │
│  │ 状态编码器      │   │ 路径后验 (仅训练期)            │  │
│  │ 活跃 Patch +   │   │ q(c | z_t, path, X_{t+k})   │  │
│  │ 全局摘要       │   └──────────────┬───────────────┘  │
│  └───────┬────────┘                   │                  │
│          │ z_t                 ┌──────┴───────┐          │
│          ▼                    │ 路径先验      │          │
│  ┌───────────────────┐        │ p(c | z_t, k) │          │
│  │ 宏步动力学        │◄───────┘ (推理期)      │          │
│  │ z_{t+k} = G(z,c,k)│        └──────────────┘          │
│  └───────┬───────────┘                                   │
│          ▼                                               │
│  ┌───────────────────┐  ┌────────────────────────────┐  │
│  │ 编辑解码器        │  │ 时长预测头                  │  │
│  │ Mask + Type logits│  │ LogNormal(μ_τ, σ_τ)       │  │
│  └───────┬───────────┘  └────────────────────────────┘  │
│          ▼                                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 可达性 + 守恒投影                                  │  │
│  │ • 仅在 C_t^(k) 内允许编辑                          │  │
│  │ • 库存守恒                                         │  │
│  │ • 编辑规模受 fixed-k 可达上界约束                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 物理约束

三条不可违背的约束同时执行：
- **库存守恒**：各类型原子与 vacancy 总数守恒
- **局部可达性**：编辑必须落在由合法 vacancy-hop 序列导出的可达候选集合 $C_t^{(k)}$ 内
- **连续时间一致性**：时间标签使用路径条件化的累积期望时间 $\tau_{\text{exp}} = \sum_{j=0}^{k-1} 1/\Gamma_{\text{tot}}(s_{t+j})$，而非任意端点回归

#### 训练路线

- **Phase 0**：离线统计——验证任务可学性（稀疏度、时间分布、候选集大小）
- **Phase 1**：fixed-k 宏步探针，使用 teacher path summary 和候选集约束
- **Phase 2**：引入投影训练，将守恒与可达性违反率降至零
- **Phase 3**：Dreamer rollout 验证——通过 prior 进行连续宏步想象
- **Phase 4**：multi-k 扩展（$k \in \{2, 4, 8, 16\}$）
- **Phase 5**：可选的 learned micro teacher 蒸馏

### 项目结构

```
SwarmEcosystem/
├── LightZero-main/              # Gumbel MuZero 框架
│   ├── lzero/model/             # MuZero + GNN 模型
│   └── zoo/kmc/                 # KMC 训练脚本
├── dreamer4-main/               # DreamerV4 框架
│   ├── dreamer4/                # Dreamer 核心模块
│   │   └── macro_edit.py        # 宏步编辑世界模型
│   ├── train_dreamer_standalone.py
│   └── train_dreamer_macro_edit.py  # 宏步编辑训练
├── RLKMC-MASSIVE-main/          # KMC 环境 + PPO 基线
│   └── RL4KMC/                  # 环境与图编码
├── doc/                         # 参考文献与报告
│   └── idea.md                  # 宏步编辑设计文档
├── eval_all_models.py           # 模型评估
├── eval_time_alignment.py       # 物理时间对齐评估
└── results/                     # 训练结果
```

### 快速开始

#### 环境要求

- Python 3.10+
- PyTorch 2.0+ (CUDA 12.1)
- torch_geometric
- numpy, scipy

#### 训练

```bash
# 训练 Gumbel MuZero + GNN
cd LightZero-main
python zoo/kmc/train_muzero_standalone.py \
  --lattice_size 40 40 40 --cu_density 0.0134 --v_density 0.0002 \
  --max_shells 16 --neighbor_order 2NN \
  --total_iterations 500 --use_physics_discount

# 训练 DreamerV4 + GNN
cd dreamer4-main
python train_dreamer_standalone.py \
  --lattice_size 40 40 40 --cu_density 0.0134 --v_density 0.0002 \
  --max_shells 16 --neighbor_order 2NN \
  --total_iterations 500

# 训练 Dreamer 宏步编辑世界模型（fixed-k 离线）
cd dreamer4-main
python train_dreamer_macro_edit.py \
  --macro_k 4 --use_teacher_path_summary \
  --use_reachability_constraint
```

#### 评估

```bash
# 评估所有模型（PPO、MuZero、Dreamer）
python eval_all_models.py

# 物理时间对齐评估
python eval_time_alignment.py \
  --muzero_ckpt path/to/muzero.pt \
  --dreamer_ckpt path/to/dreamer.pt
```

### 参考文献

- Li et al. (2025). *SwarmThinkers: 群体智能增强的动力学蒙特卡洛模拟*
- Schrittwieser et al. (2020). *MuZero: 通过学习模型进行规划*
- Danihelka et al. (2022). *Gumbel MuZero: 基于 Gumbel 采样的策略改进*
- Hafner et al. (2024). *DreamerV3/V4: 通过世界模型掌握多领域任务*





本项目基于 MIT 许可证开源 — 详见 [LICENSE](LICENSE) 文件。
