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

### Key Innovations

- **Physics-Time Discount**: $\gamma = \exp(-\Delta t / \tau)$ replaces fixed step discount, giving agents native temporal awareness
- **Log-Space Time Prediction**: Time head predicts $\log(\Delta t)$ via regression, correctly modeling the Poisson process ($\Delta t \sim \text{Exp}(\Gamma_{\text{tot}})$)
- **Gradient Isolation**: Two-pass backward prevents time head's large gradients from starving policy learning
- **State-Only Time Prediction**: $\Delta t$ depends on $\Gamma_{\text{tot}}$ (state property), not on action choice — enforcing KMC physics

### Project Structure

```
SwarmEcosystem/
├── LightZero-main/              # Gumbel MuZero framework
│   ├── lzero/model/             # MuZero + GNN model
│   └── zoo/kmc/                 # KMC training scripts
├── dreamer4-main/               # DreamerV4 framework
│   ├── dreamer4/                # Core Dreamer modules
│   └── train_dreamer_standalone.py
├── RLKMC-MASSIVE-main/          # KMC environment + PPO baseline
│   └── RL4KMC/                  # Environment & graph encoding
├── doc/                         # Reference papers & reports
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

### 核心创新

- **物理时间折扣**：$\gamma = \exp(-\Delta t / \tau)$ 替代固定步数折扣，赋予智能体原生的时间感知能力
- **对数空间时间预测**：time_head 预测 $\log(\Delta t)$，正确建模泊松过程
- **梯度隔离**：两次独立反向传播，防止 time_head 的大梯度吞噬策略学习的梯度
- **状态级时间预测**：$\Delta t$ 取决于 $\Gamma_{\text{tot}}$（状态属性），与动作选择无关——严格遵循 KMC 物理

### 快速开始

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

# 评估所有模型
python eval_all_models.py

# 物理时间对齐评估
python eval_time_alignment.py
```

### 参考文献

- Li et al. (2025). *SwarmThinkers: 群体智能增强的动力学蒙特卡洛模拟*
- Schrittwieser et al. (2020). *MuZero: 通过学习模型进行规划*
- Danihelka et al. (2022). *Gumbel MuZero: 基于 Gumbel 采样的策略改进*
- Hafner et al. (2024). *DreamerV3/V4: 通过世界模型掌握多领域任务*





本项目基于 MIT 许可证开源 — 详见 [LICENSE](LICENSE) 文件。
