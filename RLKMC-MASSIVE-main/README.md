# KMC-RL: Kinetic Monte Carlo with Reinforcement Learning

A Python implementation of Kinetic Monte Carlo (KMC) simulation enhanced with Reinforcement Learning for studying atomic diffusion and phase separation in alloys.

## Project Overview

This project combines traditional KMC methods with modern deep reinforcement learning techniques to optimize the simulation of atomic diffusion processes. It specifically focuses on Fe-Cu alloy systems with vacancy-mediated diffusion.

### Key Features

- BCC lattice structure simulation
- Vacancy-mediated diffusion
- PPO (Proximal Policy Optimization) based RL control
- Graph Neural Network for atomic environment embedding
- Real-time visualization of atomic configurations
- Energy evolution tracking

## Installation

### Prerequisites

- Python >= 3.8
- CUDA compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kmc-rl.git
cd kmc-rl
```

2. Create and activate a conda environment (recommended):
```bash
conda create -n kmc-rl python=3.8 -y
conda activate kmc-rl
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── config.py           # Configuration settings
├── embedding.py        # GNN-based vacancy environment embedding
├── kmc.py             # Core KMC simulation engine
├── lattice.py         # Lattice structure implementation
├── ppo.py             # PPO algorithm implementation
├── train.py           # Training script
└── utils/
    ├── plotter.py     # Visualization utilities
    └── visualizer.py  # 3D lattice visualization
```

## Training Metrics

During training, the following metrics are tracked:

- Average reward per epoch
- Policy and value losses
- Action entropy
- System energy evolution

## 大规模试验（Scalability Bench）

这套“大规模试验”脚本用于在多卡/多节点下跑分布式 KMC（或 KMC+RL）任务，并输出每个 task 的计时、通信开销等统计。

### 1) 如何运行

**使用 Slurm**

- 直接提交 [slurm_scalability.sbatch](scripts/slurm_scalability.sbatch)

```bash
sbatch scripts/slurm_scalability.sbatch
```

**不使用 Slurm**

- 运行 [run_scalability_noslurm.sh](scripts/run_scalability_noslurm.sh)

```bash
# 单机多卡（示例）
GPUS_PER_NODE=4 NNODES=1 MASTER_ADDR=127.0.0.1 NODE_RANK=0 \
scripts/run_scalability_noslurm.sh
```

`run_scalability_noslurm.sh` 会把分布式环境变量准备好后，转调统一入口 [run_scalability.sh](scripts/run_scalability.sh) 来启动 `torchrun scripts/scalability_bench.py`。

**使用 salloc（交互式）+ 节点筛选（在 allocation 内检测）**

- 运行 [salloc_scalability.sh](scripts/salloc_scalability.sh)，它会：
  - 先 `salloc` 申请资源（拿到 Slurm allocation）
  - 在 allocation 内对 `$SLURM_JOB_NODELIST` 中的节点逐个 `srun -w <node> ...` 做可用性检测
  - 输出 good/bad 节点，并仅用 good 节点启动正式的 `torchrun`

```bash
# 示例：申请 8 节点，然后在 allocation 内筛选好节点并运行
bash scripts/salloc_scalability.sh
```

可选环境变量（外层申请资源）：

- `PARTITION`（默认 `debug`）
- `NODES`（默认 `8`）
- `CPUS_PER_TASK`（默认 `16`）
- `GRES`（默认 `dcu:4`）
- `TIME_LIMIT`（默认 `02:00:00`）

可选环境变量（allocation 内检测/运行）：

- `CHECK_TIMEOUT_SEC`：每个节点检测 `srun` 的超时秒数（默认 `60`）
- `PYTHON_BIN`：用于节点检测的小 Python 解释器（默认优先 `python3`，否则 `python`）
- `CHECK_GRES`：节点检测时额外传给 `srun --gres` 的值（默认继承 `GRES`）
- `PYTORCH_MODULE`：正式运行阶段加载的 module（默认 `apps/PyTorch/2.1.0/pytorch-2.1.0-dtk2404`）

输出文件：

- good 节点：`/public/home/ictapp_jhp/user/zrg/voxel_2/SwarmThinkers/hostfiles/good-$SLURM_JOB_ID`
- bad 节点：`/public/home/ictapp_jhp/user/zrg/voxel_2/SwarmThinkers/hostfiles/bad-$SLURM_JOB_ID`
- bad 原因：`/public/home/ictapp_jhp/user/zrg/voxel_2/SwarmThinkers/hostfiles/bad-reasons-$SLURM_JOB_ID`

### 2) 修改实验参数

所有关键参数都在 [run_scalability.sh](scripts/run_scalability.sh) 顶部通过环境变量控制（`VAR=${VAR:-default}`），你可以在运行前 `export` 或直接在命令前面覆盖。

- **单卡 lattice 大小**：修改 `LATTICE_BASE`（默认 40）  
  位置：[run_scalability.sh:L4](scripts/run_scalability.sh#L4)
- **通信后端**：修改 `DIST_BACKEND`（默认 `nccl`，也可用 `gloo`/`tccl`）  
  位置：[run_scalability.sh:L5](scripts/run_scalability.sh#L5)
- **Cu/V 浓度**：修改 `CU_DENSITY`、`V_DENSITY`  
  位置：[run_scalability.sh:L7-L8](scripts/run_scalability.sh#L7-L8)
- **模拟物理时间（秒，注意是 rescaled 之后）**：修改 `RESALED_SIM_TIME`  
  位置：[run_scalability.sh:L6](scripts/run_scalability.sh#L6)  
  对应常见重标定关系（文献中常写作）：`t = (Cv_sim / Cv_real) * t_MC`
- **任务网格数（RPV 径向/轴向）**：修改 `N_RADIAL`、`N_AXIAL`  
  位置：[run_scalability.sh:L9-L10](scripts/run_scalability.sh#L9-L10)  
  总任务数 = `N_RADIAL * N_AXIAL`

### 2.1) 单节点多 worker 绑核（与 mpidist-mp-torch.py 一致）

当每个节点只有 1 个 MPI 进程、但进程内使用多个本地 worker 时，绑核逻辑与 [mpidist-mp-torch.py](scripts/mpidist-mp-torch.py) 的 `build_pin_plan` 完全一致，采用 NUMA 感知分配。

可用参数（名称与脚本一致）：

- `--numa-per-rank`：每个 rank 使用多少个 NUMA 节点（0 表示全部）
- `--cores-per-numa`：每个 NUMA 取多少个核心（0 表示取所选 NUMA 中最小核数）
- `--workers-per-numa`：每个 NUMA 分多少个 worker
- `--numa-nodes`：显式 NUMA 节点列表，例如 `0,1`
- `--pin-policy`：`spread`（均匀分散分配）或 `compact`（连续块分配）

示例（2 个 NUMA，每个 NUMA 32 核，每个 NUMA 8 个 worker，轮询分配）：

```bash
python scripts/scalability_bench.py \
  --numa-per-rank 2 --cores-per-numa 32 --workers-per-numa 8 \
  --pin-policy spread
```

示例：覆盖参数再运行（不使用 slurm）

```bash
LATTICE_BASE=64 DIST_BACKEND=gloo CU_DENSITY=1.34e-2 V_DENSITY=2e-4 RESALED_SIM_TIME=1800 \
N_RADIAL=10 N_AXIAL=10 \
scripts/run_scalability_noslurm.sh
```

补充：生成每个 rank 的总执行时间（用于看可扩展性和负载均衡）

```bash
python scripts/sum_task_times_rank.py --dir output/L40x40x40_N1/
```

- `--dir` 指定的是生成的输出路径（里面包含 `task_times_rank*.csv`）

### 2.5) 任务调度（mpi_rma / static）

当前任务列表在 [task_manager.py](RL4KMC/runner/task_manager.py) 内生成，并且**先按任务时长（simulation_time）降序排序**。

在新的实现里：

- 默认使用 `mpi_rma`（mpi4py + MPI RMA 动态调度）：各 rank 对称，使用全局 RMA 状态做 chunk 认领/续租/回收，并通过本地 `/dev/shm` 队列把 task id 分发给节点内 worker。
- 可选使用 `static`（静态划分）：按 `task_id % WORLD_SIZE == rank` 进行分配，最后用 `dist.barrier()` 汇合。

配置方式：

- `--task_scheduler_mode mpi_rma`（默认）或环境变量 `TASK_SCHEDULER_MODE=mpi_rma`
- `--task_scheduler_mode static` 或环境变量 `TASK_SCHEDULER_MODE=static`
- `--task_num_groups` / `TASK_NUM_GROUPS`：**已废弃（ignored）**，保留仅为兼容旧脚本

### 3) 温度设置（热老化）

- 如果直接跑热老化温度分布，温度矩阵在 `generate_aging_temps(...)` 内设置：  
  [task_manager.py:L35](RL4KMC/runner/task_manager.py#L35)
- 如果要跑指定温度/自定义温度分布，可以在 [task_manager.py](RL4KMC/runner/task_manager.py) 里改写生成温度（`generate_kmc_tasks` 相关逻辑）。

### 4) 太初平台移植（torch.sdaa）

在太初平台上，通常需要把代码中的 `torch.cuda` 替换为 `torch.sdaa`。

项目提供了一个批量替换脚本：

```bash
# 先 dry-run 看会改哪些文件
python3 scripts/taichu_replace_cuda_to_sdaa.py .

# 确认无误后再真正写回
python3 scripts/taichu_replace_cuda_to_sdaa.py --apply .
```

替换范围：

- `torch.cuda` → `torch.sdaa`
- 字符串设备名：`"cuda"` / `'cuda'` → `"sdaa"` / `'sdaa'`
- 带编号的设备：`"cuda:0"` / `'cuda:0'` → `"sdaa:0"` / `'sdaa:0'`
- 带表达式的设备：`f"cuda:{local_rank}"` → `f"sdaa:{local_rank}"`

分布式后端说明：

- 使用 `sdaa` 时，`torch.distributed` 后端请使用 `tccl` 或 `gloo`（不要用 `nccl`）。
- 本项目的跑分脚本可通过环境变量设置后端，例如：

```bash
DIST_BACKEND=gloo bash scripts/salloc_scalability.sh
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. [KMC Method Overview](relevant_paper_link)
2. [PPO Algorithm](relevant_paper_link)
3. [Graph Neural Networks](relevant_paper_link)

## Acknowledgments

This project was developed at [Your Institution/Organization].
