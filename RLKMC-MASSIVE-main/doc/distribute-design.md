# 分布式重构设计（草案）

> 说明：本文档用于指导接下来的重构与实现选择，内容不要求与当前仓库实现完全一致。

## 1. 目标与约束

### 1.1 目标

1. 落实 OOP 设计
   - 抽离“计算过程”（可替换）
   - 抽象“分布式框架”（可单独测试、可替换后端）
2. 同时适配两类平台
   - GPU 超算：单节点 8–32 进程，需要 GPU 资源感知（进程/设备映射）
   - 众核平台：单节点数百进程，需要更高效、鲁棒的通信与进程管理（分层/分组通信，可靠收尾）
3. 设计稳定、清晰的接口
   - 计算接口：输入输出清晰，便于替换计算逻辑
   - 通信接口：屏蔽后端差异，提供统一通信原语
4. 提供可验证的测试用例
   - 单元测试：计算过程、通信框架、调度器可独立验证
   - 集成测试：在真实后端（torch.distributed/mpi4py）下验证整体行为

## 2. 术语

- **Leader**：分布式进程（例如 MPI rank / torch rank），负责通信协调；在纯 Leader 模式下也负责计算
- **Worker**：Leader 派生的本地 Python `multiprocessing` 子进程（仅在 Leader-Worker 模式存在），负责计算
- **Backend**：通信后端实现（`torch.distributed` 或 `mpi4py`）
- **Task**：可调度的最小工作单元（例如一个 KMC 样本/批次/step），由调度器分发

## 3. 总体设计概览

整体以“三层抽象”组织：

1. **计算层（Compute）**：只关心“如何计算”，不关心通信/进程模型
2. **通信层（Comm）**：只关心“提供统一通信原语”，屏蔽 `torch.distributed` / `mpi4py`
3. **运行时与调度层（Runtime/Scheduler）**：决定进程模型、任务分发策略、资源绑定与收尾

> 核心原则：运行时框架可以在“替换计算过程”的情况下单独测试；计算过程也可以在“替换通信后端/进程模型”的情况下复用。

## 4. 进程模型（两套）

### 4.1 纯 Leader 模式（Leader = 通信 + 计算）

- **适用场景**：进程数不极端大，计算与通信耦合较紧，或希望简化进程管理
- **特点**：每个 rank 自己计算自己负责的任务，leader 之间用通信原语协调（可静态划分，也可动态协调）

### 4.2 Leader-Worker 模式（Leader = 通信，Worker = 计算）

- **适用场景**：众核单机上每个 rank 管理大量 CPU 核心，且希望减少通信对计算进程的干扰
- **特点**：
  - Leader 维护任务队列、负责通信与汇总
  - Worker 只做计算，和 Leader 通过共享内存/队列交互
  - Worker 的生命周期必须可控（异常退出不遗留孤儿进程）

## 5. 通信后端抽象（两套后端，统一接口）

### 5.1 后端选型

- `torch.distributed`：主要适配 GPU 平台通信（NCCL/Gloo 等），保留不同通信方案兼容
- `mpi4py`：主要适配 CPU 平台通信，适合大规模 rank，生态与调试工具成熟

### 5.2 统一通信接口（建议最小集合）

建议只抽象“运行时真正需要的最小原语”，避免把某个后端的全部能力搬进接口：

- 生命周期：`init()` / `finalize()`
- 进程信息：`rank()` / `world_size()` / `local_rank()`（可选）
- 同步：`barrier()`
- 集合通信：`broadcast()` / `allreduce()` / `allgather()` / `gather()`（按需）
- 点对点（用于动态调度时）：`send()` / `recv()` 或 `isend()` / `irecv()`（按需）

> 说明：接口以“上层调度必需”为准；如果某一阶段只用到 `allreduce`/`broadcast`，就不要提前抽象更多。

## 6. 任务分发与调度（两套策略）

### 6.1 Leader 静态分配（适合 Leader-Worker 模式）

- 启动时预先划分任务范围（例如按样本区间、按步数区间）
- 每个 rank 只处理分配到的范围，减少 rank 间动态协调

### 6.2 Leader 间动态协调（适合纯 Leader 模式）

- 通过异步通信在 leader 之间动态借/还任务
- 目标：应对任务时间方差大、静态划分负载不均
- 要点：
  - 通信协议要“可终止”（所有 rank 能达成一致收尾条件）
  - 避免全局锁或单点瓶颈

### 6.3 Leader-Worker 内部机制（共享内存 + 队列）

- Leader 负责接收/生成 task，维护 Task Queue
- Worker 从 Task Queue 取任务，写回结果到共享内存的固定槽位（slot）

## 7. 资源感知（NUMA / GPU）

### 7.1 NUMA 亲和性（建议）

- 建议每个 Leader 绑定到一个 NUMA 节点；该 Leader 管理的 Worker 绑定到同一 NUMA 节点的 CPU 核心
- 示例：16 NUMA 节点、608 核系统 → 启动 16 个 Leader，每个 Leader 管理一个 NUMA 节点上的 Worker

### 7.2 GPU 资源感知（GPU 超算场景）

- Worker 进程需要设置 device 信息（例如 `CUDA_VISIBLE_DEVICES` 或 `torch.cuda.set_device(local_rank)`）
- 需要明确“进程 → GPU”的映射策略（1 进程 1 GPU / 多进程共享 GPU 等），并在运行时统一配置

## 8. Worker 与 Leader 职责（Leader-Worker 模式）

### 8.1 Worker（本地子进程）

- Python `multiprocessing` 子进程
- 通过共享内存与 Leader 交换结果与统计信息
- 所有任务完成后，将计时结果/统计信息写入共享内存，供 Leader 读取汇总
- 需要具备“父进程异常退出时自动终止”的机制，避免孤儿进程

### 8.2 Leader（分布式进程）

- 管理 Worker 生命周期：启动、分发、收集、终止、异常处理
- 负责与其他 Leader 的通信（MPI/torch.distributed），并汇总输出结果
- 需要在分布式后端初始化前启动 Worker（见下文原因）

> 约束：Leader 在 MPI 初始化前启动 Worker，避免 Worker 继承/污染 MPI 运行时状态，确保行为可控。

## 9. 收尾与容错（重点：众核下的可靠退出）

### 9.1 任务终止通知：哨兵（Sentinel / Poison Pill）

- 推荐：Leader 在确认任务全部分发完成后，向 Task Queue 放入 `num_workers` 个 "Poison 值" 作为哨兵
  - FIFO 保证：哨兵在所有真实任务之后
  - 竞争可控：每个 Worker 取到一个 "Poison 值" 后退出

### 9.2 结果回传：共享内存 Slot

- Leader 预分配共享内存区域
- 每个 Worker 按 `worker_id` 写回到固定 slot（通过 offset 计算），避免锁竞争
- Worker 取到 "Poison 值" 后：完成最后一次写回 → `shm.close()` → 进程自然结束

### 9.3 标准收尾流程（Leader-Worker）

**(1) Leader 投递哨兵**

```python
for _ in range(num_workers):
    task_queue.put("Poison")  # "Poison" 值作为哨兵，Worker 取到后退出
```

**(2) Worker 响应并清理（保证 close 在 finally 中执行）**

```python
from multiprocessing import shared_memory

def worker(worker_id, task_queue, result_shm_name, data_size):
    shm = shared_memory.SharedMemory(name=result_shm_name)
    try:
        while True:
            task = task_queue.get()
            if task == "Poison":
                final_results = compute_final_stats()
                write_to_shm_at_offset(
                    shm,
                    offset=worker_id * data_size,
                    data=final_results,
                )
                break
            run_one_task(task)
    finally:
        shm.close()
```

**(3) Leader join 并汇总；最后由 Leader unlink**

```python
for p in processes:
    p.join(timeout=join_timeout)

final_summary = aggregate_results(shm.buf)
shm.unlink()
```

### 9.4 关键注意事项

1. **只允许 Leader `unlink`**
   - `SharedMemory.unlink()` 会删除资源，若由 Worker 提前 unlink，其他 Worker 可能写回失败（`FileNotFoundError` / `Bus Error`）
2. **Slot 的边界与对齐**
   - 每个 Worker 必须写入不重叠区域，offset 与 data_size 计算需严格一致
3. **防止 Worker 在“写回阶段”挂死**
   - Worker 用 `try...finally` 保证 `shm.close()`
   - Leader `join()` 建议设置超时与异常处理策略（记录、重试或主动结束）

### 9.5 父进程死亡自动清理：`prctl(PR_SET_PDEATHSIG, SIGKILL)`

在 Linux（ARM/x86）上，可使用 `prctl` 让子进程在父进程死亡时自动收到 `SIGKILL`，提升鲁棒性。

```python
import ctypes
import signal

libc = ctypes.CDLL("libc.so.6")
PR_SET_PDEATHSIG = 1

def enable_parent_death_kill():
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

def worker_wrapper(...):
    enable_parent_death_kill()
    ...
```

## 10. 测试与验证

### 10.1 单元测试（无需真实后端）

- **Compute 层**：给定固定输入（含随机种子），输出/统计应可重复
- **Comm 抽象层**：用 FakeBackend/LoopbackBackend 验证接口契约（rank/world_size、barrier 调用顺序、allreduce 语义等）
- **Scheduler**：验证静态划分/动态协调策略在小规模（2–4 ranks）的行为

### 10.2 集成测试（真实后端）

- `mpi4py`：用 `mpirun -n 2/4` 跑最小任务，验证初始化、通信原语、收尾
- `torch.distributed`：用 `torchrun --nproc_per_node=2/4` 跑最小任务，验证 GPU/CPU 路径
- Leader-Worker：验证哨兵退出、共享内存写回、Leader join/unlink

### 10.3 性能验证（建议）

- micro-benchmark：对 `broadcast/allreduce/allgather` 做延迟/吞吐对比
- 众核场景：观察 task queue 争用、共享内存写回耗时、收尾耗时分布


## 11. 环境变量及配置项
- `ENV` 用于配置运行时行为
- `CONFIG` 用于配置计算过程（例如 KMC 业务逻辑参数）

## 12. 待决策项（实现前需要明确）

1. 默认进程模型选择：GPU 平台是否默认纯 Leader？众核平台是否默认 Leader-Worker？
2. leader 数量与 worker 数量的配置来源（命令行/环境变量/配置文件）
3. 动态协调的协议：请求/响应消息格式、终止条件、是否需要可靠重试
4. 共享内存布局：slot 格式、对齐、版本号（避免读取旧格式）
5. 异常策略：Worker 异常如何上报？Leader 超时如何处理？是否允许部分 rank 失败？