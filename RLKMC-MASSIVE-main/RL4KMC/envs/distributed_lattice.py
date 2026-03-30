"""RL4KMC 分布式晶格环境混入类（Mixin）。

这个文件提供了一个 `DistributedMixin`，用于给晶格/环境类补充分布式运行所需的：

- rank / world_size / local_rank 的识别
- 进程网格(proc_grid)与本地几何(local geometry)的推导
- vacancy/cu 的本地划分（以及 halo 的概念，当前默认禁用）
- 基于本地 vacancy 列表计算邻居类型特征、能量增量、扩散能垒与扩散速率

重要说明（请务必先读）：

1) 这是一个 Mixin：它依赖宿主类提供一组方法/字段，例如：
    - `get_vacancy_array()` / `get_cu_array()`：返回当前 vacancy/cu 的坐标数组（通常 shape=(N,3)）。
    - `NN1` / `NN2`：一阶/二阶邻居偏移（shape=(8,3)/(6,3)）。
    - `dims`：周期边界条件(PBC)下的晶格尺寸（坐标空间的尺度需与宿主一致）。
    - `_get_pbc_coord(...)`、`_batch_get_type_from_coords(...)`：坐标 PBC 映射与“给定坐标 -> 原子类型”的批量查询。
    - `pair_energies`、`E_a0`、`k_V_idx`、`V_TYPE/CU_TYPE/FE_TYPE` 等物理参数。
    - `step_local(...)` / `step_fast_local(...)`、`get_system_stats()`、`time` 等环境交互接口。

2) 当前 `_distribute_local_positions()` 在开头 `return`，表示“默认单进程/不真正划分 vacancy”。
    下面保留了一份更完整的分布式 scatter 实现草案（含 halo），但目前不会执行。
    如果你需要启用真正的分布式划分，请删除早退并完善 `_wrap_global_to_local_with_halo` 等逻辑。

3) 本文件的“坐标”普遍以整数表示；其中多处乘以 2（例如 local_min_global * 2）。
    这通常意味着使用了“半晶格/半格点(half-site)”编码：
    - 单元格(cell)索引在坐标空间会被映射为偶数格点；
    - (奇/偶)可能表示子晶格或占位类型。
    具体约定取决于宿主类的实现。
"""

import time
import os
import itertools
from typing import Tuple, Optional

import numpy as np
import torch
from RL4KMC.utils.env import EnvKeys, env_int, env_str
from RL4KMC.config import CONFIG
import torch.distributed as dist

class DistributedMixin:
    """为晶格环境提供“分布式语义”的混入类。

    设计目标：
    - 让宿主环境在单卡/多卡（torch.distributed）下尽可能复用一套接口。
    - 把“进程拓扑”、“本地子域几何”、“局部 vacancy 特征计算”等逻辑从环境主体中抽离。

    使用方式：
    - 让你的环境类继承这个 Mixin（以及你的基础环境类）。
    - 在环境初始化时调用 `DistributedMixin.__init__(args)`。
    - 确保宿主类实现本文件 docstring 中列出的依赖接口。
    """
    def __init__(self, args):
        """初始化分布式相关字段，并构建本地 vacancy 的特征缓存。

        参数
        - args: 运行参数对象（通常是 argparse.Namespace 或类似对象）。

        关键字段（会被后续方法频繁使用）：
        - self.rank/self.world_size/self.local_rank: 分布式身份（优先从环境变量读；可通过 `_sync_rank_world` 纠正）。
        - self.device: 当前 rank 的计算设备；GPU 时通常是 `cuda:local_rank`。
        - self.comm_device: 通信张量放置的设备（默认与 device 一致）。
        - self.proc_grid_dim: 进程网格维度 (gx,gy,gz)。当前默认 (1,1,1)。
        - self.processor_dim: 每个进程负责的“cell 空间”尺寸（非 *2 坐标）。
        - self.sub_block_grid_dim: 每个进程内部再划分的子块网格（用于 RL 的分块采样/更新）。
        - self.halo_depth: halo 深度（单位通常为 cell；下游会换算成 *2 坐标）。当前默认 0。
        """
        self.args = args

        # debug 默认打开：大量日志仅在 debug=True 时打印。
        self.debug = bool(getattr(args, "debug", True))

        # torchrun/torch.distributed 通常会注入这些环境变量。
        # 这里先用环境变量初始化；如果后续 dist 已初始化，可用 `_sync_rank_world()` 纠正。
        self.rank = env_int(EnvKeys.RANK, 0)
        self.local_rank = env_int(EnvKeys.LOCAL_RANK, 0)
        self.world_size = env_int(EnvKeys.WORLD_SIZE, 1, min_value=1)

        self.device = torch.device(CONFIG.runner.device)

        # 通信设备：all_reduce 等通信使用的张量设备。
        # 在 NCCL 下通常需要 GPU Tensor；在 GLOO 下 CPU/GPU 都可能。
        self.comm_device = self.device

        # 进程网格：rank -> (x,y,z) 的映射依赖该维度。
        self.proc_grid_dim = (1, 1, 1)

        # 每个进程负责的“cell 空间”尺寸。
        # 注意：本文件中很多地方会把 cell 尺寸乘以 2 转成“坐标空间”(half-site)尺寸。
        self.processor_dim = tuple(getattr(args, "processor_dim", tuple(args.lattice_size)))

        # 进程内部子块划分：默认仅 1 个子块（即不分块）。
        self.sub_block_grid_dim = (1, 1, 1)

        # halo 深度：用于在分布式下引入邻域通信（ghost cells）。当前默认关闭。
        self.halo_depth = 0

        # 分发/划分本地数据（当前默认早退：相当于单进程或“每个进程持有全部 vacancy”）。
        self._distribute_local_positions()

        # 根据本地 vacancy 构建本地 ID 以及预计算的邻居类型缓存。
        self._build_local_ids()

    def local_to_global(self, pos):
        """把本地坐标转换为全局坐标。

        目前实现是“恒等映射”：直接把输入转成 int ndarray。
        - 如果未来启用真正的域分解(local sub-domain)，这里通常需要加上 `local_min_global` 偏移。
        - 注意 PBC 情况下还可能需要取模。
        """
        return np.array(pos, dtype=int)

    def global_to_local(self, pos):
        """把全局坐标转换为本地坐标。

        目前实现是“恒等映射”。
        真正的域分解下通常需要做：local = global - local_min_global（并处理 halo）。
        """
        return np.array(pos, dtype=int)



    def _log(self, msg):
        """调试日志：仅 debug=True 时输出。"""
        if self.debug:
            print(f"[rank {self.rank}] {msg}", flush=True)

    def _print_color(self, msg, color):
        """带颜色的日志输出（ANSI 颜色码）。

        仅用于交互式调试；在某些日志系统/重定向环境下颜色码可能不可读。
        """
        if not self.debug:
            return
        codes = {
            "red": "\x1b[31m",
            "green": "\x1b[32m",
            "yellow": "\x1b[33m",
            "blue": "\x1b[34m",
            "magenta": "\x1b[35m",
            "cyan": "\x1b[36m",
            "reset": "\x1b[0m",
        }
        print(f"{codes.get(color, '')}[rank {self.rank}] {msg}{codes['reset']}", flush=True)

    def _log_comm(self, msg):
        """通信/分布式相关日志（青色）。"""
        # self._print_color(msg, "cyan")

    def _log_check_ok(self, msg):
        """检查通过日志（绿色）。"""
        # self._print_color(msg, "green")

    def _log_check_fail(self, msg):
        """检查失败日志（红色）。"""
        # self._print_color(msg, "red")

    def _factor3(self, n: int):
        """把整数 n 分解为 3 个尽量均衡的因子 (a,b,c)。

        用途：把 GPU 数/节点数等资源数量，拆成 3D 网格维度。
        思路：
        - 先用立方根附近的因子作为 a
        - 再把剩余 m=n/a 在平方根附近找因子 b
        - c=m/b

        注意：这里不要求 a*b*c == n 的顺序最优，仅追求接近立方体。
        """
        if n <= 1:
            return (1, 1, max(1, n))
        import math
        x = max(1, int(math.floor(n ** (1.0 / 3.0))))
        def lf(m, k):
            # largest factor: 在 <=k 的范围内向下找 m 的因子。
            d = max(1, min(k, m))
            while d > 1 and m % d != 0:
                d -= 1
            return d
        a = lf(n, x)
        m = n // a
        y = max(1, int(math.floor(m ** 0.5)))
        b = lf(m, y)
        c = m // b
        return (int(a), int(b), int(c))

    def _get_gpu_grid_dim(self, g):
        """给定 GPU 总数 g，返回一个经验性的 3D 网格划分。

        特判 4 与 8：常见的多 GPU 形状更偏向 2x2 平面或 2x2x2 立方。
        其它情况回退到 `_factor3`。
        """
        g = max(1, int(g))
        if g == 4:
            return (1, 2, 2)
        if g == 8:
            return (2, 2, 2)
        return self._factor3(g)

    def _get_node_grid_dim(self, n):
        """给定节点数 n，返回 3D 网格划分。"""
        return self._factor3(max(1, int(n)))

    def _sync_rank_world(self):
        """如果 torch.distributed 已初始化，则以 dist 的信息覆盖 rank/world_size。

        背景：
        - 有时环境变量可能缺失或不一致；dist 初始化后才是权威来源。
        - 这里用 try/except 保持鲁棒性（在非分布式运行时不报错）。
        """
        try:
            if dist.is_available() and dist.is_initialized():
                self.rank = int(dist.get_rank())
                self.world_size = int(dist.get_world_size())
        except Exception:
            pass

    def _setup_proc_id(self):
        """根据 `rank` 和 `proc_grid_dim` 计算 3D 进程坐标 `proc_id_3d`。

        rank -> (x,y,z) 的线性展开方式：
        - x 为最高维：rank // (gy*gz)
        - y 次之： (rank % (gy*gz)) // gz
        - z 最低维：rank % gz

        这要求 proc_grid_dim 的乘积 >= world_size；否则部分 rank 会落在同一坐标或越界。
        """
        gx, gy, gz = self.proc_grid_dim
        x = self.rank // max(1, (gy * gz)) if gy * gz > 0 else 0
        rem = self.rank % max(1, (gy * gz)) if gy * gz > 0 else 0
        y = rem // max(1, gz) if gz > 0 else 0
        z = rem % max(1, gz) if gz > 0 else 0
        self.proc_id_3d = (x, y, z)
        try:
            self._log_comm(f"proc_grid_dim={(gx, gy, gz)} rank={self.rank} proc_id_3d={self.proc_id_3d}")
        except Exception:
            pass

    def _setup_local_geometry(self):
        """根据 `proc_id_3d` 和 `processor_dim` 推导本地子域的全局范围。

        约定：
        - `processor_dim` 是 cell 空间尺寸 (sx,sy,sz)
        - 真实坐标空间（half-site）使用 *2 的尺度

        输出字段：
        - `local_min_global`: 本 rank 子域在“全局坐标空间”中的最小角点（含）。
        - `local_max_global_ex`: 本 rank 子域在“全局坐标空间”中的最大角点（不含，exclusive）。
        - `sub_block_size`: 每个子块的 cell 空间尺寸（processor_dim / sub_block_grid_dim）。
        """
        sx, sy, sz = self.processor_dim
        px, py, pz = self.proc_id_3d
        self.local_min_global = np.array([px * sx, py * sy, pz * sz], dtype=int) * 2
        self.local_max_global_ex = self.local_min_global + np.array(self.processor_dim, dtype=int) * 2
        self.sub_block_size = tuple((np.array(self.processor_dim) // np.array(self.sub_block_grid_dim)).tolist())

    def _validate_geometry(self):
        """对本地几何推导进行一致性检查（主要用于 debug）。

        检查点：
        - `local_min_global` 是否等于 proc_id_3d * processor_dim * 2
        - 如果宿主存在 `size`（全局 lattice_size，cell 空间），则检查：
            processor_dim * proc_grid_dim == size
        """
        try:
            px, py, pz = self.proc_id_3d
            sx, sy, sz = self.processor_dim
            exp = np.array([px * sx, py * sy, pz * sz], dtype=int) * 2
            ok = bool(np.array_equal(exp, self.local_min_global))
            if ok:
                self._log_check_ok(f"geometry_ok expected={exp.tolist()} actual={self.local_min_global.tolist()}")
            else:
                self._log_check_fail(f"geometry_mismatch proc_id_3d={self.proc_id_3d} processor_dim={self.processor_dim} expected={exp.tolist()} actual={self.local_min_global.tolist()}")
            if hasattr(self, 'size'):
                prod = np.array(self.processor_dim, dtype=int) * np.array(self.proc_grid_dim, dtype=int)
                if not np.array_equal(prod, np.array(self.size, dtype=int)):
                    self._log_check_fail(f"size_mismatch processor_dim={self.processor_dim} proc_grid_dim={self.proc_grid_dim} product={prod.tolist()} lattice_size={list(self.size)}")
        except Exception:
            pass

    def _build_sub_block_indices(self):
        """构建“子块 -> vacancy/cu 坐标列表”的索引结构。

        当前实现是最简版本：
        - 只创建一个子块 (0,0,0)
        - 把所有 vacancy/cu 坐标都归入这个子块

        如果要启用真正的 sub_block 分块，需要：
        - 定义 `_coord_to_sub_id` 或类似方法
        - 按子块范围把坐标分桶
        """
        vac_arr = np.array(self.get_vacancy_array(), dtype=np.int32)
        self.vac_pos_to_index = {tuple(map(int, p)): i for i, p in enumerate(vac_arr.tolist())}
        self.sub_block_vacancy_pos = {(0, 0, 0): list(self.vac_pos_to_index.keys())}
        cu_arr = np.array(self.get_cu_array(), dtype=np.int32)
        self.sub_block_cu_pos = {(0, 0, 0): [tuple(map(int, p)) for p in cu_arr.tolist()]}

    def _build_local_ids(self):
        """构建“本地 vacancy ID”并预计算本地 vacancy 的邻居类型张量。

        重要：这里的“本地 ID”与“坐标数组中的行号/全局 vacancy id”可能不是同一概念。
        当前实现把 `get_vacancy_array()` 的行号当作本地 vacancy id（0..Nv_local-1）。
        真正的分布式划分下，通常需要维护：
        - local_id -> global vacancy index 的映射（代码中提到 `local_vac_id_to_global_index`）
        - global vacancy index -> local row 的映射

        目前文件里仍保留了这些概念的注释，但默认 `_distribute_local_positions()` 早退，
        因而大多数情况下本地数组就是全局数组。
        """
        # 进程内重新分配本地唯一 ID（vac 从 0，cu 从 Nv_local 起）
        vac_src = np.array(self.get_vacancy_array(), dtype=np.int32)
        cu_src = np.array(self.get_cu_array(), dtype=np.int32)
        local_vac = [tuple(map(int, p)) for p in vac_src.tolist()]
        local_cu = [tuple(map(int, p)) for p in cu_src.tolist()]
        Nv_local = len(local_vac)
        # 统一使用全局索引 self.cu_pos_index，无需维护本地映射
        # 提供本地 ID -> 全局 vacancy 数组索引的映射（由分发阶段提供）
        # self.local_vac_id_to_global_index 已在 _distribute_local_positions 中设置
        self._log(f"local_ids Nv_local={Nv_local} Nc_local={len(local_cu)}")
        self.V_nums = Nv_local
        # 预计算 vacancy 的局部环境（邻居类型），以减少 step 中的重复计算。
        self.nn1_types, self.nn2_types, self.nn1_nn1_types, self.nn1_nn2_types = self._calculate_vacancy_local_environments_sparse_local()
        # self._dump_local_nn_types("init")

    def _validate_local_positions_range(self):
        """预留：验证本地坐标是否都落在合法范围内。

        目前直接 return，表示不做任何检查。
        真正启用 halo/域分解时建议实现：
        - pure local 坐标是否在 [0, local_size2)
        - halo 坐标是否在 [-halo2, local_size2+halo2)（或等价窗口）
        """
        return

    def _rank_to_proc3d(self, r):
        """把线性 rank 映射到 3D 进程坐标。

        与 `_setup_proc_id` 采用相同的展开/折叠规则。
        """
        nx, ny, nz = self.proc_grid_dim
        x = r // (ny * nz)
        rem = r % (ny * nz)
        y = rem // nz
        z = rem % nz
        return (x, y, z)

    def _proc_local_min_global(self, proc3d):
        """给定进程坐标(proc3d)计算其全局最小角点（坐标空间、*2）。"""
        sx, sy, sz = self.processor_dim
        return np.array([proc3d[0] * sx, proc3d[1] * sy, proc3d[2] * sz], dtype=int) * 2

    def _wrap_global_to_local_with_halo(self, p_global, lm, lx, dims2, halo2):
        """把全局坐标映射到某个 rank 的“带 halo 的本地坐标系”。

        这是分布式域分解的关键函数之一：
        - 输入 `p_global`：全局坐标（坐标空间，通常为 *2 half-site）。
        - 输入 `lm/lx`：该 rank 子域在全局坐标空间的 [min, max_ex) 范围。
        - 输入 `dims2`：全局坐标空间维度（一般为 lattice_size * 2）。
        - 输入 `halo2`：halo 深度（坐标空间），通常为 halo_depth * 2。

        期望输出：
        - 若 `p_global` 落在该 rank 的 (pure local + halo) 覆盖范围内，则返回其“本地坐标”
            （本地坐标通常以 lm 为原点：p_local = p_global - lm，并考虑 PBC wrap）。
        - 否则返回 None。

        目前实现是最简占位：直接返回原坐标（不做映射、不做过滤）。
        这会导致下面的分布式 scatter 逻辑无法正确区分 pure/halo。
        """
        return list(map(int, p_global))

    def _distribute_local_positions(self):
        """把 vacancy/cu 从“全局数组”分发成当前 rank 的本地数组。

        当前函数分两层含义：
        - 顶部“实际执行的版本”：只设置一些字段并立即 return，相当于禁用分布式划分。
        - 底部“保留的完整实现草案”：
            - rank0 负责准备全局 vacancy/cu 数组
            - 使用 `dist.scatter_object_list` 把每个 rank 应得的 pure/halo 坐标散发出去
            - rank0 还会构建全局索引结构（vac_pos_set、v_pos_to_id、cu_pos_index 等）

        如果你准备启用底部逻辑：
        1) 删除顶部 return
        2) 实现 `_wrap_global_to_local_with_halo` 的正确映射/过滤
        3) 明确宿主类中 `get_vacancy_array()/get_cu_array()` 与这里字段的关系
                （目前顶部早退意味着 `get_vacancy_array()` 仍返回宿主自己的数组）。
        """
        # -------------------------
        # 默认路径：不做真正的分布式划分
        # -------------------------
        cu_local = np.array(self.get_cu_array(), dtype=np.int32)
        self.pure_local_cu_pos = cu_local
        self.halo_vacancy_pos = np.empty((0, 3), dtype=np.int32)
        self.halo_cu_pos = np.empty((0, 3), dtype=np.int32)
        self.local_cu_pos = self.pure_local_cu_pos
        return

        # -------------------------
        # 下面是完整分布式版本（当前不会执行）
        # -------------------------
        world = dist.get_world_size()
        rank = dist.get_rank()
        if rank == 0:
            init_by_sb = bool(getattr(self.args, "init_via_subblocks", False))
            if init_by_sb:
                nx, ny, nz = tuple(self.size) if hasattr(self, 'size') else tuple(self.processor_dim)
                total_half_sites = int(nx) * int(ny) * int(nz) * 2
                cu_d = getattr(self.args, "cu_density", None)
                v_d = getattr(self.args, "v_density", None)
                cu_count = int(round(float(cu_d) * total_half_sites)) if cu_d is not None else int(getattr(self.args, "lattice_cu_nums", total_half_sites // 2))
                v_count = int(round(float(v_d) * total_half_sites)) if v_d is not None else int(getattr(self.args, "lattice_v_nums", total_half_sites // 2))
                blocks_per_rank = int(np.prod(np.array(self.sub_block_grid_dim, dtype=int)))
                total_blocks = int(world * blocks_per_rank)
                base_v = v_count // total_blocks
                rem_v = v_count % total_blocks
                base_c = cu_count // total_blocks
                rem_c = cu_count % total_blocks
                vac_global_list = []
                cu_global_list = []
                sb_size_cells = np.array(self.sub_block_size, dtype=int)
                def sample_unique_indices(n_need: int, total_sites: int) -> np.ndarray:
                    if n_need <= 0:
                        return np.empty((0,), dtype=np.int32)
                    if n_need >= total_sites:
                        return np.arange(total_sites, dtype=np.int32)
                    device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
                    seed0 = int(getattr(self.args, "seed", 1))
                    gen = torch.Generator(device=device)
                    gen.manual_seed(seed0)
                    rand_nums = torch.rand(n_need * 2, device=device, dtype=torch.float64, generator=gen)
                    indices = (rand_nums * total_sites).long()
                    unique_indices = torch.unique(indices, sorted=False)
                    while len(unique_indices) < n_need:
                        need = n_need - len(unique_indices)
                        new_rand = torch.rand(need * 2, device=device, dtype=torch.float64, generator=gen)
                        new_indices = (new_rand * total_sites).long()
                        unique_indices = torch.unique(torch.cat([unique_indices, new_indices]), sorted=False)
                    return unique_indices[:n_need].to(dtype=torch.int32).cpu().numpy()
                # 为每个 rank 的每个子块生成坐标
                block_counter = 0
                for r in range(world):
                    proc3d = self._rank_to_proc3d(r)
                    lm = self._proc_local_min_global(proc3d)
                    for bx in range(self.sub_block_grid_dim[0]):
                        for by in range(self.sub_block_grid_dim[1]):
                            for bz in range(self.sub_block_grid_dim[2]):
                                start_cell = np.array([bx, by, bz], dtype=int) * sb_size_cells
                                sb_min_global = lm + start_cell * 2
                                nx_sb, ny_sb, nz_sb = sb_size_cells.tolist()
                                total_sites_sb = int(nx_sb) * int(ny_sb) * int(nz_sb) * 2
                                v_i = base_v + (1 if block_counter < rem_v else 0)
                                c_i = base_c + (1 if block_counter < rem_c else 0)
                                block_counter += 1
                                v_idx = sample_unique_indices(v_i, total_sites_sb)
                                c_idx = sample_unique_indices(c_i, total_sites_sb)
                                v_local_coords = self.get_coords_vectorized_local(v_idx, nx_sb, ny_sb, nz_sb)
                                c_local_coords = self.get_coords_vectorized_local(c_idx, nx_sb, ny_sb, nz_sb)
                                v_global_coords = v_local_coords + sb_min_global
                                c_global_coords = c_local_coords + sb_min_global
                                vac_global_list.append(v_global_coords)
                                cu_global_list.append(c_global_coords)
                vac_global = np.vstack(vac_global_list).astype(np.int32) if len(vac_global_list) > 0 else np.empty((0, 3), dtype=np.int32)
                cu_global = np.vstack(cu_global_list).astype(np.int32) if len(cu_global_list) > 0 else np.empty((0, 3), dtype=np.int32)
                obj_list_src = [{"vac_global": vac_global, "cu_global": cu_global}]
            else:
                vac_global = self.get_vacancy_array()
                cu_global = self.get_cu_array()
                obj_list_src = [{"vac_global": vac_global, "cu_global": cu_global}]
        else:
            obj_list_src = [None]
        recv_list_global = [None]
        dist.scatter_object_list(recv_list_global, [obj_list_src[0]] * world if rank == 0 else None, src=0)
        payload_global = recv_list_global[0]
        # 所有 rank接收全局数组；仅 rank0 保留全局数组与索引，其他 rank 不持有全局原子信息
        vac_global = np.array(payload_global["vac_global"], dtype=np.int32)
        cu_global = np.array(payload_global["cu_global"], dtype=np.int32)
        if rank == 0:
            self.vac_pos_set = {tuple(map(int, p)) for p in vac_global.tolist()}
            self.cu_pos_set = {tuple(map(int, p)) for p in cu_global.tolist()}
            self.v_pos_to_id = {tuple(vac_global[idx]): idx for idx in range(vac_global.shape[0])}
            self.v_pos_of_id = {idx: tuple(vac_global[idx]) for idx in range(vac_global.shape[0])}
            self.cu_pos_of_id = {idx + int(vac_global.shape[0]): tuple(cu_global[idx]) for idx in range(cu_global.shape[0])}
            self._build_cu_pos_index()
        else:
            self.vac_pos_set = set()
            self.cu_pos_set = set()
            self.v_pos_to_id = {}
            self.v_pos_of_id = {}
            self.cu_pos_of_id = {}
        # 延后到 _build_local_ids 完成后再初始化环境缓冲区
        if rank == 0:
            vac_global_index = {tuple(map(int, p)): i for i, p in enumerate(vac_global.tolist())}
            dims2 = np.array(self.dims, dtype=int) if hasattr(self, 'dims') else (np.array(self.processor_dim, dtype=int) * np.array(self.proc_grid_dim, dtype=int) * 2)
            halo2 = np.array([self.halo_depth, self.halo_depth, self.halo_depth], dtype=int)
            if hasattr(self, 'size'):
                nx, ny, nz = tuple(self.size)
                total_half_sites = int(nx) * int(ny) * int(nz) * 2
                cu_d = getattr(self.args, "cu_density", None)
                v_d = getattr(self.args, "v_density", None)
                cu_count = int(round(float(cu_d) * total_half_sites)) if cu_d is not None else int(getattr(self.args, "lattice_cu_nums", len(self.get_cu_array())))
                v_count = int(round(float(v_d) * total_half_sites)) if v_d is not None else int(getattr(self.args, "lattice_v_nums", len(self.get_vacancy_array())))
                self._log_comm(f"init totals size=({nx},{ny},{nz}) half_sites={total_half_sites} cu_density={cu_d} v_density={v_d} cu_count={cu_count} v_count={v_count} actual_arrays V={len(vac_global)} C={len(cu_global)} dims2={dims2.tolist()}")
            scatter_inputs = []
            for r in range(world):
                proc3d = self._rank_to_proc3d(r)
                lm = self._proc_local_min_global(proc3d)
                lx = lm + np.array(self.processor_dim, dtype=int) * 2
                vac_pure_local = []
                vac_halo_local = []
                vac_idx_list_pure = []
                for pos in vac_global:
                    lp = self._wrap_global_to_local_with_halo(pos, lm, lx, dims2, halo2)
                    if lp is not None:
                        lparr = np.array(lp, dtype=int)
                        local_size2 = np.array(self.processor_dim, dtype=int) * 2
                        if np.all(lparr >= 0) and np.all(lparr < local_size2):
                            vac_pure_local.append(tuple(map(int, lp)))
                            vac_idx_list_pure.append(vac_global_index[tuple(map(int, pos))])
                        else:
                            vac_halo_local.append(tuple(map(int, lp)))
                cu_pure_local = []
                cu_halo_local = []
                for pos in cu_global:
                    lp = self._wrap_global_to_local_with_halo(pos, lm, lx, dims2, halo2)
                    if lp is not None:
                        lparr = np.array(lp, dtype=int)
                        local_size2 = np.array(self.processor_dim, dtype=int) * 2
                        if np.all(lparr >= 0) and np.all(lparr < local_size2):
                            cu_pure_local.append(tuple(map(int, lp)))
                        else:
                            cu_halo_local.append(tuple(map(int, lp)))
                payload = {
                    "vac_pure": vac_pure_local,
                    "vac_idx_pure": vac_idx_list_pure,
                    "vac_halo": vac_halo_local,
                    "cu_pure": cu_pure_local,
                    "cu_halo": cu_halo_local,
                }
                scatter_inputs.append(payload)
            recv_list = [None]
            dist.scatter_object_list(recv_list, scatter_inputs, src=0)
            payload = recv_list[0]
        else:
            recv_list = [None]
            dist.scatter_object_list(recv_list, None, src=0)
            payload = recv_list[0]
        vp = payload.get("vac_pure", [])
        vh = payload.get("vac_halo", [])
        cp = payload.get("cu_pure", [])
        ch = payload.get("cu_halo", [])
        vi = payload.get("vac_idx_pure", [])
        self.halo_vacancy_pos = np.array(vh, dtype=np.int32).reshape(-1, 3) if len(vh) > 0 else np.empty((0, 3), dtype=np.int32)
        self.pure_local_cu_pos = np.array(cp, dtype=np.int32).reshape(-1, 3) if len(cp) > 0 else np.empty((0, 3), dtype=np.int32)
        self.halo_cu_pos = np.array(ch, dtype=np.int32).reshape(-1, 3) if len(ch) > 0 else np.empty((0, 3), dtype=np.int32)
        self.local_cu_pos = self.pure_local_cu_pos
        
        self._log_comm(f"distribute local positions Nv={len(self.get_vacancy_array())} Nc={len(self.get_cu_array())} Nv_halo={len(self.halo_vacancy_pos)} Nc_halo={len(self.halo_cu_pos)}")
        if self.debug:
            self._log_comm(f"halo vacancy pos: {self.halo_vacancy_pos.tolist()}")
            self._log_comm(f"pure local cu pos: {self.pure_local_cu_pos.tolist()}")
            self._log_comm(f"halo cu pos: {self.halo_cu_pos.tolist()}")
        _barrier_nccl_safe()
        if self.debug:
            self.check_system_consistency()
        hv = getattr(self, 'halo_vacancy_pos', np.empty((0,3), dtype=np.int32))
        pc = getattr(self, 'pure_local_cu_pos', np.empty((0,3), dtype=np.int32))
        hc = getattr(self, 'halo_cu_pos', np.empty((0,3), dtype=np.int32))
        

    def check_system_consistency(self):
        """检查 vacancy/cu 的数量在多进程下是否一致。

        单进程/未初始化 dist 时：
        - 直接用本地数组长度与 args 里的期望值比对。

        多进程/已初始化 dist 时：
        - 先统计每个 rank 的本地 vacancy/cu 数量 (lv/lc)
        - all_reduce 求和得到全局总数
        - rank0 额外对比：全局数组长度（如果 rank0 仍持有）是否与总和一致

        返回一个 dict，包含各类统计数字与 ok 标记。
        """
        if not dist.is_initialized():
            total_v = int(len(self.get_vacancy_array()))
            total_c = int(len(self.get_cu_array()))
            expected_v = int(getattr(self.args, "lattice_v_nums", total_v))
            expected_c = int(getattr(self.args, "lattice_cu_nums", total_c))
            ok = (total_v == expected_v) and (total_c == expected_c)
            return {"V_total": total_v, "C_total": total_c, "expected_V": expected_v, "expected_C": expected_c, "ok": bool(ok)}
        lv = int(len(self.get_vacancy_array()))
        lc = int(len(self.get_cu_array()))
        lv_t = torch.tensor([lv], dtype=torch.int64, device=self.comm_device)
        lc_t = torch.tensor([lc], dtype=torch.int64, device=self.comm_device)
        dist.all_reduce(lv_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(lc_t, op=dist.ReduceOp.SUM)
        total_v = int(lv_t.item())
        total_c = int(lc_t.item())
        expected_v = int(getattr(self.args, "lattice_v_nums", total_v))
        expected_c = int(getattr(self.args, "lattice_cu_nums", total_c))
        gv = int(len(self.get_vacancy_array())) if self.rank == 0 else None
        gc = int(len(self.get_cu_array())) if self.rank == 0 else None
        if self.rank == 0:
            msg = f"check totals sum_local V={total_v} C={total_c} expected V={expected_v} C={expected_c} global_array V={gv} C={gc}"
            ok = (total_v == expected_v) and (total_c == expected_c)
            if gv is not None:
                ok = ok and (gv == total_v)
            if gc is not None:
                ok = ok and (gc == total_c)
            if ok:
                self._log_check_ok(msg)
            else:
                self._log_check_fail(msg)
            return {"V_total": total_v, "C_total": total_c, "expected_V": expected_v, "expected_C": expected_c, "global_array_V": gv, "global_array_C": gc, "ok": bool(ok)}
        else:
            ok = (total_v == expected_v) and (total_c == expected_c)
            return {"V_total": total_v, "C_total": total_c, "expected_V": expected_v, "expected_C": expected_c, "ok": bool(ok)}

    def _calculate_vacancy_local_environments_sparse_local(self):
        """为本地 vacancy 预计算邻居类型张量（稀疏/批量版本）。

        输出：
        - nn1_types: shape=(Nv_local, 8)
            每个 vacancy 的 8 个一阶邻居类型（类型编码由宿主定义）。
        - nn2_types: shape=(Nv_local, 6)
            每个 vacancy 的 6 个二阶邻居类型。
        - nn1_nn1_types: shape=(Nv_local, 8, 8)
            以 vacancy 的每个 NN1 邻居作为中心，再看的 8 个 NN1 邻居类型。
        - nn1_nn2_types: shape=(Nv_local, 8, 6)
            以 vacancy 的每个 NN1 邻居作为中心，再看的 6 个 NN2 邻居类型。

        这些张量通常用于快速计算扩散能垒/能量变化，避免每步重复查邻居坐标。
        这里大量依赖宿主提供：
        - `_get_pbc_coord`：做 PBC wrap 后的邻居坐标计算
        - `_batch_get_type_from_local_coords`：给定坐标批量查询类型
        """
        local_v = np.array(self.get_vacancy_array(), dtype=np.int32)
        Nv_local = int(local_v.shape[0])
        nn1_types = np.zeros((Nv_local, 8), dtype=np.int8)
        nn2_types = np.zeros((Nv_local, 6), dtype=np.int8)
        nn1_nn1_types = np.zeros((Nv_local, 8, 8), dtype=np.int8)
        nn1_nn2_types = np.zeros((Nv_local, 8, 6), dtype=np.int8)
        if Nv_local == 0:
            self._log_comm(f"local_env_sparse rows=0/0")
            return nn1_types, nn2_types, nn1_nn1_types, nn1_nn2_types
        NN1 = self.NN1
        NN2 = self.NN2
        V_nn1_coords = self._get_pbc_coord(local_v[:, None, :], NN1[None, :, :], self.dims)
        V_nn2_coords = self._get_pbc_coord(local_v[:, None, :], NN2[None, :, :], self.dims)
        types_nn1_flat = self._batch_get_type_from_local_coords(V_nn1_coords.reshape(-1, 3)).reshape(Nv_local, 8)
        types_nn2_flat = self._batch_get_type_from_local_coords(V_nn2_coords.reshape(-1, 3)).reshape(Nv_local, 6)
        nn1_types[:, :] = types_nn1_flat
        nn2_types[:, :] = types_nn2_flat
        A_nn1_nn1_coords = self._get_pbc_coord(V_nn1_coords[:, :, None, :], NN1[None, None, :, :], self.dims)
        A_nn1_nn2_coords = self._get_pbc_coord(V_nn1_coords[:, :, None, :], NN2[None, None, :, :], self.dims)
        types_A_nn1 = self._batch_get_type_from_local_coords(A_nn1_nn1_coords.reshape(-1, 3)).reshape(Nv_local, 8, 8)
        types_A_nn2 = self._batch_get_type_from_local_coords(A_nn1_nn2_coords.reshape(-1, 3)).reshape(Nv_local, 8, 6)
        nn1_nn1_types[:, :, :] = types_A_nn1
        nn1_nn2_types[:, :, :] = types_A_nn2
        self._log_comm(f"local_env_sparse rows={Nv_local}/{Nv_local}")
        return nn1_types, nn2_types, nn1_nn1_types, nn1_nn2_types

    def _dump_local_nn_types(self, tag=None, sub_block_id=None):
        """调试工具：打印/导出本地 vacancy 的邻居类型张量。

        当前函数末尾是 `pass`，表示尚未实现真正的 dump。
        你可以在这里把 nn1/nn2/a11/a12 写到文件或打印出来做对照。
        """
        try:
            msg = f"dump nn tensors tag={tag} sub_block={sub_block_id}"
            self._log(msg)
        except Exception:
            pass
        if not hasattr(self, 'nn1_types') or not hasattr(self, 'nn2_types'):
            return
        Nv_local = int(np.array(self.get_vacancy_array(), dtype=np.int32).shape[0])
        local_ids = list(range(Nv_local))
        if len(local_ids) == 0:
            return
        global_rows = local_ids
        if len(global_rows) == 0:
            return
        nn1 = self.nn1_types[global_rows, :]
        nn2 = self.nn2_types[global_rows, :]
        a11 = self.nn1_nn1_types[global_rows, :, :] if hasattr(self, 'nn1_nn1_types') else None
        a12 = self.nn1_nn2_types[global_rows, :, :] if hasattr(self, 'nn1_nn2_types') else None
        
        pass

    def get_local_vacancy_ids(self):
        """返回当前 rank 的本地 vacancy id 列表（0..Nv_local-1）。

        注意：这里返回的是“行号意义上的本地 id”。
        如果未来启用真正的分布式划分与全局 id，需要改成返回“本 rank 持有的 vacancy 的全局 id 列表”。
        """
        Nv_local = int(np.array(self.get_vacancy_array(), dtype=np.int32).shape[0])
        return np.arange(Nv_local, dtype=int)

    def get_vacancy_neighbor_features(self, as_torch: bool = True, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """返回 vacancy 的“邻居类型”特征向量（NN1+NN2，共 14 维）。

        这函数提取的特征是什么？
        - 对每一个 vacancy（空位）位置 $v_i=(x_i,y_i,z_i)$，取它周围两层邻居：
            - NN1：8 个一阶邻居（self.NN1，形如 (8,3) 的偏移表）
            - NN2：6 个二阶邻居（self.NN2，形如 (6,3) 的偏移表）
        - 对每个邻居坐标查询该位置“是什么类型”的离散编码（type code），通常是：
            - self.FE_TYPE：Fe
            - self.CU_TYPE：Cu
            - self.V_TYPE：Vacancy
            （具体编码由宿主/基类 Lattice 约定；这里只负责批量查询并拼接。）

        输出特征的形状与顺序（非常重要）：
        - 输出为 (Nv_local, 14)
        - 每行对应一个 vacancy
        - 列的顺序固定为：
            - 0..7：按 self.NN1 的行顺序排列的 8 个 NN1 邻居的 type
            - 8..13：按 self.NN2 的行顺序排列的 6 个 NN2 邻居的 type
        因而它本质上是一个“邻域的离散类型指纹/one-hot 的前置输入”，常用于：
        - 作为 policy/value 网络的离散环境特征输入
        - 或作为能垒/速率模型的局部结构特征

        参数
        - as_torch: True 时返回 torch.Tensor；False 时返回 np.ndarray。
        - device/dtype: as_torch=True 时生效，用于指定输出张量的设备与类型。

        计算路径与缓存策略（按优先级）：
        1) 若存在 `self.nn_features_t`（已拼接好的 torch 特征），直接取并做 device/dtype 对齐。
        2) 若存在 `self.nn1_types_t`/`self.nn2_types_t`（torch 版本的分块特征），cat 拼接并对齐。
        3) 若存在 numpy 版本 `self.nn1_types`/`self.nn2_types`（通常由预计算静态环境得到），
           直接拼接为 numpy 再视需要转 torch。
        4) 若都不存在，则：
           - 读取 vacancy 坐标数组 local_pos
           - 用 `_get_pbc_coord` 计算 NN1/NN2 邻居坐标（含 PBC wrap）
           - 用 `_batch_get_type_from_local_coords` 批量查询这些邻居坐标的 type
           - 最后拼接成 (Nv_local,14)

        关于 Nv_local：
        - 这里优先用 `vac_pos_set` 的长度（O(1)）来避免构造数组开销。
        - 但某些“默认早退/尚未初始化集合”的路径下 `vac_pos_set` 可能不存在，
        """
        # Nv_local = int(np.array(self.vac_pos_set, dtype=np.int32).shape[0])
        Nv_local = int(len(self.vac_pos_set))
        if Nv_local == 0:
            if as_torch:
                dev = device if device is not None else (self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                dt = dtype if dtype is not None else torch.float32
                return torch.empty((0, 14), dtype=dt, device=dev)
            return np.empty((0, 14), dtype=np.int32)
        if as_torch:
            dev = device if device is not None else (self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            dt = dtype if dtype is not None else torch.float32
            if hasattr(self, 'nn_features_t') and isinstance(getattr(self, 'nn_features_t'), torch.Tensor):
                # 最快：直接返回缓存好的 (Nv,14) torch 特征。
                out = self.nn_features_t
                if out.device != dev:
                    out = out.to(device=dev)
                if out.dtype != dt:
                    out = out.to(dtype=dt)
                return out
            if hasattr(self, 'nn1_types_t') and hasattr(self, 'nn2_types_t') and isinstance(getattr(self, 'nn1_types_t'), torch.Tensor) and isinstance(getattr(self, 'nn2_types_t'), torch.Tensor):
                # 次快：已有分块张量 (Nv,8) 与 (Nv,6)，拼接成 (Nv,14)。
                out = torch.cat([self.nn1_types_t, self.nn2_types_t], dim=1)
                if out.device != dev:
                    out = out.to(device=dev)
                if out.dtype != dt:
                    out = out.to(dtype=dt)
                return out
            if hasattr(self, 'nn1_types') and hasattr(self, 'nn2_types'):
                # numpy 版本已存在（通常由静态环境预计算得到）；避免重复查邻居。
                nn1 = self.nn1_types
                nn2 = self.nn2_types
                nn_features = np.empty((nn1.shape[0], nn1.shape[1] + nn2.shape[1]), dtype=np.int32)
                nn_features[:, :nn1.shape[1]] = nn1
                nn_features[:, nn1.shape[1]:] = nn2
                return torch.as_tensor(nn_features, dtype=dt, device=dev)
        if hasattr(self, 'nn1_types') and hasattr(self, 'nn2_types'):
            t0 = time.time()
            nn1 = self.nn1_types
            nn2 = self.nn2_types
            t1 = time.time()
            nn_features = np.empty((nn1.shape[0], nn1.shape[1] + nn2.shape[1]), dtype=np.int32)
            nn_features[:, :nn1.shape[1]] = nn1
            nn_features[:, nn1.shape[1]:] = nn2
            t2 = time.time()
            print(f"get_vacancy_neighbor_features nn_features={nn_features.shape} t1-t0 time={t1-t0:.6f}s t2-t1 time={t2-t1:.6f}s")
            return nn_features
        # 最慢路径：即时计算。
        # 1) 收集本地 vacancy 坐标 (Nv,3)
        local_pos = np.array(self.get_vacancy_array(), dtype=np.int32)
        # 2) 通过偏移表生成邻居坐标（PBC wrap 在 _get_pbc_coord 内处理）
        #    形状：nn1_coords=(Nv,8,3), nn2_coords=(Nv,6,3)
        nn1_coords = self._get_pbc_coord(local_pos[:, None, :], self.NN1[None, :, :], self.dims)
        nn2_coords = self._get_pbc_coord(local_pos[:, None, :], self.NN2[None, :, :], self.dims)
        # 3) 批量查类型：把 (Nv,K,3) 展平成 (Nv*K,3) 查询，再 reshape 回来
        #    得到离散类型编码矩阵：nn1_types=(Nv,8), nn2_types=(Nv,6)
        nn1_types = self._batch_get_type_from_local_coords(nn1_coords.reshape(-1, 3)).reshape(Nv_local, 8)
        nn2_types = self._batch_get_type_from_local_coords(nn2_coords.reshape(-1, 3)).reshape(Nv_local, 6)
        # 4) 拼接成最终特征 (Nv,14)：[NN1_types | NN2_types]
        nn_features = np.empty((nn1_types.shape[0], nn1_types.shape[1] + nn2_types.shape[1]), dtype=np.int32)
        nn_features[:, :nn1_types.shape[1]] = nn1_types
        nn_features[:, nn1_types.shape[1]:] = nn2_types
        return nn_features

    def _batch_get_type_from_coords_local(self, positions: np.ndarray) -> np.ndarray:
        """一个“本地集合”的类型查询实现（备用）。

        给定一组坐标 positions，逐个检查：
        - 若坐标在 `vac_pos_set` 中 -> vacancy
        - 若坐标在 `cu_pos_index` 中 -> Cu
        - 否则默认 FE

        这是 Python for-loop 版本，速度慢但逻辑直观；
        实际运行中通常由宿主提供更快的 `_batch_get_type_from_coords`。
        """
        # print(f"rank {self.rank} _batch_get_type_from_coords_local positions={positions}")
        N = len(positions)
        if N == 0:
            return np.empty(0, dtype=np.int8)
        positions_int = np.round(positions).astype(np.int32)
        types = np.full(N, self.FE_TYPE, dtype=np.int8)
        vset = getattr(self, 'vac_pos_set', set())
        cindex = getattr(self, 'cu_pos_index', {})
        pts = [tuple(map(int, p)) for p in positions_int.tolist()]
        for i, tp in enumerate(pts):
            if tp in vset:
                # print(f"rank {self.rank} tp={tp} is vacancy")
                types[i] = int(self.V_TYPE)
            elif cindex.get(tp) is not None:
                types[i] = int(self.CU_TYPE)
        return types

    def _batch_get_type_from_local_coords(self, positions: np.ndarray) -> np.ndarray:
        """把“本地坐标”映射到类型编码。

        当前直接调用宿主的 `_batch_get_type_from_coords`。
        如果你想强制使用本文件的 set/index 逻辑，可切换到 `_batch_get_type_from_coords_local`。
        """
        return self._batch_get_type_from_coords(positions)
        # return self._batch_get_type_from_coords_local(positions)
        

    def _batch_vacancy_diffusion_energy(self, vacancies_index: Optional[np.ndarray] = None) -> np.ndarray:
        """批量计算 vacancy 沿 8 个 NN1 方向跳跃的扩散“能垒”(energy barrier)。

        输出 shape=(M,8)：
        - M 为参与计算的 vacancy 数（由 vacancies_index 指定；None 则使用全部本地 vacancy）。
        - 每行对应一个 vacancy，8 列对应 8 个 NN1 方向。

        计算结构（高层解释）：
        1) 取 vacancy 当前坐标 vac_local。
        2) 计算 vacancy 的 NN1/NN2 邻居坐标，并查询这些邻居的类型编码 nn1/nn2。
        3) 以 vacancy 的每个 NN1 邻居作为中心，进一步查询其 NN1/NN2 邻居类型 nn1_nn1/nn1_nn2。
        4) 使用 pair_energies 计算跳跃前后局部能量 E_before/E_after（按对相互作用累加并除以 2 去重）。
        5) delta_base = E_after - E_before。
        6) 加上与元素类型相关的基准能垒项 E_a0（只对“非 vacancy 类型”的中心 A 生效）。
        7) 对非法跳跃（目标位也是 vacancy，即 nn1==V_TYPE）置 inf。

        注意：该函数依赖宿主对 pair_energies 的组织方式：
        - pe[0]：NN1 配对能量矩阵
        - pe[1]：NN2 配对能量矩阵
        矩阵索引通常是 [type_i, type_j]。
        """
        if vacancies_index is None:
            indices = self.get_local_vacancy_ids()
        else:
            indices = np.asarray(vacancies_index, dtype=int)
        M_prime = len(indices)
        if M_prime == 0:
            return np.zeros((0, 8), dtype=float)
        NN1 = self.NN1
        NN2 = self.NN2
        vac_src = np.array(self.get_vacancy_array(), dtype=np.int32)
        vac_local = np.array(vac_src[indices], dtype=int)
        V_nn1_coords = self._get_pbc_coord(vac_local[:, None, :], NN1[None, :, :], self.dims)
        V_nn2_coords = self._get_pbc_coord(vac_local[:, None, :], NN2[None, :, :], self.dims)
        nn1 = self._batch_get_type_from_local_coords(V_nn1_coords.reshape(-1, 3)).reshape(M_prime, 8)
        nn2 = self._batch_get_type_from_local_coords(V_nn2_coords.reshape(-1, 3)).reshape(M_prime, 6)
        A_nn1_nn1_coords = self._get_pbc_coord(V_nn1_coords[:, :, None, :], NN1[None, None, :, :], self.dims)
        A_nn1_nn2_coords = self._get_pbc_coord(V_nn1_coords[:, :, None, :], NN2[None, None, :, :], self.dims)
        nn1_nn1 = self._batch_get_type_from_local_coords(A_nn1_nn1_coords.reshape(-1, 3)).reshape(M_prime, 8, 8)
        nn1_nn2 = self._batch_get_type_from_local_coords(A_nn1_nn2_coords.reshape(-1, 3)).reshape(M_prime, 8, 6)
        pe = self.pair_energies
        try:
            pair1 = pe[0]
            pair2 = pe[1]
        except Exception:
            pair1, pair2 = pe
        centerA = nn1
        centerA_rep_8 = np.repeat(centerA[:, :, None], 8, axis=2)
        centerA_rep_6 = np.repeat(centerA[:, :, None], 6, axis=2)
        E1A = pair1[centerA_rep_8, nn1_nn1].sum(axis=2)
        E2A = pair2[centerA_rep_6, nn1_nn2].sum(axis=2)
        E_A = E1A + E2A
        centerV = np.full((M_prime,), int(self.V_TYPE), dtype=np.int32)
        E1V = pair1[centerV[:, None], nn1].sum(axis=1)
        E2V = pair2[centerV[:, None], nn2].sum(axis=1)
        E_V = np.repeat((E1V + E2V)[:, None], 8, axis=1)
        E_before = (E_A + E_V) / 2.0
        V_new_nn1 = np.repeat(nn1[:, None, :], 8, axis=1)
        idx = np.arange(8)
        V_new_nn1[np.arange(M_prime)[:, None], idx[None, :], idx[None, :]] = int(self.V_TYPE)
        centerVnew = np.full((M_prime, 8), int(self.V_TYPE), dtype=np.int32)
        centerVnew_rep_8 = np.repeat(centerVnew[:, :, None], 8, axis=2)
        centerVnew_rep_6 = np.repeat(centerVnew[:, :, None], 6, axis=2)
        E1Vn = pair1[centerVnew_rep_8, V_new_nn1].sum(axis=2)
        E2Vn = pair2[centerVnew_rep_6, np.repeat(nn2[:, None, :], 8, axis=1)].sum(axis=2)
        A_new_nn1 = nn1_nn1.copy()
        kV = np.asarray(self.k_V_idx, dtype=int)
        for j in range(8):
            A_new_nn1[:, j, kV[j]] = nn1[:, j]
        centerAnew = np.full((M_prime, 8), int(self.V_TYPE), dtype=np.int32)
        centerAnew_rep_8 = np.repeat(centerAnew[:, :, None], 8, axis=2)
        centerAnew_rep_6 = np.repeat(centerAnew[:, :, None], 6, axis=2)
        E1An = pair1[centerAnew_rep_8, A_new_nn1].sum(axis=2)
        E2An = pair2[centerAnew_rep_6, nn1_nn2].sum(axis=2)
        E_after = (E2Vn + E1Vn + E1An + E2An) / 2.0
        delta_base = E_after - E_before
        Ea0 = np.asarray(self.E_a0, dtype=float)
        Ea0_mask = np.zeros((M_prime, 8), dtype=float)
        non_v = (nn1 != int(self.V_TYPE))
        if np.any(non_v):
            # 仅对非空位类型进行索引，避免越界
            Ea0_mask[non_v] = Ea0[nn1[non_v]]
        out = Ea0_mask + 0.5 * delta_base
        illegal = (nn1 == int(self.V_TYPE))
        out = np.where(illegal, np.inf, out)
        return out.astype(float)

    def calculate_diffusion_rate_for_sub_block(self, sub_block_id):
        """计算子块的扩散速率矩阵 rates（shape=(Nv_local,8)）。

        目前 sub_block_id 未参与筛选：
        - 直接对所有本地 vacancy 计算能垒
        - 使用 Arrhenius 形式：rate = 1e13 * exp(-E / (kB*T))

        kB 单位：eV/K（这里取 8.617e-5）
        T 来源：args.temperature，默认 500。
        """
        indices = self.get_local_vacancy_ids()
        # 加上全局同步 并输出对各个rank局部indices的对比
        # 比对 indices 是否一致
        # Begin
        # if self.world_size > 1 and dist.is_initialized():
        #     try:
        #         _barrier_nccl_safe()
        #     except Exception:
        #         dist.barrier()
            # 输出对比结果
            # print(f"Rank {dist.get_rank()} local indices: {indices}")

        if len(indices) == 0:
            return np.empty((0, 8), dtype=float)
        energies = self._batch_vacancy_diffusion_energy(indices)
        # print(f"Rank {dist.get_rank()} energies: {energies}")
        kB = 8.617e-5
        T = float(getattr(self.args, "temperature", 500))
        rates = 1e13 * np.exp(-energies / (kB * T))
        # rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        # rates = np.maximum(rates, 0.0)
        # print(f"Rank {dist.get_rank()} diffusion rates: {rates}")

        # if self.world_size > 1 and dist.is_initialized():
        #     try:
        #         _barrier_nccl_safe()
        #     except Exception:
        #         dist.barrier()
        # End

        return rates

    def calculate_energy_delta_for_sub_block(self, sub_block_id):
        """计算 vacancy 沿 8 个方向跳跃的能量变化 ΔE（不含基准能垒项）。

        与 `_batch_vacancy_diffusion_energy` 类似，但返回 delta_base（并对非法跳跃置 inf）。
        该函数常用于：
        - 分析/可视化能量地形
        - 或把 ΔE 单独作为特征/奖励的一部分
        """
        indices = self.get_local_vacancy_ids()
        M_prime = len(indices)
        if M_prime == 0:
            return np.zeros((0, 8), dtype=float)
        NN1 = self.NN1
        NN2 = self.NN2
        vac_src = np.array(self.get_vacancy_array(), dtype=np.int32)
        vac_local = np.array(vac_src[indices], dtype=int)
        V_nn1_coords = self._get_pbc_coord(vac_local[:, None, :], NN1[None, :, :], self.dims)
        V_nn2_coords = self._get_pbc_coord(vac_local[:, None, :], NN2[None, :, :], self.dims)
        nn1 = self._batch_get_type_from_local_coords(V_nn1_coords.reshape(-1, 3)).reshape(M_prime, 8)
        nn2 = self._batch_get_type_from_local_coords(V_nn2_coords.reshape(-1, 3)).reshape(M_prime, 6)
        A_nn1_nn1_coords = self._get_pbc_coord(V_nn1_coords[:, :, None, :], NN1[None, None, :, :], self.dims)
        A_nn1_nn2_coords = self._get_pbc_coord(V_nn1_coords[:, :, None, :], NN2[None, None, :, :], self.dims)
        nn1_nn1 = self._batch_get_type_from_local_coords(A_nn1_nn1_coords.reshape(-1, 3)).reshape(M_prime, 8, 8)
        nn1_nn2 = self._batch_get_type_from_local_coords(A_nn1_nn2_coords.reshape(-1, 3)).reshape(M_prime, 8, 6)
        pe = self.pair_energies
        try:
            pair1 = pe[0]
            pair2 = pe[1]
        except Exception:
            pair1, pair2 = pe
        centerA = nn1
        centerA_rep_8 = np.repeat(centerA[:, :, None], 8, axis=2)
        centerA_rep_6 = np.repeat(centerA[:, :, None], 6, axis=2)
        E1A = pair1[centerA_rep_8, nn1_nn1].sum(axis=2)
        E2A = pair2[centerA_rep_6, nn1_nn2].sum(axis=2)
        E_A = E1A + E2A
        centerV = np.full((M_prime,), int(self.V_TYPE), dtype=np.int32)
        E1V = pair1[centerV[:, None], nn1].sum(axis=1)
        E2V = pair2[centerV[:, None], nn2].sum(axis=1)
        E_V = np.repeat((E1V + E2V)[:, None], 8, axis=1)
        E_before = (E_A + E_V) / 2.0
        V_new_nn1 = np.repeat(nn1[:, None, :], 8, axis=1)
        idx = np.arange(8)
        V_new_nn1[np.arange(M_prime)[:, None], idx[None, :], idx[None, :]] = int(self.V_TYPE)
        centerVnew = np.full((M_prime, 8), int(self.V_TYPE), dtype=np.int32)
        centerVnew_rep_8 = np.repeat(centerVnew[:, :, None], 8, axis=2)
        centerVnew_rep_6 = np.repeat(centerVnew[:, :, None], 6, axis=2)
        E1Vn = pair1[centerVnew_rep_8, V_new_nn1].sum(axis=2)
        E2Vn = pair2[centerVnew_rep_6, np.repeat(nn2[:, None, :], 8, axis=1)].sum(axis=2)
        A_new_nn1 = nn1_nn1.copy()
        kV = np.asarray(self.k_V_idx, dtype=int)
        for j in range(8):
            A_new_nn1[:, j, kV[j]] = nn1[:, j]
        centerAnew = np.full((M_prime, 8), int(self.V_TYPE), dtype=np.int32)
        centerAnew_rep_8 = np.repeat(centerAnew[:, :, None], 8, axis=2)
        centerAnew_rep_6 = np.repeat(centerAnew[:, :, None], 6, axis=2)
        E1An = pair1[centerAnew_rep_8, A_new_nn1].sum(axis=2)
        E2An = pair2[centerAnew_rep_6, nn1_nn2].sum(axis=2)
        E_after = (E2Vn + E1Vn + E1An + E2An) / 2.0
        delta_base = E_after - E_before
        illegal = (nn1 == int(self.V_TYPE))
        delta_base = np.where(illegal, np.inf, delta_base)
        return delta_base.astype(float)

    def update_diffusion_rates_for_sub_block(self, sub_block_id, changed_positions):
        """增量更新子块的扩散速率缓存（sub_block_rates）。

        目标：当系统发生局部变化（比如一次 vacancy jump）后，只重算受影响的 vacancy 行。

        输入
        - sub_block_id: 要更新的子块 id。
        - changed_positions: 一组发生变化的坐标（通常包含 old_pos/new_pos）。

        当前实现要点：
        - 先把 changed_positions 扩展到“可能受影响的邻域坐标集合”affected_site_set
            （中心点 + NN1 + NN2）。
        - 对所有 vacancy 坐标逐个判断是否落在 affected_site_set 中。
        - 若命中，则把对应行的 8 个方向能垒重算并写回缓存。

        注意：这里混合使用了坐标->id 映射：
        - `v_pos_to_id` 被当成 vacancy 的 id 映射，但它只在完整分布式路径的 rank0 构建。
        - 默认早退路径下，如果宿主不初始化这些字段，这个函数可能无法正常工作。
        """
        vac_coords = [coord for lst in self.sub_block_vacancy_pos.values() for coord in lst]
        if not vac_coords or changed_positions is None or len(changed_positions) == 0:
            return
        dims = np.array(self.dims, dtype=int)
        offsets = np.vstack((np.array([[0, 0, 0]], dtype=int), self.NN1, self.NN2))
        changed = np.array(changed_positions, dtype=int)
        neigh_coords = self._get_pbc_coord(changed[:, None, :], offsets[None, :, :], dims)
        affected_site_set = {tuple(map(int, p)) for p in neigh_coords.reshape(-1, 3).tolist()}
        indices = self.get_local_vacancy_ids()
        # print(f"Rank {dist.get_rank()} update indices: {indices}, affected_site_set: {affected_site_set}")
        if len(indices) == 0:
            return
        vac_coords_arr = np.array(vac_coords, dtype=int)
        # print(f"Rank {dist.get_rank()} vac_coords_arr: {vac_coords_arr}")
        vac_global_arr = vac_coords_arr
        pos_map = {int(indices[k]): k for k in range(len(indices))}
        row_positions = []
        # print(f"Rank {dist.get_rank()} vac_global_arr: {vac_global_arr}")
        for i in range(vac_global_arr.shape[0]):
            gpos = tuple(map(int, vac_global_arr[i].tolist()))
            if gpos in affected_site_set:
                vid = self.v_pos_to_id.get(tuple(vac_coords_arr[i].tolist()))
                if vid is not None:
                    pos = pos_map.get(int(vid), None)
                    if pos is not None:
                        row_positions.append(int(pos))
        # print(f"Rank {dist.get_rank()} update row_positions: {row_positions}")
        
        if len(row_positions) > 0:
            local_subset = [int(indices[p]) for p in row_positions]
            energies = self._batch_vacancy_diffusion_energy(local_subset)
            kB = 8.617e-5
            T = float(getattr(self.args, "temperature", 500))
            new_rates = 1e13 * np.exp(-energies / (kB * T))
            if not hasattr(self, 'sub_block_rates') or self.sub_block_rates.get(sub_block_id) is None:
                self.sub_block_rates = getattr(self, 'sub_block_rates', {})
                self.sub_block_rates[sub_block_id] = self.calculate_diffusion_rate_for_sub_block(sub_block_id)
            base = self.sub_block_rates[sub_block_id]
            for j, ri in enumerate(row_positions): 
                # print(f"Rank {dist.get_rank()} j: {j}, ri: {ri}")
                if ri < base.shape[0]:
                    base[ri, :] = new_rates[j, :]
            # print(f"Rank {dist.get_rank()} update base: {base}")
        try:
            old_pos = tuple(map(int, changed_positions[0])) if len(changed_positions) > 0 else None
            new_pos = tuple(map(int, changed_positions[1])) if len(changed_positions) > 1 else None
            # print(f"Rank {dist.get_rank()} 1151 old_pos: {old_pos}, new_pos: {new_pos}")
            # if old_pos is not None:
            #     print(f"Rank {dist.get_rank()} 1153 old_pos: {old_pos}")
            #     local_old = tuple(self.global_to_local_coord(old_pos))
            #     if self._is_pure_local(local_old):
            #         old_sb = self._coord_to_sub_id(local_old)
            #         print(f"Rank {dist.get_rank()} 1157 old_sb: {old_sb}")
            #         idx_old = self.get_vacancy_indices_in_sub_block(old_sb)
            #         # lid_old = self.local_vac_pos_to_id.get(tuple(map(int, local_old)))
            #         lid_old = self.local_vac_pos_to_id.get(tuple(map(int, local_new)))
                    
            #         if lid_old is not None:
            #             pos_map_old = {int(idx_old[k]): k for k in range(len(idx_old))}
            #             pos_old = pos_map_old.get(int(lid_old), None)
            #             cached = getattr(self, 'sub_block_rates', {})
            #             base_old = cached.get(old_sb, None)
            #             if base_old is not None and pos_old is not None and 0 <= pos_old < base_old.shape[0]:
            #                 self.sub_block_rates[old_sb] = np.delete(base_old, pos_old, axis=0)
            if new_pos is not None:
                # print(f"Rank {dist.get_rank()} update new_pos: {new_pos}")
                local_new = tuple(map(int, new_pos))
                # print(f"Rank {dist.get_rank()} update local_new: {local_new}")
                new_sb = self._coord_to_sub_id(local_new)
                idx_new = self.get_local_vacancy_ids()
                lid_new = self.v_pos_to_id.get(tuple(map(int, local_new)))
                if lid_new is not None:
                    pos_map_new = {int(idx_new[k]): k for k in range(len(idx_new))}
                    pos_new = pos_map_new.get(int(lid_new), None)
                    cached = getattr(self, 'sub_block_rates', {})
                    base_new = cached.get(new_sb, None)
                    if base_new is not None and pos_new is not None:
                        e_new = self._batch_vacancy_diffusion_energy([int(lid_new)])
                        kB2 = 8.617e-5
                        T2 = float(getattr(self.args, "temperature", 500))
                        row_new = 1e13 * np.exp(-(e_new) / (kB2 * T2))
                        M_new = len(idx_new)
                        out = np.zeros((M_new, 8), dtype=float)
                        j_old = 0
                        for i in range(M_new):
                            if i == int(pos_new):
                                out[i, :] = row_new[0, :]
                            else:
                                out[i, :] = base_new[j_old, :]
                                j_old += 1
                        self.sub_block_rates[new_sb] = out
        except Exception:
            pass

    def get_sub_block_propensity_info(self, sub_block_id):
        """返回子块总倾向率（propensity）与 vacancy 数量。

        - total_rate: 子块中所有 vacancy、所有方向的 rates 之和
        - count: vacancy 数量

        会优先使用缓存 `self.sub_block_rates[sub_block_id]`；没有则计算并缓存。
        同时会把 NaN/inf 清理为 0，并截断为非负。
        """
        indices = self.get_local_vacancy_ids()
        if len(indices) == 0:
            self._log(f"propensity {sub_block_id} empty")
            return 0.0, 0
        # 优先使用缓存的子块速率
        cached = getattr(self, 'sub_block_rates', None)
        if cached is not None and (sub_block_id in cached) and cached[sub_block_id] is not None:
            rates = cached[sub_block_id]
        else:
            rates = self.calculate_diffusion_rate_for_sub_block(sub_block_id)
            self.sub_block_rates = getattr(self, 'sub_block_rates', {})
            self.sub_block_rates[sub_block_id] = rates
        if rates.size > 0:
            rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
            rates = np.maximum(rates, 0.0)
        total = float(np.sum(rates)) if rates.size > 0 else 0.0
        count = int(len(indices))
        self._log(f"propensity {sub_block_id} total_rate={total} count={count}")
        return total, count

    def step_for_sub_block(self, sub_block_id, action, episode):
        """对子块执行一步环境交互（RL step 接口的一个分块版本）。

        action 编码方式：
        - action 被展开成 (vac_local, dir_idx)
            - vac_local: 选择第几个本地 vacancy（在 indices 列表中的位置）
            - dir_idx: 0..7，选择 NN1 的哪个方向

        执行策略：
        - 若 args.use_full_step_local=True：调用宿主的 `step_local`（通常更完整、更慢）。
        - 否则调用 `step_fast_local`（通常更快，返回更少信息）。

        返回值约定：
        - obs: 观测（可能是 dict 或 ndarray，取决于宿主）
        - full_obs: 展平后的观测（如果 skip_stats=False 会拼上系统统计量）
        - positions: vacancy 坐标数组
        - reward, done, infos: RL 常见返回

        注意：这里的 reward/done 目前固定返回 0.0/False；时间增量也固定为 0.0。
        说明该函数更像是“动作执行通道”，奖励/终止逻辑可能在外部控制。
        """
        indices = self.get_local_vacancy_ids()
        if len(indices) == 0:
            obs = self.get_vacancy_neighbor_features()
            full_obs = obs.flatten()
            return obs, full_obs, np.array([]), 0.0, False, {"error": "empty_sub_block"}
        vac_local, dir_idx = divmod(action, 8)
        if vac_local < 0 or vac_local >= len(indices):
            obs = self.get_vacancy_neighbor_features()
            full_obs = obs.flatten()
            return obs, full_obs, np.array([]), 0.0, False, {"error": "invalid_action"}
        vac_local_id = int(indices[vac_local])
        use_full = bool(getattr(self.args, "use_full_step_local", False))
        if use_full:
            return self.step_local(vac_local_id, int(dir_idx), episode)
        obs = self.step_fast_local(vac_local_id, int(dir_idx), episode)
        if bool(getattr(self.args, "skip_stats", False)):
            full_obs = obs.get('V_features_local', np.array([], dtype=float)).flatten()
        else:
            full_obs = np.concatenate([obs.get('V_features_local', np.array([], dtype=float)).flatten(), self.get_system_stats()], axis=0)
        positions = self.get_vacancy_array()
        infos = {"delta_t": float(0.0), "time": float(self.time)}
        return obs, full_obs, positions, 0.0, False, infos

    def step_fast_for_sub_block(self, sub_block_id, action, episode):
        """兼容接口：忽略 sub_block_id，直接走 step_for_sub_block。"""
        return self.step_for_sub_block(None, action, episode)

    # 兼容旧名
    def step_only_jump_for_sub_block(self, sub_block_id, action, episode):
        """兼容旧名：历史版本可能用该函数名表达“只做 jump”。"""
        return self.step_fast_for_sub_block(sub_block_id, action, episode)

    def local_to_global_coord(self, local_pos):
        """把本地坐标(可迭代)规范化为全局坐标 tuple[int,int,int]。

        目前为恒等映射；若启用域分解，应在此处加偏移。
        """
        p = np.array(local_pos, dtype=int)
        return tuple(map(int, p))

def _barrier_nccl_safe():
    """在 NCCL + CUDA 下更安全的 barrier。

    背景：
    - `dist.barrier()` 在 NCCL 后端通常期望 GPU 参与。
    - 某些版本/环境下，如果不传 device_ids，可能出现 hang 或报错。

    策略：
    - 若 backend 是 nccl 且有 cuda，则尝试 `dist.barrier(device_ids=[current_device])`。
    - 失败或非 nccl 时回退到普通 `dist.barrier()`。
    """
    if not dist.is_initialized():
        return
    try:
        b = str(dist.get_backend()).lower()
    except Exception:
        b = str(env_str(EnvKeys.DIST_BACKEND, "gloo") or "gloo").lower()
    if b == "nccl" and torch.cuda.is_available():
        try:
            dev_id = torch.cuda.current_device()
            dist.barrier(device_ids=[dev_id])
            return
        except Exception:
            pass
    dist.barrier()
