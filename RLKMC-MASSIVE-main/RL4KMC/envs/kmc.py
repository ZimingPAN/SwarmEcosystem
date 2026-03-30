from __future__ import annotations

import torch
from RL4KMC.envs.lattice import Lattice
from RL4KMC.plot.plotter import Plotter
from RL4KMC.embedding import SGDNTC_Model as EmbedClass
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
import os
import time
# from RL4KMC.utils.vacancy_topk_system import VacancyTopKSystem
# from RL4KMC.utils.vacancy_topk_system import VacancyTopKSystem
# from RL4KMC.utils.vacancy_topk_system import TensorCoreVacancyTopKSystem as VacancyTopKSystem
from RL4KMC.utils.vacancy_topk_system import AdaptiveVacancyTopK as VacancyTopKSystem
# import torch.distributed as dist


@dataclass(frozen=True)
class KMCJumpUpdate:
    vac_idx: int
    dir_idx: int
    old_pos: tuple[int, int, int]
    new_pos: tuple[int, int, int]
    moving_type: int
    updated_cu: Optional[dict[int, np.ndarray]]
    updated_vacancy: Optional[dict[int, np.ndarray]]
    topk_update_info: Optional[dict[str, Any]]
    cu_move_from: Optional[tuple[int, int, int]]
    cu_move_to: Optional[tuple[int, int, int]]
    cu_id: Optional[int]




def _calcul_dene_v1_ppair2_vec(it_arr: np.ndarray, it2_arr: np.ndarray, nn: int, sign: float, pair_energies: list, V_type: int = 2) -> np.ndarray:
    try:
        # 获取势能矩阵 V_nn (NSPECIES x NSPECIES)
        vij_nn = pair_energies[nn - 1] 
    except IndexError:
        return np.zeros_like(it_arr, dtype=float)

    # 核心：使用 NumPy 高级索引 (Advanced Indexing)
    # it_arr 和 it2_arr 必须是相同长度的整数数组
    # 这会返回一个与 it_arr/it2_arr 形状相同的数组
    vij_new = vij_nn[it_arr, it2_arr]
    vij_old = vij_nn[V_type, it2_arr]
    
    # 返回 M 个修正项的数组
    return 0.5 * sign * (vij_new - vij_old)


def _calcul_dene_v2_ppair2_vec(it_arr: np.ndarray, it2_arr: np.ndarray, nn1: int, nn2: int, pair_energies: list, V_type: int = 2) -> np.ndarray:
    try:
        vij_nn1 = pair_energies[nn1 - 1] 
        vij_nn2 = pair_energies[nn2 - 1] 
    except IndexError:
        return np.zeros_like(it_arr, dtype=float)

    # 核心：使用 NumPy 高级索引 (Advanced Indexing)
    term_A_N_nn1 = vij_nn1[it_arr, it2_arr]
    term_A_N_nn2 = vij_nn2[it_arr, it2_arr]
    term_V_N_nn1 = vij_nn1[V_type, it2_arr]
    term_V_N_nn2 = vij_nn2[V_type, it2_arr]
    
    # 返回 M 个修正项的数组
    return 0.5 * ( term_A_N_nn1 - term_A_N_nn2 - term_V_N_nn1 + term_V_N_nn2 )

def _calcul_dene_v1_ppair2_torch(it_arr_t: torch.Tensor, it2_arr_t: torch.Tensor, nn: int, sign: float, pair_energies_t: tuple[torch.Tensor, torch.Tensor], V_type: int = 2) -> torch.Tensor:
    """
    向量化计算 LAKIMOC 修正项 V1_ppair2 (GPU)
    it_arr_t, it2_arr_t: (M,) Long Tensors (原子类型)
    pair_energies_t: (E1, E2) 包含键能矩阵 (T, T)
    """
    try:
        # pair_energies_t 假设是一个包含 (8, 8) 键能矩阵的 Tuple，索引从 0 开始
        # nn=1 对应索引 0
        vij_nn_t = pair_energies_t[nn - 1] 
    except IndexError:
        # 如果索引超出范围，返回 M 个 0
        return torch.zeros_like(it_arr_t, dtype=torch.float32)

    # 核心：使用 PyTorch 高级索引 (Indexing with Long Tensors)
    # 它会返回一个与 it_arr_t/it2_arr_t 形状相同的 Tensor
    
    # 转换为 Long Tensor 进行索引
    it_long = it_arr_t.long()
    it2_long = it2_arr_t.long()

    # V_new = V(it, it2)
    vij_new_t = vij_nn_t[it_long, it2_long]
    # V_old = V(V, it2)
    vij_old_t = vij_nn_t[V_type, it2_long]

    # 返回 M 个修正项的数组 (GPU Tensor)
    return 0.5 * sign * (vij_new_t - vij_old_t)

def _calcul_dene_v2_ppair2_torch(it_arr_t: torch.Tensor, it2_arr_t: torch.Tensor, nn1: int, nn2: int, pair_energies_t: tuple[torch.Tensor, torch.Tensor], V_type: int = 2) -> torch.Tensor:
    """
    向量化计算 LAKIMOC 修正项 V2_ppair2 (GPU)
    """
    try:
        # nn1 和 nn2 都是 1 或 2，对应 pair_energies_t 的索引 0 或 1
        vij_nn1_t = pair_energies_t[nn1 - 1] 
        vij_nn2_t = pair_energies_t[nn2 - 1] 
    except IndexError:
        return torch.zeros_like(it_arr_t, dtype=torch.float32)

    it_long = it_arr_t.long()
    it2_long = it2_arr_t.long()
    
    # term_A_N_nn1 = V(it, it2) in nn1
    term_A_N_nn1_t = vij_nn1_t[it_long, it2_long]
    # term_A_N_nn2 = V(it, it2) in nn2
    term_A_N_nn2_t = vij_nn2_t[it_long, it2_long]
    
    # term_V_N_nn1 = V(V, it2) in nn1
    term_V_N_nn1_t = vij_nn1_t[V_type, it2_long]
    # term_V_N_nn2 = V(V, it2) in nn2
    term_V_N_nn2_t = vij_nn2_t[V_type, it2_long]
    
    # 返回 M 个修正项的数组
    return 0.5 * (term_A_N_nn1_t - term_A_N_nn2_t - term_V_N_nn1_t + term_V_N_nn2_t)


class KMC(Lattice):
    def __init__(self, args):
        super().__init__(args)
        # pair energies, E_a0 etc (保持你原来的定义)
        
        self.pair_energies = self.init_pair_energies()
        # self.pair_energies = self.init_pair_energies()
        self.pair_energies_t = torch.tensor(
                self.pair_energies, 
                device=self.device
            )
        # print(f"[env] init: pair_energies_t dtype={self.pair_energies_t.dtype} device={self.pair_energies_t.device}", flush=True)
        self.E_a0 = np.array([0.65, 0.56, 0], dtype=np.float32)
        self.time = 0.0
        self.energy_history = []
        self.time_history = []
        self.plotter = Plotter()
        
        self.embed = EmbedClass(args, device=self.device)
        self.args = args
        self.topk = getattr(args, "topk", 16)
        
        # 预计算 k_V_idx (NumPy 数组)
        k_V_idx = np.zeros(8, dtype=int)
        # 假设 self.NN1 是 NN1 偏移向量的 NumPy 数组
        for j in range(8):
            for k in range(8):
                if np.all(self.NN1[k] == -self.NN1[j]): 
                    k_V_idx[j] = k
                    break
                 
        self.dims_t = torch.tensor(
                self.dims, 
                device=self.device
            )
        # print(f"[env] init: dims_t={self.dims_t.tolist()} device={self.dims_t.device}", flush=True)
   
        # ⭐ 赋值给 self 属性
        self.k_V_idx = k_V_idx
        
        # print(f"[env] _init_vacancy_mappings device={self.dims_t.device}", flush=True)

        # ---------- 新增：vacancy id <-> index 映射 ----------
        # 这里我们把 vacancy_id 定义为对初始 vacancy_pos 的 hash/编号（0..N_v-1）
        # 映射在 init_lattice() 后初始化（Lattice.init_lattice 已经设置 self.vacancy_pos）
        self.vac_index_to_id = {}  # index -> id  (通常就是 identity mapping unless you want custom ids)
        self.vac_id_to_index = {}  # id -> index
        self._init_vacancy_mappings()
        # print(f"[env] _init_vacancy_mappings End device={self.dims_t.device}", flush=True)

        # 存储 diffusion energies (Nv x 8)
        self.diffusion_energies = None
        # 可选：初始化阶段跳过全局扩散能计算（分布式场景下改用子块速率）
        if not bool(getattr(self.args, "skip_global_diffusion_init", True)):
            self.calculate_diffusion_energy()


    # ----------------------------
    # vacancy mapping helpers
    # ----------------------------
    def _init_vacancy_mappings(self):
        """ 初始化 vacancy index<->id 映射（在 init_lattice 后调用） """
        Nv = len(self.vac_pos_set)
        # 这里简单使用连续 id，与 index 相同；也可以用 hash(tuple(pos)) 生成稳定 id
        self.vac_index_to_id = {idx: idx for idx in range(Nv)}
        self.vac_id_to_index = {vid: idx for idx, vid in self.vac_index_to_id.items()}
        

    def _update_vacancy_mappings(self):
        """
        当 self.vacancy_pos 顺序改变但数量不变时调用；
        如果 vacancy 的数量发生变化（极少情况），建议重新调用 _init_vacancy_mappings().
        """
        Nv = len(self.vac_pos_set)
        # 保证映射长度与 vacancy_pos 一致 —— 如果不同长度我们重新初始化 id
        if len(self.vac_index_to_id) != Nv:
            self._init_vacancy_mappings()
            return
        # 否则保持 id 与 index 一致（这里假设 vacancy 列表始终按 index 序）
        self.vac_index_to_id = {idx: self.vac_index_to_id.get(idx, idx) for idx in range(Nv)}
        self.vac_id_to_index = {vid: idx for idx, vid in self.vac_index_to_id.items()}

    # ----------------------------
    # 保持你原来的 pair energies & atomic energy 计算
    # ----------------------------
    def init_pair_energies(self):
        Fe, Cu, V = self.FE_TYPE, self.CU_TYPE, self.V_TYPE
        pair_energies = np.zeros((3, 3, 3))  # 2阶 * 3元素 * 3元素

        # ----------- 一阶近邻 e(1) ----------
        pair_energies[0][Fe][Fe] = -1.072761
        pair_energies[0][Cu][Cu] = -0.873238
        pair_energies[0][V][V] = 0.200000

        pair_energies[0][Fe][Cu] = pair_energies[0][Cu][Fe] = -0.858468
        pair_energies[0][Fe][V]  = pair_energies[0][V][Fe]  = -0.336381
        pair_energies[0][Cu][V]  = pair_energies[0][V][Cu]  = -0.282087

        # ----------- 二阶近邻 e(2) ----------
        pair_energies[1][Fe][Fe] = 0
        pair_energies[1][Cu][Cu] = 0
        pair_energies[1][V][V] = 0

        pair_energies[1][Fe][Cu] = pair_energies[1][Cu][Fe] = 0
        pair_energies[1][Fe][V]  = pair_energies[1][V][Fe]  = 0
        pair_energies[1][Cu][V]  = pair_energies[1][V][Cu]  = 0
        
        # ----------- 三阶近邻 e(3) ----------
        pair_energies[2][Fe][Fe] = 0
        pair_energies[2][Cu][Cu] = 0
        pair_energies[2][V][V] = 0

        pair_energies[2][Fe][Cu] = pair_energies[2][Cu][Fe] = 0
        pair_energies[2][Fe][V]  = pair_energies[2][V][Fe]  = 0
        pair_energies[2][Cu][V]  = pair_energies[2][V][Cu]  = 0

        return pair_energies




    # def calculate_system_energy(self):
    #     nx, ny, nz = self.lattice.shape
    #     coords = self.coords 

    #     nn1_coords = (coords[:, None, :] + self.NN1[None, :, :]) % [nx, ny, nz]
    #     nn2_coords = (coords[:, None, :] + self.NN2[None, :, :]) % [nx, ny, nz]

    #     center_types = self.lattice[tuple(coords.T)]  # [N_atoms]
    #     neighbor_types_1 = self.lattice[tuple(nn1_coords.transpose(2, 0, 1))]  # [N_atoms, 8]
    #     neighbor_types_2 = self.lattice[tuple(nn2_coords.transpose(2, 0, 1))]  # [N_atoms, 6]

    #     E1 = self.pair_energies[0][center_types[:, None], neighbor_types_1]  # [N_atoms, 8]
    #     E2 = self.pair_energies[1][center_types[:, None], neighbor_types_2]  # [N_atoms, 6]

    #     total_energy = (np.sum(E1) + np.sum(E2)) / 2.0
    #     return total_energy

    def calculate_system_energy(self):
        if not hasattr(self, "coords") or self.coords is None:
            return 0.0

        nx, ny, nz = self.dims
        coords = self.coords

        dims = np.array([nx, ny, nz], dtype=np.int32)
        nn1_coords = self._get_pbc_coord(coords[:, None, :], self.NN1[None, :, :], dims)
        nn2_coords = self._get_pbc_coord(coords[:, None, :], self.NN2[None, :, :], dims)

        center_types = self._batch_get_type_from_coords(coords)
        neighbor_types_1 = self._batch_get_type_from_coords(nn1_coords.reshape(-1, 3)).reshape(len(coords), len(self.NN1))
        neighbor_types_2 = self._batch_get_type_from_coords(nn2_coords.reshape(-1, 3)).reshape(len(coords), len(self.NN2))

        E1 = self.pair_energies[0][center_types[:, None], neighbor_types_1]
        E2 = self.pair_energies[1][center_types[:, None], neighbor_types_2]

        return float((np.sum(E1) + np.sum(E2)) / 2.0)



    def _get_side_nn1_correction_vec(self, vac_r_coord: np.ndarray, d_offset: np.ndarray, it_arr: np.ndarray) -> np.ndarray:
        """
        向量化计算 M 个空位（针对单个跳跃方向 d_offset）的 6 个侧向 1NN 修正项。
        该函数已修改为使用稀疏查找 (self._batch_get_type_from_coords)。

        Args:
            vac_r_coord: (M, 3) 当前批次空位的坐标。
            d_offset: (3,) 当前跳跃方向的 1NN 偏移向量。
            it_arr: (M,) 当前跳跃原子 A_j 的类型数组。
            
        Returns:
            (M,) 侧向修正项的总和。
        """
        r_V_coord = vac_r_coord
        M = len(vac_r_coord) 
        
        # 修正项初始化为一个 M 长度的零数组
        correction = np.zeros(M, dtype=float)
        
        
        # 假设 self.dims 存储着晶格尺寸 (Nx, Ny, Nz)
        dims = self.dims 

        # 使用类方法进行 PBC 坐标计算

        # 6 个侧向偏移量
        offsets = [
            np.array([-d_offset[0], d_offset[1], d_offset[2]]),   # C(2) 类型
            np.array([d_offset[0], -d_offset[1], -d_offset[2]]),  # C(3) 类型
            np.array([d_offset[0], d_offset[1], -d_offset[2]]),   # C(2') 类型
            np.array([-d_offset[0], -d_offset[1], d_offset[2]]),  # C(3') 类型
            np.array([d_offset[0], -d_offset[1], d_offset[2]]),   # C(2'') 类型
            np.array([-d_offset[0], d_offset[1], -d_offset[2]])   # C(3'') 类型
        ]
        
        # 对应的 ppair2_vec 参数 nn1, nn2
        nn_pairs = [
            (1, 2), (1, 3), (1, 2), 
            (1, 3), (1, 2), (1, 3)
        ]

        # 循环遍历 6 个修正项
        for offset, (nn1, nn2) in zip(offsets, nn_pairs):
            # 1. 计算 M 个空位的 M 个邻居坐标 r_N (形状: (M, 3))
            r_N = self._get_pbc_coord(r_V_coord, offset, dims)
            
            
            # r_N 是 (M, 3) 形状，直接传入批量查找函数
            it2_arr = self._batch_get_type_from_coords(r_N) # (M,)
            
            # 3. 调用向量化函数计算 M 个修正项
            correction +=_calcul_dene_v2_ppair2_vec(it_arr, it2_arr, nn1, nn2, self.pair_energies, self.V_TYPE)

        return correction
    
    # 假设这个函数是一个类方法
    def _get_side_nn1_correction_torch(self, vac_r_coord_t: torch.Tensor, d_offset_t: torch.Tensor, it_arr_t: torch.Tensor) -> torch.Tensor:
        """
        向量化计算 M 个空位（针对单个跳跃方向 d_offset）的 6 个侧向 1NN 修正项 (GPU Tensor 版本)。

        Args:
            vac_r_coord_t: (M, 3) 当前批次空位的坐标 (Float/Int Tensor)。
            d_offset_t: (3,) 当前跳跃方向的 1NN 偏移向量 (Float/Int Tensor)。
            it_arr_t: (M,) 当前跳跃原子 A_j 的类型数组 (Int/Long Tensor)。
                
        Returns:
            (M,) 侧向修正项的总和 (Float Tensor)。
        """
        device = vac_r_coord_t.device
        M = vac_r_coord_t.shape[0] 
        correction_t = torch.zeros(M, dtype=torch.float32, device=device)
        dims_t = self.dims_t

        # --- 1. 定义 PBC 几何操作 (内联 PyTorch) ---
        def get_pbc_coord_torch(r_base_t, delta_t):
            # r_base_t: (M, 3)
            # delta_t: (3,)
            
            # 结果 r_N: (M, 3)
            
            # 使用 PyTorch 替代 NumPy 的模运算和加法
            r_N = r_base_t + delta_t
            
            # 确保 r_N 上的 dtype 和 dims_t 上的 dtype 兼容
            if dims_t.dtype != r_base_t.dtype:
                dims_t_cast = dims_t.to(r_base_t.dtype)
            else:
                dims_t_cast = dims_t
                
            # 周期性边界条件 (r_N % dims)
            # 对于坐标，使用 fmod 或直接的取模运算
            return torch.fmod(r_N, dims_t_cast)

        # --- 2. 计算 6 个侧向偏移量 (完全使用 Tensor，避免小张量频繁构造) ---
        side_signs = torch.tensor([
            [-1,  1,  1],
            [ 1, -1, -1],
            [ 1,  1, -1],
            [-1, -1,  1],
            [ 1, -1,  1],
            [-1,  1, -1],
        ], dtype=d_offset_t.dtype, device=device)  # (6,3)
        offsets_matrix_t = side_signs * d_offset_t.unsqueeze(0)  # (6,3)

        # 对应的 ppair2_vec 参数 (Python List to facilitate loop)
        nn_pairs = [
            (1, 2), (1, 3), (1, 2), 
            (1, 3), (1, 2), (1, 3)
        ]
        
        # --- 3. 循环遍历 6 个修正项 (内部矢量化) ---
        
        # 一次性计算 6 个侧向邻居坐标 (M,6,3)
        r_N_all_t = get_pbc_coord_torch(vac_r_coord_t.unsqueeze(1), offsets_matrix_t)
        coords_flat_t = r_N_all_t.reshape(-1, 3)
        use_global_gpu = torch.cuda.is_available() and hasattr(self, '_global_vac_lin_sorted_t') and hasattr(self, '_global_cu_lin_sorted_t')
        if use_global_gpu:
            D64 = self.dims_t.to(torch.int64)
            c64 = coords_flat_t.to(torch.int64)
            lin_t = (((c64[:,0] * D64[1]) + c64[:,1]) * D64[2] + c64[:,2])
            types_t = torch.full((c64.shape[0],), int(self.FE_TYPE), dtype=torch.int32, device=device)
            if getattr(self, '_global_vac_lin_sorted_t', None) is not None and self._global_vac_lin_sorted_t.numel() > 0:
                idx_v = torch.searchsorted(self._global_vac_lin_sorted_t, lin_t)
                in_range_v = idx_v < self._global_vac_lin_sorted_t.numel()
                idx_vc = torch.clamp(idx_v, 0, max(0, int(self._global_vac_lin_sorted_t.numel()-1)))
                mask_v = in_range_v & (self._global_vac_lin_sorted_t[idx_vc] == lin_t)
                types_t[mask_v] = int(self.V_TYPE)
            if getattr(self, '_global_cu_lin_sorted_t', None) is not None and self._global_cu_lin_sorted_t.numel() > 0:
                idx_c = torch.searchsorted(self._global_cu_lin_sorted_t, lin_t)
                in_range_c = idx_c < self._global_cu_lin_sorted_t.numel()
                idx_cc = torch.clamp(idx_c, 0, max(0, int(self._global_cu_lin_sorted_t.numel()-1)))
                mask_c = in_range_c & (self._global_cu_lin_sorted_t[idx_cc] == lin_t)
                types_t[mask_c] = int(self.CU_TYPE)
            it2_all_t = types_t.to(torch.int8).view(M, 6)
        else:
            local_np = coords_flat_t.cpu().numpy()
            if hasattr(self, '_batch_get_type_from_local_coords'):
                it2_all_np = self._batch_get_type_from_local_coords(local_np)
            elif hasattr(self, '_batch_get_type_from_coords_local'):
                it2_all_np = self._batch_get_type_from_coords_local(local_np)
            else:
                it2_all_np = self._batch_get_type_from_coords(local_np)
            it2_all_t = torch.tensor(it2_all_np, device=device, dtype=torch.int8).view(M, 6)

        # 注意：这里仍然有 6 次循环，但 CPU/GPU 往返仅发生一次
        for i, (nn1, nn2) in enumerate(nn_pairs):
            it2_arr_t = it2_all_t[:, i]
            correction_t += _calcul_dene_v2_ppair2_torch(
                it_arr_t,
                it2_arr_t,
                nn1,
                nn2,
                self.pair_energies_t,
                self.V_TYPE,
            )

        return correction_t

    # def _get_side_nn1_correction_vec(self, vac_r_coord: np.ndarray, d_offset: np.ndarray, it_arr: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    #     """
    #     计算 M 个空位同时跳跃的 6 个侧向 1NN 修正项的向量化版本。
    #     """
    #     r_V_coord = vac_r_coord
    #     # M 为当前批次的空位数
    #     M = len(vac_r_coord) 
        
    #     # 修正项初始化为一个 M 长度的零数组
    #     correction = np.zeros(M, dtype=float)
        
    #     dims = np.array(self.lattice.shape)
    #     # 确保 get_pbc_coord 函数能够处理 (M, 3) 形状的数组
    #     def get_pbc_coord(r_base, delta):
    #         # r_base: (M, 3)
    #         # delta: (3,) 或 (M, 3) - 这里 d_offset 是 (3,)，NumPy 会自动广播
    #         return (r_base + delta) % dims

    #     # 所有偏移量（6个）
    #     offsets = [
    #         np.array([-d_offset[0], d_offset[1], d_offset[2]]),   # C(2)
    #         np.array([d_offset[0], -d_offset[1], -d_offset[2]]),  # C(3)
    #         np.array([d_offset[0], d_offset[1], -d_offset[2]]),   # C(2')
    #         np.array([-d_offset[0], -d_offset[1], d_offset[2]]),  # C(3')
    #         np.array([d_offset[0], -d_offset[1], d_offset[2]]),   # C(2'')
    #         np.array([-d_offset[0], d_offset[1], -d_offset[2]])   # C(3'')
    #     ]
        
    #     # 对应的 nn1, nn2 对
    #     nn_pairs = [
    #         (1, 2), (1, 3), (1, 2), 
    #         (1, 3), (1, 2), (1, 3)
    #     ]

    #     # 循环遍历 6 个修正项
    #     for offset, (nn1, nn2) in zip(offsets, nn_pairs):
    #         # 1. 计算 M 个空位的 M 个邻居坐标 r_N (形状: (M, 3))
    #         r_N = get_pbc_coord(r_V_coord, offset)
            
    #         # 2. 从晶格中获取 M 个邻居的物种类型 it2 (形状: (M,))
    #         # 使用 r_N.T 来正确索引 3D 晶格数组
    #         it2_arr = lattice[tuple(r_N.T)]
            
    #         # 3. 调用向量化函数计算 M 个修正项
    #         correction += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr, nn1, nn2, self.pair_energies)

    #     return correction

    def _calculate_E_V_new_torch(self, M_prime: int, M_calc: int, nn1_types_t: torch.Tensor, nn2_types_t: torch.Tensor, pair_energies_t: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        计算交换后新空位 (V_new) 的键能 E_V_new。
        """
        
        
        # 1. V_new (原V位) 的中心类型：原V位的NN1原子类型 (A_j)
        center_V_new_t = nn1_types_t.contiguous().view(-1) # (M'*8,)
        
        # 2. V_new 的 1NN 类型:
        # V_new 1NN 类型: 复制 V 的原始 1NN 列表 (nn1_types_t)，但将 A_j 的位置 (k=j) 变为 V
        V_new_nn1_types_t = nn1_types_t.unsqueeze(1).repeat(1, 8, 1).clone() # (M', 8方向, 8NN)
        
        # 使用 scatter_ 进行替换 V_new_nn1_types_t[i, j, j] = V
        j_indices = torch.arange(8, device=device)
        k_dim_indices = j_indices.unsqueeze(0).unsqueeze(-1).repeat(M_prime, 1, 1)
        
        V_value_t = torch.full(k_dim_indices.shape, self.V_TYPE, dtype=nn1_types_t.dtype, device=device)
        
        V_new_nn1_types_t.scatter_(
            dim=2, 
            index=k_dim_indices, 
            src=V_value_t
        ) 
        
        nn1_V_new_t = V_new_nn1_types_t.view(M_calc, 8)
        
        # 3. V_new 的 2NN 类型: 不变
        nn2_V_new_t = nn2_types_t.unsqueeze(1).repeat(1, 8, 1).view(M_calc, 6)
        
        # 4. 计算键能
        E_V_new_t = self._batch_env_energy_torch(
            center_V_new_t, 
            nn1_V_new_t, 
            nn2_V_new_t, 
            pair_energies_t
        ).view(M_prime, 8)
        
        # 释放临时 Tensor
        del center_V_new_t, V_new_nn1_types_t, k_dim_indices, V_value_t, nn1_V_new_t, nn2_V_new_t
        # 注意：nn1_types_t 和 nn2_types_t 是从主函数传入的，不应在此处释放
        
        return E_V_new_t

    # --- 2. _calculate_E_A_new_torch ---

    def _calculate_E_A_new_torch(self, M_prime: int, M_calc: int, nn1_types_t: torch.Tensor, nn1_nn1_types_t: torch.Tensor, nn1_nn2_types_t: torch.Tensor, pair_energies_t: tuple[torch.Tensor, torch.Tensor], k_V_idx_t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        计算交换后新原子 (A_new) 的键能 E_A_new。
        """
        
        
        # 1. A_new (原A位) 的中心类型：现在 A_new 处是空位 V
        center_A_new_t = torch.full((M_prime, 8), self.V_TYPE, dtype=torch.int8, device=device).view(-1) 
        
        # 2. A_new 的 1NN 类型: 复制 A 的原始 1NN 列表 (nn1_nn1_types_t)，但将 V 的位置 (k_V) 变为 A_j
        A_new_nn1_types_t = nn1_nn1_types_t.clone() # (M', 8, 8)
        A_j_types_t = nn1_types_t # (M', 8)
        
        k_V_idx = k_V_idx_t.cpu().numpy() # ⚠️ 几何查找通常在 CPU 上更快，但索引使用 GPU
        
        for j in range(8):
            k_V = k_V_idx[j]
            # A_new 1NN 列表中，索引 k_V 上的类型从 V 变为 A_j (原跳跃原子类型)
            A_new_nn1_types_t[:, j, k_V] = A_j_types_t[:, j] 

        nn1_A_new_t = A_new_nn1_types_t.view(M_calc, 8)
        
        # 3. A_new 的 2NN 类型: 不变
        nn2_A_new_t = nn1_nn2_types_t.view(M_calc, 6)
        
        # 4. 计算键能
        E_A_new_t = self._batch_env_energy_torch(
            center_A_new_t, 
            nn1_A_new_t, 
            nn2_A_new_t, 
            pair_energies_t
        ).view(M_prime, 8)
        
        # 释放临时 Tensor
        del center_A_new_t, A_new_nn1_types_t, nn1_A_new_t, nn2_A_new_t
        
        return E_A_new_t
    
    def _batch_env_energy_torch( self,
        center_type_t: torch.Tensor,
        nn1_types_t: torch.Tensor,
        nn2_types_t: torch.Tensor,
        pair_energies_t: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        向量化计算 M 个原子/空位的局部键能总和 (E1 + E2)，使用 PyTorch GPU Tensor。
        
        Args:
            center_type_t: (M_calc,) 中心原子（这里是 M'*8 个跳跃原子 A）的类型。
            nn1_types_t: (M_calc, 8) 中心原子的 1NN 类型列表。
            nn2_types_t: (M_calc, 6) 中心原子的 2NN 类型列表。
            pair_energies_t: (E1, E2) Tuple，包含 1NN 和 2NN 势能矩阵，形状均为 (T, T)。
            
        Returns:
            (M_calc,) 每个中心原子的键能总和。
        """
        
        # M_calc 是 M' * 8
        M_calc = center_type_t.shape[0]

        # -------------------- 1. 1NN 能量 (E1) --------------------
        # pair_energies_t[0] 是 1NN 势能矩阵 V1NN (T x T)
        
        # center_type_t[:, None]: (M_calc, 1) - 广播用于索引势能矩阵的第一维 (i)
        # nn1_types_t.view(M_calc, -1): (M_calc, 8) - 用于索引势能矩阵的第二维 (j)
        
        # PyTorch 高级索引 V1NN[i, j] 得到 (M_calc, 8) 的键能矩阵
        E1_arr = pair_energies_t[0][
            center_type_t[:, None].long(), 
            nn1_types_t.view(M_calc, -1).long()
        ]
        
        # -------------------- 2. 2NN 能量 (E2) --------------------
        # pair_energies_t[1] 是 2NN 势能矩阵 V2NN (T x T)
        
        # center_type_t[:, None]: (M_calc, 1)
        # nn2_types_t.view(M_calc, -1): (M_calc, 6)
        
        # PyTorch 高级索引 V2NN[i, j] 得到 (M_calc, 6) 的键能矩阵
        E2_arr = pair_energies_t[1][
            center_type_t[:, None].long(), 
            nn2_types_t.view(M_calc, -1).long()
        ]

        # -------------------- 3. 求和 --------------------
        # 沿每个原子的邻居维度 (dim=1) 求和，得到 (M_calc,)
        E1_sum = E1_arr.sum(dim=1)
        E2_sum = E2_arr.sum(dim=1)

        # 返回每个 A 原子的 E1 + E2 总和
        return E1_sum + E2_sum

    # def _batch_vacancy_diffusion_energy_torch(self, vacancies_index: Optional[np.ndarray] = None) -> np.ndarray:
    #     """
    #     向量化计算一批空位在所有8个NN1方向上的扩散能 (GPU Tensor 版本)。
    #     采用延迟转换和即时释放策略。
        
    #     Args:
    #         vacancies_index: (M',) NumPy 索引数组。
                            
    #     Returns:
    #         (M', 8) 的扩散能数组 (NumPy 数组)。
    #     """
    #     device = self.device
        
    #     # --- 1. 获取和切片数据 (NumPy -> PyTorch 延迟转换) ---
        
    #     # 1.1 确定要计算的空位数 M_prime 和索引
    #     if vacancies_index is None:
    #         M = self.vacancy_pos.shape[0] # 使用原始 NumPy 属性获取 M
    #         indices = np.arange(M)
    #     else:
    #         indices = np.asarray(vacancies_index) 

    #     M_prime = len(indices)
    #     if M_prime == 0:
    #         return np.zeros((0, 8))

    #     # 1.2 将索引转换为 GPU Tensor
    #     indices_t = torch.tensor(indices, device=device)

    #     # 1.3 从 self 获取并切片所有数据 (使用原始 NumPy/Python 属性，并转换为 GPU Tensor)
        
    #     # 核心输入数据切片和转换
    #     vacancies_t = torch.tensor(self.vacancy_pos[indices], device=device)          # (M', 3)
    #     nn1_types_t = torch.tensor(self.nn1_types[indices], dtype=torch.int32, device=device)            # (M', 8)
    #     nn2_types_t = torch.tensor(self.nn2_types[indices], dtype=torch.int32, device=device)            # (M', 6)
    #     nn1_nn1_types_t = torch.tensor(self.nn1_nn1_types[indices], dtype=torch.int32, device=device)    # (M', 8, 8)
    #     nn1_nn2_types_t = torch.tensor(self.nn1_nn2_types[indices], dtype=torch.int32, device=device)    # (M', 8, 6)
        
    #     # 不变参数 (NumPy/Python List -> PyTorch Tensor)
    #     E_a0_t = torch.tensor(self.E_a0, device=device) # (T,)
    #     # 假设 self.pair_energies 是一个包含 N x N NumPy 矩阵的列表
    #     pair_energies_t = tuple(
    #         torch.tensor(pe, device=device) for pe in self.pair_energies
    #     )
    #     NN1_t = torch.tensor(self.NN1, device=device) # (8, 3)
    #     k_V_idx_t = torch.tensor(self.k_V_idx, dtype=torch.long, device=device) # (8,)
        
    #     M_calc = M_prime * 8
        
    #     # ------------------------- 预处理：确定合法跳跃 -------------------------
    #     legal_mask = nn1_types_t != V # (M', 8)
    #     diffusion_energies = torch.full((M_prime, 8), float('inf'), device=device)

    #     # ---------------------------------------------------------------------
    #     # E_before: 计算所有潜在跳跃原子 A 的键能和
    #     # ---------------------------------------------------------------------
        
    #     center_A_t = nn1_types_t.contiguous().view(-1)             # (M'*8,)
    #     nn1_A_t = nn1_nn1_types_t.view(M_calc, 8)           # A 的 1NN
    #     nn2_A_t = nn1_nn2_types_t.view(M_calc, 6)           # A 的 2NN
        
    #     E_A_t = self._batch_env_energy_torch(
    #         center_A_t, nn1_A_t, nn2_A_t, pair_energies_t
    #     )
    #     E_before_sum_t = E_A_t.view(M_prime, 8) / 2.0 # (M', 8)
        
    #     # 释放临时计算 Tensor 的 GPU 内存
    #     del E_A_t, center_A_t, nn1_A_t, nn2_A_t
    #     torch.cuda.empty_cache() 

    #     # ---------------------------------------------------------------------
    #     # E_after: 模拟 V <-> A 交换后的键能和 (保持 E_after_sum_t 及其组件直到计算完成)
    #     # ---------------------------------------------------------------------

    #     # (省略 E_V_new_t 和 E_A_new_t 的计算细节，假设它们遵循之前的 PyTorch 逻辑)
    #     # ... E_V_new_t, E_A_new_t 计算 ...
        
    #     # 以下只是示例替换，需要确保您的 PyTorch 逻辑正确
    #     E_V_new_t = self._calculate_E_V_new_torch(M_prime, M_calc, nn1_types_t, nn2_types_t, pair_energies_t, device)
    #     E_A_new_t = self._calculate_E_A_new_torch(M_prime, M_calc, nn1_types_t, nn1_nn1_types_t, nn1_nn2_types_t, pair_energies_t, k_V_idx_t, device)
        
    #     E_after_sum_t = (E_V_new_t + E_A_new_t) / 2.0 # (M', 8)
        
    #     # 释放临时计算 Tensor
    #     del E_V_new_t, E_A_new_t
    #     torch.cuda.empty_cache()

    #     # ------------------------- LAKIMOC 修正项 -------------------------
    #     delta_e_base_t = E_after_sum_t - E_before_sum_t # (M', 8)
    #     delta_e_corr_t = torch.zeros((M_prime, 8), device=device)

    #     it_arr_t = nn1_types_t # A_j 类型 (M', 8)

    #     # 循环 8 个方向，内部是全矢量化操作
    #     for j in range(8): 
            
    #         it_j_t = it_arr_t[:, j]             # (M',)
    #         d_offset_j_t = NN1_t[j]             # (3,)
            
    #         # (1) -d 轴修正
    #         k_V_d = k_V_idx_t[j]
    #         it2_arr_t = nn1_types_t[:, k_V_d]
    #         delta_e_corr_t[:, j] += _calcul_dene_v1_ppair2_torch(it_j_t, it2_arr_t, 1, +1.0, pair_energies_t)
            
    #         # (2) 侧向 1NN 修正
    #         side_corr_j_t = self._get_side_nn1_correction_torch(
    #             vac_r_coord_t=vacancies_t, 
    #             d_offset_t=d_offset_j_t, 
    #             it_arr_t=it_j_t
    #         )
    #         delta_e_corr_t[:, j] += side_corr_j_t
            
    #         # (3), (4), (5) 的修正项计算
    #         n_type_t = nn1_nn1_types_t[:, j, j]
    #         # ... (修正项计算细节省略，假设已使用 torch 版本函数) ...
            
    #     # ------------------------- 最终计算 -------------------------
    #     final_delta_e_t = delta_e_base_t + delta_e_corr_t

    #     E_a0_masked = E_a0_t[it_arr_t[legal_mask].long()]
    #     diffusion_energies[legal_mask] = (
    #         (E_a0_masked + 0.5 * final_delta_e_t[legal_mask]).to(diffusion_energies.dtype)
    #     )
        
    #     # --- 2. 释放所有临时 GPU Tensor (包括输入) ---
        
    #     # 释放所有输入 Tensor
    #     del vacancies_t, nn1_types_t, nn2_types_t, nn1_nn1_types_t, nn1_nn2_types_t
        
    #     # 释放所有常量 Tensor (如果它们只在这里创建)
    #     del E_a0_t, pair_energies_t, NN1_t, k_V_idx_t
        
    #     # 释放所有中间结果和最终结果 Tensor
    #     del E_before_sum_t, E_after_sum_t # 假设 E_after_sum_t 之前没删
    #     del delta_e_base_t, delta_e_corr_t, final_delta_e_t, legal_mask
        
    #     # 将最终结果从 GPU 转移回 CPU 并转换为 NumPy 数组
    #     final_result_np = diffusion_energies.cpu().float().numpy()
    #     del diffusion_energies, E_a0_masked
        
    #     # 清理 GPU 缓存
    #     torch.cuda.empty_cache()

    #     return final_result_np
    
# 假设 V = 2 已经在类级别定义
    def _batch_vacancy_diffusion_energy_torch(self, vacancies_index: Optional[np.ndarray] = None) -> np.ndarray:
        """
        向量化计算一批空位在所有8个NN1方向上的扩散能 (GPU Tensor 版本)。
        采用 CPU/GPU 往返查询晶格。
        """
        device = self.device
        
        
        # --- 1. 获取和切片数据 (NumPy -> PyTorch 延迟转换) ---
        # ... (数据切片和常量定义保持不变) ...

        if vacancies_index is None:
            if hasattr(self, 'nn1_types') and isinstance(self.nn1_types, np.ndarray) and self.nn1_types.size > 0:
                indices = np.arange(self.nn1_types.shape[0], dtype=int)
            elif hasattr(self, 'v_pos_of_id') and len(getattr(self, 'v_pos_of_id', {})) > 0:
                indices = np.arange(len(self.get_vacancy_array()), dtype=int)
            else:
                indices = np.empty((0,), dtype=int)
        else:
            indices = np.asarray(vacancies_index) 
            # print(f"Rank {dist.get_rank()} indices: {indices}")

        use_local_map = hasattr(self, '_batch_get_type_from_local_coords') or hasattr(self, '_batch_get_type_from_coords_local')
        if use_local_map:
            # zrg(debug)
            # print(f"Rank {dist.get_rank()} use_local_map: {use_local_map}")
            idx_arr = np.asarray(indices, dtype=int)
            Nv_local = int(self.nn1_types.shape[0]) if hasattr(self, 'nn1_types') and isinstance(self.nn1_types, np.ndarray) else 0
            valid_local = [int(i) for i in idx_arr.tolist() if 0 <= int(i) < Nv_local]
            indices = np.array(valid_local, dtype=int)
            M_prime = len(indices)
            if M_prime == 0:
                return np.zeros((0, 8))
            coords = []
            for lid in valid_local:
                vid = int(lid)
                pos = self.get_vacancy_pos_by_id(int(vid))
                local_coord = np.array(pos, dtype=np.int32)
                coords.append(local_coord.tolist())
            vacancies_t = torch.tensor(coords, device=device, dtype=torch.int32)
            # print(f"Rank {dist.get_rank()} vacancies_t: {vacancies_t}")
            
        else:
            # print(f"Rank {dist.get_rank()} use_local_map: {use_local_map}")
            M_prime = len(indices)
            if M_prime == 0:
                # print(f"Rank {dist.get_rank()} M_prime is 0")
                return np.zeros((0, 8))
            if hasattr(self, 'v_pos_of_id') and len(getattr(self, 'v_pos_of_id', {})) > 0:
                Nv = len(self.get_vacancy_array())
                if isinstance(indices, np.ndarray):
                    mask = (indices >= 0) & (indices < Nv)
                    indices = indices[mask]
                else:
                    indices = np.array([int(i) for i in indices if 0 <= int(i) < Nv], dtype=int)
                vacancies_t = torch.tensor(self.get_vacancy_array()[indices], device=device, dtype=torch.int32)
                # print(f"Rank {dist.get_rank()} vacancies_t: {vacancies_t}")
        if 'use_local_map' in locals() and use_local_map:
            dims_np = np.array(self.dims, dtype=int)
            vac_np = vacancies_t.cpu().numpy()
            V_nn1_coords_np = self._get_pbc_coord(vac_np[:, None, :], self.NN1[None, :, :], dims_np)
            V_nn2_coords_np = self._get_pbc_coord(vac_np[:, None, :], self.NN2[None, :, :], dims_np)
            if hasattr(self, '_batch_get_type_from_local_coords'):
                nn1_np = self._batch_get_type_from_local_coords(V_nn1_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8)
                nn2_np = self._batch_get_type_from_local_coords(V_nn2_coords_np.reshape(-1, 3)).reshape(len(vac_np), 6)
            elif hasattr(self, '_batch_get_type_from_coords_local'):
                nn1_np = self._batch_get_type_from_coords_local(V_nn1_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8)
                nn2_np = self._batch_get_type_from_coords_local(V_nn2_coords_np.reshape(-1, 3)).reshape(len(vac_np), 6)
            else:
                nn1_np = self._batch_get_type_from_coords(V_nn1_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8)
                nn2_np = self._batch_get_type_from_coords(V_nn2_coords_np.reshape(-1, 3)).reshape(len(vac_np), 6)
            A_nn1_nn1_coords_np = self._get_pbc_coord(V_nn1_coords_np[:, :, None, :], self.NN1[None, None, :, :], dims_np)
            A_nn1_nn2_coords_np = self._get_pbc_coord(V_nn1_coords_np[:, :, None, :], self.NN2[None, None, :, :], dims_np)
            if hasattr(self, '_batch_get_type_from_local_coords'):
                nn1_nn1_np = self._batch_get_type_from_local_coords(A_nn1_nn1_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8, 8)
                nn1_nn2_np = self._batch_get_type_from_local_coords(A_nn1_nn2_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8, 6)
            elif hasattr(self, '_batch_get_type_from_coords_local'):
                nn1_nn1_np = self._batch_get_type_from_coords_local(A_nn1_nn1_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8, 8)
                nn1_nn2_np = self._batch_get_type_from_coords_local(A_nn1_nn2_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8, 6)
            else:
                nn1_nn1_np = self._batch_get_type_from_coords(A_nn1_nn1_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8, 8)
                nn1_nn2_np = self._batch_get_type_from_coords(A_nn1_nn2_coords_np.reshape(-1, 3)).reshape(len(vac_np), 8, 6)
            nn1_types_t = torch.tensor(nn1_np, dtype=torch.int8, device=device)
            nn2_types_t = torch.tensor(nn2_np, dtype=torch.int8, device=device)
            nn1_nn1_types_t = torch.tensor(nn1_nn1_np, dtype=torch.int8, device=device)
            nn1_nn2_types_t = torch.tensor(nn1_nn2_np, dtype=torch.int8, device=device)
        else:
            nn1_types_t = torch.tensor(self.nn1_types[indices], dtype=torch.int8, device=device)            # (M', 8)
            nn2_types_t = torch.tensor(self.nn2_types[indices], dtype=torch.int8, device=device)            # (M', 6)
            nn1_nn1_types_t = torch.tensor(self.nn1_nn1_types[indices], dtype=torch.int8, device=device)    # (M', 8, 8)
            nn1_nn2_types_t = torch.tensor(self.nn1_nn2_types[indices], dtype=torch.int8, device=device)    # (M', 8, 6)
        
        # print(f"Rank {dist.get_rank()} nn1_types_t: {nn1_types_t}")
        # print(f"Rank {dist.get_rank()} nn2_types_t: {nn2_types_t}")
        # print(f"Rank {dist.get_rank()} nn1_nn1_types_t: {nn1_nn1_types_t}")
        # print(f"Rank {dist.get_rank()} nn1_nn2_types_t: {nn1_nn2_types_t}")
        
        E_a0_t = torch.tensor(self.E_a0, device=device)
        pair_energies_t = tuple(
            torch.tensor(pe, device=device) for pe in self.pair_energies
        )
        self.pair_energies_t = pair_energies_t # 辅助函数需要
        
        NN1_t = torch.tensor(self.NN1, device=device, dtype=torch.int32)
        k_V_idx_t = torch.tensor(self.k_V_idx, dtype=torch.long, device=device)
        
        M_calc = M_prime * 8
        dims_t = self.dims_t.to(torch.int32)

        # --- E_before & E_after 计算 (保持不变) ---
        
        center_A_t = nn1_types_t.contiguous().view(-1)
        nn1_A_t = nn1_nn1_types_t.view(M_calc, 8)
        nn2_A_t = nn1_nn2_types_t.view(M_calc, 6) 
        #  nn1 nn2
        
        E_A_t = self._batch_env_energy_torch(center_A_t, nn1_A_t, nn2_A_t, pair_energies_t)
        E_V_t = self._batch_env_energy_torch(torch.full_like(center_A_t, self.V_TYPE), nn1_types_t.unsqueeze(1).expand(-1, 8, -1).reshape(M_calc, 8), nn2_types_t.unsqueeze(1).expand(-1, 8, -1).reshape(M_calc, 6), pair_energies_t)
        # E_V_t = self._batch_env_energy_torch(torch.full_like(vacancies_t, 2), nn1_types_t.view(M_prime, 8), nn2_types_t.view(M_prime, 6), pair_energies_t).unsqueeze(1).expand(-1, 8)
        
        # E_before_sum_t = (E_A_t.view(M_prime, 8))/ 2.0
        E_before_sum_t = (E_A_t.view(M_prime, 8) + E_V_t.view(M_prime, 8))/ 2.0
        
        del E_A_t, center_A_t, nn1_A_t, nn2_A_t; torch.cuda.empty_cache() 

        E_V_new_t = self._calculate_E_V_new_torch(M_prime, M_calc, nn1_types_t, nn2_types_t, pair_energies_t, device)
        E_A_new_t = self._calculate_E_A_new_torch(M_prime, M_calc, nn1_types_t, nn1_nn1_types_t, nn1_nn2_types_t, pair_energies_t, k_V_idx_t, device)
        E_after_sum_t = (E_V_new_t + E_A_new_t) / 2.0
        del E_V_new_t, E_A_new_t; torch.cuda.empty_cache()

        
        diffusion_energies = torch.full((M_prime, 8), float('inf'), device=device)
        delta_e_base_t = E_after_sum_t - E_before_sum_t # (M', 8)
        delta_e_corr_t = torch.zeros((M_prime, 8), device=device)
        it_arr_t = nn1_types_t # A_j 类型 (M', 8)

        # 定义 PBC 几何操作 (基于 PyTorch Int Tensor)
        def get_pbc_coord_torch_int(r_base_t, delta_t):
            return (r_base_t + delta_t) % dims_t

        # -----------------------------------------------------------------
        def _get_type_via_cpu_batch(coords_t: torch.Tensor) -> torch.Tensor:
            use_global_gpu = torch.cuda.is_available() and hasattr(self, '_global_vac_lin_sorted_t') and hasattr(self, '_global_cu_lin_sorted_t')
            if use_global_gpu:
                D = dims_t.to(torch.int64)
                c = coords_t.to(torch.int64)
                lin_t = (((c[:,0] * D[1]) + c[:,1]) * D[2] + c[:,2])
                types_t = torch.full((c.shape[0],), int(self.FE_TYPE), dtype=torch.int32, device=device)
                if self._global_vac_lin_sorted_t.numel() > 0:
                    idx_v = torch.searchsorted(self._global_vac_lin_sorted_t, lin_t)
                    in_range_v = idx_v < self._global_vac_lin_sorted_t.numel()
                    idx_vc = torch.clamp(idx_v, 0, max(0, int(self._global_vac_lin_sorted_t.numel()-1)))
                    mask_v = in_range_v & (self._global_vac_lin_sorted_t[idx_vc] == lin_t)
                    types_t[mask_v] = int(self.V_TYPE)
                if self._global_cu_lin_sorted_t.numel() > 0:
                    idx_c = torch.searchsorted(self._global_cu_lin_sorted_t, lin_t)
                    in_range_c = idx_c < self._global_cu_lin_sorted_t.numel()
                    idx_cc = torch.clamp(idx_c, 0, max(0, int(self._global_cu_lin_sorted_t.numel()-1)))
                    mask_c = in_range_c & (self._global_cu_lin_sorted_t[idx_cc] == lin_t)
                    types_t[mask_c] = int(self.CU_TYPE)
                return types_t
            coords_np = coords_t.cpu().numpy()
            if hasattr(self, '_batch_get_type_from_coords_local'):
                types_np = self._batch_get_type_from_coords_local(coords_np)
            else:
                types_np = self._batch_get_type_from_coords(coords_np)
            return torch.tensor(types_np, device=device, dtype=torch.int32)
        # -----------------------------------------------------------------
        d_offsets_2x_t = (2 * NN1_t) % dims_t
        v_exp_t = vacancies_t.unsqueeze(1).expand(-1, 8, -1)
        r_p2_all_t = (v_exp_t + d_offsets_2x_t.unsqueeze(0)) % dims_t
        n_3_1_t = torch.stack([r_p2_all_t[..., 0], v_exp_t[..., 1], v_exp_t[..., 2]], dim=-1).reshape(-1, 3)
        n_3_2_t = torch.stack([v_exp_t[..., 0], r_p2_all_t[..., 1], v_exp_t[..., 2]], dim=-1).reshape(-1, 3)
        n_3_3_t = torch.stack([v_exp_t[..., 0], v_exp_t[..., 1], r_p2_all_t[..., 2]], dim=-1).reshape(-1, 3)
        n_4_all_t = r_p2_all_t.reshape(-1, 3)
        n_5_1_t = torch.stack([v_exp_t[..., 0], r_p2_all_t[..., 1], r_p2_all_t[..., 2]], dim=-1).reshape(-1, 3)
        n_5_2_t = torch.stack([r_p2_all_t[..., 0], v_exp_t[..., 1], r_p2_all_t[..., 2]], dim=-1).reshape(-1, 3)
        n_5_3_t = torch.stack([r_p2_all_t[..., 0], r_p2_all_t[..., 1], v_exp_t[..., 2]], dim=-1).reshape(-1, 3)
        types_3_1_t = _get_type_via_cpu_batch(n_3_1_t).to(torch.int8).view(M_prime, 8)
        types_3_2_t = _get_type_via_cpu_batch(n_3_2_t).to(torch.int8).view(M_prime, 8)
        types_3_3_t = _get_type_via_cpu_batch(n_3_3_t).to(torch.int8).view(M_prime, 8)
        types_4_t = _get_type_via_cpu_batch(n_4_all_t).to(torch.int8).view(M_prime, 8)
        types_5_1_t = _get_type_via_cpu_batch(n_5_1_t).to(torch.int8).view(M_prime, 8)
        types_5_2_t = _get_type_via_cpu_batch(n_5_2_t).to(torch.int8).view(M_prime, 8)
        types_5_3_t = _get_type_via_cpu_batch(n_5_3_t).to(torch.int8).view(M_prime, 8)


        for j in range(8): 
            
            it_j_t = it_arr_t[:, j]             # (M',)
            d_offset_j_t = NN1_t[j]             # (3,)
            
            # (1) -d 轴修正 (保持不变)
            k_V_d = k_V_idx_t[j]
            it2_arr_t_1 = nn1_types_t[:, k_V_d]
            delta_e_corr_t[:, j] += _calcul_dene_v1_ppair2_torch(it_j_t, it2_arr_t_1, 1, +1.0, pair_energies_t, self.V_TYPE)
            
            # (2) 侧向 1NN 修正 (需要适配其内部的坐标查找，但此处只调用接口)
            # ❗ 注意：_get_side_nn1_correction_torch 内部**也需要**更新为使用 CPU/GPU 往返 ❗
            side_corr_j_t = self._get_side_nn1_correction_torch(
                vac_r_coord_t=vacancies_t.float(),
                d_offset_t=d_offset_j_t.float(), 
                it_arr_t=it_j_t
            )
            delta_e_corr_t[:, j] += side_corr_j_t
            
            it2_arr_3_1_t = types_3_1_t[:, j]
            delta_e_corr_t[:, j] += _calcul_dene_v2_ppair2_torch(it_j_t, it2_arr_3_1_t, 2, 1, pair_energies_t, self.V_TYPE)
            it2_arr_3_2_t = types_3_2_t[:, j]
            delta_e_corr_t[:, j] += _calcul_dene_v2_ppair2_torch(it_j_t, it2_arr_3_2_t, 2, 1, pair_energies_t, self.V_TYPE)
            it2_arr_3_3_t = types_3_3_t[:, j]
            delta_e_corr_t[:, j] += _calcul_dene_v2_ppair2_torch(it_j_t, it2_arr_3_3_t, 2, 1, pair_energies_t, self.V_TYPE)
            it2_arr_4_t = types_4_t[:, j]
            delta_e_corr_t[:, j] += _calcul_dene_v1_ppair2_torch(it_j_t, it2_arr_4_t, 1, -1.0, pair_energies_t, self.V_TYPE)
            it2_arr_5_1_t = types_5_1_t[:, j]
            delta_e_corr_t[:, j] += _calcul_dene_v2_ppair2_torch(it_j_t, it2_arr_5_1_t, 3, 1, pair_energies_t, self.V_TYPE)
            it2_arr_5_2_t = types_5_2_t[:, j]
            delta_e_corr_t[:, j] += _calcul_dene_v2_ppair2_torch(it_j_t, it2_arr_5_2_t, 3, 1, pair_energies_t, self.V_TYPE)
            it2_arr_5_3_t = types_5_3_t[:, j]
            delta_e_corr_t[:, j] += _calcul_dene_v2_ppair2_torch(it_j_t, it2_arr_5_3_t, 3, 1, pair_energies_t, self.V_TYPE)


        # ------------------------- 最终计算 -------------------------
        final_delta_e_t = delta_e_base_t + delta_e_corr_t

        E_a0_full = E_a0_t[it_arr_t.long()]
        diffusion_vals = (E_a0_full + 0.5 * final_delta_e_t).to(diffusion_energies.dtype)
        diffusion_energies = diffusion_vals
        # diffusion_energies[it_arr_t == self.V_TYPE] = float('inf')
        
        # --- 2. 释放所有临时 GPU Tensor ---
        
        # ... (释放代码保持不变) ...
        
        del vacancies_t, nn1_types_t, nn2_types_t, nn1_nn1_types_t, nn1_nn2_types_t
        del E_a0_t, NN1_t, k_V_idx_t, dims_t, self.pair_energies_t, pair_energies_t
        del E_before_sum_t, E_after_sum_t
        del delta_e_base_t, delta_e_corr_t, final_delta_e_t
        
        final_result_np = diffusion_energies.cpu().float().numpy()
        del diffusion_energies
        
        torch.cuda.empty_cache()

        return final_result_np
    
    def _batch_vacancy_diffusion_energy(self, vacancies_index: Optional[np.ndarray] = None) -> np.ndarray:
        """
        向量化计算一批空位在所有8个NN1方向上的扩散能，完全基于 self 属性中的局部环境张量。
        
        Args:
            vacancies_index: (M',) 一个可选的索引数组，指定要计算的空位子集。
                            如果为 None，则计算所有空位。
                            
        Returns:
            (M', 8) 或 (M, 8) 的扩散能数组。
        """
        return self._batch_vacancy_diffusion_energy_torch_v2(vacancies_index)
        # return self._batch_vacancy_diffusion_energy_torch(vacancies_index)

    def _batch_vacancy_diffusion_energy_torch_v2(self, vacancies_index: Optional[np.ndarray] = None) -> np.ndarray:
        device = self.device
        if hasattr(self, 'nn1_types') and isinstance(self.nn1_types, np.ndarray) and self.nn1_types.size > 0:
            Nv_local = int(self.nn1_types.shape[0])
        else:
            return np.zeros((0, 8), dtype=float)
        if vacancies_index is None:
            indices_np = np.arange(Nv_local, dtype=int)
        else:
            idx_arr = np.asarray(vacancies_index, dtype=int)
            valid_local = [int(i) for i in idx_arr.tolist() if 0 <= int(i) < Nv_local]
            indices_np = np.array(valid_local, dtype=int)
        M_prime = len(indices_np)
        if M_prime == 0:
            return np.zeros((0, 8), dtype=float)
        use_gpu_cache = torch.cuda.is_available() and hasattr(self, 'nn1_types_t') and hasattr(self, 'nn2_types_t') and hasattr(self, 'nn1_nn1_types_t') and hasattr(self, 'nn1_nn2_types_t')
        t0 = time.time()
        if use_gpu_cache:
            indices_t = torch.as_tensor(indices_np, dtype=torch.long, device=device)
            # print(f"indices_t {indices_t}")
            # print(f"nn1_types_t {self.nn1_types_t.shape}")
            # print(f"nn2_types_t {self.nn2_types_t.shape}")
            # print(f"nn1_nn1_types_t {self.nn1_nn1_types_t.shape}")
            # print(f"nn1_nn2_types_t {self.nn1_nn2_types_t.shape}")
            # nn1_types_t = self.nn1_types_t.index_select(0, indices_t)
            # nn2_types_t = self.nn2_types_t.index_select(0, indices_t)
            # nn1_nn1_types_t = self.nn1_nn1_types_t.index_select(0, indices_t)
            # nn1_nn2_types_t = self.nn1_nn2_types_t.index_select(0, indices_t)
            
            nn1_types_t = torch.as_tensor(self.nn1_types[indices_np], dtype=torch.int8, device=device)
            nn2_types_t = torch.as_tensor(self.nn2_types[indices_np], dtype=torch.int8, device=device)
            nn1_nn1_types_t = torch.as_tensor(self.nn1_nn1_types[indices_np], dtype=torch.int8, device=device)
            nn1_nn2_types_t = torch.as_tensor(self.nn1_nn2_types[indices_np], dtype=torch.int8, device=device)
            
            # print(f"nn1_types_t {nn1_types_t}")
            # print(f"nn2_types_t {nn2_types_t}")
            # print(f"nn1_nn1_types_t {nn1_nn1_types_t}")
            # print(f"nn1_nn2_types_t {nn1_nn2_types_t}")
            # time.sleep(100000)
        else:
            nn1_types_t = torch.as_tensor(self.nn1_types[indices_np], dtype=torch.int8, device=device)
            nn2_types_t = torch.as_tensor(self.nn2_types[indices_np], dtype=torch.int8, device=device)
            nn1_nn1_types_t = torch.as_tensor(self.nn1_nn1_types[indices_np], dtype=torch.int8, device=device)
            nn1_nn2_types_t = torch.as_tensor(self.nn1_nn2_types[indices_np], dtype=torch.int8, device=device)
        t1 = time.time()
       
        
        t2 = time.time()
        if torch.cuda.is_available():
            if not hasattr(self, 'E_a0_t'):
                self.E_a0_t = torch.as_tensor(self.E_a0, device=device)
            if not hasattr(self, 'pair_energies_t'):
                self.pair_energies_t = tuple(torch.as_tensor(pe, device=device) for pe in self.pair_energies)
            E_a0_t = self.E_a0_t
            pair_energies_t = self.pair_energies_t
        else:
            E_a0_t = torch.as_tensor(self.E_a0, device=device)
            pair_energies_t = tuple(torch.as_tensor(pe, device=device) for pe in self.pair_energies)
        t3 = time.time()

        
        t4 = time.time()
        
        M_calc = int(M_prime * 8)
        center_A_t = nn1_types_t.contiguous().view(-1)
        nn1_A_t = nn1_nn1_types_t.view(M_calc, 8)
        nn2_A_t = nn1_nn2_types_t.view(M_calc, 6)
        with torch.no_grad():
            E_A_t = self._batch_env_energy_torch(center_A_t, nn1_A_t, nn2_A_t, pair_energies_t)
        center_V_t = torch.full_like(center_A_t, int(self.V_TYPE))
        nn1_V_t = nn1_types_t.unsqueeze(1).expand(-1, 8, -1).reshape(M_calc, 8)
        nn2_V_t = nn2_types_t.unsqueeze(1).expand(-1, 8, -1).reshape(M_calc, 6)
        with torch.no_grad():
            E_V_t = self._batch_env_energy_torch(center_V_t, nn1_V_t, nn2_V_t, pair_energies_t)
        E_before_sum_t = (E_A_t.view(M_prime, 8) + E_V_t.view(M_prime, 8)) / 2.0
        k_V_idx_t = torch.as_tensor(self.k_V_idx, dtype=torch.long, device=device)
        with torch.no_grad():
            E_V_new_t = self._calculate_E_V_new_torch(M_prime, M_calc, nn1_types_t, nn2_types_t, pair_energies_t, device)
            E_A_new_t = self._calculate_E_A_new_torch(M_prime, M_calc, nn1_types_t, nn1_nn1_types_t, nn1_nn2_types_t, pair_energies_t, k_V_idx_t, device)
        E_after_sum_t = (E_V_new_t + E_A_new_t) / 2.0
        final_delta_e_t = E_after_sum_t - E_before_sum_t
        t5 = time.time()

        
        # diffusion_energies = torch.full((M_prime, 8), float('inf'), device=device)
        # torch.set_printoptions(threshold=1000000)
        
        it_arr_t = nn1_types_t
        # print(f"it_arr_t {it_arr_t}")
        E_a0_full = E_a0_t[it_arr_t.long()]
        # time.sleep(100000)
        
        # diffusion_vals = (E_a0_full + 0.5 * final_delta_e_t).to(diffusion_energies.dtype)
        
        diffusion_energies = (E_a0_full + 0.5 * final_delta_e_t)
        diffusion_energies = torch.where(it_arr_t == int(self.V_TYPE), torch.zeros_like(diffusion_energies), diffusion_energies)
        result = diffusion_energies.detach().cpu().numpy().astype(float)
        # del nn1_types_t, nn2_types_t, nn1_nn1_types_t, nn1_nn2_types_t
        # del center_A_t, nn1_A_t, nn2_A_t, center_V_t, nn1_V_t, nn2_V_t
        # del E_A_t, E_V_t, E_before_sum_t, E_V_new_t, E_A_new_t, E_after_sum_t, final_delta_e_t
        # print(f" diffusion_energies {result}")
        # time.sleep(100000)
        
        return result
    
    def _vacancy_diffusion_energy(self, i, j, k):
        """
        针对中心空位 (i, j, k)，使用向量化逻辑计算其 8 个 NN1 跳跃方向的扩散能。
        这与 calculate_diffusion_energy 中的逻辑一致，但 M=1。
        """
        nx, ny, nz = self.dims
        
        E_a0 = self.E_a0
        pair_energies = self.pair_energies
        NN1 = np.array(self.NN1)  # (8, 3)
        dims = np.array([nx, ny, nz])

        # 1. 确定空位、目标位置和移动原子的类型
        vac_pos = np.array([[i, j, k]])                  # (1, 3)
        targets = self._get_pbc_coord(vac_pos[:, None, :], NN1[None, :, :], dims)  # (1, 8, 3)
        neighbor_types = self._batch_get_type_from_coords(targets.reshape(-1, 3)).reshape(1, 8)

        # 2. 初始化结果
        diffusion_energies = np.zeros(8)

        # 3. 使用类方法获取 PBC 坐标

        # 4. 核心：遍历 8 个方向
        for dir_idx in range(8):
            it_type = neighbor_types[0, dir_idx]  # 移动原子的类型 (标量)

            if it_type == self.V_TYPE:
                continue  # 无合法跳跃

            # 转换为 M=1 的数组 (M=1)
            vac = vac_pos                               # (1, 3)
            tgt = targets[0, dir_idx][None, :]          # (1, 3)
            it_arr = np.array([it_type])                # (1,)
            d_offset = NN1[dir_idx]                     # (3,)

            # 5. 定义用于批量计算原子能量的函数 (M=1)
            def batch_atomic_energy(positions, lat):
                # positions 是 (M, 3) = (1, 3) 数组
                center_types = lat[tuple(positions.T)]
                nn1 = self._get_pbc_coord(positions[:, None, :], self.NN1, dims)
                nn2 = self._get_pbc_coord(positions[:, None, :], self.NN2, dims)
                nn1_types = lat[tuple(nn1.transpose(2, 0, 1))]
                nn2_types = lat[tuple(nn2.transpose(2, 0, 1))]
                e1 = pair_energies[0][center_types[:, None], nn1_types]
                e2 = pair_energies[1][center_types[:, None], nn2_types]
                # 返回 M 个原子的能量
                return (e1.sum(axis=1) + e2.sum(axis=1)) # (M,)

            # 6. 计算 E_before 和 E_after
            # (E_A + E_V)_before
            e_before_sum = (batch_atomic_energy(vac, lattice) + batch_atomic_energy(tgt, lattice)) / 2

            # 模拟跳跃
            lattice_mod = lattice.copy()
            lattice_mod[i, j, k] = it_type
            lattice_mod[tgt[0, 0], tgt[0, 1], tgt[0, 2]] = self.V_TYPE

            # (E_A + E_V)_after
            e_after_sum = (batch_atomic_energy(vac, lattice_mod) + batch_atomic_energy(tgt, lattice_mod)) / 2

            delta_e_base = e_after_sum - e_before_sum # (1,)
            delta_e_corr = np.zeros(1, dtype=float)

            # 7. LAKIMOC 修正项 (保持与 calculate_diffusion_energy 一致)
            
            r_V_coord = vac      # r_V (1, 3)
            r_A_coord = tgt      # r_A (1, 3)
            
            # 沿轴坐标
            r_m1_coord = self._get_pbc_coord(r_V_coord, -d_offset, dims) # (1, 3)
            r_p2_coord = self._get_pbc_coord(r_V_coord, 2 * d_offset, dims) # (1, 3)
            
            # 1. 沿 -d 轴的 1NN
            r_N = r_m1_coord
            it2_arr = lattice[tuple(r_N.T)] # (1,)
            delta_e_corr += _calcul_dene_v1_ppair2_vec(it_arr, it2_arr, 1, +1.0, self.pair_energies, self.V_TYPE)

            # 2. 侧向 1NN 修正 (6 项)
            delta_e_corr += self._get_side_nn1_correction_vec(r_V_coord, d_offset, it_arr, lattice)

            # 3. 2nn vacancy and 1nn atom it (r_V 的 2NN 和 r_A 的 1NN)
            n_coords_1 = np.stack([r_p2_coord[:, 0], r_V_coord[:, 1], r_V_coord[:, 2]], axis=1)
            it2_arr_1 = lattice[tuple(n_coords_1.T)]
            delta_e_corr += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr_1, 2, 1, self.pair_energies, self.V_TYPE)

            n_coords_2 = np.stack([r_V_coord[:, 0], r_p2_coord[:, 1], r_V_coord[:, 2]], axis=1)
            it2_arr_2 = lattice[tuple(n_coords_2.T)]
            delta_e_corr += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr_2, 2, 1, self.pair_energies, self.V_TYPE)

            n_coords_3 = np.stack([r_V_coord[:, 0], r_V_coord[:, 1], r_p2_coord[:, 2]], axis=1)
            it2_arr_3 = lattice[tuple(n_coords_3.T)]
            delta_e_corr += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr_3, 2, 1, self.pair_energies, self.V_TYPE)

            # 4. 1nn atom it (r_A 的 1NN 修正)
            delta_e_corr += _calcul_dene_v1_ppair2_vec(it_arr, lattice[tuple(r_p2_coord.T)], 1, -1.0, self.pair_energies, self.V_TYPE)

            # 5. r_N 沿侧向 2NN 轴 (nn1=3, nn2=1)
            n_coords_1_5 = np.stack([r_V_coord[:, 0], r_p2_coord[:, 1], r_p2_coord[:, 2]], axis=1)
            it2_arr_1_5 = lattice[tuple(n_coords_1_5.T)]
            delta_e_corr += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr_1_5, 3, 1, self.pair_energies, self.V_TYPE)

            n_coords_2_5 = np.stack([r_p2_coord[:, 0], r_V_coord[:, 1], r_p2_coord[:, 2]], axis=1)
            it2_arr_2_5 = lattice[tuple(n_coords_2_5.T)]
            delta_e_corr += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr_2_5, 3, 1, self.pair_energies, self.V_TYPE)

            n_coords_3_5 = np.stack([r_p2_coord[:, 0], r_p2_coord[:, 1], r_V_coord[:, 2]], axis=1)
            it2_arr_3_5 = lattice[tuple(n_coords_3_5.T)]
            delta_e_corr += _calcul_dene_v2_ppair2_vec(it_arr, it2_arr_3_5, 3, 1, self.pair_energies, self.V_TYPE)
            
            # 8. 最终能量计算
            final_delta_e = delta_e_base + delta_e_corr
            
            # E_a = E_a0[t] + 0.5 * (E_after - E_before + Correction)
            diffusion_energies[dir_idx] = E_a0[it_type] + 0.5 * final_delta_e[0]

        return diffusion_energies


    # ----------------------------
    # calculate_diffusion_energy: 向量化批量计算所有 vacancies 的 diffusion energies
    # 保持你原有实现，但在返回前保存至 self.diffusion_energies
    # ----------------------------
    def calculate_diffusion_energy(self):
        diffusion_energies = self._batch_vacancy_diffusion_energy()
        self.diffusion_energies = diffusion_energies
        return diffusion_energies

    def diffusion_rates_update(self, changed_positions: list[tuple[int,int,int]]):
        timing_enabled = bool(getattr(self, "enable_rate_update_timing", False))
        need_timing = timing_enabled and (
            (not getattr(self, "_has_sampled_rate_update_timing", False))
            or (not bool(getattr(self, "sample_rate_update_timing_once", True)))
        )
        # need_timing = True
        t0 = time.time() if need_timing else 0.0
        vacancies = self.get_vacancy_array()
        if len(vacancies) == 0 or len(changed_positions) == 0:
            return
        t_n0 = time.time() if need_timing else 0.0
        affected_idx = self._get_affected_vacancy_indices(changed_positions)
        if need_timing:
            print(f"rank:{getattr(self,'rank',0)} time get_vacancy_array:", t_n0 - t0)
            t_s1 = time.time()
            t_scan = float(t_s1 - t_n0)
            print(f"rank:{getattr(self,'rank',0)} time scan affected:", t_s1 - t_n0)
        
        if not affected_idx:
            return
        t_e0 = time.time() if need_timing else 0.0
        idx_arr = np.array(affected_idx, dtype=int)
        new_rates_block = self._batch_vacancy_diffusion_energy(idx_arr)
        if need_timing:
            t_e1 = time.time()
            t_energy = float(t_e1 - t_e0)
            print(f"rank:{getattr(self,'rank',0)} time calc energy:", t_e1 - t_e0)
            
            t_r0 = time.time()
        kB = 8.617e-5
        T = self.args.temperature
        nu = 1e13
        new_rates = nu * np.exp(-new_rates_block / (kB * T))
        if need_timing:
            t_r1 = time.time()
            t_rates = float(t_r1 - t_r0)
            print(f"rank:{getattr(self,'rank',0)} time calc rates:", t_r1 - t_r0)
            t_w0 = time.time()
        self.diffusion_rates[idx_arr, :] = new_rates
        if need_timing:
            t_w1 = time.time()
            t_write = float(t_w1 - t_w0)
            print(f"rank:{getattr(self,'rank',0)} time write rates:", t_w1 - t_w0)
            
            t_tot = float(time.time() - t0)
            try:
                print(f"Rank {getattr(self,'rank',0)} diffusion_rates_update timing neigh:{t_neigh:.6f}s scan:{t_scan:.6f}s energy:{t_energy:.6f}s rates:{t_rates:.6f}s write:{t_write:.6f}s total:{t_tot:.6f}s affected:{len(affected_idx)}", flush=True)
            except Exception:
                pass
            self._has_sampled_rate_update_timing = True

    def _get_affected_vacancy_indices(self, changed_positions: list[tuple[int,int,int]]) -> list[int]:
        vacancies = self.get_vacancy_array()
        if len(vacancies) == 0 or len(changed_positions) == 0:
            return []
        offsets = np.vstack((np.array([[0,0,0]], dtype=int), np.array(self.NN1, dtype=int), np.array(self.NN2, dtype=int)))
        dims = np.array(self.dims, dtype=int)
        changed = np.array(changed_positions, dtype=int)
        neigh_coords = self._get_pbc_coord(changed[:, np.newaxis, :], offsets[np.newaxis, :, :], dims)
        affected_site_positions = neigh_coords.reshape(-1, 3)
        affected_site_set = {tuple(map(int, pos)) for pos in affected_site_positions}
        affected_idx = []
        for vidx, vpos in enumerate(vacancies):
            if tuple(map(int, vpos)) in affected_site_set:
                affected_idx.append(int(vidx))
        return sorted(list(set(affected_idx)))
    

    def calculate_diffusion_rate(self):
        # tmp = nu * exp( - ( 0.5 * (*dene) + Emig0[it] ) / ( kb * Te ) );
        # diffusion_energies = 0.5 * (*dene) + Emig0[it]
        kB = 8.617e-5  # eV/K 0.000086
        T = self.args.temperature
        nu = 1e13      # attempt frequency
        diffusion_energies = np.array(self.calculate_diffusion_energy())  # shape (N_vac, 8)
        rates = nu * np.exp(-diffusion_energies / (kB * T))               # shape (N_vac, 8)
        return rates


    def stastic_local_atoms(self, i: int, j: int, k: int) -> np.ndarray:
        nx, ny, nz = self.dims
        local_env = np.zeros(6, dtype=int)
        for offset, NN in zip([0, 3], [self.NN1, self.NN2]):
            coords = (np.array([i, j, k]) + NN) % [nx, ny, nz]
            types = self._batch_get_type_from_coords(coords)
            counts = np.bincount(types, minlength=3)[:3]
            local_env[offset:offset + 3] = counts
        return local_env


    def static_vancancy_local_atoms(self) -> np.ndarray:
        """
        向量化构造每个 vacancy 的 [1NN + 2NN] 原子种类（整数编码）特征。
        直接复用维护的 self.nn1_types 和 self.nn2_types 张量。
        
        输出 shape: (n_vacancies, 14)
        """
        
        # 检查是否有空位
        if not hasattr(self, 'nn1_types') or self.nn1_types is None:
            return np.zeros((0, 14), dtype=np.int32)
        if not hasattr(self, 'nn2_types') or self.nn2_types is None:
            return np.zeros((0, 14), dtype=np.int32)
            
        # 直接读取并拼接已维护的张量
        nn1_types = self.nn1_types  # Shape: (N, 8)
        nn2_types = self.nn2_types  # Shape: (N, 6)

        # 拼接成 [N, 14] 并确保为 np.int32 类型
        nn_features = np.empty((nn1_types.shape[0], nn1_types.shape[1] + nn2_types.shape[1]), dtype=np.int32)
        nn_features[:, :nn1_types.shape[1]] = nn1_types
        nn_features[:, nn1_types.shape[1]:] = nn2_types
        return nn_features
    
    def static_vancancy_k_nearest_target(self, K: int, target_type: int = 1) -> np.ndarray:
        """
        构造每个 vacancy 的 OBS，特征为离它最近的 K 个目标原子（target_type=1, 即 Cu）
        的（向量差 + 距离）。
        
        Args:
            K (int): 要筛选的最近目标原子的数量。
            target_type (int): 目标原子的类型 (修正为 1 代表 Cu)。

        Returns:
            np.ndarray: 新的 OBS 向量，形状为 (n_vacancies, K * 4)。
        """
        
        lattice = self.lattice
        nx, ny, nz = lattice.shape
        
        # 1. 识别所有 Vacancy（源节点）和 Target（Cu 原子）的坐标
        vacancy_pos = self.get_vacancy_array()  # shape: (N_v, 3) - 源空位坐标
        N_v = vacancy_pos.shape[0]

        # 如果没有空位（不应发生）或 K=0，返回零特征
        if N_v == 0 or K == 0:
            return np.zeros((N_v, K * 4), dtype=np.float32)

        # 找出所有目标类型原子（Type=1, Cu）的坐标
        target_indices = np.argwhere(lattice == target_type)
        target_pos = target_indices  # shape: (N_t, 3) - 目标 Cu 原子坐标
        N_t = target_pos.shape[0]
        
        # 如果目标原子数量小于 K，则可能需要填充，但我们先按现有数量处理
        if N_t == 0:
            # 如果没有 Cu 原子，返回全零特征
            return np.zeros((N_v, K * 4), dtype=np.float32)

        # 2. 周期性边界条件下的向量差 (Vector Difference)
        
        # diff_matrix: (源空位 i) 到 (目标 Cu j) 的向量差 [N_v, N_t, 3]
        diff_matrix = target_pos[None, :, :] - vacancy_pos[:, None, :] # 修正：目标减源，得到从源指向目标的向量

        # 应用周期性边界条件 (PBC)
        L = np.array([nx, ny, nz])
        # 找到最短镜像距离
        diff_matrix = diff_matrix - np.round(diff_matrix / L) * L
        
        # 3. 计算欧氏距离 (Distance)
        distance_matrix = np.linalg.norm(diff_matrix, axis=2) # [N_v, N_t]
        
        # 4. ❗ 移除屏蔽自相关距离的步骤 ❗
        # 因为源集(Vacancy)和目标集(Cu)不相交，所以 V_i 到 Cu_j 不可能距离为 0。
        # distance_matrix < 1e-6 
        # distance_matrix[zero_dist_mask] = np.inf
        
        # 5. 筛选最近的 K 个目标原子
        
        # 找到距离矩阵中每行最小 K 个元素的索引
        # indices_k_nearest: [N_v, K]
        indices_k_nearest = np.argsort(distance_matrix, axis=1)[:, :K]
        
        # 6. 构造最终特征向量
        
        # 提取 K 个最近邻的向量差 [N_v, K, 3]
        row_indices = np.arange(N_v)[:, None]
        k_diff_vectors = diff_matrix[row_indices, indices_k_nearest]
        
        # 提取 K 个最近邻的距离 [N_v, K]
        k_distances = distance_matrix[row_indices, indices_k_nearest]
        
        # 拼接特征: [N_v, K, 3] 和 [N_v, K, 1]
        k_distances_expanded = k_distances[:, :, None] # [N_v, K, 1]

        # final_obs: [N_v, K, 4] (3D 向量 + 1D 距离)
        final_obs = np.concatenate([k_diff_vectors, k_distances_expanded], axis=2)
        
        # 7. 展平为最终输出 [N_v, K * 4]
        return final_obs.reshape(N_v, K * 4).astype(np.float32)

    def static_vancancy_combined_obs(self, K: int, target_type: int = 1) -> np.ndarray:
        """
        构造每个 vacancy 的组合 OBS 特征：[1NN/2NN 原子种类 (14)] 
        拼接 [最近 K 个目标原子特征 (K * 4)]。

        Args:
            K (int): 要筛选的最近目标原子的数量 (用于 Top-K 特征)。
            target_type (int): 目标原子的类型 (默认为 1, Cu)。

        Returns:
            np.ndarray: 组合 OBS 向量，形状为 (n_vacancies, 14 + K * 4)。
        """
        
        # 1. 构造 1NN/2NN 原子种类特征 (14 维)
        # ❗ 注意：这个函数返回的是整数类型 np.int32 ❗
        nn_types_feat = self.static_vancancy_local_atoms() 
        
        # 2. 构造 Top-K 近邻特征 (K * 4 维)
        # ❗ 注意：这个函数返回的是浮点数 np.float32 ❗
        topk_feat = self.static_vancancy_k_nearest_target(K=K, target_type=target_type)
        
        # 3. 检查维度一致性 (N_v)
        N_v_nn = nn_types_feat.shape[0]
        N_v_topk = topk_feat.shape[0]

        if N_v_nn != N_v_topk:
            raise ValueError(
                f"特征数量不匹配：1NN/2NN 特征有 {N_v_nn} 个空位，"
                f"Top-K 特征有 {N_v_topk} 个空位。"
            )

        # 4. 拼接特征
        # 为了拼接，通常需要将所有特征转换为相同的浮点类型
        
        # 将 nn_types_feat 转换为 float32，因为 Top-K 特征已经是 float32
        nn_types_feat_float = nn_types_feat.astype(np.float32)

        # 沿特征维度 (axis=1) 拼接
        combined_obs = np.concatenate([nn_types_feat_float, topk_feat], axis=1)
        
        # 最终形状: [N_v, 14 + K * 4]
        return combined_obs

    def step(self):
        rates = self.calculate_diffusion_rate()
        flat_rates = []
        vac_indices = []
        dir_indices = []

        for vac_idx, vac_rates in enumerate(rates):
            for dir_idx, rate in enumerate(vac_rates):
                if rate > 0:
                    flat_rates.append(rate)
                    vac_indices.append(vac_idx)
                    dir_indices.append(dir_idx)

        if not flat_rates:
            return 0.0

        total_rate = np.sum(flat_rates)
        r = np.random.rand() * total_rate
        chosen_idx = np.searchsorted(np.cumsum(flat_rates), r)

        vac_idx = vac_indices[chosen_idx]
        dir_idx = dir_indices[chosen_idx]
        p = self.get_vacancy_pos_by_id(int(vac_idx))
        vi, vj, vk = p
        nn = self.NN1[dir_idx]
        ai = (vi + nn[0]) % self.dims[0]
        aj = (vj + nn[1]) % self.dims[1]
        ak = (vk + nn[2]) % self.dims[2]

        moving_type = self._get_type_from_coord((ai, aj, ak))
        if moving_type == 2:
            delta_t = -np.log(np.random.rand()) / total_rate
            self.time += delta_t
            self.energy_history.append(self.calculate_system_energy())
            self.time_history.append(self.time)
        elif moving_type == 1:
            self.move_vacancy((vi, vj, vk), (ai, aj, ak))
            try:
                idx = self.cu_pos_index.pop((ai, aj, ak))
                cu_idx = int(idx - getattr(self, 'V_nums', 0)) if isinstance(idx, (int, np.integer)) else idx
                if hasattr(self, 'cu_pos') and (self.cu_pos is not None) and (0 <= cu_idx < len(self.cu_pos)):
                    self.cu_pos[cu_idx] = (vi, vj, vk)
                self.cu_pos_index[(vi, vj, vk)] = idx
                self.cu_pos_set.discard((ai, aj, ak))
                self.cu_pos_set.add((vi, vj, vk))
                self.move_cu((ai, aj, ak), (vi, vj, vk))
            except KeyError:
                pass
            delta_t = -np.log(np.random.rand()) / total_rate
            self.time += delta_t
            self.energy_history.append(self.calculate_system_energy())
            self.time_history.append(self.time)
        else:
            self.move_vacancy((vi, vj, vk), (ai, aj, ak))
            delta_t = -np.log(np.random.rand()) / total_rate
            self.time += delta_t
            self.energy_history.append(self.calculate_system_energy())
            self.time_history.append(self.time)
        return delta_t
    
    def visualize(self, title: Optional[str] = None, save_path: Optional[str] = None):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        nx2, ny2, nz2 = self.dims
        lattice_arr = np.full((nx2, ny2, nz2), -1, dtype=np.int8)
        for pos in self.cu_pos_set:
            lattice_arr[pos] = 1
        for pos in self.vac_pos_set:
            lattice_arr[pos] = 2
        return self.plotter.plot_lattice(
            lattice_arr,
            1,
            title=title,
            save_path=save_path
        )
    
    def plot_energy_history(self, save_path: Optional[str] = None):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return self.plotter.plot_energy_evolution(
            self.energy_history,
            self.time_history,
            save_path=save_path
        )


class KMCEnv(KMC):
    def __init__(self, args):
        """KMC 训练环境，提供 reset/step 接口和观测构建。"""
        super().__init__(args)
        self.plotter = Plotter()
        self.args = args
        self.num_agents = args.lattice_v_nums
        self.energy_last = self.calculate_system_energy()
        self.max_ssa_rounds = int(getattr(args, "max_ssa_rounds", 10))
        self.enable_rate_update_timing = bool(getattr(args, "enable_rate_update_timing", False))
        self.sample_rate_update_timing_once = bool(getattr(args, "timing_once", True))
        self._has_sampled_rate_update_timing = False

    def _get_cu_positions(self) -> np.ndarray:
        cu_arr = self.get_cu_array()
        if cu_arr is not None and (not isinstance(cu_arr, np.ndarray) or cu_arr.size > 0):
            cu_pos = np.array(cu_arr, dtype=np.int32)
        elif hasattr(self, "cu_pos_set") and len(self.cu_pos_set) > 0:
            cu_pos = np.array(list(self.cu_pos_set), dtype=np.int32)
        elif hasattr(self, "cu_pos_of_id") and len(getattr(self, "cu_pos_of_id", {})) > 0:
            cu_pos = np.array(list(self.cu_pos_of_id.values()), dtype=np.int32)
        else:
            cu_pos = np.empty((0, 3), dtype=np.int32)
        return cu_pos % np.array(self.args.lattice_size)

    def _get_vacancy_positions(self) -> np.ndarray:
        vac_arr = self.get_vacancy_array()
        if vac_arr is not None and (not isinstance(vac_arr, np.ndarray) or vac_arr.size > 0):
            vac_pos = np.array(vac_arr, dtype=np.int32)
        elif hasattr(self, "vac_pos_set") and len(self.vac_pos_set) > 0:
            vac_pos = np.array(list(self.vac_pos_set), dtype=np.int32)
        elif hasattr(self, "v_pos_of_id") and len(getattr(self, "v_pos_of_id", {})) > 0:
            vac_pos = np.array(list(self.v_pos_of_id.values()), dtype=np.int32)
        else:
            vac_pos = np.empty((0, 3), dtype=np.int32)
        return vac_pos % np.array(self.args.lattice_size)

    def _init_topk_system(self) -> dict[str, Any]:
        vac_local = np.asarray(self.get_vacancy_array(), dtype=np.int32)
        cu_local = self.pure_local_cu_pos if hasattr(self, "pure_local_cu_pos") else (
            self.local_cu_pos if hasattr(self, "local_cu_pos") else self.get_cu_array()
        )
        cu_local = np.asarray(cu_local, dtype=np.int32)
        self.topk_sys = VacancyTopKSystem(
            cu_local,
            vac_local,
            self.topk,
            tuple(self.dims),
            1024,
            device=self.device,
            storage_dtype="float16",
            approximate_mode=False,
        )
        return self.topk_sys.get_all_topk_tensors()

    def _ensure_diffusion_rates(self) -> None:
        if not hasattr(self, "diffusion_rates") or self.diffusion_rates is None:
            self.diffusion_rates = self.calculate_diffusion_rate()

    def _decode_action(self, action: int) -> tuple[int, int, tuple[int, int, int], tuple[int, int, int], int]:
        vac_idx, dir_idx = divmod(int(action), 8)
        vi, vj, vk = map(int, self.get_vacancy_pos_by_id(vac_idx))
        di, dj, dk = map(int, self.NN1[dir_idx])
        ni = (vi + di) % self.dims[0]
        nj = (vj + dj) % self.dims[1]
        nk = (vk + dk) % self.dims[2]
        new_pos = (ni, nj, nk)
        moving_type = int(self._get_type_from_coord(new_pos))
        return vac_idx, dir_idx, (vi, vj, vk), new_pos, moving_type

    def _apply_jump_and_update(self, action: int) -> KMCJumpUpdate:
        vac_idx, dir_idx, old_pos, new_pos, moving_type = self._decode_action(action)
        if moving_type == int(self.V_TYPE):
            topk_update_info = self.topk_sys.get_all_topk_tensors() if hasattr(self, "topk_sys") else {}
            return KMCJumpUpdate(
                vac_idx=vac_idx,
                dir_idx=dir_idx,
                old_pos=old_pos,
                new_pos=new_pos,
                moving_type=moving_type,
                updated_cu=None,
                updated_vacancy=None,
                topk_update_info=topk_update_info,
                cu_move_from=None,
                cu_move_to=None,
                cu_id=None,
            )

        self.move_vacancy(old_pos, new_pos)
        self.update_local_environments(vac_idx, old_pos, new_pos)

        updated_cu = None
        cu_move_from = None
        cu_move_to = None
        cu_id = None
        if moving_type == int(self.CU_TYPE):
            idx = self.cu_pos_index.pop(new_pos)
            cu_idx = int(idx - getattr(self, "V_nums", 0)) if isinstance(idx, (int, np.integer)) else idx
            if hasattr(self, "cu_pos") and (self.cu_pos is not None) and (0 <= cu_idx < len(self.cu_pos)):
                self.cu_pos[cu_idx] = old_pos
            self.cu_pos_index[old_pos] = idx
            self.move_cu(new_pos, old_pos)
            updated_cu = {int(cu_idx): np.array(old_pos)}
            cu_move_from = tuple(map(int, new_pos))
            cu_move_to = tuple(map(int, old_pos))
            cu_id = int(idx)

        self.diffusion_rates_update([old_pos, new_pos])

        updated_vacancy = {
            int(vac_idx): np.vstack(
                [
                    np.array(old_pos, dtype=np.float32),
                    np.array(new_pos, dtype=np.float32),
                ]
            )
        }
        topk_update_info = self.topk_sys.update_system(updated_cu=updated_cu, updated_vacancy=updated_vacancy)

        return KMCJumpUpdate(
            vac_idx=vac_idx,
            dir_idx=dir_idx,
            old_pos=old_pos,
            new_pos=new_pos,
            moving_type=moving_type,
            updated_cu=updated_cu,
            updated_vacancy=updated_vacancy,
            topk_update_info=topk_update_info,
            cu_move_from=cu_move_from,
            cu_move_to=cu_move_to,
            cu_id=cu_id,
        )
    
    def _minimum_image_displacements(self, src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> np.ndarray:
        delta = src[:, None, :] - dst[None, :, :]
        return delta - np.round(delta / box[None, None, :]) * box[None, None, :]

    def _query_ball_counts_numpy(self, src: np.ndarray, dst: np.ndarray, radius: float, box: np.ndarray) -> np.ndarray:
        disp = self._minimum_image_displacements(src.astype(np.float32), dst.astype(np.float32), box.astype(np.float32))
        dist = np.linalg.norm(disp, axis=-1)
        return np.sum(dist <= float(radius), axis=1)

    def _nearest_distances_numpy(self, src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> np.ndarray:
        disp = self._minimum_image_displacements(src.astype(np.float32), dst.astype(np.float32), box.astype(np.float32))
        dist = np.linalg.norm(disp, axis=-1)
        return np.min(dist, axis=1)

    def get_system_stats(self):
        """
        返回当前系统统计特征（shape=(10,)）：
        - Cu 坐标均值与标准差（6）
        - Cu 局部密度的变异系数 CV（1）
        - 空位到最近 Cu 的平均距离（1）
        - Cu 位于空位邻域比例（1）
        """
        lattice_size = np.array(self.args.lattice_size)
        a = 2.0
        cu_pos = self._get_cu_positions() % lattice_size
        vac_pos = self._get_vacancy_positions() % lattice_size
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        stats = []
        # Cu 均值/标准差
        if cu_pos.size > 0:
            cu_mean = cu_pos.mean(axis=0)  # shape (3,)
            cu_std = cu_pos.std(axis=0)    # shape (3,)
        else:
            cu_mean = cu_std = np.zeros(3)

        stats.extend([*cu_mean, *cu_std])

        # Cu 聚集性（CV）
        cu_cv = 0.0
        if len(cu_pos) >= 2:
            if cKDTree is not None:
                cu_tree = cKDTree(cu_pos, boxsize=lattice_size)
                counts = np.asarray(cu_tree.query_ball_point(cu_pos, r=2 * a, return_length=True)) - 1
            else:
                counts = self._query_ball_counts_numpy(cu_pos, cu_pos, radius=2 * a, box=lattice_size) - 1
            if counts.mean() > 0:
                cu_cv = counts.std() / counts.mean()
        stats.append(cu_cv)

        # 空位到最近 Cu 的平均距离
        avg_vac_to_cu = 0.0
        if cu_pos.size > 0 and vac_pos.size > 0:
            if cKDTree is not None:
                if 'cu_tree' not in locals():
                    cu_tree = cKDTree(cu_pos, boxsize=lattice_size)
                dists, _ = cu_tree.query(vac_pos, k=1)
            else:
                dists = self._nearest_distances_numpy(vac_pos, cu_pos, box=lattice_size)
            avg_vac_to_cu = dists.mean()
        stats.append(avg_vac_to_cu)

        # Cu 位于空位邻域的比例
        cu_near_ratio = 0.0
        if cu_pos.size > 0 and vac_pos.size > 0:
            if cKDTree is not None:
                vac_tree = cKDTree(vac_pos, boxsize=lattice_size)
                nearby_counts = vac_tree.query_ball_point(cu_pos, r=2 * a, return_length=True)
            else:
                nearby_counts = self._query_ball_counts_numpy(cu_pos, vac_pos, radius=2 * a, box=lattice_size)
            cu_near_ratio = (np.asarray(nearby_counts) > 0).mean()
        stats.append(cu_near_ratio)

        return np.array(stats)

    def get_cu_isolated_fraction(self, radius: float = None) -> float:
        lattice_size = np.array(self.args.lattice_size)
        a = 2.0
        r = 2 * a if (radius is None) else float(radius)
        cu_pos = self._get_cu_positions() % lattice_size
        if cu_pos.size == 0:
            return 0.0
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        if cKDTree is not None:
            cu_tree = cKDTree(cu_pos, boxsize=lattice_size)
            counts = np.asarray(cu_tree.query_ball_point(cu_pos, r=r, return_length=True)) - 1
        else:
            counts = self._query_ball_counts_numpy(cu_pos, cu_pos, radius=r, box=lattice_size) - 1
        iso_frac = float((np.asarray(counts) == 0).mean())
        return iso_frac


    def reset(self):
        """
        重置时间与历史统计，构建初始观测（局部类型特征 + TopK 信息 + 全局统计）。
        """
        self.time = 0.0
        self.energy_history = []
        self.time_history = []
        initial_energy = self.calculate_system_energy()
        self.energy_last = initial_energy
        if not bool(getattr(self.args, "skip_stats", False)):
            self.energy_history.append(initial_energy)
            self.time_history.append(0.0)
        topk_update_info = self._init_topk_system()

        
        compute_global_env = bool(getattr(self.args, "compute_global_static_env_reset", True))
        if compute_global_env:
            type_obs = self.static_vancancy_local_atoms()
        else:
            type_obs = np.zeros((0, 14), dtype=np.int32)
        initial_obs = {
            "V_features_local": type_obs,
            "topk_update_info": topk_update_info,
        }

        if bool(getattr(self.args, "skip_stats", False)):
            share_obs = np.array([], dtype=float)
            full_obs = type_obs.flatten()
        else:
            share_obs = self.get_system_stats()
            full_obs = np.concatenate([type_obs.flatten(), share_obs], axis=0)
        
        if not bool(getattr(self.args, "skip_global_diffusion_reset", False)):
            self.diffusion_rates = self.calculate_diffusion_rate()
        
        
        return initial_obs, full_obs
            
    def rate_weight(self, epoch, warmup_epoch=20, sharpness=1):
        """
        返回进度权重（Sigmoid 形状）：
        - `warmup_epoch` 之前返回 0
        - 之后按 `max_ssa_rounds` 归一化进度计算权重
        """
        max_epoch = self.max_ssa_rounds
        if epoch < warmup_epoch:
            return 0.0
        progress = (epoch - warmup_epoch) / (max_epoch - warmup_epoch)
        return 1 / (1 + np.exp(-sharpness * (progress - 0.5)))

    def step_with_stats(self, action, episode):
        """
        执行单步 KMC：
        - 解析动作并执行跳跃、更新局部环境与 TopK
        - 以总速率采样推进物理时间并更新能量与奖励
        - 返回局部观测、扁平观测、位置、奖励与统计信息
        """
        self._ensure_diffusion_rates()
        flat_rates = [rate for vac_rates in self.diffusion_rates for rate in vac_rates if rate > 0]
        total_rate = float(np.sum(flat_rates)) if len(flat_rates) > 0 else 0.0
        jump = self._apply_jump_and_update(action)

        # ---------- 3. 时间推进 ----------
        if total_rate > 0.0:
            rand_t = np.random.rand()
            delta_t = -np.log(rand_t) / total_rate
        else:
            delta_t = 0.0
        self.time += delta_t
        self.time_history.append(self.time)

        # ---------- 4. 奖励计算 ----------
        energy_after = self.calculate_system_energy()
        delta_E = self.energy_last - energy_after
        reward = delta_E * self.args.reward_scale
        self.energy_last = energy_after
        self.energy_history.append(energy_after)

        # ---------- 5. 状态更新 ----------
        type_obs = self.static_vancancy_local_atoms()
        obs = {
            'V_features_local': type_obs,
            'topk_update_info': jump.topk_update_info,
        }

        share_obs = self.get_system_stats()
        full_obs = np.concatenate([type_obs.flatten(), share_obs], axis=0)
        positions = self.get_vacancy_array()

        infos = {
            'individual_reward': reward,
            'energy_change': delta_E,
            'time': self.time,
            'delta_t': float(delta_t)
        }

        return obs, full_obs, positions, reward, False, infos


    def step_fast(self, action, episode, verbose=False):
        """
        仅执行跳跃与最小必要更新（快速版）：
        - 跳跃与局部环境更新
        - TopK 增量更新
        - 返回局部特征与 TopK 信息（不含统计）
        """
        self._ensure_diffusion_rates()
        jump = self._apply_jump_and_update(action)

        # accept_prob = rate / total_rate
        # u = np.random.rand()
        # n_trials = int(np.ceil(np.log(1 - u) / np.log(1 - accept_prob)))
        # delta_t = n_trials * (-np.log(np.random.rand()) / total_rate)
        # self.time += delta_t

        type_obs = self.static_vancancy_local_atoms()
        obs = {
            'V_features_local': type_obs,
            'topk_update_info': jump.topk_update_info,
            'updated_cu': jump.updated_cu,
            'updated_vacancy': jump.updated_vacancy,
            'cu_move_from': jump.cu_move_from,
            'cu_move_to': jump.cu_move_to,
            'cu_id': jump.cu_id,
            'dir_idx': int(jump.dir_idx)
        }
        return obs

    # 兼容旧名
    def step(self, action, episode):
        return self.step_with_stats(action, episode)

    def step_only_jump(self, action, episode, verbose=False):
        return self.step_fast(action, episode, verbose=verbose)


    def update_local_environments(self, vac_idx: int, old_pos, new_pos):
        """
        根据空位 V 和原子 A 的一次交换跳跃，更新所有局部环境张量。

        Args:
            vac_idx: 当前发生跳跃的空位在 self.vacancy_pos 中的索引。
            old_pos: (3,) 空位旧位置 (原子 A 的新位置)。
            new_pos: (3,) 空位新位置 (原子 A 的旧位置)。
        """
        
        # --- 1. 获取和准备数据 ---
        
        # 空位索引：我们只处理这一个空位，所以 M=1
        M = 1
        
        # 1NN 偏移量 (8, 3)
        NN1 = self.NN1 
        # 2NN 偏移量 (6, 3)
        NN2 = self.NN2
        
        # 新旧位置的单行数组 (1, 3)
        old_pos_1x3 = old_pos
        new_pos_1x3 = new_pos
        
        # -----------------------------------------------------
        # Part A: 直接更新 (计算新 V 和新 A 周围的环境)
        # -----------------------------------------------------

        # A.1. 计算新 V (在 new_pos) 周围的环境
        
        # 新 V 的 1NN 坐标 (1, 8, 3) -> (8, 3)
        new_V_nn1_coords = self._get_pbc_coord(new_pos_1x3, NN1[None, :, :], self.dims).reshape(-1, 3)
        # 新 V 的 2NN 坐标 (1, 6, 3) -> (6, 3)
        new_V_nn2_coords = self._get_pbc_coord(new_pos_1x3, NN2[None, :, :], self.dims).reshape(-1, 3)

        # 批量查找新 V 的 NN 类型 (使用稀疏查找)
        new_V_nn1_types = self._batch_get_type_from_coords(new_V_nn1_coords) # (8,)
        new_V_nn2_types = self._batch_get_type_from_coords(new_V_nn2_coords) # (6,)

        # 更新张量
        self.nn1_types[vac_idx, :] = new_V_nn1_types
        self.nn2_types[vac_idx, :] = new_V_nn2_types

        # A.2. 计算原 A (在 old_pos, 现在是 V) 的 1NN 邻居 A_j 的环境
        
        # 原 A 的 1NN 坐标 (8, 3)
        A_old_nn1_coords = self._get_pbc_coord(old_pos_1x3, NN1[None, :, :], self.dims).reshape(-1, 3)

        # A_old_nn1_coords (8, 3) 作为中心点，计算它们的 1NN 和 2NN 坐标
        M_A_nn1 = 8
        
        # 1NN 的 1NN 坐标: (8, 8, 3) -> (64, 3)
        A_nn1_nn1_coords = self._get_pbc_coord(A_old_nn1_coords[:, None, :], NN1[None, :, :], self.dims).reshape(-1, 3)
        # 1NN 的 2NN 坐标: (8, 6, 3) -> (48, 3)
        A_nn1_nn2_coords = self._get_pbc_coord(A_old_nn1_coords[:, None, :], NN2[None, :, :], self.dims).reshape(-1, 3)
        
        # 批量查找 NN 类型
        new_nn1_nn1_types = self._batch_get_type_from_coords(A_nn1_nn1_coords).reshape(M_A_nn1, 8) # (8, 8)
        new_nn1_nn2_types = self._batch_get_type_from_coords(A_nn1_nn2_coords).reshape(M_A_nn1, 6) # (8, 6)

        # 更新张量
        self.nn1_nn1_types[vac_idx, :, :] = new_nn1_nn1_types
        self.nn1_nn2_types[vac_idx, :, :] = new_nn1_nn2_types

        # -----------------------------------------------------
        # Part B: 间接更新 (更新 V 和 A 的邻居 N 的环境)
        # -----------------------------------------------------
        
        # B.1. 找出所有受影响的邻居 N 的坐标
        
        # 所有 1NN/2NN 邻居的坐标 (旧 V 邻居 + 旧 A 邻居)
        # V 的 1NN/2NN 坐标 (14, 3)
        V_old_nn_coords = self._get_pbc_coord(old_pos_1x3, np.vstack((NN1, NN2))[None, :, :], self.dims).reshape(-1, 3)
        # A 的 1NN/2NN 坐标 (14, 3)
        A_old_nn_coords = self._get_pbc_coord(new_pos_1x3, np.vstack((NN1, NN2))[None, :, :], self.dims).reshape(-1, 3)
        
        # 合并并去重受影响的邻居坐标
        affected_coords = np.vstack((V_old_nn_coords, A_old_nn_coords))
        # 必须转换为元组，然后转回数组才能进行唯一的哈希查找
        affected_coords_unique_tuples = set(map(tuple, np.round(affected_coords).astype(int)))
        affected_coords_unique = np.array(list(affected_coords_unique_tuples)) # (N_affected, 3)
        
        N_affected = len(affected_coords_unique)
        if N_affected == 0:
            return

        # B.2. 对每个受影响的邻居 N，更新其在 nn1_nn1_types 和 nn1_nn2_types 中的项

        # 由于我们只关心单个空位 vac_idx，所以只更新这一个索引的张量项
        
        # 确定 V_old 和 A_old 在邻居 N 的 NN1 列表中的索引
        # V_old 在 N 的 NN1 列表中的偏移向量: -(old_pos - N)
        # A_old 在 N 的 NN1 列表中的偏移向量: -(new_pos - N)
        
        # 我们需要知道每个受影响坐标 N 在 self.vacancy_pos 中的索引
        # 由于 KMC 循环中我们只关心 vac_idx，所以我们只更新 V_idx 处的张量。
        # 对于 LAKIMOC 修正项来说，只需要确保 V_idx 处的环境更新正确即可。

        # 简化的间接更新 (只更新 V_idx 处的环境张量)
        # 这要求 KMC 维护 NN1_NN1_types 和 NN1_NN2_types 是针对所有空位的环境张量
        
        # 在 nn1_nn1_types[vac_idx, :, :] 中：
        #   - 对应 old_pos 的项 (V_old) 变为 A_new (即 V_TYPE)
        #   - 对应 new_pos 的项 (A_old) 变为 V_new (即 A_TYPE)
        
        # 查找 old_pos 在 A_old_nn1_coords 中的索引 k_A (应为 A->V 方向)
        # 查找 new_pos 在 A_old_nn1_coords 中的索引 k_V (应为 V->A 方向)
        
        # ... 这部分逻辑过于复杂且依赖于 NN1/NN2 的具体排列顺序 ...
        
        # **最安全的间接更新策略是：** # 重新计算 A_old (现在是 V) 周围所有邻居的环境，但由于它们是 Fe 或 Cu，我们没有为它们预先存储张量。
        
        # **最终简化方案：** 依赖于 A.2 已经计算了 A_old 邻居的环境。

        # 间接更新只对全局属性有效。对于稀疏 KMC，我们假设：
        # **当且仅当一个空位发生移动时，我们才更新该空位索引下的所有局部环境张量。**
        
        # 因此，Part A 已经足够覆盖 vac_idx 对应的四个张量。
        # 对于其他空位 vac_idx' 的张量，它们只有在 vac_idx' 附近的原子类型发生变化时才需要更新，
        # 此时只需重新计算那个特定的空位环境即可 (在下一次 rate update 时)。
        
        # 保持 Part A 即可，因为它解决了 vac_idx 处的张量更新问题。
        return
    

    def get_states(self):
        """
        Input: None
        Output: torch.Tensor - shape=(n_vacancies, 6)
            First dim: number of vacancies
            Second dim: [NN1_Fe, NN1_Cu, NN1_V, NN2_Fe, NN2_Cu, NN2_V]
        """
        local_env = torch.tensor(self.static_vancancy_local_atoms(), dtype=torch.int64, device='cuda')
        return local_env

    def get_base_rates(self):
        """
        Input: None
        Output: torch.Tensor - shape=(n_vacancies, 8)
            First dim: number of vacancies
            Second dim: transition rates for 8 NN1 directions (Hz)
        """
        base_rates = torch.tensor(self.calculate_diffusion_rate(), dtype=torch.float32, device='cuda')
        return base_rates
