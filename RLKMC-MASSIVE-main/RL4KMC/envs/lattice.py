from typing import Tuple
import numpy as np
import torch
from RL4KMC.config import CONFIG
import os
import time

FE_TYPE = 0
CU_TYPE = 1
V_TYPE = 2 # 空位

class Lattice:
    def __init__(self, args):
        self.args = args
        self.size = args.lattice_size
        nx, ny, nz = self.size
        
        self.dims = [nx*2, ny*2, nz*2]
        
        total_half_sites = nx * ny * nz * 2
        cu_nums_arg = getattr(args, 'lattice_cu_nums', None)
        v_nums_arg = getattr(args, 'lattice_v_nums', None)
        cu_density = getattr(args, 'cu_density', None)
        v_density = getattr(args, 'v_density', None)
        if (cu_nums_arg is None or int(cu_nums_arg) <= 0) and cu_density is not None:
            self.Cu_nums = int(round(float(cu_density) * total_half_sites))
        else:
            self.Cu_nums = int(cu_nums_arg)
        if (v_nums_arg is None or int(v_nums_arg) <= 0) and v_density is not None:
            self.V_nums = int(round(float(v_density) * total_half_sites))
        else:
            self.V_nums = int(v_nums_arg)
        self.NN1 = np.array([
            [1, 1, 1],  [-1, -1, -1],  
            [1, 1, -1], [-1, -1, 1],   
            [1, -1, 1], [-1, 1, -1],   
            [-1, 1, 1], [1, -1, -1]    
        ], dtype=np.int32)
        self.NN2 = np.array([
            [2, 0, 0], [-2, 0, 0], 
            [0, 2, 0], [0, -2, 0],
            [0, 0, 2], [0, 0, -2]
        ], dtype=np.int32)
        
        
        # self.coords = self.generate_lattice_coordinates()
        self.coords = None
        
        
        self.FE_TYPE = 0
        self.CU_TYPE = 1
        self.V_TYPE = 2
        self.cu_pos = None
        self.vac_pos_set = set()
        self.cu_pos_set = set()
        self.cu_pos_index = None
        self.device = torch.device(CONFIG.runner.device)
        self.init_lattice()
        

    # def generate_lattice_coordinates(self):
    #     nx, ny, nz = self.size
    #     corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
    #     body = corner + 1
    #     return np.vstack((corner, body))

    # def generate_lattice_coordinates(self):
        
    #     # 1. 确定设备和尺寸
    #     device = self.device if hasattr(self, 'device') else "cuda" 
        
    #     # 确保 self.size 是可迭代的
    #     nx, ny, nz = self.size
        
    #     # 2. 在 GPU 上生成角点 (Corner) 坐标
        
    #     # 生成 x, y, z 三个维度上的坐标网格
    #     X, Y, Z = torch.meshgrid(
    #         torch.arange(nx, device=device),
    #         torch.arange(ny, device=device),
    #         torch.arange(nz, device=device),
    #         indexing='ij' # 使用 'ij' 索引
    #     )
        
    #     # 将三个网格堆叠成 [3, nx, ny, nz] 的 Tensor
    #     indices = torch.stack((X, Y, Z), dim=0) 
        
    #     # 重新塑形：将 (nx, ny, nz) 维度展平，得到 [3, N_cells]
    #     corner_indices = indices.reshape(3, -1) 
        
    #     # 核心计算：角点坐标 = 2 * 晶胞索引
    #     # 形状：[3, N_cells]
    #     corner = corner_indices * 2
        
    #     # 3. 在 GPU 上生成体心 (Body) 坐标
        
    #     # 核心计算：体心坐标 = 角点坐标 + 1
    #     body = corner + 1
        
    #     # 4. 在 GPU 上合并和转置
        
    #     # 水平堆叠：[3, 2 * N_cells]
    #     all_coords_T = torch.hstack((corner, body))
        
    #     # 转置成 [2 * N_cells, 3]，即 [N_atoms, 3]
    #     final_coords_tensor = all_coords_T.T
        
    #     # 5. 转移到 CPU 并转换为 NumPy 数组 (确保输出格式)
        
    #     # .cpu() 将 Tensor 移回 CPU
    #     # .numpy() 将 CPU Tensor 转换为 NumPy ndarray
    #     return final_coords_tensor.cpu().numpy()

    def generate_lattice_coordinates(self):
        
        # 1. 确定设备和尺寸
        device = self.device
        nx, ny, nz = self.size
        
        # ----------------------------------------------------
        # 确定分块策略
        # 我们按 Z 维度进行分块，因为 Z 维度通常是生成坐标网格的内层循环
        # 假设一次处理 50 个 Z 维度，即 500*500*50 个晶胞，约 2500 万个晶胞
        N_Z_CHUNK = 50 
        
        # 最终用于收集所有坐标的列表
        all_coords_list = []
        # ----------------------------------------------------
        
        # 2. 按 Z 维度分块循环
        for z_start in range(0, nz, N_Z_CHUNK):
            z_end = min(z_start + N_Z_CHUNK, nz)
            current_nz = z_end - z_start
            
            # --- 2.1 在 GPU 上生成当前块的坐标网格 ---
            
            # 生成 X, Y, Z 三个维度上的坐标网格 (仅限于当前 Z 范围)
            X, Y, Z = torch.meshgrid(
                torch.arange(nx, device=device),
                torch.arange(ny, device=device),
                torch.arange(z_start, z_end, device=device), # Z 维度分块
                indexing='ij'
            )
            
            # 将三个网格堆叠成 [3, nx, ny, current_nz] 的 Tensor
            indices = torch.stack((X, Y, Z), dim=0) 
            
            # 重新塑形：得到 [3, N_cells_chunk]
            corner_indices_chunk = indices.reshape(3, -1) 
            
            # --- 2.2 计算当前块的原子坐标 ---
            
            # 角点坐标 = 2 * 晶胞索引
            corner_chunk = corner_indices_chunk * 2
            
            # 体心坐标 = 角点坐标 + 1
            body_chunk = corner_chunk + 1
            
            # 合并和转置成 [N_atoms_chunk, 3]
            all_coords_T_chunk = torch.hstack((corner_chunk, body_chunk))
            final_coords_tensor_chunk = all_coords_T_chunk.T
            
            # --- 2.3 收集结果并清理内存 ---
            
            # 移回 CPU 并转为 NumPy 数组
            all_coords_list.append(final_coords_tensor_chunk.cpu().numpy())
            
            # 清理 GPU 内存
            del X, Y, Z, indices, corner_indices_chunk, corner_chunk, body_chunk, all_coords_T_chunk, final_coords_tensor_chunk
            torch.cuda.empty_cache()

        # 3. 合并所有 NumPy 数组并返回
        
        # 使用 np.vstack 合并列表中的所有 NumPy 数组
        return np.vstack(all_coords_list)

    def _build_cu_pos_index(self):
            self.cu_pos_index = {tuple(pos): (idx+self.V_nums) for idx, pos in enumerate(self.cu_pos)} 
            # print("self.cu_pos_index", self.cu_pos_index)
            # self.cu_pos_index = {tuple(pos): idx for idx, pos in enumerate(self.cu_pos)} 

    def update_cu_pos(self):
        return None

    def get_coords_vectorized(self, indices, nx, ny, nz):
        """
        [向量化方法] 根据一维索引数组计算其对应的三维晶格坐标数组。
        """
        num_half_sites = nx * ny * nz
        
        # 1. 确定子晶格类型 (corner=0, body=1)
        sub_lattice_type = indices // num_half_sites 
        
        # 2. 确定在 nx*ny*nz 空间中的局部索引
        sub_idx = indices % num_half_sites 

        # 3. 向量化计算 (x, y, z) 在 nx*ny*nz 空间中的坐标
        z = sub_idx // (nx * ny)
        y = (sub_idx % (nx * ny)) // nx
        x = sub_idx % nx
        
        # 4. 映射到实际的 (2nx, 2ny, 2nz) 晶格坐标
        # base_coords 形状为 (N, 3)，使用 int16 节省空间
        base_coords = np.stack([2*x, 2*y, 2*z], axis=1).astype(np.int32)
        
        # 5. 应用 body 偏移 (1, 1, 1)
        # body_offset 是一个 N x 3 的数组，值为 0 或 1
        body_offset = np.stack([sub_lattice_type] * 3, axis=1).astype(np.int32)
        
        final_coords = base_coords + body_offset
        
        return final_coords

    def get_coords_vectorized_local(self, indices, nx, ny, nz, offset=None):
        indices = np.asarray(indices, dtype=np.int64)
        num_half_sites = int(nx) * int(ny) * int(nz)
        sub_lattice_type = indices // num_half_sites
        sub_idx = indices % num_half_sites
        z = sub_idx // (int(nx) * int(ny))
        y = (sub_idx % (int(nx) * int(ny))) // int(nx)
        x = sub_idx % int(nx)
        base_coords = np.stack([2 * x, 2 * y, 2 * z], axis=1).astype(np.int32)
        body_offset = np.stack([sub_lattice_type] * 3, axis=1).astype(np.int32)
        coords = base_coords + body_offset
        if offset is not None:
            o = np.asarray(offset, dtype=np.int64)
            coords = (coords.astype(np.int64) + o).astype(np.int32)
        return coords

    def check_coords_vectorized_local_negatives(self, indices, nx, ny, nz, offset=None):
        indices = np.asarray(indices, dtype=np.int64)
        num_half_sites = int(nx) * int(ny) * int(nz)
        sub_lattice_type = indices // num_half_sites
        sub_idx = indices % num_half_sites
        z = sub_idx // (int(nx) * int(ny))
        y = (sub_idx % (int(nx) * int(ny))) // int(nx)
        x = sub_idx % int(nx)
        base_coords = np.stack([2 * x, 2 * y, 2 * z], axis=1).astype(np.int32)
        body_offset = np.stack([sub_lattice_type] * 3, axis=1).astype(np.int32)
        coords = base_coords + body_offset
        def neg_info(arr, name):
            neg_mask = (arr < 0) if arr.ndim == 1 else (arr < 0).any(axis=1)
            print(f"{name}: neg_count={int(neg_mask.sum())}")
            if neg_mask.any():
                print(f"{name}: neg_samples={arr[neg_mask][:min(5, len(arr[neg_mask]))].tolist()}")
        print(f"indices_len={int(len(indices))} nx={int(nx)} ny={int(ny)} nz={int(nz)} num_half_sites={int(num_half_sites)}")
        neg_info(indices, "indices")
        neg_info(sub_lattice_type, "sub_lattice_type")
        neg_info(base_coords, "base_coords")
        neg_info(body_offset, "body_offset")
        neg_info(coords, "coords_before_offset")
        if offset is not None:
            o = np.asarray(offset, dtype=np.int64)
            coords_off = (coords.astype(np.int64) + o).astype(np.int32)
            neg_info(coords_off, "coords_after_offset")
        return coords

    # def _get_type_from_coord(self, positions: np.ndarray) -> np.ndarray:
    #     """
    #     批量查找给定 (N, 3) 坐标数组中每个位置的原子类型。

    #     Args:
    #         positions: (N, 3) 整数坐标数组。
            
    #     Returns:
    #         (N,) 原子类型数组。
    #     """
    #     N = len(positions)
    #     if N == 0:
    #         return np.empty(0, dtype=int)
            
    #     # 1. 确保坐标是整数并转换为元组列表
    #     # 这一步非常关键，因为 PBC 运算可能产生浮点数
    #     # positions_int = np.round(positions).astype(int)
    #     # # positions_int = np.round(positions).astype(int)
    #     print("positions", positions)
    #     pos_tuples = [tuple(p) for p in positions]
        
    #     types = np.full(N, FE_TYPE, dtype=int) # 默认类型为 Fe (0)

    #     # 2. 遍历查找并更新类型
    #     # 由于依赖哈希表，这里必须使用循环，但它是 O(N) 的高效查找
    #     for i in range(N):
    #         pos_tuple = pos_tuples[i]
            
    #         if pos_tuple in self.vac_pos_set:
    #             types[i] = V_TYPE
    #         elif pos_tuple in self.cu_pos_set:
    #             types[i] = CU_TYPE
    #         # 否则保持 FE_TYPE (0)
            
    #     return types
    
    def _get_type_from_coord(self, pos: np.ndarray) -> int:
        """
        根据坐标 (x, y, z) 查找原子类型。
        这是替代 lattice[x, y, z] 的核心函数。
        
        Args:
            pos: (3,) 坐标数组 [x, y, z]。
        
        Returns:
            原子类型 (0: Fe, 1: Cu, 2: V)。
        """
        # print("pos", pos)
        pos_tuple = tuple(pos)

        # for i in range(N):
        #     pos_tuple = pos_tuples[i]
            
        if pos_tuple in self.vac_pos_set:
            return self.V_TYPE
        elif pos_tuple in self.cu_pos_set:
            return self.CU_TYPE
                
        # # 1. 查询 Cu 原子
        # if pos_tuple in self.cu_pos_index:
        #     return CU_TYPE
        
        # # 2. 查询空位
        # # 由于 vacancy_pos 是一个数组，查询效率较低。
        # # 更好的做法是维护一个 self.vac_pos_set 或 self.vac_pos_index。
        # # 这里为了演示，我们假设空位数量 Nv 较小，可以接受遍历查找。
        # # 在实际高性能代码中，应使用哈希表。
        
        # # 检查坐标是否是当前空位之一
        # # WARNING: 这是一个慢速操作 (O(N_v))。建议在类中维护 self.vac_pos_set
        
        # if np.any(np.all(self.vacancy_pos == pos, axis=1)):
        #      return V_TYPE
        
        # 3. 否则，假设它是 Fe 原子 (背景原子)
        return self.FE_TYPE

    def _batch_get_type_from_coords(self, positions: np.ndarray) -> np.ndarray:
        """
        批量查找给定 (N, 3) 坐标数组中每个位置的原子类型。

        Args:
            positions: (N, 3) 整数坐标数组。
            
        Returns:
            (N,) 原子类型数组。
        """
        # print(f"rank {self.rank} _batch_get_type_from_coords positions={positions}")
        N = len(positions)
        if N == 0:
            return np.empty(0, dtype=np.int8)
        positions_int = np.round(positions).astype(np.int32)
        D = np.array(self.dims, dtype=int)
        lin_idx = (((positions_int[:,0] * D[1]) + positions_int[:,1]) * D[2] + positions_int[:,2]).astype(np.int64)
        types = np.full(N, self.FE_TYPE, dtype=np.int8)
        use_gpu = torch.cuda.is_available() and hasattr(self, '_global_vac_lin_sorted_t') and hasattr(self, '_global_cu_lin_sorted_t')
        if use_gpu:
            lin_t = torch.as_tensor(lin_idx, dtype=torch.int64, device=self.device)
            if getattr(self, '_global_vac_lin_sorted_t', None) is not None and self._global_vac_lin_sorted_t.numel() > 0:
                idx_v = torch.searchsorted(self._global_vac_lin_sorted_t, lin_t)
                in_range_v = idx_v < self._global_vac_lin_sorted_t.numel()
                idx_vc = torch.clamp(idx_v, 0, max(0, int(self._global_vac_lin_sorted_t.numel()-1)))
                mask_v = in_range_v & (self._global_vac_lin_sorted_t[idx_vc] == lin_t)
                types[mask_v.cpu().numpy()] = int(self.V_TYPE)
            if getattr(self, '_global_cu_lin_sorted_t', None) is not None and self._global_cu_lin_sorted_t.numel() > 0:
                idx_c = torch.searchsorted(self._global_cu_lin_sorted_t, lin_t)
                in_range_c = idx_c < self._global_cu_lin_sorted_t.numel()
                idx_cc = torch.clamp(idx_c, 0, max(0, int(self._global_cu_lin_sorted_t.numel()-1)))
                mask_c = in_range_c & (self._global_cu_lin_sorted_t[idx_cc] == lin_t)
                types[mask_c.cpu().numpy()] = int(self.CU_TYPE)
            return types
        if hasattr(self, '_global_vac_lin_sorted') and self._global_vac_lin_sorted.size > 0:
            idx_v = np.searchsorted(self._global_vac_lin_sorted, lin_idx)
            in_range_v = (idx_v < self._global_vac_lin_sorted.size)
            mask_v = in_range_v & (self._global_vac_lin_sorted[np.clip(idx_v, 0, self._global_vac_lin_sorted.size-1)] == lin_idx)
            types[mask_v] = int(self.V_TYPE)
        if hasattr(self, '_global_cu_lin_sorted') and self._global_cu_lin_sorted.size > 0:
            idx_c = np.searchsorted(self._global_cu_lin_sorted, lin_idx)
            in_range_c = (idx_c < self._global_cu_lin_sorted.size)
            mask_c = in_range_c & (self._global_cu_lin_sorted[np.clip(idx_c, 0, self._global_cu_lin_sorted.size-1)] == lin_idx)
            types[mask_c] = int(self.CU_TYPE)
        return types

    def _rebuild_global_lin_cache(self):
        D = np.array(self.dims, dtype=int)
        vac_arr = self.get_vacancy_array()
        cu_arr = self.get_cu_array()
        if vac_arr is None:
            vac_arr = np.empty((0,3), dtype=np.int32)
        if cu_arr is None:
            cu_arr = np.empty((0,3), dtype=np.int32)
        if vac_arr.size > 0:
            lin_v = (((vac_arr[:,0] * D[1]) + vac_arr[:,1]) * D[2] + vac_arr[:,2]).astype(np.int64)
            self._global_vac_lin_sorted = np.array(sorted(lin_v.tolist()), dtype=np.int64)
            if torch.cuda.is_available():
                self._global_vac_lin_sorted_t = torch.as_tensor(self._global_vac_lin_sorted, dtype=torch.int64, device=self.device)
        else:
            self._global_vac_lin_sorted = np.empty((0,), dtype=np.int64)
            if torch.cuda.is_available():
                self._global_vac_lin_sorted_t = torch.as_tensor(self._global_vac_lin_sorted, dtype=torch.int64, device=self.device)
        if cu_arr.size > 0:
            lin_c = (((cu_arr[:,0] * D[1]) + cu_arr[:,1]) * D[2] + cu_arr[:,2]).astype(np.int64)
            self._global_cu_lin_sorted = np.array(sorted(lin_c.tolist()), dtype=np.int64)
            if torch.cuda.is_available():
                self._global_cu_lin_sorted_t = torch.as_tensor(self._global_cu_lin_sorted, dtype=torch.int64, device=self.device)
        else:
            self._global_cu_lin_sorted = np.empty((0,), dtype=np.int64)
            if torch.cuda.is_available():
                self._global_cu_lin_sorted_t = torch.as_tensor(self._global_cu_lin_sorted, dtype=torch.int64, device=self.device)
    def _validate_global_lin_cache(self):
        D = np.array(self.dims, dtype=int)
        vac_arr = self.get_vacancy_array()
        cu_arr = self.get_cu_array()
        if vac_arr is None:
            vac_arr = np.empty((0,3), dtype=np.int32)
        if cu_arr is None:
            cu_arr = np.empty((0,3), dtype=np.int32)
        lin_v = (((vac_arr[:,0] * D[1]) + vac_arr[:,1]) * D[2] + vac_arr[:,2]).astype(np.int64) if vac_arr.size > 0 else np.empty((0,), dtype=np.int64)
        lin_c = (((cu_arr[:,0] * D[1]) + cu_arr[:,1]) * D[2] + cu_arr[:,2]).astype(np.int64) if cu_arr.size > 0 else np.empty((0,), dtype=np.int64)
        exp_v = np.array(sorted(lin_v.tolist()), dtype=np.int64)
        exp_c = np.array(sorted(lin_c.tolist()), dtype=np.int64)
        ok_v = hasattr(self, '_global_vac_lin_sorted') and np.array_equal(exp_v, self._global_vac_lin_sorted)
        ok_c = hasattr(self, '_global_cu_lin_sorted') and np.array_equal(exp_c, self._global_cu_lin_sorted)
        
        if not ok_v or not ok_c:
            self._global_vac_lin_sorted = exp_v
            self._global_cu_lin_sorted = exp_c
            if torch.cuda.is_available():
                self._global_vac_lin_sorted_t = torch.as_tensor(self._global_vac_lin_sorted, dtype=torch.int64, device=self.device)
                self._global_cu_lin_sorted_t = torch.as_tensor(self._global_cu_lin_sorted, dtype=torch.int64, device=self.device)

        
    def _calculate_vacancy_local_environments_sparse(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        根据稀疏原子位置数据结构计算并返回四个局部环境张量。
        """
        
        M = len(self.vac_pos_set)
        if M == 0:
            # 返回空的正确形状数组
            return (
                np.empty((0, 8), dtype=np.int8), 
                np.empty((0, 6), dtype=np.int8), 
                np.empty((0, 8, 8), dtype=np.int8), 
                np.empty((0, 8, 6), dtype=np.int8)
            )

        vacancies = np.array(list(self.vac_pos_set), dtype=int)
        dims = self.dims
        NN1 = self.NN1
        NN2 = self.NN2
        
        # ------------------------- 1. 预先计算所有 NN 坐标 -------------------------
        
        # V 的 NN1 坐标: (M, 8, 3)
        V_nn1_coords = self._get_pbc_coord(vacancies[:, None, :], NN1[None, :, :], dims)
        # V 的 NN2 坐标: (M, 6, 3)
        V_nn2_coords = self._get_pbc_coord(vacancies[:, None, :], NN2[None, :, :], dims)

        # ------------------------- 2. 批量查找原子类型 -------------------------
        
        # nn1_types (M, 8)
        # nn2_types (M, 6)
        # nn1_nn1_types (M, 8, 8)
        # nn1_nn2_types (M, 8, 6)
        
        nn1_types = np.zeros((M, 8), dtype=np.int8)
        nn2_types = np.zeros((M, 6), dtype=np.int8)
        nn1_nn1_types = np.zeros((M, 8, 8), dtype=np.int8)
        nn1_nn2_types = np.zeros((M, 8, 6), dtype=np.int8)

        vac_set = self.vac_pos_set
        cu_set = self.cu_pos_set
        t0 = time.time()
        
        nn1_types[:, :] = self._batch_get_type_from_coords(V_nn1_coords.reshape(-1, 3)).reshape(M, 8)
        nn2_types[:, :] = self._batch_get_type_from_coords(V_nn2_coords.reshape(-1, 3)).reshape(M, 6)
        progress_interval = 2000
        for start in range(0, M, progress_interval):
            end = min(start + progress_interval, M)
            V_nn1_batch = V_nn1_coords[start:end]  # (B, 8, 3)
            A_nn1_nn1_coords_batch = self._get_pbc_coord(V_nn1_batch[:, :, None, :], NN1[None, None, :, :], dims)  # (B, 8, 8, 3)
            A_nn1_nn2_coords_batch = self._get_pbc_coord(V_nn1_batch[:, :, None, :], NN2[None, None, :, :], dims)  # (B, 8, 6, 3)
            types_nn1 = self._batch_get_type_from_coords(A_nn1_nn1_coords_batch.reshape(-1, 3)).reshape(end - start, 8, 8)
            types_nn2 = self._batch_get_type_from_coords(A_nn1_nn2_coords_batch.reshape(-1, 3)).reshape(end - start, 8, 6)
            nn1_nn1_types[start:end, :, :] = types_nn1
            nn1_nn2_types[start:end, :, :] = types_nn2
 
        return nn1_types, nn2_types, nn1_nn1_types, nn1_nn2_types
    
    def _get_pbc_coord(self, r_base: np.ndarray, delta: np.ndarray, dims: np.ndarray) -> np.ndarray:
        """ 获取 PBC 坐标 """
        return (r_base + delta) % dims

    def init_lattice(self):
        Cu_nums = self.Cu_nums
        V_nums = self.V_nums
        nx, ny, nz = self.size # dim
        # print(f"init_lattice size={nx} {ny} {nz}", flush=True)
        lattice_shape = (nx * 2, ny * 2, nz * 2)
        total_sites = nx * ny * nz * 2
        total_atoms = Cu_nums + V_nums
        # print(f"init_lattice total_atoms={total_atoms} total_sites={total_sites}", flush=True)

        # 1. 检查和初始化晶格
        assert total_atoms <= total_sites, "原子数超出晶格容量"
        # lattice = np.full(lattice_shape, -1, dtype=np.int8) 
        
        # print("init_lattice 202")
        # 2. 随机选择所有被占据的晶格点的**一维索引**
        # 传入整数 total_sites，避免创建大型 all_indices 数组
        # occupied_indices = np.random.choice(
        #     total_sites, 
        #     size=total_atoms, 
        #     replace=False
        # )
        
        def generate_large_scale_indices(total_sites, total_atoms, device):
            # 1. 生成随机数并缩放到 [0, total_sites-1]，用 float64 保证精度（避免超大 total_sites 精度丢失）
            rand_nums = torch.rand(total_atoms * 2, device=device, dtype=torch.float64)  # 多生成一倍，减少补全次数
            indices = (rand_nums * total_sites).long()  # 缩放到目标范围，转成整数（long=int64，兼容超大 total_sites）
            
            # 2. 去重（GPU 上高效去重，2w 数据耗时 < 0.1ms）
            unique_indices = torch.unique(indices, sorted=False)
            
            # 3. 若去重后数量不够，补全（极端情况才需要，概率极低）
            while len(unique_indices) < total_atoms:
                # 补全需要的数量
                need = total_atoms - len(unique_indices)
                # 生成新的随机数
                new_rand = torch.rand(need * 2, device=device, dtype=torch.float64)
                new_indices = (new_rand * total_sites).long()
                # 合并去重
                unique_indices = torch.unique(torch.cat([unique_indices, new_indices]), sorted=False)
            
            # 4. 取前 total_atoms 个（保证数量正确）
            return unique_indices[:total_atoms].to(dtype=torch.int32).cpu().numpy()  # 转 int32 省内存

        occupied_indices = generate_large_scale_indices(total_sites, total_atoms, self.device)
   
        # occupied_indices = torch.randperm(total_sites, device="cuda")[:total_atoms]
        # import cupy as cp
        # occupied_indices = cp.random.choice(
        #     total_sites,
        #     size=total_atoms,
        #     replace=False
        #     )
        
        
        # 3. 在被占据的索引中，随机分配 V (空位)
        occupied_relative_indices = np.arange(total_atoms)
        
        
        # 随机选择 V_nums 个相对索引作为 V 的位置
        vacancy_relative_indices = np.random.choice(
            occupied_relative_indices,
            size=V_nums,
            replace=False
        )
        
        # 4. 确定 V 和 Cu 的**最终索引**
        vacancy_final_indices = occupied_indices[vacancy_relative_indices]
        
        
        cu_relative_indices = np.setdiff1d(occupied_relative_indices, vacancy_relative_indices)
        cu_final_indices = occupied_indices[cu_relative_indices]
        
        
        # 立即释放中间索引数组，帮助节省内存
        del occupied_indices
        del occupied_relative_indices
        del vacancy_relative_indices
        del cu_relative_indices

        # ----------------------------------------------------
        # 5. 向量化计算坐标并赋值给 lattice
        # ----------------------------------------------------
        
        # V (空位) 的坐标计算和赋值
        vacancy_pos = self.get_coords_vectorized(vacancy_final_indices, nx, ny, nz)
        
        
        # 使用 NumPy 高级索引进行赋值：将 N x 3 数组转置为 3 x N 索引元组
        # x_vac, y_vac, z_vac = vacancy_pos.T 
        # lattice[(x_vac, y_vac, z_vac)] = 2 # 赋值 V (2)

        # Cu 的坐标计算和赋值
        cu_pos = self.get_coords_vectorized(cu_final_indices, nx, ny, nz)
        
        
        # 使用 NumPy 高级索引进行赋值
        # x_cu, y_cu, z_cu = cu_pos.T
        # lattice[(x_cu, y_cu, z_cu)] = 1 # 赋值 Cu (1)

        # 6. 存储结果
        self.vac_pos_set = {tuple(map(int, p)) for p in vacancy_pos}
        self.cu_pos_set = {tuple(map(int, p)) for p in cu_pos}
        self.cu_pos = cu_pos.astype(np.int32)
        if self.coords is None:
            self.coords = self.generate_lattice_coordinates().astype(np.int32, copy=False)
        self.v_pos_to_id = {tuple(vacancy_pos[idx]): idx for idx in range(vacancy_pos.shape[0])}
        self.v_pos_of_id = {idx: tuple(vacancy_pos[idx]) for idx in range(vacancy_pos.shape[0])}
        self.cu_pos_of_id = {idx + self.V_nums: tuple(self.cu_pos[idx]) for idx in range(self.cu_pos.shape[0])}
        self._build_cu_pos_index()
        self._rebuild_global_lin_cache()
        do_static_env = True
        try:
            do_static_env = bool(getattr(self.args, "compute_global_static_env_reset", True))
        except Exception:
            do_static_env = True
        if do_static_env:
            self.nn1_types, self.nn2_types, self.nn1_nn1_types, self.nn1_nn2_types = self._calculate_vacancy_local_environments_sparse()
        
        
        
    def get_vacancy_array(self) -> np.ndarray:
        # return np.array(list(self.vac_pos_set), dtype=np.int32)
        
        if hasattr(self, 'v_pos_of_id') and len(self.v_pos_of_id) > 0:
            # 以 id 顺序生成数组，保证稳定性
            vals = [self.v_pos_of_id[i] for i in sorted(self.v_pos_of_id.keys())]
            return np.array(vals, dtype=np.int32)
        elif hasattr(self, 'vac_pos_set') and len(self.vac_pos_set) > 0:
            return np.array(list(self.vac_pos_set), dtype=np.int32)
        else:
            return np.empty((0, 3), dtype=np.int32)

    def get_cu_array(self) -> np.ndarray:
        return self.cu_pos

    def get_vacancy_ids_array(self) -> np.ndarray:
        if hasattr(self, 'v_pos_of_id') and len(self.v_pos_of_id) > 0:
            return np.array(sorted(self.v_pos_of_id.keys()), dtype=np.int32)
        return np.empty((0,), dtype=np.int32)

    def get_cu_ids_array(self) -> np.ndarray:
        if hasattr(self, 'cu_pos') and hasattr(self, 'cu_pos_index'):
            return np.array([self.cu_pos_index[tuple(p)] for p in self.cu_pos], dtype=np.int32)
        return np.empty((0,), dtype=np.int32)

    def get_vacancy_pos_by_id(self, vid: int):
        return self.v_pos_of_id.get(int(vid))

    def get_cu_pos_by_id(self, cid: int):
        return self.cu_pos_of_id.get(int(cid))

    def move_vacancy(self, old_pos: tuple, new_pos: tuple):
        # if self.rank == 0:
            
        if old_pos in self.vac_pos_set:
            self.vac_pos_set.discard(old_pos)
        self.vac_pos_set.add(tuple(new_pos))
        
        idx = self.v_pos_to_id.pop(old_pos)
        self.v_pos_to_id[tuple(new_pos)] = idx
        self.v_pos_of_id[idx] = tuple(new_pos)
        try:
            D = np.array(self.dims, dtype=int)
            ov = np.array(old_pos, dtype=np.int64)
            nv = np.array(new_pos, dtype=np.int64)
            old_lin = int(((ov[0] * D[1] + ov[1]) * D[2] + ov[2]))
            new_lin = int(((nv[0] * D[1] + nv[1]) * D[2] + nv[2]))
            if hasattr(self, '_global_vac_lin_sorted'):
                arr = self._global_vac_lin_sorted
                if arr.size > 0:
                    i = int(np.searchsorted(arr, old_lin))
                    if i < arr.size and arr[i] == old_lin:
                        arr = np.delete(arr, i)
                j = int(np.searchsorted(arr, new_lin))
                self._global_vac_lin_sorted = np.insert(arr, j, new_lin).astype(np.int64)
                if torch.cuda.is_available():
                    self._global_vac_lin_sorted_t = torch.as_tensor(self._global_vac_lin_sorted, dtype=torch.int64, device=self.device)
        except Exception:
            # pass
            self._rebuild_global_lin_cache()
        
        # if hasattr(self, 'vac_pos_index'):
        #     if old_pos in self.vac_pos_index:
        #         idx = self.vac_pos_index.pop(old_pos)
        #         self.vac_pos_index[tuple(new_pos)] = idx
        #         if hasattr(self, 'v_pos_of_id'):
        #             self.v_pos_of_id[idx] = tuple(new_pos)

    def move_cu(self, old_pos: tuple, new_pos: tuple):
        if old_pos in self.cu_pos_set:
            self.cu_pos_set.discard(old_pos)
        self.cu_pos_set.add(tuple(new_pos))
        if hasattr(self, 'cu_pos_index'):
            if old_pos in self.cu_pos_index:
                cid = self.cu_pos_index.pop(old_pos)
                self.cu_pos_index[tuple(new_pos)] = cid
                if hasattr(self, 'cu_pos_of_id'):
                    self.cu_pos_of_id[cid] = tuple(new_pos)
                if hasattr(self, 'cu_pos') and (self.cu_pos is not None):
                    cu_idx = int(cid - getattr(self, 'V_nums', 0)) if isinstance(cid, (int, np.integer)) else cid
                    if 0 <= cu_idx < len(self.cu_pos):
                        self.cu_pos[cu_idx] = tuple(new_pos)
        try:
            D = np.array(self.dims, dtype=int)
            oc = np.array(old_pos, dtype=np.int64)
            nc = np.array(new_pos, dtype=np.int64)
            old_lin = int(((oc[0] * D[1] + oc[1]) * D[2] + oc[2]))
            new_lin = int(((nc[0] * D[1] + nc[1]) * D[2] + nc[2]))
            if hasattr(self, '_global_cu_lin_sorted'):
                arr = self._global_cu_lin_sorted
                if arr.size > 0:
                    i = int(np.searchsorted(arr, old_lin))
                    if i < arr.size and arr[i] == old_lin:
                        arr = np.delete(arr, i)
                j = int(np.searchsorted(arr, new_lin))
                self._global_cu_lin_sorted = np.insert(arr, j, new_lin).astype(np.int64)
                if torch.cuda.is_available():
                    self._global_cu_lin_sorted_t = torch.as_tensor(self._global_cu_lin_sorted, dtype=torch.int64, device=self.device)
        except Exception:
            self._rebuild_global_lin_cache()

    # def init_lattice(self):
    #     Cu_nums = self.Cu_nums
    #     V_nums = self.V_nums
    #     nx, ny, nz = self.size
    #     lattice_shape = (nx * 2, ny * 2, nz * 2)
    #     total_sites = nx * ny * nz * 2
    #     total_atoms = Cu_nums + V_nums

    #     # 1. 确保原子数不超过总晶格点数
    #     assert total_atoms <= total_sites, "原子数超出晶格容量"

    #     # 2. 计算所有可能占据的晶格点坐标 (all_coords)
    #     corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
    #     body = corner + 1
    #     all_coords = np.vstack((corner, body))  # 总共 total_sites 个坐标点

    #     # 3. 随机选择总共要被 Cu/V 占据的位置
    #     # 从 [0, total_sites - 1] 中随机选择 total_atoms 个不重复的索引
    #     all_indices = np.arange(total_sites)
        
    #     # 随机选择 total_atoms 个索引作为所有被占据的晶格点
    #     occupied_indices = np.random.choice(
    #         all_indices, 
    #         size=total_atoms, 
    #         replace=False  # 确保选择的位置不重复
    #     )
        
    #     # 获取这批被占据点的实际坐标
    #     occupied_coords = all_coords[occupied_indices]
        
    #     # 4. 在这批被占据的位置中，随机分配 V (空位)
        
    #     # 从 occupied_coords 的索引 [0, total_atoms - 1] 中选择 V_nums 个索引
    #     occupied_relative_indices = np.arange(total_atoms)
        
    #     # 随机选择 V_nums 个索引作为 V 的位置
    #     vacancy_relative_indices = np.random.choice(
    #         occupied_relative_indices,
    #         size=V_nums,
    #         replace=False  # 确保 V 的位置不重复
    #     )

    #     # 5. 确定 V 和 Cu 的最终坐标
        
    #     # V (空位) 的坐标
    #     vacancy_pos = occupied_coords[vacancy_relative_indices]
        
    #     # Cu 的坐标：未被 V 选中的剩余位置
    #     cu_relative_indices = np.setdiff1d(occupied_relative_indices, vacancy_relative_indices)
    #     cu_pos = occupied_coords[cu_relative_indices]

    #     # 6. 构建晶格数组
        
    #     # 初始化一个所有点为 -1 (未被占据) 的大晶格
    #     lattice = np.full(lattice_shape, -1, dtype=np.int8)

    #     # 标记 V (2) 和 Cu (1) 的位置
    #     # V (2)
    #     lattice[tuple(vacancy_pos.T)] = 2
    #     # Cu (1)
    #     lattice[tuple(cu_pos.T)] = 1
        
    #     # 7. 存储结果
    #     self.lattice = lattice
    #     self.vacancy_pos = vacancy_pos
    #     self.cu_pos = cu_pos
    #     self._build_cu_pos_index()
        

    # def init_lattice(self):
    #     Cu_nums = self.Cu_nums
    #     V_nums = self.V_nums
    #     nx, ny, nz = self.size
    #     lattice_shape = (nx * 2, ny * 2, nz * 2)
    #     total_sites = nx * ny * nz * 2
    #     assert Cu_nums + V_nums <= total_sites, "原子数超出晶格容量"
    #     atom_list = np.array(
    #         [1] * Cu_nums + [2] * V_nums + [0] * (total_sites - Cu_nums - V_nums),
    #         dtype=np.int8
    #     )
    #     np.random.shuffle(atom_list)

    #     corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
    #     body = corner + 1
    #     all_coords = np.vstack((corner, body))

    #     lattice = np.full(lattice_shape, -1, dtype=np.int8)
    #     lattice[tuple(all_coords.T)] = atom_list

    #     vacancy_pos = all_coords[atom_list == 2]
    #     cu_pos = all_coords[atom_list == 1]

    #     self.lattice = lattice
    #     self.vacancy_pos = vacancy_pos
    #     self.cu_pos = cu_pos
    #     self._build_cu_pos_index()

    # def init_lattice(self):
    #     Cu_nums = self.Cu_nums
    #     V_nums = self.V_nums
    #     nx, ny, nz = self.size
    #     # 晶格的尺寸是 (2nx, 2ny, 2nz)
    #     lattice_shape = (nx * 2, ny * 2, nz * 2)
    #     total_sites = nx * ny * nz * 2
        
    #     # 1. 生成所有可能的原子/空位位置 (坐标)
    #     # 每个原胞有2个格点：(i*2, j*2, k*2) [corner] 和 (i*2+1, j*2+1, k*2+1) [body]
    #     corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
    #     body = corner + 1
    #     all_coords = np.vstack((corner, body))
        
    #     # 确保总格点数匹配
    #     assert all_coords.shape[0] == total_sites
    #     # 确保原子数不超出容量
    #     assert Cu_nums + V_nums <= total_sites, "原子数超出晶格容量"

    #     # 2. **直接打乱所有格点坐标**
    #     np.random.shuffle(all_coords)

    #     # 3. **直接从打乱后的坐标中分配** Cu, V, (未占用位)
        
    #     # 前 Cu_nums 个坐标分配给 Cu 原子
    #     self.cu_pos = all_coords[:Cu_nums]
        
    #     # 接着 V_nums 个坐标分配给 V (空位)
    #     self.vacancy_pos = all_coords[Cu_nums : Cu_nums + V_nums]

    #     # 剩余的坐标 (如果有) 保持为未占用 (或默认的其他原子类型)
    #     # unassigned_pos = all_coords[Cu_nums + V_nums:]

    #     # 4. **（可选）创建并填充完整的晶格**
    #     # 只有在后续模拟中需要快速访问任意位置的原子类型时才需要
        
    #     # 初始化晶格，用 0 (或 -1) 表示未分配或默认原子类型
    #     lattice = np.full(lattice_shape, 0, dtype=np.int8) 
        
    #     # 填充 Cu 原子 (类型 1)
    #     lattice[tuple(self.cu_pos.T)] = 1
        
    #     # 填充 V (空位) (类型 2)
    #     lattice[tuple(self.vacancy_pos.T)] = 2
        
    #     self.lattice = lattice
        
    #     # 5. 调用其他初始化方法
    #     self._build_cu_pos_index()

    # def init_lattice(self):
    #     # ... (定义 Cu_nums, V_nums, size, total_sites)
    #     Cu_nums = self.Cu_nums
    #     V_nums = self.V_nums
    #     nx, ny, nz = self.size 
    #     # 1. 生成所有可能的原子/空位位置 (坐标)
    #     corner = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
    #     body = corner + 1
    #     all_coords = np.vstack((corner, body))
        
    #     # 2. 直接打乱所有格点坐标
    #     np.random.shuffle(all_coords)

    #     # 3. 直接从打乱后的坐标中分配 Cu 和 V
    #     self.cu_pos = all_coords[:Cu_nums]
    #     self.vacancy_pos = all_coords[Cu_nums : Cu_nums + V_nums]

    #     # 移除 self.lattice 的创建和赋值代码
    #     # self.lattice = None # 或者保持不定义，取决于你的代码结构
        
    #     self._build_cu_pos_index()

if __name__ == "__main__":
    pass
