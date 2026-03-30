import numpy as np
import heapq
from collections import defaultdict
import torch

# -------------------- Top-K 维护单元 --------------------
class TopK:
    def __init__(self, K):
        self.K = K
        self.heap = []          # max-heap of (-distance, atom_id)
        self.lookup = {}        # atom_id -> distance
    
    def try_update(self, atom_id, dist):
        if atom_id in self.lookup:
            old_dist = self.lookup[atom_id]
            if dist < old_dist:
                self.lookup[atom_id] = dist
                # lazy rebuild heap
                self.heap = [(-d,a) for a,d in self.lookup.items()]
                heapq.heapify(self.heap)
        else:
            if len(self.heap) < self.K:
                heapq.heappush(self.heap, (-dist, atom_id))
                self.lookup[atom_id] = dist
            else:
                max_dist = -self.heap[0][0]
                if dist < max_dist:
                    _, removed_atom = heapq.heappop(self.heap)
                    del self.lookup[removed_atom]
                    heapq.heappush(self.heap, (-dist, atom_id))
                    self.lookup[atom_id] = dist
    
    def get_topk(self):
        return sorted([(atom_id, dist) for atom_id, dist in self.lookup.items()], key=lambda x: x[1])

# -------------------- Vacancy-TopK 系统 --------------------
class VacancyTopKSystem:
    def __init__(self, cu_positions, vacancy_positions, K, box, cell_size):
        """
        cu_positions: np.array(N,3)
        vacancy_positions: np.array(M,3)
        K: 每个 vacancy 的 Top-K
        box: tuple(Lx,Ly,Lz) 周期边界
        cell_size: float, cell list 边长
        """
        # -------------------- 位置 --------------------
        self.P_cu = np.array(cu_positions, dtype=np.float64)
        self.P_vac = np.array(vacancy_positions, dtype=np.float64)
        self.N = len(cu_positions)
        self.M = len(vacancy_positions)
        self.K = K
        self.box = np.array(box, dtype=np.float64)
        self.cell_size = float(cell_size)
        
        # -------------------- cell list --------------------
        self.cell_dim = np.ceil(self.box / self.cell_size).astype(int)
        self.cell_list = defaultdict(list)
        self.atom_cell = np.zeros(self.N, dtype=int)
        self._build_cell_list()
        self.vac_cell_list = defaultdict(list)
        self.vacancy_cell = np.zeros(self.M, dtype=int)
        self._build_vac_cell_list()
        self.cu_in_vac_topk = defaultdict(set)
        self.vac_topk_members = {}
        
        # -------------------- Top-K per vacancy --------------------
        self.topks = [TopK(K) for _ in range(self.M)]
        self._initialize_topk()
    
    # -------------------- 辅助函数 --------------------
    def _compute_distance(self, pos1, pos2):
        delta = np.array(pos1, dtype=np.float64) - np.array(pos2, dtype=np.float64)
        delta -= np.round(delta / self.box) * self.box
        return np.linalg.norm(delta)
    
    def _get_cell_index(self, pos):
        idx = np.floor(pos / self.cell_size).astype(int)
        idx = np.mod(idx, self.cell_dim)
        linear = idx[0]*self.cell_dim[1]*self.cell_dim[2] + idx[1]*self.cell_dim[2] + idx[2]
        return linear
    
    def _build_cell_list(self):
        for i, pos in enumerate(self.P_cu):
            cell_idx = self._get_cell_index(pos)
            self.cell_list[cell_idx].append(i)
            self.atom_cell[i] = cell_idx

    def _build_vac_cell_list(self):
        for i, pos in enumerate(self.P_vac):
            cell_idx = self._get_cell_index(pos)
            self.vac_cell_list[cell_idx].append(i)
            self.vacancy_cell[i] = cell_idx

    def _update_cell_list(self, atom_id, new_pos):
        old_cell = self.atom_cell[atom_id]
        new_cell = self._get_cell_index(new_pos)
        if old_cell != new_cell:
            self.cell_list[old_cell].remove(atom_id)
            self.cell_list[new_cell].append(atom_id)
            self.atom_cell[atom_id] = new_cell

    def _update_vac_cell_list(self, vac_id, new_pos):
        old_cell = self.vacancy_cell[vac_id]
        new_cell = self._get_cell_index(new_pos)
        if old_cell != new_cell:
            self.vac_cell_list[old_cell].remove(vac_id)
            self.vac_cell_list[new_cell].append(vac_id)
            self.vacancy_cell[vac_id] = new_cell

    def _get_neighbor_vacancy_ids(self, idx, ring=1):
        neighbors = []
        for dx in range(-ring, ring+1):
            for dy in range(-ring, ring+1):
                for dz in range(-ring, ring+1):
                    cx = (idx[0] + dx) % self.cell_dim[0]
                    cy = (idx[1] + dy) % self.cell_dim[1]
                    cz = (idx[2] + dz) % self.cell_dim[2]
                    linear = cx*self.cell_dim[1]*self.cell_dim[2] + cy*self.cell_dim[2] + cz
                    if linear in self.vac_cell_list:
                        neighbors.extend(self.vac_cell_list[linear])
        return neighbors
    
    # -------------------- 初始化 Top-K --------------------
    def _initialize_topk(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        box_t = torch.tensor(self.box, dtype=torch.float32, device=device)
        P_cu_t = torch.from_numpy(self.P_cu.astype(np.float32)).to(device)

        for vid, v_pos in enumerate(self.P_vac):
            idx = np.floor(v_pos / self.cell_size).astype(int)

            neighbors = []
            ring = 1
            max_ring = max(int(self.cell_dim[0]), int(self.cell_dim[1]), int(self.cell_dim[2]))
            while len(neighbors) < self.K and ring <= max_ring:
                for dx in range(-ring, ring+1):
                    for dy in range(-ring, ring+1):
                        for dz in range(-ring, ring+1):
                            cx = (idx[0] + dx) % self.cell_dim[0]
                            cy = (idx[1] + dy) % self.cell_dim[1]
                            cz = (idx[2] + dz) % self.cell_dim[2]
                            linear = cx*self.cell_dim[1]*self.cell_dim[2] + cy*self.cell_dim[2] + cz
                            neighbors.extend(self.cell_list[linear])
                ring += 1

            if len(neighbors) == 0:
                continue

            if use_cuda:
                v_t = torch.tensor(v_pos, dtype=torch.float32, device=device).unsqueeze(0)
                idx_t = torch.tensor(neighbors, dtype=torch.long, device=device)
                cu_t = P_cu_t[idx_t]
                delta = cu_t - v_t
                delta -= torch.round(delta / box_t) * box_t
                d_sq = torch.sum(delta * delta, dim=1)
                num_k = min(self.K, d_sq.shape[0])
                topk_vals, topk_pos = torch.topk(d_sq, k=num_k, largest=False)
                topk_vals = torch.sqrt(topk_vals)
                for i in range(num_k):
                    aid = int(neighbors[int(topk_pos[i])])
                    dist = float(topk_vals[i].item())
                    self.topks[vid].try_update(aid, dist)
            else:
                cu_arr = self.P_cu[np.array(neighbors, dtype=int)]
                delta = cu_arr - v_pos
                delta -= np.round(delta / self.box) * self.box
                d_sq = np.sum(delta*delta, axis=1)
                num_k = min(self.K, d_sq.shape[0])
                pos = np.argpartition(d_sq, num_k-1)[:num_k]
                dists = np.sqrt(d_sq[pos])
                for aid_idx, dist in zip(pos.tolist(), dists.tolist()):
                    aid = int(neighbors[aid_idx])
                    self.topks[vid].try_update(aid, float(dist))
            members = set(self.topks[vid].lookup.keys())
            self.vac_topk_members[vid] = members
            for aid in members:
                self.cu_in_vac_topk[aid].add(vid)
    
    # -------------------- Cu 更新 --------------------
    def update_cu(self, updated_cu: dict):
        """
        updated_cu: {cu_id: new_pos}
        """
        affected_vacancies = set()
        for cu_id, new_pos in updated_cu.items():
            old_pos = self.P_cu[cu_id].copy()
            self.P_cu[cu_id] = np.array(new_pos, dtype=np.float64)
            self._update_cell_list(cu_id, self.P_cu[cu_id])
            idx_new = np.floor(self.P_cu[cu_id] / self.cell_size).astype(int)
            idx_old = np.floor(old_pos / self.cell_size).astype(int)
            vids_new = self._get_neighbor_vacancy_ids(idx_new, ring=1)
            vids_old = self._get_neighbor_vacancy_ids(idx_old, ring=1)
            vids_map = list(self.cu_in_vac_topk.get(cu_id, set()))
            candidate = set(vids_new) | set(vids_old) | set(vids_map)
            if len(candidate) == 0:
                continue
            vids_list = sorted(list(candidate))
            v_arr = self.P_vac[vids_list]
            delta = v_arr - self.P_cu[cu_id]
            delta -= np.round(delta / self.box) * self.box
            dists = np.sqrt(np.sum(delta*delta, axis=1))
            for i, vid in enumerate(vids_list):
                topk = self.topks[vid]
                prev_in = cu_id in topk.lookup
                thr = max(topk.lookup.values()) if len(topk.lookup) >= self.K and len(topk.lookup) > 0 else float('inf')
                di = float(dists[i])
                if prev_in or di < thr:
                    topk.try_update(cu_id, di)
                    after_in = cu_id in topk.lookup
                    if prev_in or after_in:
                        affected_vacancies.add(vid)
                    prev_members = self.vac_topk_members.get(vid, set())
                    new_members = set(topk.lookup.keys())
                    if new_members != prev_members:
                        removed = prev_members - new_members
                        added = new_members - prev_members
                        for aid in removed:
                            self.cu_in_vac_topk[aid].discard(vid)
                        for aid in added:
                            self.cu_in_vac_topk[aid].add(vid)
                        self.vac_topk_members[vid] = new_members
        return affected_vacancies
    
    # -------------------- Vacancy 更新 --------------------
    def update_vacancy(self, updated_vacancy: dict):
        """
        updated_vacancy: {vac_id: new_pos}
        """
        affected_vacancies = set()
        for vid, new_pos in updated_vacancy.items():
            self.P_vac[vid] = np.array(new_pos, dtype=np.float64)
            self._update_vac_cell_list(vid, self.P_vac[vid])
            affected_vacancies.add(vid)
            
            idx = np.floor(new_pos / self.cell_size).astype(int)
            neighbors = []
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        cx = (idx[0]+dx) % self.cell_dim[0]
                        cy = (idx[1]+dy) % self.cell_dim[1]
                        cz = (idx[2]+dz) % self.cell_dim[2]
                        linear = cx*self.cell_dim[1]*self.cell_dim[2] + cy*self.cell_dim[2] + cz
                        neighbors.extend(self.cell_list[linear])
            topk = self.topks[vid]
            for aid in neighbors:
                dist = self._compute_distance(self.P_cu[aid], new_pos)
                topk.try_update(aid, dist)
            prev_members = self.vac_topk_members.get(vid, set())
            new_members = set(topk.lookup.keys())
            if new_members != prev_members:
                removed = prev_members - new_members
                added = new_members - prev_members
                for aid in removed:
                    self.cu_in_vac_topk[aid].discard(vid)
                for aid in added:
                    self.cu_in_vac_topk[aid].add(vid)
                self.vac_topk_members[vid] = new_members
        return affected_vacancies
    
    # -------------------- 更新系统 --------------------
    def update_system(self, updated_cu=None, updated_vacancy=None):
        """
        返回:
            {
                "vid_list": [...],
                "diff_k": tensor([num_vid,K,3]),
                "dist_k": tensor([num_vid,K])
            }
        """
        affected_vacancies = set()
        if updated_cu:
            affected_vacancies |= self.update_cu(updated_cu)
        if updated_vacancy:
            affected_vacancies |= self.update_vacancy(updated_vacancy)
        
        vid_list = sorted(list(affected_vacancies))
        diff_list = []
        dist_list = []

        for vid in vid_list:
            topk = self.topks[vid].get_topk()
            topk = sorted(topk, key=lambda x: x[1])
            atom_ids = [a for a,d in topk]
            dists = [d for a,d in topk]
            vec = self.P_cu[atom_ids] - self.P_vac[vid]
            diff_list.append(vec)
            dist_list.append(dists)
        
        diff_k = torch.tensor(diff_list, dtype=torch.float32)
        dist_k = torch.tensor(dist_list, dtype=torch.float32)

        return {
            "vid_list": vid_list,
            "diff_k": diff_k,
            "dist_k": dist_k
        }
    
    # -------------------- 查询某 vacancy Top-K --------------------
    def get_topk_for_vacancy(self, vac_id):
        return self.topks[vac_id].get_topk()
    
    # -------------------- 获取所有 Top-K 张量 --------------------
    def get_all_topk_tensors(self):
        diff_list = []
        dist_list = []
        for vid in range(self.M):
            topk = self.topks[vid].get_topk()
            topk = sorted(topk, key=lambda x:x[1])
            atom_ids = [a for a,d in topk]
            dists = [d for a,d in topk]
            vec = self.P_cu[atom_ids] - self.P_vac[vid]
            diff_list.append(vec)
            dist_list.append(dists)
        # diff_k = torch.tensor(diff_list, dtype=torch.float32)
        # dist_k = torch.tensor(dist_list, dtype=torch.float32)
        diff_k = torch.from_numpy(np.array(diff_list, dtype=np.float32))
        dist_k = torch.from_numpy(np.array(dist_list, dtype=np.float32))

        return {
            "vid_list": list(range(self.M)),
            "diff_k": diff_k,
            "dist_k": dist_k
        }

    def verify_update(self, updated_cu=None, updated_vacancy=None):
        """
        验证 update_system() 的输出是否和从头计算得到的所有 Top-K 一致。
        如果一致返回 True，否则返回 False 并打印差异。
        """
        # 先用 update_system 得到增量更新结果
        result_update = self.update_system(updated_cu, updated_vacancy)
        vid_list = result_update["vid_list"]
        diff_k_update = result_update["diff_k"]
        dist_k_update = result_update["dist_k"]

        # 从头计算整个系统
        result_full = self.get_all_topk_tensors()
        diff_k_full = result_full["diff_k"][vid_list]
        dist_k_full = result_full["dist_k"][vid_list]

        # 比较
        diff_ok = torch.allclose(diff_k_update, diff_k_full, atol=1e-6)
        dist_ok = torch.allclose(dist_k_update, dist_k_full, atol=1e-6)

        if diff_ok and dist_ok:
            print("✅ 验证通过: update_system 与全量计算结果一致")
            return True
        else:
            print("❌ 验证失败!")
            for i, vid in enumerate(vid_list):
                if not torch.allclose(diff_k_update[i], diff_k_full[i], atol=1e-6):
                    print(f"  Vacancy {vid} diff_k 不一致")
                if not torch.allclose(dist_k_update[i], dist_k_full[i], atol=1e-6):
                    print(f"  Vacancy {vid} dist_k 不一致")
            return False
        
        
import numpy as np
import torch
from collections import defaultdict

# # -------------------- PyTorch GEMM TopK 系统 --------------------
# class TensorCoreVacancyTopKSystem:
#     def __init__(self, cu_positions, vacancy_positions, K, box, cell_size, device='cuda'):
#         """
#         cu_positions: np.array(N,3)
#         vacancy_positions: np.array(M,3)
#         K: 每个 vacancy 的 Top-K
#         box: tuple(Lx,Ly,Lz) 周期边界
#         cell_size: float, cell list 边长
#         device: 'cuda' 或 'cpu'，推荐 'cuda' 以利用 Tensor Core
#         """
#         self.device = torch.device(device)
#         self.dtype = torch.float32 # PyTorch会自动使用FP16/BF16进行Tensor Core加速
        
#         # -------------------- 位置 (存储在GPU上) --------------------
#         # P_cu: (N, 3), P_vac: (M, 3)
#         self.P_cu = torch.tensor(cu_positions, dtype=self.dtype, device=self.device)
#         self.P_vac = torch.tensor(vacancy_positions, dtype=self.dtype, device=self.device)
#         self.N = self.P_cu.shape[0]
#         self.M = self.P_vac.shape[0]
#         self.K = K
#         self.box = torch.tensor(box, dtype=self.dtype, device=self.device)
#         self.cell_size = float(cell_size)
        
#         # -------------------- Cell List --------------------
#         self.cell_dim = torch.ceil(self.box / self.cell_size).long()
#         self.cell_list = defaultdict(list)
#         self.atom_cell = torch.zeros(self.N, dtype=torch.long, device=self.device)
#         self._build_cell_list()
        
#         # -------------------- Top-K 结果 (原子ID和距离) --------------------
#         # 保存每个 vacancy 的 Top-K 原子的 ID 和 距离
#         # self.topk_indices: (M, K) Long Tensor
#         # self.topk_dists: (M, K) Float Tensor
#         self.topk_indices = torch.zeros((self.M, self.K), dtype=torch.long, device=self.device)
#         self.topk_dists = torch.full((self.M, self.K), float('inf'), dtype=self.dtype, device=self.device)
        
#         # self._initialize_topk()
#         self._initialize_topk_chunked()

#     # # -------------------- 辅助函数 (CUDA 优化) --------------------
#     # def _compute_pbc_sq_distance(self, pos1, pos2):
#     #     """
#     #     计算周期边界条件下的平方欧氏距离 (pos1: (A, 3), pos2: (B, 3))
#     #     利用广播机制，并确保所有计算都在 GPU 上进行
#     #     """
#     #     # (A, 1, 3) - (1, B, 3) -> (A, B, 3)
#     #     delta = pos1.unsqueeze(1) - pos2.unsqueeze(0)
#     #     # 周期边界条件
#     #     delta -= torch.round(delta / self.box) * self.box
#     #     # 平方欧氏距离
#     #     sq_dist = torch.sum(delta**2, dim=-1) # (A, B)
#     #     return sq_dist

#     def _compute_pbc_sq_distance(self, pos1, pos2):
#         """
#         计算周期边界条件下的平方欧氏距离，优化内存占用。
#         pos1: (A, 3), pos2: (B, 3)
#         """
        
#         # --- 1. 计算初始差值 (A, B, 3) ---
#         # 这一步必须创建一个新的张量 `delta`
#         delta = pos1.unsqueeze(1) - pos2.unsqueeze(0)
        
#         # --- 2. 周期边界条件 (原地修正) ---
        
#         # box: (1, 3) 或 (3,)
#         # 2a. 计算修正项 factor = torch.round(delta / self.box)
#         #     注意：如果 self.box 维度是 (3,)，torch 会自动广播
        
#         # 创建修正项张量 (A, B, 3)
#         correction_factor = torch.round(delta / self.box)
        
#         # 2b. 原地计算修正项 (correction_factor * self.box)
#         #     并使用原地操作 delta -= correction_term
        
#         # 修正项的计算需要临时内存
#         correction_term = correction_factor * self.box
        
#         # 原地更新 delta (避免创建新的 delta 张量)
#         delta.sub_(correction_term)
        
#         # 释放修正计算的临时张量
#         del correction_factor, correction_term
#         # torch.cuda.empty_cache() # 频繁调用开销大，在循环体外或主函数中调用更合适
        
#         # --- 3. 平方欧氏距离 (优化内存) ---
        
#         # 3a. 原地计算 delta 的平方 (delta = delta**2)
#         # 警告：此操作后 delta 包含了平方距离！
#         delta.pow_(2) # delta 现在是 (A, B, 3) 形状的平方差
        
#         # 3b. 沿最后一维求和
#         # torch.sum(..., dim=-1) 会创建一个新的 (A, B) 张量
#         sq_dist = torch.sum(delta, dim=-1) # (A, B)
        
#         # --- 4. 显式释放 delta ---
#         # 这一步是关键，释放最大的临时张量
#         del delta
        
#         return sq_dist
    
#     def _get_cell_index_cuda(self, pos):
#         """CUDA优化的cell index计算 (pos: (N, 3))"""
#         # (N, 3)
#         idx = torch.floor(pos / self.cell_size).long()
#         idx = torch.remainder(idx, self.cell_dim)
        
#         # 线性索引 (N,)
#         cd = self.cell_dim
#         linear = idx[:, 0] * (cd[1] * cd[2]) + idx[:, 1] * cd[2] + idx[:, 2]
#         return linear

#     # -------------------- Cell List --------------------
#     def _build_cell_list(self):
#         self.atom_cell = self._get_cell_index_cuda(self.P_cu)
#         self.cell_list.clear()
#         for i, cell_idx in enumerate(self.atom_cell.tolist()):
#             self.cell_list[cell_idx].append(i)
    
#     def _update_cell_list(self, atom_id, new_pos):
#         # 注意: 这里的原子更新频率不高，可以使用CPU list和dict辅助更新，避免大规模GPU操作的开销
#         new_cell_idx = self._get_cell_index_cuda(new_pos.unsqueeze(0)).item()
#         atom_id_long = int(atom_id)
#         old_cell_idx = self.atom_cell[atom_id_long].item()

#         if old_cell_idx != new_cell_idx:
#             if atom_id_long in self.cell_list[old_cell_idx]:
#                 self.cell_list[old_cell_idx].remove(atom_id_long)
#             self.cell_list[new_cell_idx].append(atom_id_long)
#             self.atom_cell[atom_id_long] = new_cell_idx

#     # -------------------- 初始化 Top-K (GEMM 核心) --------------------
#     def _initialize_topk(self):
#         # 1. 计算所有 vacancy 和所有 cu 原子的平方距离矩阵 D_sq (M, N)
#         # Tensor Core: 在 FP16/BF16 模式下，这个距离计算（通过内积）将由 Tensor Core 加速
#         D_sq = self._compute_pbc_sq_distance(self.P_vac, self.P_cu)
        
#         # 2. 批量 Top-K (PyTorch CUDA优化)
#         # 找到最小的 K 个距离及其对应的 cu 原子索引
#         # values: (M, K) 是距离，indices: (M, K) 是原子ID
#         # torch.topk 针对 GPU 进行了高度优化
#         self.topk_dists, self.topk_indices = torch.topk(
#             D_sq, 
#             k=self.K, 
#             dim=1, 
#             largest=False  # 找最小值
#         )
#         # 开根号得到真实距离
#         self.topk_dists = torch.sqrt(self.topk_dists)

#     def _initialize_topk_chunked(self, chunk_size=64):
#         """
#         分块执行全量 Top-K 计算，以减少显存压力。

#         chunk_size: 每次计算的 vacancy 块大小 (M_b)。
#                     通常选择 2048, 4096, 8192 等与 GPU 架构相关的尺寸。
#         """
        
#         M = self.M
#         if M == 0:
#             return

#         # 初始化 Top-K 结果存储（如果未初始化）
#         # self.topk_indices 和 self.topk_dists 已经在 __init__ 中初始化为 M x K 的大小
#         # 且 dists 填充了 float('inf')，不需要额外初始化。
        
#         K = self.K
        
#         # 沿着 M (vacancy) 维度进行分块迭代
#         num_chunks = (M + chunk_size - 1) // chunk_size
        
#         print(f"--- 启动分块初始化：M={M}, N={self.N}, K={K}, Chunk Size={chunk_size}, 块数={num_chunks} ---")

#         for i in range(num_chunks):
#             start_idx = i * chunk_size
#             end_idx = min((i + 1) * chunk_size, M)
#             current_M_b = end_idx - start_idx
            
#             # 1. 提取当前块的 vacancy 位置
#             # P_vac_chunk: (M_b, 3)
#             P_vac_chunk = self.P_vac[start_idx:end_idx]
            
#             # 2. 计算当前块到所有 Cu 原子的平方距离矩阵 D_sq (M_b, N)
#             # D_sq 计算仍然完全在 GPU 上矢量化执行
#             # 这一步计算量大，但内存占用为 M_b * N，受到 chunk_size 限制
#             D_sq = self._compute_pbc_sq_distance(P_vac_chunk, self.P_cu)
            
#             # 3. 批量 Top-K (PyTorch CUDA优化)
#             # 找到最小的 K 个距离及其对应的 Cu 原子索引
#             # chunk_dists: (M_b, K), chunk_indices: (M_b, K)
            
#             # 确保 K 不超过 N
#             k_eff = min(K, self.N)
            
#             chunk_dists_sq, chunk_indices = torch.topk(
#                 D_sq, 
#                 k=k_eff, 
#                 dim=1, 
#                 largest=False
#             )
            
#             # 4. 更新全局存储
#             # 开根号得到真实距离
#             self.topk_dists[start_idx:end_idx, :k_eff] = torch.sqrt(chunk_dists_sq)
#             self.topk_indices[start_idx:end_idx, :k_eff] = chunk_indices

#             # 如果 K > N，剩余部分保持为 inf (已在 __init__ 中处理)
            
#             # 释放临时张量以减轻 VRAM 压力
#             del D_sq, chunk_dists_sq, chunk_indices
#             if i < num_chunks - 1:
#                  # 建议在循环结束时手动调用 PyTorch 内存管理，强制清理缓存
#                  torch.cuda.empty_cache()

#         print("--- 分块初始化完成 ---")
        
#     # -------------------- Cu 更新 (局部更新) --------------------
#     def update_cu(self, updated_cu: dict):
#         affected_vacancies = set()
        
#         for cu_id, new_pos_np in updated_cu.items():
#             new_pos = torch.tensor(new_pos_np, dtype=self.dtype, device=self.device)
#             # 1. 更新位置
#             self.P_cu[cu_id] = new_pos
#             self._update_cell_list(cu_id, new_pos)

#             # 2. 检查所有 vacancy 是否受影响
#             for vid in range(self.M):
#                 # 计算新位置到该 vacancy 的距离
#                 dist = self._compute_pbc_sq_distance(self.P_vac[vid].unsqueeze(0), new_pos.unsqueeze(0)).squeeze().sqrt()
                
#                 # 检查这个 cu 原子是否能替换当前的 Top-K
#                 max_dist_in_k = self.topk_dists[vid, -1] # 当前第 K 大的距离
                
#                 # 如果距离小于当前第 K 大的距离，或者该原子已经在 Top-K 列表中
#                 if dist < max_dist_in_k or cu_id in self.topk_indices[vid]:
#                     # 重新计算该 vacancy 的所有 Top-K
#                     affected_vacancies.add(vid)
        
#         # 3. 批量更新受影响的 vacancy 的 Top-K
#         if affected_vacancies:
#             vid_list = sorted(list(affected_vacancies))
#             vids_tensor = torch.tensor(vid_list, device=self.device, dtype=torch.long)
            
#             # 提取受影响的 vacancy 位置 (V', 3)
#             P_vac_affected = self.P_vac[vids_tensor]
            
#             # 重新计算这些 affected_vacancies 到所有 cu 原子的距离 (V', N)
#             D_sq_affected = self._compute_pbc_sq_distance(P_vac_affected, self.P_cu)
            
#             # 批量 Top-K 搜索
#             new_dists, new_indices = torch.topk(
#                 D_sq_affected, k=self.K, dim=1, largest=False
#             )
            
#             # 更新全局 Top-K 存储
#             self.topk_dists[vids_tensor] = torch.sqrt(new_dists)
#             self.topk_indices[vids_tensor] = new_indices

#         return affected_vacancies

#     # -------------------- Vacancy 更新 (Cell List 筛选) --------------------
#     def update_vacancy(self, updated_vacancy: dict):
#         affected_vacancies = set()
#         vids_to_recalc = []
        
#         for vid, new_pos_np in updated_vacancy.items():
#             new_pos = torch.tensor(new_pos_np, dtype=self.dtype, device=self.device)
#             self.P_vac[vid] = new_pos
#             affected_vacancies.add(vid)
#             vids_to_recalc.append(vid)

#         if not vids_to_recalc:
#             return affected_vacancies
        
#         # 批量筛选候选原子（使用 Cell List）
#         all_candidate_aids = []
#         cu_indices_map = {} # 映射回原始 cu_id
        
#         for vid in vids_to_recalc:
#             v_pos = self.P_vac[vid].cpu().numpy()
#             idx = np.floor(v_pos / self.cell_size).astype(int)
            
#             # Cell List 邻居搜索
#             neighbors = []
#             cd_np = self.cell_dim.cpu().numpy()
#             for dx in [-1,0,1]:
#                 for dy in [-1,0,1]:
#                     for dz in [-1,0,1]:
#                         cx = (idx[0]+dx) % cd_np[0]
#                         cy = (idx[1]+dy) % cd_np[1]
#                         cz = (idx[2]+dz) % cd_np[2]
#                         linear = cx*cd_np[1]*cd_np[2] + cy*cd_np[2] + cz
#                         neighbors.extend(self.cell_list[linear])
            
#             # 将当前 Top-K 原子也加入候选集 (确保原子ID未重复)
#             current_topk = set(self.topk_indices[vid].tolist())
#             candidate_aids = sorted(list(set(neighbors) | current_topk))
#             all_candidate_aids.append(candidate_aids)
#             cu_indices_map[vid] = candidate_aids

#         # 批量计算距离和 Top-K
#         for i, vid in enumerate(vids_to_recalc):
#             candidate_aids = cu_indices_map[vid]
#             if not candidate_aids:
#                 continue

#             # 提取候选 cu 原子的位置 (N', 3)
#             P_cu_candidates = self.P_cu[candidate_aids]
            
#             # 计算该 vacancy 到所有候选原子的距离 (1, N')
#             D_sq_candidate = self._compute_pbc_sq_distance(self.P_vac[vid].unsqueeze(0), P_cu_candidates).squeeze(0)
            
#             # Top-K 搜索
#             num_k = min(self.K, D_sq_candidate.shape[0])
#             new_dists_sq, candidate_rank = torch.topk(
#                 D_sq_candidate, k=num_k, largest=False
#             )
            
#             # 映射回原始 cu 原子 ID (K,)
#             new_indices = torch.tensor(candidate_aids, device=self.device, dtype=torch.long)[candidate_rank]
            
#             # 更新全局 Top-K 存储
#             self.topk_dists[vid, :num_k] = torch.sqrt(new_dists_sq)
#             self.topk_indices[vid, :num_k] = new_indices
#             # 处理 K 个数不够的情况
#             if num_k < self.K:
#                 self.topk_dists[vid, num_k:] = float('inf')
#                 self.topk_indices[vid, num_k:] = 0 # 填充占位符

#         return affected_vacancies
    
#     # -------------------- 更新系统 --------------------
#     def update_system(self, updated_cu=None, updated_vacancy=None):
#         affected_vacancies = set()
#         if updated_cu:
#             affected_vacancies |= self.update_cu(updated_cu)
#         if updated_vacancy:
#             affected_vacancies |= self.update_vacancy(updated_vacancy)
        
#         vid_list = sorted(list(affected_vacancies))
        
#         if not vid_list:
#             return {
#                 "vid_list": [],
#                 "diff_k": torch.empty((0, self.K, 3), dtype=self.dtype, device=self.device),
#                 "dist_k": torch.empty((0, self.K), dtype=self.dtype, device=self.device)
#             }

#         # 批量提取结果
#         vids_tensor = torch.tensor(vid_list, device=self.device, dtype=torch.long)
        
#         # 提取 Top-K 距离 (V', K)
#         dist_k = self.topk_dists[vids_tensor]
        
#         # 提取对应的原子ID (V', K)
#         indices_k = self.topk_indices[vids_tensor]
        
#         # 计算位移向量 (Difference Vector)
#         # P_vac_affected: (V', 1, 3)
#         P_vac_affected = self.P_vac[vids_tensor].unsqueeze(1)
        
#         # P_cu_topk: (V', K, 3)
#         P_cu_topk = self.P_cu[indices_k.reshape(-1)].reshape(len(vids_tensor), self.K, 3)

#         # diff_k: (V', K, 3)
#         diff_k = P_cu_topk - P_vac_affected
#         diff_k -= torch.round(diff_k / self.box) * self.box # 周期性边界条件 (PBC)

#         return {
#             "vid_list": vid_list,
#             "diff_k": diff_k.float(),
#             "dist_k": dist_k.float()
#         }
    
#     # -------------------- 查询某 vacancy Top-K --------------------
#     def get_topk_for_vacancy(self, vac_id):
#         dists = self.topk_dists[vac_id].cpu().tolist()
#         indices = self.topk_indices[vac_id].cpu().tolist()
#         # 返回 (atom_id, dist) 列表，并根据距离排序 (与原接口一致)
#         return sorted([(indices[i], dists[i]) for i in range(self.K)], key=lambda x: x[1])
    
#     # -------------------- 获取所有 Top-K 张量 --------------------
#     def get_all_topk_tensors(self):
#         # 批量计算所有 vacancy 的位移向量
#         # P_vac: (M, 1, 3)
#         P_vac_all = self.P_vac.unsqueeze(1)
        
#         # indices_k: (M, K)
#         indices_k = self.topk_indices
        
#         # P_cu_topk: (M, K, 3)
#         P_cu_topk = self.P_cu[indices_k.reshape(-1)].reshape(self.M, self.K, 3)

#         # diff_k: (M, K, 3)
#         diff_k = P_cu_topk - P_vac_all
#         diff_k -= torch.round(diff_k / self.box) * self.box # 周期性边界条件 (PBC)
        
#         dist_k = self.topk_dists

#         return {
#             "vid_list": list(range(self.M)),
#             "diff_k": diff_k.float(),
#             "dist_k": dist_k.float()
#         }
        
#     def verify_update(self, updated_cu=None, updated_vacancy=None):
#         # 复制对象以便验证
#         temp_system = TensorCoreVacancyTopKSystem(
#             self.P_cu.cpu().numpy(), 
#             self.P_vac.cpu().numpy(), 
#             self.K, 
#             self.box.cpu().tolist(), 
#             self.cell_size, 
#             device='cpu' # 验证时使用 CPU 避免影响原始 GPU 状态
#         )
        
#         # 先用 update_system 得到增量更新结果
#         result_update = self.update_system(updated_cu, updated_vacancy)
#         vid_list = result_update["vid_list"]
#         diff_k_update = result_update["diff_k"]
#         dist_k_update = result_update["dist_k"]

#         # 从头计算整个系统 (在复制对象上执行初始化)
#         # 注意: 这里需要重新计算 full topk，避免使用自身状态
#         temp_system._initialize_topk() 
#         result_full = temp_system.get_all_topk_tensors()
        
#         if not vid_list:
#              print("✅ 验证通过: 没有受影响的 vacancy")
#              return True
        
#         # 筛选出需要对比的 vid
#         vids_tensor = torch.tensor(vid_list, device=self.device, dtype=torch.long)
#         diff_k_full = result_full["diff_k"].to(self.device)[vids_tensor]
#         dist_k_full = result_full["dist_k"].to(self.device)[vids_tensor]

#         # 比较
#         diff_ok = torch.allclose(diff_k_update, diff_k_full, atol=1e-5)
#         dist_ok = torch.allclose(dist_k_update, dist_k_full, atol=1e-5)

#         if diff_ok and dist_ok:
#             print("✅ 验证通过: update_system 与全量计算结果一致")
#             return True
#         else:
#             print("❌ 验证失败!")
#             for i, vid in enumerate(vid_list):
#                 if not torch.allclose(diff_k_update[i], diff_k_full[i], atol=1e-5):
#                     print(f"  Vacancy {vid} diff_k 不一致. Max Diff: {torch.max(torch.abs(diff_k_update[i] - diff_k_full[i]))}")
#                 if not torch.allclose(dist_k_update[i], dist_k_full[i], atol=1e-5):
#                     print(f"  Vacancy {vid} dist_k 不一致. Max Diff: {torch.max(torch.abs(dist_k_update[i] - dist_k_full[i]))}")
#             return False
        
        
# class TensorCoreVacancyTopKSystem:
#     def __init__(self, cu_positions: np.ndarray, vacancy_positions: np.ndarray, K: int, box: tuple[float, float, float], cell_size: float, device: str = 'cuda'):
#         """
#         初始化 Top-K 最近邻系统。
        
#         Args:
#             cu_positions: np.array(N,3) - Cu 原子位置
#             vacancy_positions: np.array(M,3) - 空位位置
#             K: 每个 vacancy 的 Top-K 数量
#             box: tuple(Lx,Ly,Lz) - 周期边界
#             cell_size: float - Cell List 边长。必须 >= R_K^max，否则稀疏初始化结果错误。
#             device: 'cuda' 或 'cpu'
#         """
#         self.device = torch.device(device)
#         # self.dtype = torch.float32 
#         self.dtype = torch.bfloat16
        
        
#         # 全局设置 PyTorch 默认 dtype，避免 Runtime Error
#         if self.device.type == 'cuda':
#             torch.set_default_dtype(self.dtype)
        
#         # -------------------- 位置 (存储在GPU上) --------------------
#         self.P_cu = torch.tensor(cu_positions, dtype=self.dtype, device=self.device)
#         self.P_vac = torch.tensor(vacancy_positions, dtype=self.dtype, device=self.device)
#         self.N = self.P_cu.shape[0]
#         self.M = self.P_vac.shape[0]
#         self.K = K
#         self.box = torch.tensor(box, dtype=self.dtype, device=self.device)
#         self.cell_size = float(cell_size)
        
#         # -------------------- Cell List --------------------
#         self.cell_dim = torch.ceil(self.box / self.cell_size).long()
#         self.cell_list = defaultdict(list)
#         self.atom_cell = torch.zeros(self.N, dtype=torch.long, device=self.device)
#         self._build_cell_list()
        
#         # -------------------- Top-K 结果 --------------------
#         self.topk_indices = torch.zeros((self.M, self.K), dtype=torch.long, device=self.device)
#         # 使用 torch.full 而不是 float('inf') 来确保 dtype 一致性
#         self.topk_dists = torch.full((self.M, self.K), float('inf'), dtype=self.dtype, device=self.device)
        
#         # 使用稀疏初始化以避免 VRAM 溢出 (Chunk Size 根据您的 VRAM 限制调整)
#         self._initialize_topk_sparse(chunk_size=8192)
#         # self._verify_init_with_dense()

#     # ################### 辅助函数 (CUDA 优化) ###################
    
#     def _compute_pbc_sq_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
#         """
#         计算周期边界条件下的平方欧氏距离，采用原地操作优化内存占用。
#         pos1: (A, 3), pos2: (B, 3)
#         """
        
#         # 1. 计算初始差值 delta (A, B, 3)
#         delta = pos1.unsqueeze(1) - pos2.unsqueeze(0)
        
#         # 2. 周期边界条件 (原地修正)
#         correction_factor = torch.round(delta / self.box)
#         correction_term = correction_factor * self.box
        
#         delta.sub_(correction_term) # 原地更新 delta
        
#         del correction_factor, correction_term
        
#         # 3. 平方欧氏距离 (优化内存)
#         delta.pow_(2) # 原地计算平方
#         sq_dist = torch.sum(delta, dim=-1) # (A, B)
        
#         # 4. 显式释放 delta
#         del delta
        
#         return sq_dist
    
#     def _get_cell_index_cuda(self, pos: torch.Tensor) -> torch.Tensor:
#         """CUDA优化的cell index计算 (pos: (N, 3))"""
#         idx = torch.floor(pos / self.cell_size).long()
#         idx = torch.remainder(idx, self.cell_dim)
        
#         # 线性索引 (N,)
#         cd = self.cell_dim
#         linear = idx[:, 0] * (cd[1] * cd[2]) + idx[:, 1] * cd[2] + idx[:, 2]
#         return linear

#     # ################### Cell List & 更新 ###################
    
#     def _build_cell_list(self):
#         """在 CPU 上构建 Cell List 字典"""
#         self.atom_cell = self._get_cell_index_cuda(self.P_cu)
#         self.cell_list.clear()
        
#         for i, cell_idx in enumerate(self.atom_cell.cpu().tolist()):
#             self.cell_list[cell_idx].append(i)
    
#     def _update_cell_list(self, atom_id: int, new_pos: torch.Tensor):
#         """局部更新 Cell List (在 CPU 字典上操作)"""
#         new_cell_idx = self._get_cell_index_cuda(new_pos.unsqueeze(0)).item()
#         atom_id_long = int(atom_id)
#         old_cell_idx = self.atom_cell[atom_id_long].item()

#         if old_cell_idx != new_cell_idx:
#             try:
#                 self.cell_list[old_cell_idx].remove(atom_id_long)
#                 if not self.cell_list[old_cell_idx]:
#                     del self.cell_list[old_cell_idx]
#             except ValueError:
#                 pass 
                
#             self.cell_list[new_cell_idx].append(atom_id_long)
#             self.atom_cell[atom_id_long] = new_cell_idx

#     # ################### 核心方法：稀疏初始化 ###################
    
#     def _initialize_topk_sparse(self, chunk_size: int = 16384):
#         """
#         基于 Cell List 搜索的稀疏 Top-K 初始化，解决 VRAM 溢出问题。
#         """
#         M = self.M
#         if M == 0:
#             return

#         device = self.device
        
#         neighbor_offsets = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]]
#         cd_np = self.cell_dim.cpu().numpy()
        
#         num_chunks = (M + chunk_size - 1) // chunk_size
        
#         print(f"--- 启动稀疏初始化：M={M}, N={self.N}, K={self.K}, Chunk Size={chunk_size}, Cell Size={self.cell_size:.2f}, 块数={num_chunks} ---")

#         for i in range(num_chunks):
#             print("num_chunks id ", i)
#             start_idx = i * chunk_size
#             end_idx = min((i + 1) * chunk_size, M)
#             vids_chunk_list = list(range(start_idx, end_idx))
            
#             P_vac_chunk = self.P_vac[start_idx:end_idx] # (M_b, 3) 
            
#             all_candidate_aids = [] 
#             M_b_to_N_prime_map = [] 
            
#             # 1. CPU Cell List 查找
#             for vid in vids_chunk_list:
#                 v_pos_np = self.P_vac[vid].cpu().float().numpy()
#                 idx = np.floor(v_pos_np / self.cell_size).astype(int)
                
#                 candidate_aids = set()
                
#                 for dx, dy, dz in neighbor_offsets:
#                     cx = (idx[0]+dx) % cd_np[0]
#                     cy = (idx[1]+dy) % cd_np[1]
#                     cz = (idx[2]+dz) % cd_np[2]
#                     linear = cx*cd_np[1]*cd_np[2] + cy*cd_np[2] + cz
#                     candidate_aids.update(self.cell_list.get(linear, []))
                
#                 candidate_aids_list = sorted(list(candidate_aids))
#                 all_candidate_aids.append(candidate_aids_list)
#                 M_b_to_N_prime_map.append(candidate_aids_list)

#             if not any(M_b_to_N_prime_map): continue

#             # 2. 收集 unique_cu_ids 并提取位置 (GPU)
#             unique_cu_ids = sorted(list(set(aid for sublist in M_b_to_N_prime_map for aid in sublist)))
            
#             if not unique_cu_ids: continue
                
#             unique_cu_ids_t = torch.tensor(unique_cu_ids, device=device, dtype=torch.long)
#             P_cu_candidates = self.P_cu[unique_cu_ids_t] # (N', 3)
            
#             # 3. 计算距离 D_sq (M_b, N')
#             D_sq = self._compute_pbc_sq_distance(P_vac_chunk, P_cu_candidates)
            
#             # 4. 针对每个 vacancy (M_b)，从 D_sq 中提取其需要的列并 Top-K
#             id_to_col_map = {id: idx for idx, id in enumerate(unique_cu_ids)}
            
#             for m in range(len(vids_chunk_list)):
#                 vid = vids_chunk_list[m]
#                 cu_ids = M_b_to_N_prime_map[m]
                
#                 if not cu_ids: continue
                
#                 # 提取正确的距离列 (1, C)
#                 col_indices = torch.tensor([id_to_col_map[id] for id in cu_ids], device=device, dtype=torch.long)
#                 dists_to_candidates_sq = D_sq[m, col_indices]
                
#                 # Top-K 搜索
#                 num_k = min(self.K, dists_to_candidates_sq.shape[0])
#                 new_dists_sq, candidate_rank = torch.topk(
#                     dists_to_candidates_sq, k=num_k, largest=False
#                 )
                
#                 # 映射回原始 cu 原子 ID
#                 candidate_aids_t = torch.tensor(cu_ids, device=device, dtype=torch.long)
#                 new_indices = candidate_aids_t[candidate_rank]
                
#                 # 7. 更新全局存储
#                 self.topk_dists[vid, :num_k] = torch.sqrt(new_dists_sq).to(self.dtype)
#                 self.topk_indices[vid, :num_k] = new_indices
                
#                 # 处理 K 个数不够的情况
#                 if num_k < self.K:
#                     self.topk_dists[vid, num_k:] = torch.full_like(self.topk_dists[vid, num_k:], float('inf'))
#                     self.topk_indices[vid, num_k:] = 0

#             # 5. 显式释放内存
#             del P_vac_chunk, D_sq, P_cu_candidates, unique_cu_ids_t
#             torch.cuda.empty_cache()

#         print("--- 稀疏初始化完成 ---")
        
#     # ################### 增量更新 (Vacancy) ###################
    
#     def update_vacancy(self, updated_vacancy: dict[int, np.ndarray]) -> set[int]:
#         vids_to_recalc = list(updated_vacancy.keys())
#         if not vids_to_recalc:
#             return set()
        
#         vids_tensor = torch.tensor(vids_to_recalc, device=self.device, dtype=torch.long)
        
#         # 1. 批量更新位置
#         new_pos_np_list = [updated_vacancy[vid] for vid in vids_to_recalc]
#         new_pos_t = torch.tensor(new_pos_np_list, dtype=self.dtype, device=self.device)
#         self.P_vac[vids_tensor] = new_pos_t
        
#         all_updates = []
        
#         neighbor_offsets = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]]
#         cd_np = self.cell_dim.cpu().numpy()

#         for vid in vids_to_recalc:
#             v_pos = self.P_vac[vid].cpu().float().numpy()
#             idx = np.floor(v_pos / self.cell_size).astype(int)
            
#             # 1. Cell List 邻居搜索 (CPU)
#             candidate_aids = set()
#             for dx, dy, dz in neighbor_offsets:
#                 cx = (idx[0]+dx) % cd_np[0]
#                 cy = (idx[1]+dy) % cd_np[1]
#                 cz = (idx[2]+dz) % cd_np[2]
#                 linear = cx*cd_np[1]*cd_np[2] + cy*cd_np[2] + cz
#                 candidate_aids.update(self.cell_list.get(linear, []))
            
#             # 2. 加入当前 Top-K 原子
#             current_topk = set(self.topk_indices[vid].cpu().tolist())
#             candidate_aids.update(current_topk)
            
#             if not candidate_aids:
#                 continue
                
#             candidate_aids_list = sorted(list(candidate_aids))
            
#             # 3. 提取候选 cu 原子的位置 (N', 3) (GPU)
#             candidate_aids_t = torch.tensor(candidate_aids_list, device=self.device, dtype=torch.long)
#             P_cu_candidates = self.P_cu[candidate_aids_t] 
            
#             # 4. 计算距离 D_sq (1, N') (GPU)
#             D_sq_candidate = self._compute_pbc_sq_distance(self.P_vac[vid].unsqueeze(0), P_cu_candidates).squeeze(0)
            
#             # 5. Top-K 搜索 (GPU)
#             num_k = min(self.K, D_sq_candidate.shape[0])
#             new_dists_sq, candidate_rank = torch.topk(
#                 D_sq_candidate, k=num_k, largest=False
#             )
            
#             # 6. 映射回原始 cu 原子 ID (GPU)
#             new_indices = candidate_aids_t[candidate_rank]
            
#             all_updates.append((vid, new_dists_sq, new_indices, num_k))
            
#             # 7. 释放临时 Tensor
#             del P_cu_candidates, D_sq_candidate, new_dists_sq, candidate_rank, new_indices, candidate_aids_t
#             torch.cuda.empty_cache()

#         # 8. 最终更新全局存储
#         affected_vids = set()
#         for vid, new_dists_sq, new_indices, num_k in all_updates:
#             self.topk_dists[vid, :num_k] = torch.sqrt(new_dists_sq).to(self.dtype)
#             self.topk_indices[vid, :num_k] = new_indices
#             if num_k < self.K:
#                 self.topk_dists[vid, num_k:] = float('inf')
#                 self.topk_indices[vid, num_k:] = 0
#             affected_vids.add(vid)

#         return affected_vids
    
#     # ################### 增量更新 (Cu) ###################

#     # def update_cu(self, updated_cu: dict[int, np.ndarray]) -> set[int]:
        
#     #     affected_vacancies = set()
        
#     #     for cu_id, new_pos_np in updated_cu.items():
#     #         new_pos = torch.tensor(new_pos_np, dtype=self.dtype, device=self.device)
#     #         self.P_cu[cu_id] = new_pos
#     #         self._update_cell_list(cu_id, new_pos)

#     #         for vid in range(self.M):
#     #             # O(1) 距离检查
#     #             D_sq = self._compute_pbc_sq_distance(self.P_vac[vid].unsqueeze(0), new_pos.unsqueeze(0))
#     #             dist = D_sq.squeeze().sqrt()
                
#     #             max_dist_in_k = self.topk_dists[vid, -1]
                
#     #             # 检查受影响的 vacancy (CPU 传输)
#     #             if dist.item() < max_dist_in_k.item() or cu_id in self.topk_indices[vid].cpu().tolist():
#     #                 affected_vacancies.add(vid)
                
#     #             del D_sq

#     #     # 3. 批量更新受影响的 vacancy 的 Top-K
#     #     if affected_vacancies:
#     #         vid_list = sorted(list(affected_vacancies))
#     #         vids_tensor = torch.tensor(vid_list, device=self.device, dtype=torch.long)
            
#     #         P_vac_affected = self.P_vac[vids_tensor]
            
#     #         # 重新计算距离 (V', N) - 仍需谨慎 V' * N 的大小
#     #         D_sq_affected = self._compute_pbc_sq_distance(P_vac_affected, self.P_cu)
            
#     #         new_dists_sq, new_indices = torch.topk(
#     #             D_sq_affected, k=self.K, dim=1, largest=False
#     #         )
            
#     #         self.topk_dists[vids_tensor] = torch.sqrt(new_dists_sq).to(self.dtype)
#     #         self.topk_indices[vids_tensor] = new_indices
            
#     #         del D_sq_affected, new_dists_sq, new_indices
#     #         torch.cuda.empty_cache()

#     #     return affected_vacancies
    
#     # ################### 结果查询 ###################

#     def update_system(self, updated_cu: dict[int, np.ndarray] = None, updated_vacancy: dict[int, np.ndarray] = None):
#         """
#         处理所有更新并返回受影响 vacancy 的 Top-K 结果。
#         """
#         affected_vacancies = set()
#         if updated_cu:
#             affected_vacancies |= self.update_cu(updated_cu)
#         if updated_vacancy:
#             affected_vacancies |= self.update_vacancy(updated_vacancy)
        
#         vid_list = sorted(list(affected_vacancies))
        
#         if not vid_list:
#             return {
#                 "vid_list": [],
#                 "diff_k": torch.empty((0, self.K, 3), dtype=self.dtype, device=self.device),
#                 "dist_k": torch.empty((0, self.K), dtype=self.dtype, device=self.device)
#             }

#         vids_tensor = torch.tensor(vid_list, device=self.device, dtype=torch.long)
        
#         dist_k = self.topk_dists[vids_tensor]
#         indices_k = self.topk_indices[vids_tensor]
        
#         P_vac_affected = self.P_vac[vids_tensor].unsqueeze(1)
#         P_cu_topk = self.P_cu[indices_k.reshape(-1)].reshape(len(vids_tensor), self.K, 3)

#         diff_k = P_cu_topk - P_vac_affected
#         diff_k -= torch.round(diff_k / self.box) * self.box # PBC

#         return {
#             "vid_list": vid_list,
#             "diff_k": diff_k.to(self.dtype),
#             "dist_k": dist_k.to(self.dtype)
#         }
    
#     def get_all_topk_tensors(self):
#         """返回所有 M 个 vacancy 的 Top-K 结果"""
#         P_vac_all = self.P_vac.unsqueeze(1)
#         indices_k = self.topk_indices
#         P_cu_topk = self.P_cu[indices_k.reshape(-1)].reshape(self.M, self.K, 3)

#         diff_k = P_cu_topk - P_vac_all
#         diff_k -= torch.round(diff_k / self.box) * self.box # PBC
        
#         dist_k = self.topk_dists

#         return {
#             "vid_list": list(range(self.M)),
#             "diff_k": diff_k.to(self.dtype),
#             "dist_k": dist_k.to(self.dtype)
#         }
        
#   # ################### 增量更新 (Cu) ###################

#     def update_cu(self, updated_cu: dict[int, np.ndarray]) -> set[int]:
        
#         affected_vacancies = set()
        
#         # 1. 局部更新 Cu 位置和 Cell List，并标记受影响的 Vacancy
#         for cu_id, new_pos_np in updated_cu.items():
#             new_pos = torch.tensor(new_pos_np, dtype=self.dtype, device=self.device)
#             self.P_cu[cu_id] = new_pos
#             self._update_cell_list(cu_id, new_pos)

#             for vid in range(self.M):
#                 # O(1) 距离检查
#                 D_sq = self._compute_pbc_sq_distance(self.P_vac[vid].unsqueeze(0), new_pos.unsqueeze(0))
#                 dist = D_sq.squeeze().sqrt()
                
#                 max_dist_in_k = self.topk_dists[vid, -1]
                
#                 # 检查受影响的 vacancy (CPU 传输)
#                 if dist.item() < max_dist_in_k.item() or cu_id in self.topk_indices[vid].cpu().tolist():
#                     affected_vacancies.add(vid)
                
#                 del D_sq

#         # 2. 对所有受影响的 vacancy 进行稀疏 Top-K 重新计算 (关键修复!)
#         if affected_vacancies:
#             vids_to_recalc = sorted(list(affected_vacancies))
            
#             # 使用稀疏搜索逻辑重新计算 Top-K，保证 Inf 填充的一致性
#             self._recalculate_topk_sparse_vids(vids_to_recalc)

#         return affected_vacancies
    
#     def verify_update(self, updated_cu=None, updated_vacancy=None):
#         # 复制对象以便验证
#         temp_system = TensorCoreVacancyTopKSystem(
#             self.P_cu.cpu().float().numpy(), 
#             self.P_vac.cpu().float().numpy(), 
#             self.K, 
#             self.box.cpu().tolist(), 
#             self.cell_size, 
#             device='cpu' # 验证时使用 CPU 避免影响原始 GPU 状态
#         )
        
#         # 先用 update_system 得到增量更新结果
#         result_update = self.update_system(updated_cu, updated_vacancy)
#         vid_list = result_update["vid_list"]
#         diff_k_update = result_update["diff_k"]
#         dist_k_update = result_update["dist_k"]

#         # 从头计算整个系统 (在复制对象上执行初始化)
#         # 注意: 这里需要重新计算 full topk，避免使用自身状态
#         temp_system._initialize_topk_sparse() 
#         result_full = temp_system.get_all_topk_tensors()
        
#         if not vid_list:
#              print("✅ 验证通过: 没有受影响的 vacancy")
#              return True
        
#         # 筛选出需要对比的 vid
#         vids_tensor = torch.tensor(vid_list, device=self.device, dtype=torch.long)
#         diff_k_full = result_full["diff_k"].to(self.device)[vids_tensor]
#         dist_k_full = result_full["dist_k"].to(self.device)[vids_tensor]

#         # 比较
#         diff_ok = torch.allclose(diff_k_update, diff_k_full, atol=1e-5)
#         dist_ok = torch.allclose(dist_k_update, dist_k_full, atol=1e-5)

#         if diff_ok and dist_ok:
#             print("✅ 验证通过: update_system 与全量计算结果一致")
#             return True
#         else:
#             print("❌ 验证失败!")
#             for i, vid in enumerate(vid_list):
#                 if not torch.allclose(diff_k_update[i], diff_k_full[i], atol=1e-5):
#                     print(f"  Vacancy {vid} diff_k 不一致. Max Diff: {torch.max(torch.abs(diff_k_update[i] - diff_k_full[i]))}")
#                 if not torch.allclose(dist_k_update[i], dist_k_full[i], atol=1e-5):
#                     print(f"  Vacancy {vid} dist_k 不一致. Max Diff: {torch.max(torch.abs(dist_k_update[i] - dist_k_full[i]))}")
#             return False
        
#     def _verify_init_with_dense(self):
#         """
#         验证稀疏初始化 (_initialize_topk_sparse) 的结果与朴素全量计算
#         (M x N 稠密矩阵) 的结果是否一致。
        
#         警告: 如果 M*N 过大，此函数会导致 GPU 内存溢出 (OOM)！
#         """
        
#         # --- 1. 获取稀疏初始化结果 ---
#         sparse_dists = self.topk_dists.clone()
#         sparse_indices = self.topk_indices.clone()
        
#         # --- 2. 朴素的全量计算 (Dense Calculation) ---
#         print("\n--- 启动朴素全量 Top-K 验证 (M x N 稠密计算) ---")
        
#         # 计算 M x N 的平方距离矩阵
#         try:
#             D_sq_dense = self._compute_pbc_sq_distance(self.P_vac, self.P_cu)
#         except RuntimeError as e:
#             if "out of memory" in str(e):
#                 print("❌ 验证失败：GPU 内存不足 (OOM) 无法执行 M x N 稠密计算。")
#                 print("请减小 M 或 N 进行测试，或信任稀疏算法的逻辑正确性。")
#                 return False
#             raise e
        
#         # 找到 Top-K
#         dense_dists_sq, dense_indices = torch.topk(
#             D_sq_dense, k=self.K, dim=1, largest=False
#         )
        
#         dense_dists = torch.sqrt(dense_dists_sq).to(self.dtype)
        
#         # 释放内存
#         del D_sq_dense, dense_dists_sq
#         torch.cuda.empty_cache()
        
#         # --- 3. 对比结果 ---
        
#         # 使用 atol=1e-5 允许浮点误差
#         dist_ok = torch.allclose(sparse_dists, dense_dists, atol=1e-5, equal_nan=True)
#         # 索引必须完全相同 (注意：如果距离完全相等，topk的tie-breaking可能导致索引顺序不同，但通常不应发生)
#         indices_ok = torch.equal(sparse_indices, dense_indices)
        
#         print("--- 朴素全量 Top-K 验证完成 ---")
        
#         if dist_ok and indices_ok:
#             print("✅ 稀疏初始化验证成功：距离和索引与朴素全量计算一致。")
#             return True
#         else:
#             print("❌ 稀疏初始化验证失败！稀疏结果与朴素稠密结果不符。")
            
#             # 查找不一致的 Vacancy ID
#             dist_diff = torch.abs(sparse_dists - dense_dists)
#             max_dist_diff = torch.max(dist_diff).item()
            
#             # 查找索引或距离不一致的行
#             idx_diff_mask = torch.any(sparse_indices != dense_indices, dim=1)
#             dist_diff_mask = torch.any(dist_diff > 1e-5, dim=1)
            
#             failed_vids = torch.where(dist_diff_mask | idx_diff_mask)[0].cpu().tolist()
            
#             print(f"  Max Absolute Distance Difference: {max_dist_diff:.6e}")
#             print(f"  Total Vacancies failed: {len(failed_vids)} / {self.M}")
            
#             if failed_vids:
#                 vid = failed_vids[0]
#                 print(f"\n  首次失败 Vacancy ID: {vid}")
#                 print("    稀疏距离 (Sparse Dists):", sparse_dists[vid])
#                 print("    全量距离 (Dense Dists):", dense_dists[vid])
#                 print("    稀疏索引 (Sparse Indices):", sparse_indices[vid])
#                 print("    全量索引 (Dense Indices):", dense_indices[vid])
                
#             return False

# def _recalculate_topk_sparse_vids(self, vids_to_recalc: list[int]):
#         """
#         对给定的 Vacancy ID 列表执行 Cell List 增强的稀疏 Top-K 计算，
#         并更新 self.topk_dists/indices。
#         """
#         if not vids_to_recalc:
#             return

#         all_updates = []
#         neighbor_offsets = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]]
#         cd_np = self.cell_dim.cpu().numpy()

#         for vid in vids_to_recalc:
#             # 1. 获取 Vacancy 位置
#             v_pos = self.P_vac[vid].cpu().float().numpy()
#             idx = np.floor(v_pos / self.cell_size).astype(int)
            
#             # 2. Cell List 邻居搜索 (CPU)
#             candidate_aids = set()
#             for dx, dy, dz in neighbor_offsets:
#                 cx = (idx[0]+dx) % cd_np[0]
#                 cy = (idx[1]+dy) % cd_np[1]
#                 cz = (idx[2]+dz) % cd_np[2]
#                 linear = cx*cd_np[1]*cd_np[2] + cy*cd_np[2] + cz
#                 candidate_aids.update(self.cell_list.get(linear, []))
            
#             # 3. 加入当前 Top-K 原子 (关键：保证 update_vacancy 的行为)
#             current_topk = set(self.topk_indices[vid].cpu().tolist())
#             candidate_aids.update(current_topk)
            
#             if not candidate_aids:
#                 all_updates.append((vid, None, None, 0)) # 记录 num_k=0
#                 continue
                
#             candidate_aids_list = sorted(list(candidate_aids))
            
#             # 4. 提取候选 cu 原子的位置和计算距离 (GPU)
#             candidate_aids_t = torch.tensor(candidate_aids_list, device=self.device, dtype=torch.long)
#             P_cu_candidates = self.P_cu[candidate_aids_t] 
            
#             D_sq_candidate = self._compute_pbc_sq_distance(self.P_vac[vid].unsqueeze(0), P_cu_candidates).squeeze(0)
            
#             # 5. Top-K 搜索 (GPU)
#             num_k = min(self.K, D_sq_candidate.shape[0])
#             new_dists_sq, candidate_rank = torch.topk(
#                 D_sq_candidate, k=num_k, largest=False
#             )
            
#             # 6. 映射回原始 cu 原子 ID (GPU)
#             new_indices = candidate_aids_t[candidate_rank]
            
#             all_updates.append((vid, new_dists_sq, new_indices, num_k))
            
#             # 7. 释放临时 Tensor
#             del P_cu_candidates, D_sq_candidate, new_dists_sq, candidate_rank, new_indices, candidate_aids_t
#             torch.cuda.empty_cache()

#         # 8. 最终更新全局存储 (Inf 填充逻辑)
#         vids_tensor = torch.tensor([u[0] for u in all_updates], device=self.device, dtype=torch.long)
        
#         for vid, new_dists_sq, new_indices, num_k in all_updates:
            
#             # 注意：这里需要对 new_dists_sq 取根号
#             if new_dists_sq is not None:
#                  final_dists = torch.sqrt(new_dists_sq).to(self.dtype)
#             else:
#                  final_dists = None

#             self.topk_indices[vid, :num_k] = new_indices
#             self.topk_dists[vid, :num_k] = final_dists

#             if num_k < self.K:
#                 self.topk_dists[vid, num_k:] = float('inf')
#                 self.topk_indices[vid, num_k:] = 0


import torch
import numpy as np
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# class AdaptiveVacancyTopK:
#     """
#     高性能、自适应邻域搜索的 Top-K 系统（适用于大规模：M ~ 1e5, N ~ 8e4）。
#     设计要点：
#       - 存储位置可选 bfloat16/float32（storage_dtype）
#       - 计算使用 float32（compute_dtype）
#       - 先 1-ring (27 cell) 搜索，取 top-K 的 r_max；再用 cell AABB vs sphere 测试扩展候选 cell（向量化）
#       - 对移动的 cu 使用增量 cell 映射更新
#     """

#     def __init__(
#         self,
#         cu_positions: np.ndarray,
#         vacancy_positions: np.ndarray,
#         K: int,
#         box: Tuple[float, float, float],
#         cell_size: float,
#         device: str = "cuda",
#         storage_dtype: str = "float32",
#         max_extra_ring: int = 2,  # 最多扩展到 2-ring（默认）
#     ):
#         # device & dtype
#         self.device = torch.device(device)
#         assert storage_dtype in ("bfloat16", "float32")
#         self.storage_dtype = getattr(torch, storage_dtype)
#         self.compute_dtype = torch.float32

#         # data arrays ensure float32 numpy input
#         cu_positions = np.asarray(cu_positions, dtype=np.float32)
#         vacancy_positions = np.asarray(vacancy_positions, dtype=np.float32)
#         assert cu_positions.ndim == 2 and cu_positions.shape[1] == 3
#         assert vacancy_positions.ndim == 2 and vacancy_positions.shape[1] == 3

#         self.N = cu_positions.shape[0]
#         self.M = vacancy_positions.shape[0]
#         self.K = int(K)
#         self.box = torch.tensor(box, dtype=self.compute_dtype, device=self.device)
#         self.cell_size = float(cell_size)
#         self.max_extra_ring = int(max_extra_ring)

#         # store positions on device (storage dtype)
#         self.P_cu = torch.tensor(cu_positions, dtype=self.storage_dtype, device=self.device)
#         self.P_vac = torch.tensor(vacancy_positions, dtype=self.storage_dtype, device=self.device)

#         # compute integer cell dims (nx,ny,nz) as ints
#         cell_dim = np.ceil((self.box.cpu().numpy().astype(np.float32) / self.cell_size)).astype(int)
#         self.cell_dim = tuple(int(x) for x in cell_dim)  # (nx,ny,nz)
#         self.ncells = self.cell_dim[0] * self.cell_dim[1] * self.cell_dim[2]

#         # CPU-side cell -> lists (dynamic)
#         self.cell_cu = defaultdict(list)   # cell_index -> list of cu ids (ints)
#         self.cell_vac = defaultdict(list)  # cell_index -> list of vac ids (ints)
#         # per-atom current cell (on CPU numpy int)
#         self.cu_cell = np.zeros(self.N, dtype=np.int64)
#         self.vac_cell = np.zeros(self.M, dtype=np.int64)

#         # precompute cell geometry (CPU numpy) for AABB overlap tests
#         # cell centers (ncells, 3) and half diagonal (scalar)
#         self._build_cell_geometries()

#         # build initial mappings and cells
#         self._build_cell_mappings(cu_positions, vacancy_positions)

#         # top-k storage: indices (long) -1 padded, distances float32
#         self.topk_indices = torch.full((self.M, self.K), -1, dtype=torch.long, device=self.device)
#         self.topk_dists = torch.full((self.M, self.K), float("inf"), dtype=self.compute_dtype, device=self.device)

#         # initialize sparse top-k
#         self._initialize_topk_sparse()

#     # ---------------- helpers: cell geometry & indexing ----------------
#     def _build_cell_geometries(self):
#         """
#         Build CPU numpy arrays:
#           - self.cell_centers: shape (ncells,3)
#           - self.cell_half_diag: scalar half-diagonal for cubic cell (same for all)
#         and also cell (cx,cy,cz) coordinate arrays for mapping linear indices.
#         """
#         nx, ny, nz = self.cell_dim
#         # compute cell centers by coordinates
#         centers = []
#         for cx in range(nx):
#             for cy in range(ny):
#                 for cz in range(nz):
#                     center = np.array(
#                         [(cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size, (cz + 0.5) * self.cell_size],
#                         dtype=np.float32,
#                     )
#                     centers.append(center)
#         self.cell_centers = np.stack(centers, axis=0)  # shape (ncells,3)
#         # half diagonal (cube of side cell_size): half_diag = sqrt(3*(cell_size/2)^2)
#         self.cell_half_diag = np.sqrt(3.0 * (self.cell_size * 0.5) ** 2).astype(np.float32)

#         # store linear->(cx,cy,cz) for convenience
#         linear_to_coords = np.zeros((self.ncells, 3), dtype=np.int32)
#         idx = 0
#         for cx in range(nx):
#             for cy in range(ny):
#                 for cz in range(nz):
#                     linear_to_coords[idx, 0] = cx
#                     linear_to_coords[idx, 1] = cy
#                     linear_to_coords[idx, 2] = cz
#                     idx += 1
#         self.linear_to_coords = linear_to_coords

#     def _pos_to_cell_index_np(self, pos_np: np.ndarray) -> np.ndarray:
#         """
#         Vectorized mapping numpy positions (N,3) -> linear cell indices (N,)
#         Positions are expected in [0, box) or not; we apply modulo with box.
#         """
#         box = self.box.cpu().numpy().astype(np.float32)
#         pos_mod = np.remainder(pos_np.astype(np.float32), box)
#         idx = np.floor(pos_mod / self.cell_size).astype(np.int64)
#         idx[:, 0] = np.remainder(idx[:, 0], self.cell_dim[0])
#         idx[:, 1] = np.remainder(idx[:, 1], self.cell_dim[1])
#         idx[:, 2] = np.remainder(idx[:, 2], self.cell_dim[2])
#         linear = idx[:, 0] * (self.cell_dim[1] * self.cell_dim[2]) + idx[:, 1] * self.cell_dim[2] + idx[:, 2]
#         return linear.astype(np.int64)

#     def _pos_to_cell_index_tensor(self, pos_t: torch.Tensor) -> torch.Tensor:
#         """
#         Vectorized mapping on torch Tensor (device aware). Returns long tensor.
#         """
#         p = pos_t.to(self.compute_dtype)
#         box = self.box.to(self.compute_dtype)
#         pos_mod = torch.remainder(p, box)
#         idx = torch.floor(pos_mod / self.cell_size).long()
#         cd = torch.tensor(self.cell_dim, device=pos_t.device, dtype=torch.long)
#         idx = torch.remainder(idx, cd)
#         linear = idx[:, 0] * (cd[1] * cd[2]) + idx[:, 1] * cd[2] + idx[:, 2]
#         return linear

#     # ---------------- build mappings ----------------
#     def _build_cell_mappings(self, cu_positions_np: np.ndarray, vac_positions_np: np.ndarray):
#         """
#         Build CPU-side cell_cu and cell_vac mappings (lists). Also fill cu_cell and vac_cell arrays.
#         cu_positions_np, vac_positions_np are numpy float32 arrays.
#         """
#         self.cell_cu.clear()
#         self.cell_vac.clear()
#         # compute linear indices
#         cu_cells = self._pos_to_cell_index_np(cu_positions_np)
#         vac_cells = self._pos_to_cell_index_np(vac_positions_np)
#         self.cu_cell = cu_cells.copy()
#         self.vac_cell = vac_cells.copy()

#         for aid, c in enumerate(cu_cells):
#             self.cell_cu[int(c)].append(int(aid))
#         for vid, c in enumerate(vac_cells):
#             self.cell_vac[int(c)].append(int(vid))

#     # ---------------- AABB vs sphere (vectorized CPU) ----------------
#     def _cells_overlapping_sphere(self, pos_np: np.ndarray, radius: float, exclude_set: Set[int] = None) -> np.ndarray:
#         """
#         Return numpy array of linear cell indices whose AABB overlaps sphere(center=pos_np, radius).
#         Vectorized test: for each cell center, compute axis-wise distance = max(0, abs(center - pos) - half_extent).
#         squared sum <= radius^2  => overlap
#         exclude_set: optional to exclude some cells quickly
#         """
#         # pos_np shape (3,)
#         # quickly compute center distances
#         diff = np.abs(self.cell_centers - pos_np.reshape(1, 3)).astype(np.float32)
#         # half-extent per axis
#         half_extent = self.cell_size * 0.5
#         d_axis = np.maximum(0.0, diff - half_extent)
#         d2 = np.sum(d_axis * d_axis, axis=1)  # (ncells,)
#         mask = d2 <= (radius * radius)
#         if exclude_set:
#             # mask out excluded cells
#             ex = np.fromiter(exclude_set, dtype=np.int64)
#             if ex.size > 0:
#                 mask[ex] = False
#         return np.nonzero(mask)[0].astype(np.int64)

#     # ---------------- PBC distance compute (GPU) ----------------
#     def _compute_pbc_sq_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
#         """
#         pos1: (A,3) float32 tensor (on device)
#         pos2: (B,3) float32 tensor (on device)
#         returns (A,B) squared distances with minimum image convention (float32)
#         """
#         p1 = pos1.to(self.compute_dtype)
#         p2 = pos2.to(self.compute_dtype)
#         delta = p1.unsqueeze(1) - p2.unsqueeze(0)  # (A,B,3)
#         box = self.box.to(self.compute_dtype)
#         corr = torch.round(delta / box) * box
#         delta = delta - corr
#         sq = torch.sum(delta * delta, dim=-1)
#         return sq

#     # ---------------- initialization (sparse per vacancy) ----------------
#     def _initialize_topk_sparse(self):
#         """
#         Initialize topk for all vacancies using Adaptive expansion.
#         Loop over vacancys, but heavy distance work vectorized on GPU per vacancy.
#         """
#         if self.M == 0 or self.N == 0:
#             return

#         # 1. Build GPU Cell List


#             # deterministic ordering
#             cand_list = sorted(list(cand_set), key=int)
#             cand_idx_t = torch.tensor(cand_list, device=self.device, dtype=torch.long)
#             P_cand = P_cu_f32[cand_idx_t]  # (C,3)

#             # compute distances and topk
#             D_sq = self._compute_pbc_sq_distance(vpos_dev.unsqueeze(0), P_cand).squeeze(0)
#             C = D_sq.shape[0]
#             num_k = min(self.K, C)
#             d_sq_topk, pos_topk = torch.topk(D_sq, k=num_k, largest=False)
#             idx_topk = cand_idx_t[pos_topk]

#             # write
#             self.topk_indices[vid, :num_k] = idx_topk
#             self.topk_dists[vid, :num_k] = torch.sqrt(d_sq_topk)

#             # if need expand adaptively
#             if num_k < self.K:
#                 r_max = float("inf")
#             else:
#                 r_max = float(self.topk_dists[vid, :num_k].max().item())

#             # adaptive expansion: find cells whose AABB intersects sphere radius r_max
#             # exclude cells already in candidate_cells for speed
#             if not np.isinf(r_max):
#                 # find cells overlapping sphere (vectorized CPU)
#                 extra_cells = self._cells_overlapping_sphere(vpos_np, r_max, exclude_set=set(candidate_cells))
#                 # limit expansion by max_extra_ring to avoid runaway
#                 # but _cells_overlapping_sphere already filters geometric overlap; we also restrict to rings
#                 # collect extra cells up to ring limit
#                 if extra_cells.size > 0:
#                     # further filter by ring distance <= max_extra_ring
#                     extra_cells_filtered = []
#                     for c in extra_cells:
#                         # compute manhattan ring distance in cell coordinates from vid_cell
#                         cx0, cy0, cz0 = self.linear_to_coords[vid_cell]
#                         cx1, cy1, cz1 = self.linear_to_coords[int(c)]
#                         ring_dist = max(abs(cx1 - cx0), abs(cy1 - cy0), abs(cz1 - cz0))
#                         if ring_dist <= self.max_extra_ring:
#                             extra_cells_filtered.append(int(c))
#                     # combine new cand cells
#                     for c in extra_cells_filtered:
#                         if c not in candidate_cells:
#                             candidate_cells.append(c)

#                     # recompute candidate cu set after expansion
#                     cand_set2 = set()
#                     for c in candidate_cells:
#                         cand_set2.update(self.cell_cu.get(int(c), []))
#                     cand_list2 = sorted(list(cand_set2), key=int)
#                     cand_idx_t2 = torch.tensor(cand_list2, device=self.device, dtype=torch.long)
#                     P_cand2 = P_cu_f32[cand_idx_t2]
#                     D_sq2 = self._compute_pbc_sq_distance(vpos_dev.unsqueeze(0), P_cand2).squeeze(0)
#                     C2 = D_sq2.shape[0]
#                     num_k2 = min(self.K, C2)
#                     d_sq_topk2, pos_topk2 = torch.topk(D_sq2, k=num_k2, largest=False)
#                     idx_topk2 = cand_idx_t2[pos_topk2]
#                     self.topk_indices[vid, :num_k2] = idx_topk2
#                     self.topk_dists[vid, :num_k2] = torch.sqrt(d_sq_topk2)
#                     if num_k2 < self.K:
#                         self.topk_indices[vid, num_k2:] = -1
#                         self.topk_dists[vid, num_k2:] = float("inf")
#             # pad if necessary
#             if self.topk_indices[vid, -1] == -1:
#                 # ensure dists filled with inf
#                 mask = (self.topk_indices[vid] == -1)
#                 if mask.any():
#                     self.topk_dists[vid][mask] = float("inf")

#     def _gather_ring_cells(self, linear_cell: int, ring: int = 1) -> List[int]:
#         """
#         Return list of linear cell indices within Chebyshev distance 'ring' from linear_cell.
#         ring=1 => 3x3x3, ring=2 => 5x5x5, etc.
#         Deterministic order: sorted by int.
#         """
#         nx, ny, nz = self.cell_dim
#         cx0, cy0, cz0 = self.linear_to_coords[int(linear_cell)]
#         cells = []
#         for dx in range(-ring, ring + 1):
#             for dy in range(-ring, ring + 1):
#                 for dz in range(-ring, ring + 1):
#                     nx_ = (cx0 + dx) % nx
#                     ny_ = (cy0 + dy) % ny
#                     nz_ = (cz0 + dz) % nz
#                     linear = nx_ * (ny * nz) + ny_ * nz + nz_
#                     cells.append(int(linear))
#         return sorted(list(set(cells)), key=int)

#     # ---------------- incremental updates ----------------
#     def update_vacancy(self, updated_vac: Dict[int, np.ndarray]) -> Set[int]:
#         """
#         updated_vac: dict vid -> new_pos (np.array 3,)
#         Batch update vacancies positions and recompute their top-K adaptively.
#         Returns set of vids recalculated.
#         """
#         if not updated_vac:
#             return set()
#         vids = sorted(updated_vac.keys())
#         # batch positions
#         new_pos_np = np.vstack([np.asarray(updated_vac[v], dtype=np.float32).reshape(1, 3) for v in vids])
#         new_pos_t = torch.tensor(new_pos_np, dtype=self.storage_dtype, device=self.device)
#         vids_t = torch.tensor(vids, dtype=torch.long, device=self.device)
#         # update storage
#         self.P_vac[vids_t] = new_pos_t
#         # update vac_cell mapping incrementally
#         new_vac_cells = self._pos_to_cell_index_np(new_pos_np)
#         for i, vid in enumerate(vids):
#             oldc = int(self.vac_cell[vid])
#             newc = int(new_vac_cells[i])
#             if oldc != newc:
#                 # remove from old cell list
#                 try:
#                     self.cell_vac[oldc].remove(int(vid))
#                 except ValueError:
#                     pass
#                 self.cell_vac[newc].append(int(vid))
#                 self.vac_cell[vid] = newc
#         # recompute just these vids
#         self._recalculate_topk_sparse_vids(vids)
#         return set(vids)

#     def update_cu(self, updated_cu: Dict[int, np.ndarray]) -> Set[int]:
#         """
#         updated_cu: dict cu_id -> new_pos (np.array 3,)
#         Update storage positions and incrementally update cell_cu mapping.
#         Return set of affected vacancy ids that need recalculation.
#         """
#         if not updated_cu:
#             return set()
#         aids = sorted(updated_cu.keys())
#         new_pos_np = np.vstack([np.asarray(updated_cu[a], dtype=np.float32).reshape(1, 3) for a in aids])
#         new_pos_t = torch.tensor(new_pos_np, dtype=self.storage_dtype, device=self.device)
#         aids_t = torch.tensor(aids, dtype=torch.long, device=self.device)
#         # update storage positions
#         self.P_cu[aids_t] = new_pos_t

#         # update cu_cell mapping incrementally and cell_cu lists
#         new_cu_cells = self._pos_to_cell_index_np(new_pos_np)
#         affected_vacancies = set()
#         for i, aid in enumerate(aids):
#             oldc = int(self.cu_cell[aid])
#             newc = int(new_cu_cells[i])
#             if oldc != newc:
#                 # remove from old
#                 try:
#                     self.cell_cu[oldc].remove(int(aid))
#                 except ValueError:
#                     pass
#                 self.cell_cu[newc].append(int(aid))
#                 self.cu_cell[aid] = newc
#             # for safety, even if it stayed in same cell, cu moved, so nearby vacancies may be affected
#             # find vacancies in neighbor ring of newc (1-ring) and mark
#             nx, ny, nz = self.cell_dim
#             cx, cy, cz = self.linear_to_coords[newc]
#             for dx in (-1, 0, 1):
#                 for dy in (-1, 0, 1):
#                     for dz in (-1, 0, 1):
#                         nx_ = (cx + dx) % nx
#                         ny_ = (cy + dy) % ny
#                         nz_ = (cz + dz) % nz
#                         linear = nx_ * (ny * nz) + ny_ * nz + nz_
#                         for vid in self.cell_vac.get(int(linear), []):
#                             affected_vacancies.add(int(vid))

#         if affected_vacancies:
#             self._recalculate_topk_sparse_vids(sorted(list(affected_vacancies)))

#         return affected_vacancies

#     def _recalculate_topk_sparse_vids(self, vids: List[int]):
#         """
#         Recompute topk for listed vacancy ids using adaptive expansion.
#         Uses GPU for distance computations in batches per vacancy (vectorized per candidate set).
#         """
#         if not vids:
#             return
#         P_cu_f32 = self.P_cu.to(self.compute_dtype)

#         for vid in vids:
#             vpos_dev = self.P_vac[vid].to(self.compute_dtype)
#             vpos_np = self.P_vac[vid].cpu().to(torch.float32).numpy().astype(np.float32)

#             vid_cell = int(self.vac_cell[vid])
#             candidate_cells = self._gather_ring_cells(vid_cell, ring=1)

#             # initial candidate set
#             cand_set = set()
#             for c in candidate_cells:
#                 cand_set.update(self.cell_cu.get(int(c), []))

#             # also include previous topk (to avoid losing existing neighbors)
#             prev_idx = self.topk_indices[vid].cpu().numpy().astype(int).tolist()
#             for idx in prev_idx:
#                 if idx >= 0:
#                     cand_set.add(int(idx))

#             if not cand_set:
#                 self.topk_indices[vid, :] = -1
#                 self.topk_dists[vid, :] = float("inf")
#                 continue

#             cand_list = sorted(list(cand_set), key=int)
#             cand_idx_t = torch.tensor(cand_list, device=self.device, dtype=torch.long)
#             P_cand = P_cu_f32[cand_idx_t]
#             D_sq = self._compute_pbc_sq_distance(vpos_dev.unsqueeze(0), P_cand).squeeze(0)
#             C = D_sq.shape[0]
#             num_k = min(self.K, C)
#             d_sq_topk, pos_topk = torch.topk(D_sq, k=num_k, largest=False)
#             idx_topk = cand_idx_t[pos_topk]
#             # write
#             self.topk_indices[vid, :num_k] = idx_topk
#             self.topk_dists[vid, :num_k] = torch.sqrt(d_sq_topk)
#             if num_k < self.K:
#                 self.topk_indices[vid, num_k:] = -1
#                 self.topk_dists[vid, num_k:] = float("inf")

#             # adaptive expansion
#             r_max = float("inf") if num_k < self.K else float(self.topk_dists[vid, :num_k].max().item())

#             if not np.isinf(r_max):
#                 extra_cells = self._cells_overlapping_sphere(vpos_np, r_max, exclude_set=set(candidate_cells))
#                 if extra_cells.size > 0:
#                     # filter by ring distance <= max_extra_ring
#                     extra_cells_filtered = []
#                     cx0, cy0, cz0 = self.linear_to_coords[vid_cell]
#                     for c in extra_cells:
#                         cx1, cy1, cz1 = self.linear_to_coords[int(c)]
#                         ring_dist = max(abs(cx1 - cx0), abs(cy1 - cy0), abs(cz1 - cz0))
#                         if ring_dist <= self.max_extra_ring and (int(c) not in candidate_cells):
#                             extra_cells_filtered.append(int(c))
#                     if extra_cells_filtered:
#                         # extend candidate cells and recompute
#                         for c in extra_cells_filtered:
#                             if c not in candidate_cells:
#                                 candidate_cells.append(c)
#                         cand_set2 = set()
#                         for c in candidate_cells:
#                             cand_set2.update(self.cell_cu.get(int(c), []))
#                         # include previous topk too
#                         for idx in prev_idx:
#                             if idx >= 0:
#                                 cand_set2.add(int(idx))
#                         cand_list2 = sorted(list(cand_set2), key=int)
#                         cand_idx_t2 = torch.tensor(cand_list2, device=self.device, dtype=torch.long)
#                         P_cand2 = P_cu_f32[cand_idx_t2]
#                         D_sq2 = self._compute_pbc_sq_distance(vpos_dev.unsqueeze(0), P_cand2).squeeze(0)
#                         C2 = D_sq2.shape[0]
#                         num_k2 = min(self.K, C2)
#                         d_sq_topk2, pos_topk2 = torch.topk(D_sq2, k=num_k2, largest=False)
#                         idx_topk2 = cand_idx_t2[pos_topk2]
#                         self.topk_indices[vid, :num_k2] = idx_topk2
#                         self.topk_dists[vid, :num_k2] = torch.sqrt(d_sq_topk2)
#                         if num_k2 < self.K:
#                             self.topk_indices[vid, num_k2:] = -1
#                             self.topk_dists[vid, num_k2:] = float("inf")

#             # ensure padding handled
#             mask_pad = (self.topk_indices[vid] == -1)
#             if mask_pad.any():
#                 self.topk_dists[vid][mask_pad] = float("inf")

#     # ---------------- query APIs ----------------
#     def update_system(self, updated_cu: Dict[int, np.ndarray] = None, updated_vacancy: Dict[int, np.ndarray] = None):
#         affected = set()
#         if updated_cu:
#             affected |= self.update_cu(updated_cu)
#         if updated_vacancy:
#             affected |= self.update_vacancy(updated_vacancy)
#         vid_list = sorted(list(affected))
#         if not vid_list:
#             empty_diff = torch.empty((0, self.K, 3), dtype=self.compute_dtype, device=self.device)
#             empty_dist = torch.empty((0, self.K), dtype=self.compute_dtype, device=self.device)
#             return {"vid_list": [], "diff_k": empty_diff, "dist_k": empty_dist}

#         vids_t = torch.tensor(vid_list, dtype=torch.long, device=self.device)
#         indices_k = self.topk_indices[vids_t]  # (B,K)
#         dist_k = self.topk_dists[vids_t]      # (B,K)
#         mask_valid = (indices_k >= 0)
#         idx_safe = indices_k.clone()
#         idx_safe[~mask_valid] = 0
#         P_cu_topk = self.P_cu[idx_safe.reshape(-1)].reshape(len(vid_list), self.K, 3).to(self.compute_dtype)
#         P_vac_affected = self.P_vac[vids_t].unsqueeze(1).to(self.compute_dtype)
#         diff_k = P_cu_topk - P_vac_affected
#         diff_k = diff_k - torch.round(diff_k / self.box.to(self.compute_dtype)) * self.box.to(self.compute_dtype)
#         diff_k[~mask_valid.unsqueeze(-1).expand_as(diff_k)] = 0.0
#         dist_k[~mask_valid] = float("inf")
#         return {"vid_list": vid_list, "diff_k": diff_k.to(self.compute_dtype), "dist_k": dist_k.to(self.compute_dtype)}

#     def get_all_topk_tensors(self):
#         vids_t = torch.arange(self.M, dtype=torch.long, device=self.device)
#         indices_k = self.topk_indices
#         dist_k = self.topk_dists
#         mask_valid = (indices_k >= 0)
#         idx_safe = indices_k.clone()
#         idx_safe[~mask_valid] = 0
#         P_cu_topk = self.P_cu[idx_safe.reshape(-1)].reshape(self.M, self.K, 3).to(self.compute_dtype)
#         P_vac_all = self.P_vac.unsqueeze(1).to(self.compute_dtype)
#         diff_k = P_cu_topk - P_vac_all
#         diff_k = diff_k - torch.round(diff_k / self.box.to(self.compute_dtype)) * self.box.to(self.compute_dtype)
#         diff_k[~mask_valid.unsqueeze(-1).expand_as(diff_k)] = 0.0
#         dist_k = dist_k.clone()
#         dist_k[~mask_valid] = float("inf")
#         return {"vid_list": list(range(self.M)), "diff_k": diff_k.to(self.compute_dtype), "dist_k": dist_k.to(self.compute_dtype)}

#     # ---------------- verification ----------------
#     def verify_update(self, updated_cu: Dict[int, np.ndarray] = None, updated_vacancy: Dict[int, np.ndarray] = None):
#         """
#         Verify incremental update result vs full recompute on CPU float32 copy.
#         """
#         cpu_sys = AdaptiveVacancyTopK(
#             self.P_cu.cpu().float().numpy().astype(np.float32),
#             self.P_vac.cpu().float().numpy().astype(np.float32),
#             self.K,
#             tuple(self.box.cpu().numpy().astype(np.float32).tolist()),
#             self.cell_size,
#             device="cpu",
#             storage_dtype="float32",
#             max_extra_ring=self.max_extra_ring,
#         )
#         result_update = self.update_system(updated_cu, updated_vacancy)
#         vid_list = result_update["vid_list"]
#         diff_k_update = result_update["diff_k"]
#         dist_k_update = result_update["dist_k"]
#         cpu_sys._initialize_topk_sparse()
#         result_full = cpu_sys.get_all_topk_tensors()
#         if not vid_list:
#             print("✅ 验证通过: 无受影响 vacancy")
#             return True
#         vids_np = np.array(vid_list, dtype=np.int64)
#         diff_k_full = result_full["diff_k"][vids_np].to(self.device)
#         dist_k_full = result_full["dist_k"][vids_np].to(self.device)
#         diff_ok = torch.allclose(diff_k_update.to(self.device), diff_k_full, atol=1e-5, equal_nan=True)
#         dist_ok = torch.allclose(dist_k_update.to(self.device), dist_k_full, atol=1e-5, equal_nan=True)
#         if diff_ok and dist_ok:
#             print("✅ 验证通过")
#             return True
#         else:
#             print("❌ 验证失败")
#             for i, vid in enumerate(vid_list):
#                 if not torch.allclose(dist_k_update[i].to(self.device), dist_k_full[i], atol=1e-5, equal_nan=True):
#                     m = torch.max(torch.abs(dist_k_update[i].to(self.device) - dist_k_full[i])).item()
#                     print(f"  Vacancy {vid} dist_k 不一致. Max diff: {m}")
#                 if not torch.allclose(diff_k_update[i].to(self.device), diff_k_full[i], atol=1e-5, equal_nan=True):
#                     m = torch.max(torch.abs(diff_k_update[i].to(self.device) - diff_k_full[i])).item()
#                     print(f"  Vacancy {vid} diff_k 不一致. Max diff: {m}")
#             return False

#     # ---------------- debug ----------------
#     def debug_stats(self):
#         print(f"Cells: {self.cell_dim} total {self.ncells}, N={self.N}, M={self.M}")
#         nonempty_cu = sum(1 for v in self.cell_cu.values() if v)
#         nonempty_vac = sum(1 for v in self.cell_vac.values() if v)
#         print(f"non-empty cu cells: {nonempty_cu}, vac cells: {nonempty_vac}")
#         print(f"storage_dtype={self.storage_dtype}, compute_dtype={self.compute_dtype}, max_extra_ring={self.max_extra_ring}")
#         # sample distribution
#         lens = [len(v) for v in self.cell_cu.values()]
#         if lens:
#             import statistics as _st
#             print(f"cu per non-empty cell: min={min(lens)}, median={int(_st.median(lens))}, max={max(lens)}")
#         else:
#             print("no cu in cells (weird)")

# # End of class


import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


class AdaptiveVacancyTopK:
    """
    高性能、自适应邻域搜索的 Top-K 系统（恢复为增量更新模式）。
    修正了 Cell 映射精度和验证逻辑。
    """

    def __init__(
        self,
        cu_positions: np.ndarray,
        vacancy_positions: np.ndarray,
        K: int,
        box: Tuple[float, float, float],
        cell_size: float,
        device: str = "cuda",
        storage_dtype: str = "float32",
        max_extra_ring: int = 2,
        approximate_mode: bool = False,
    ):
        self.device: torch.device = device if isinstance(device, torch.device) else torch.device(device)
        assert storage_dtype in ("bfloat16", "float16", "float32")
        self.storage_dtype = getattr(torch, storage_dtype)
        self.compute_dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self.output_dtype = torch.float32
        self.dist_store_dtype = torch.float32

        # data arrays ensure float32 numpy input
        cu_positions = np.asarray(cu_positions, dtype=np.float32)
        vacancy_positions = np.asarray(vacancy_positions, dtype=np.float32)
   
        assert cu_positions.ndim == 2 and cu_positions.shape[1] == 3
        assert vacancy_positions.ndim == 2 and vacancy_positions.shape[1] == 3

        self.N = cu_positions.shape[0]
        self.M = vacancy_positions.shape[0]
        self.K = int(K)
        self.box = torch.tensor(box, dtype=self.compute_dtype, device="cpu")
        self.box_np = self.box.numpy().astype(np.float32)
        self.cell_size = float(cell_size)
        self.max_extra_ring = int(max_extra_ring)
        self.approximate_mode = bool(approximate_mode)
        ring_limit = int(self.max_extra_ring)
        if ring_limit < 1:
            ring_limit = 1
        search_max = float((ring_limit + 1.0) * self.cell_size * np.sqrt(3.0))
        pbc_max = float(np.linalg.norm(self.box_np * 0.5))
        self.fill_dist = float(min(search_max, pbc_max))
        self.fill_dist = float(min(self.fill_dist, np.finfo(np.float32).max))

        self.P_cu = torch.tensor(cu_positions, dtype=self.storage_dtype, device="cpu")
        self.P_vac = torch.tensor(vacancy_positions, dtype=self.storage_dtype, device="cpu")
        self._use_cuda_cache = str(self.device).startswith("cuda")
        if self._use_cuda_cache:
            dev_dtype = torch.float16
            self.P_cu_dev = self.P_cu.to(device=self.device, dtype=dev_dtype)
            self.P_vac_dev = self.P_vac.to(device=self.device, dtype=dev_dtype)
        max_delta = float(np.max(self.box_np) * 0.5) if self.box_np.size > 0 else 0.0
        max_d2 = float(3.0 * max_delta * max_delta)
        target_d2_fp16 = 60000.0
        if max_d2 > 0.0:
            self._dist_fp16_scale = float(min(1.0, np.sqrt(target_d2_fp16 / max_d2)))
        else:
            self._dist_fp16_scale = 1.0
        self.use_fp16_matmul_distance = bool(str(self.device).startswith("cuda"))

        # compute integer cell dims (nx,ny,nz) as ints
        cell_dim = np.ceil((self.box_np / self.cell_size)).astype(int)
        self.cell_dim = tuple(int(x) for x in cell_dim)  # (nx,ny,nz)
        self.ncells = self.cell_dim[0] * self.cell_dim[1] * self.cell_dim[2]

        # CPU-side cell -> lists (dynamic)
        self.cell_cu = defaultdict(list)   # cell_index -> list of cu ids (ints)
        self.cell_vac = defaultdict(list)  # cell_index -> list of vac ids (ints)
        # per-atom current cell (on CPU numpy int)
        self.cu_cell = np.zeros(self.N, dtype=np.int64)
        self.vac_cell = np.zeros(self.M, dtype=np.int64)

        # precompute cell geometry (CPU numpy) for AABB overlap tests
        # cell centers (ncells, 3) and half diagonal (scalar)
        self._build_cell_geometries()

        # build initial mappings and cells
        self._build_cell_mappings(cu_positions, vacancy_positions)
        self.cell_cu_np: Dict[int, np.ndarray] = {}
        for k, lst in self.cell_cu.items():
            if lst:
                self.cell_cu_np[int(k)] = np.asarray(lst, dtype=np.int64)

        self._cell_version = np.zeros(int(self.ncells), dtype=np.uint32)
        self._cand_cache: Dict[Tuple[int, int], Tuple[np.ndarray, torch.Tensor]] = {}
        self._cand_cache_max_entries = int(getattr(self, "cand_cache_max_entries", 4096))
        w = []
        x = 0x9E3779B97F4A7C15
        for _ in range(32):
            x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            z = x
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
            z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
            z = z ^ (z >> 31)
            w.append(z & 0xFFFFFFFFFFFFFFFF)
        self._cand_cache_hash_weights = np.asarray(w, dtype=np.uint64)

        self._cu_sort_version = 0
        self._cu_sort_ready_version = -1
        self._cu_morton_sorted = torch.empty((0,), dtype=torch.int64, device="cpu")
        self._cu_aids_sorted = torch.empty((0,), dtype=torch.long, device="cpu")

        # try:
        #     print(f"[topk] init M={int(self.M)} N={int(self.N)} K={int(self.K)} box={tuple(map(float, self.box_np.tolist()))} cell_size={float(self.cell_size)} device={str(self.device)}", flush=True)
        # except Exception:
        #     pass

        # top-k storage: indices (long) -1 padded, distances float32
        self.topk_indices = torch.full((self.M, self.K), -1, dtype=torch.long, device="cpu")
        self.topk_dists = torch.full((self.M, self.K), self.fill_dist, dtype=self.dist_store_dtype, device="cpu")
        
        self._initialize_topk_sparse()


    # ---------------- helpers: cell geometry & indexing ----------------
    def _build_cell_geometries(self):
        """
        Build CPU numpy arrays:
          - self.cell_centers: shape (ncells,3)
          - self.cell_half_diag: scalar half-diagonal for cubic cell (same for all)
        and also cell (cx,cy,cz) coordinate arrays for mapping linear indices.
        """
        nx, ny, nz = self.cell_dim
        # compute cell centers by coordinates
        centers = []
        for cx in range(nx):
            for cy in range(ny):
                for cz in range(nz):
                    center = np.array(
                        [(cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size, (cz + 0.5) * self.cell_size],
                        dtype=np.float32,
                    )
                    centers.append(center)
        self.cell_centers = np.stack(centers, axis=0)  # shape (ncells,3)
        # half diagonal (cube of side cell_size): half_diag = sqrt(3*(cell_size/2)^2)
        self.cell_half_diag = np.sqrt(3.0 * (self.cell_size * 0.5) ** 2).astype(np.float32)

        # store linear->(cx,cy,cz) for convenience
        linear_to_coords = np.zeros((self.ncells, 3), dtype=np.int32)
        idx = 0
        for cx in range(nx):
            for cy in range(ny):
                for cz in range(nz):
                    linear_to_coords[idx, 0] = cx
                    linear_to_coords[idx, 1] = cy
                    linear_to_coords[idx, 2] = cz
                    idx += 1
        self.linear_to_coords = linear_to_coords
        nx, ny, nz = self.cell_dim
        neighbors = [[] for _ in range(self.ncells)]
        for idx in range(self.ncells):
            cx0, cy0, cz0 = linear_to_coords[idx]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        nx_ = (int(cx0) + dx) % nx
                        ny_ = (int(cy0) + dy) % ny
                        nz_ = (int(cz0) + dz) % nz
                        linear = int(nx_) * (ny * nz) + int(ny_) * nz + int(nz_)
                        neighbors[idx].append(linear)
            neighbors[idx] = sorted(list(set(neighbors[idx])), key=int)
        self.neighbor_cells_ring1 = neighbors

    def _hash_cell_versions(self, cells: List[int]) -> int:
        if not cells:
            return 0
        idx = np.asarray(cells, dtype=np.int64)
        v = self._cell_version[idx].astype(np.uint64, copy=False)
        w = self._cand_cache_hash_weights[: int(v.shape[0])]
        h = np.bitwise_xor.reduce((v + np.uint64(1)) * w)
        return int(h)

    def _pos_to_cell_index_np(self, pos_np: np.ndarray) -> np.ndarray:
        """
        Vectorized mapping numpy positions (N,3) -> linear cell indices (N,)
        Positions are expected in [0, box) or not; we apply modulo with box.
        --- 修正: 增加 epsilon 避免浮点精度导致的边界划分错误 ---
        """
        epsilon = 1e-7 # 解决边界精度问题
        pos_mod = np.remainder(pos_np.astype(np.float32), self.box_np)
        
        # 核心修正：添加 epsilon 避免 np.floor 结果不一致
        idx = np.floor(pos_mod / self.cell_size + epsilon).astype(np.int64)
        
        # 确保索引在合法范围内（即使添加 epsilon，边界也应保持）
        idx[:, 0] = np.remainder(idx[:, 0], self.cell_dim[0])
        idx[:, 1] = np.remainder(idx[:, 1], self.cell_dim[1])
        idx[:, 2] = np.remainder(idx[:, 2], self.cell_dim[2])
        linear = idx[:, 0] * (self.cell_dim[1] * self.cell_dim[2]) + idx[:, 1] * self.cell_dim[2] + idx[:, 2]
        return linear.astype(np.int64)

    def _pos_to_cell_index_tensor(self, pos_t: torch.Tensor) -> torch.Tensor:
        """
        Vectorized mapping on torch Tensor (device aware). Returns long tensor.
        --- 修正: 增加 epsilon 避免浮点精度导致的边界划分错误 ---
        """
        epsilon = 1e-7
        p = pos_t.to(torch.float32)
        box = self.box.to(device=pos_t.device, dtype=torch.float32)
        pos_mod = torch.remainder(p, box)
        
        # 核心修正：添加 epsilon
        idx = torch.floor(pos_mod / self.cell_size + epsilon).to(dtype=torch.long)
        
        cd = torch.tensor(self.cell_dim, device=pos_t.device, dtype=torch.long)
        idx = torch.remainder(idx, cd)
        linear = idx[:, 0] * (cd[1] * cd[2]) + idx[:, 1] * cd[2] + idx[:, 2]
        return linear

    # ---------------- build mappings ----------------
    def _build_cell_mappings(self, cu_positions_np: np.ndarray, vac_positions_np: np.ndarray):
        """
        Build CPU-side cell_cu and cell_vac mappings (lists). Also fill cu_cell and vac_cell arrays.
        cu_positions_np, vac_positions_np are numpy float32 arrays.
        """
        self.cell_cu.clear()
        self.cell_vac.clear()
        # compute linear indices
        cu_cells = self._pos_to_cell_index_np(cu_positions_np)
        vac_cells = self._pos_to_cell_index_np(vac_positions_np)
        self.cu_cell = cu_cells.copy()
        self.vac_cell = vac_cells.copy()

        for aid, c in enumerate(cu_cells):
            self.cell_cu[int(c)].append(int(aid))
        for vid, c in enumerate(vac_cells):
            self.cell_vac[int(c)].append(int(vid))
        off = np.zeros(self.ncells + 1, dtype=np.int64)
        for i in range(self.ncells):
            off[i + 1] = off[i] + len(self.cell_cu.get(int(i), []))
        flat = np.empty(int(off[-1]), dtype=np.int64)
        p = 0
        for i in range(self.ncells):
            lst = self.cell_cu.get(int(i), [])
            if lst:
                n = len(lst)
                flat[p:p + n] = np.asarray(lst, dtype=np.int64)
                p += n
        self.cell_cu_offset = off
        self.cell_cu_flat = flat

    # ---------------- AABB vs sphere (vectorized CPU) ----------------
    def _cells_overlapping_sphere(self, pos_np: np.ndarray, radius: float, exclude_set: Optional[Set[int]] = None) -> np.ndarray:
        """ (保持不变) """
        # pos_np shape (3,)
        # quickly compute center distances
        diff = np.abs(self.cell_centers - pos_np.reshape(1, 3)).astype(np.float32)
        # half-extent per axis
        half_extent = self.cell_size * 0.5
        d_axis = np.maximum(0.0, diff - half_extent)
        d2 = np.sum(d_axis * d_axis, axis=1)  # (ncells,)
        mask = d2 <= (radius * radius)
        if exclude_set:
            # mask out excluded cells
            ex = np.fromiter(exclude_set, dtype=np.int64)
            if ex.size > 0:
                mask[ex] = False
        return np.nonzero(mask)[0].astype(np.int64)

    def _cells_overlapping_sphere_ring(self, pos_np: np.ndarray, radius: float, center_cell: int, ring: int, exclude_set: Optional[Set[int]] = None) -> np.ndarray:
        ring = int(ring)
        if radius <= 0.0 or ring <= 0:
            return np.empty((0,), dtype=np.int64)
        nx, ny, nz = self.cell_dim
        cx0, cy0, cz0 = self.linear_to_coords[int(center_cell)]
        cx_min = max(int(cx0) - ring, 0)
        cx_max = min(int(cx0) + ring, int(nx) - 1)
        cy_min = max(int(cy0) - ring, 0)
        cy_max = min(int(cy0) + ring, int(ny) - 1)
        cz_min = max(int(cz0) - ring, 0)
        cz_max = min(int(cz0) + ring, int(nz) - 1)

        cx = np.arange(cx_min, cx_max + 1, dtype=np.int64)
        cy = np.arange(cy_min, cy_max + 1, dtype=np.int64)
        cz = np.arange(cz_min, cz_max + 1, dtype=np.int64)
        if cx.size == 0 or cy.size == 0 or cz.size == 0:
            return np.empty((0,), dtype=np.int64)

        Cx, Cy, Cz = np.meshgrid(cx, cy, cz, indexing='ij')
        linear = (Cx * (int(ny) * int(nz)) + Cy * int(nz) + Cz).reshape(-1)
        if linear.size == 0:
            return np.empty((0,), dtype=np.int64)

        if exclude_set:
            ex = np.fromiter(exclude_set, dtype=np.int64)
            if ex.size > 0:
                keep = ~np.isin(linear, ex)
                linear = linear[keep]
                if linear.size == 0:
                    return np.empty((0,), dtype=np.int64)

        centers = self.cell_centers[linear]
        diff = np.abs(centers - pos_np.reshape(1, 3)).astype(np.float32)
        half_extent = float(self.cell_size) * 0.5
        d_axis = np.maximum(0.0, diff - half_extent)
        d2 = np.sum(d_axis * d_axis, axis=1)
        r2 = float(radius) * float(radius)
        mask = d2 <= r2
        if not np.any(mask):
            return np.empty((0,), dtype=np.int64)
        return linear[mask].astype(np.int64, copy=False)

    # ---------------- PBC distance compute (GPU) ----------------
    def _compute_pbc_sq_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        p1 = pos1.to(self.compute_dtype)
        p2 = pos2.to(self.compute_dtype)
        box = self.box.to(device=pos1.device, dtype=self.compute_dtype)
        delta = p1.unsqueeze(1) - p2.unsqueeze(0)
        delta = delta - torch.round(delta / box) * box
        delta32 = delta.to(dtype=torch.float32)
        return torch.sum(delta32 * delta32, dim=-1)

    def _compute_pbc_sq_distance_matmul_1vN(self, vpos: torch.Tensor, cand_pos: torch.Tensor) -> torch.Tensor:
        v = vpos.reshape(1, 3)
        device = cand_pos.device
        box32 = self.box.to(device=device, dtype=torch.float32).reshape(1, 3)
        v32 = v.to(dtype=torch.float32, device=device)
        p32 = cand_pos.to(dtype=torch.float32)
        delta = v32 - p32
        corr = torch.round(delta / box32) * box32
        p_shift = (p32 + corr).to(dtype=cand_pos.dtype)
        v16 = v.to(dtype=cand_pos.dtype, device=device).reshape(3)

        scale = float(getattr(self, "_dist_fp16_scale", 1.0)) if p_shift.dtype in (torch.float16, torch.bfloat16) else 1.0
        if scale != 1.0:
            p_shift = p_shift * scale
            v16 = v16 * scale

        dot = torch.matmul(p_shift, v16.reshape(3, 1)).reshape(-1)
        p2 = torch.sum(p_shift.to(dtype=torch.float32) * p_shift.to(dtype=torch.float32), dim=1)
        v2 = torch.sum(v16.to(dtype=torch.float32) * v16.to(dtype=torch.float32))
        d2 = p2 + v2 - 2.0 * dot.to(dtype=torch.float32)
        if scale != 1.0:
            d2 = d2 / (scale * scale)
        return torch.clamp(d2, min=0.0)

    def _compute_pbc_sq_distance_matmul_MvN(self, vpos: torch.Tensor, cand_pos: torch.Tensor) -> torch.Tensor:
        v = vpos.reshape(-1, 3)
        device = cand_pos.device
        box32 = self.box.to(device=device, dtype=torch.float32).reshape(1, 1, 3)
        v32 = v.to(dtype=torch.float32, device=device).unsqueeze(1)
        p32 = cand_pos.to(dtype=torch.float32).unsqueeze(0)
        delta = v32 - p32
        corr = torch.round(delta / box32) * box32
        p_shift = (p32 + corr).to(dtype=cand_pos.dtype)
        v16 = v.to(dtype=cand_pos.dtype, device=device).unsqueeze(1)

        scale = float(getattr(self, "_dist_fp16_scale", 1.0)) if p_shift.dtype in (torch.float16, torch.bfloat16) else 1.0
        if scale != 1.0:
            p_shift = p_shift * scale
            v16 = v16 * scale

        dot = torch.bmm(p_shift, v16.transpose(1, 2)).squeeze(-1)
        p_shift32 = p_shift.to(dtype=torch.float32)
        p2 = torch.sum(p_shift32 * p_shift32, dim=-1)
        v16_32 = v16.squeeze(1).to(dtype=torch.float32)
        v2 = torch.sum(v16_32 * v16_32, dim=-1).unsqueeze(1)
        d2 = p2 + v2 - 2.0 * dot.to(dtype=torch.float32)
        if scale != 1.0:
            d2 = d2 / (scale * scale)
        return torch.clamp(d2, min=0.0)

    def _build_gpu_cell_list(self):
        """
        Build a dense GPU cell list for Cu atoms.
        Returns:
            grid: (ncells + 1, max_atoms_per_cell) tensor of atom indices, padded with -1.
                  The last row (index ncells) is the "invalid" cell, always -1.
        """
        # 1. Compute cell indices for all Cu atoms on GPU
        cu_cells = self._pos_to_cell_index_tensor(self.P_cu)
        
        # 2. Sort atoms by cell index
        sorted_idx = torch.argsort(cu_cells)
        sorted_cells = cu_cells[sorted_idx]
        
        # 3. Compute counts per cell
        unique_cells, counts = torch.unique_consecutive(sorted_cells, return_counts=True)
        
        if counts.numel() == 0:
            return torch.full((self.ncells + 1, 0), -1, dtype=torch.long, device=self.device)
            
        max_atoms = counts.max().item()
        
        # 4. Create dense grid
        # Pad with one extra row for "invalid" cell (index = self.ncells)
        grid = torch.full((self.ncells + 1, max_atoms), -1, dtype=torch.long, device=self.device)
        
        # 5. Fill grid
        # We need local_rank for each atom in its cell
        
        cell_starts = torch.zeros(self.ncells, dtype=torch.long, device=self.device)
        cumsum_counts = torch.cumsum(counts, 0)
        starts_valid = torch.cat([torch.tensor([0], device=self.device), cumsum_counts[:-1]])
        cell_starts[unique_cells] = starts_valid
        
        # For each atom in sorted_cells, its local rank is global_rank - cell_start
        global_rank = torch.arange(sorted_idx.shape[0], device=self.device)
        local_rank = global_rank - cell_starts[sorted_cells]
        
        grid[sorted_cells, local_rank] = sorted_idx
        
        return grid

    def _get_neighbor_cells_gpu(self, center_cells: torch.Tensor, ring: int = 1) -> torch.Tensor:
        """
        Vectorized neighbor cell gathering on GPU with PBC wrap.
        center_cells: (M,) linear cell indices
        Returns: (M, (2*ring+1)^3) neighbor linear cell indices (wrapped by modulo).
        """
        nx, ny, nz = self.cell_dim
        cz = center_cells % nz
        cy = (center_cells // nz) % ny
        cx = center_cells // (ny * nz)
        center_coords = torch.stack([cx, cy, cz], dim=1)
        r_range = torch.arange(-ring, ring + 1, device=self.device)
        offsets = torch.meshgrid(r_range, r_range, r_range, indexing='ij')
        offsets = torch.stack(offsets, dim=-1).reshape(-1, 3)
        neigh_coords = center_coords.unsqueeze(1) + offsets.unsqueeze(0)
        # wrap by modulo for PBC
        neigh_coords[..., 0] = torch.remainder(neigh_coords[..., 0], nx)
        neigh_coords[..., 1] = torch.remainder(neigh_coords[..., 1], ny)
        neigh_coords[..., 2] = torch.remainder(neigh_coords[..., 2], nz)
        return neigh_coords[:, :, 0] * (ny * nz) + neigh_coords[:, :, 1] * nz + neigh_coords[:, :, 2]

    def _invalidate_cu_sorted_index(self):
        self._cu_sort_version = int(self._cu_sort_version) + 1

    def _morton3d(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.int64)
        y = y.to(dtype=torch.int64)
        z = z.to(dtype=torch.int64)

        def part1by2(n: torch.Tensor) -> torch.Tensor:
            n = n & 0x1FFFFF
            n = (n | (n << 32)) & 0x1F00000000FFFF
            n = (n | (n << 16)) & 0x1F0000FF0000FF
            n = (n | (n << 8)) & 0x100F00F00F00F00F
            n = (n | (n << 4)) & 0x10C30C30C30C30C3
            n = (n | (n << 2)) & 0x1249249249249249
            return n

        return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)

    def _ensure_cu_sorted_morton_index(self, device: torch.device):
        if self._cu_sort_ready_version == self._cu_sort_version:
            if self._cu_morton_sorted.device == device and self._cu_aids_sorted.device == device and int(self._cu_morton_sorted.numel()) == int(self.N):
                return
        if getattr(self, "_use_cuda_cache", False):
            cu_pos = self.P_cu_dev.to(device=device)
        else:
            cu_pos = self.P_cu.to(device=device)
        cu_cells = self._pos_to_cell_index_tensor(cu_pos)
        nx, ny, nz = self.cell_dim
        cz = torch.remainder(cu_cells, int(nz)).to(dtype=torch.int64)
        cy = torch.remainder(cu_cells // int(nz), int(ny)).to(dtype=torch.int64)
        cx = (cu_cells // int(ny * nz)).to(dtype=torch.int64)
        morton = self._morton3d(cx, cy, cz)
        morton_sorted, perm = torch.sort(morton)
        self._cu_morton_sorted = morton_sorted
        self._cu_aids_sorted = perm.to(dtype=torch.long)
        self._cu_sort_ready_version = self._cu_sort_version

    def _recalculate_topk_sparse_vids_searchsorted(self, vids: List[int]):
        if not vids or self.N <= 0:
            return
        device: torch.device = self.device
        self._ensure_cu_sorted_morton_index(device=device)
        morton_sorted = self._cu_morton_sorted
        aids_sorted = self._cu_aids_sorted

        W = int(getattr(self, "search_width", 2048))
        if W < 1:
            W = 1
        if W > int(self.N):
            W = int(self.N)
        K = int(self.K)
        if K < 1:
            return

        batch = int(getattr(self, "recalc_sorted_batch", 4096))
        if batch < 256:
            batch = 256

        use_cuda_cache = bool(getattr(self, "_use_cuda_cache", False))
        box32 = self.box.to(device=device, dtype=torch.float32).reshape(1, 1, 3)
        offsets = torch.arange(W, device=device, dtype=torch.long).reshape(1, -1)

        vids = [int(v) for v in vids]
        with torch.no_grad():
            for start in range(0, len(vids), batch):
                end = min(start + batch, len(vids))
                vids_chunk = vids[start:end]
                vids_t_cpu = torch.tensor(vids_chunk, dtype=torch.long, device="cpu")
                if use_cuda_cache:
                    vids_t = vids_t_cpu.to(device=device)
                    vpos = self.P_vac_dev[vids_t]
                    cu_pos = self.P_cu_dev
                else:
                    vpos = self.P_vac[vids_t_cpu].to(device=device)
                    cu_pos = self.P_cu.to(device=device)

                vac_cells = self._pos_to_cell_index_tensor(vpos)
                nx, ny, nz = self.cell_dim
                cz = torch.remainder(vac_cells, int(nz)).to(dtype=torch.int64)
                cy = torch.remainder(vac_cells // int(nz), int(ny)).to(dtype=torch.int64)
                cx = (vac_cells // int(ny * nz)).to(dtype=torch.int64)
                morton_v = self._morton3d(cx, cy, cz)

                idx_in_sorted = torch.searchsorted(morton_sorted, morton_v).to(dtype=torch.long)
                if int(self.N) == W:
                    start_idx = torch.zeros_like(idx_in_sorted)
                else:
                    start_idx = (idx_in_sorted - (W // 2)).clamp(0, int(self.N) - W)
                window_idx = start_idx.unsqueeze(1) + offsets
                cand_aids = aids_sorted[window_idx]
                cand_pos = cu_pos[cand_aids]

                v32 = vpos.to(dtype=torch.float32, device=device).unsqueeze(1)
                p32 = cand_pos.to(dtype=torch.float32, device=device)
                delta = v32 - p32
                delta = delta - torch.round(delta / box32) * box32
                d2 = torch.sum(delta * delta, dim=-1)

                k_eff = min(K, W)
                d2_topk, pos_topk = torch.topk(d2, k=k_eff, largest=False, dim=1)
                idx_topk = cand_aids.gather(1, pos_topk)
                dist = torch.sqrt(torch.clamp(d2_topk, min=0.0)).to(dtype=self.dist_store_dtype)

                self.topk_indices[vids_t_cpu, :] = -1
                self.topk_dists[vids_t_cpu, :] = self.fill_dist
                self.topk_indices[vids_t_cpu, :k_eff] = idx_topk.to(device="cpu", dtype=torch.long)
                self.topk_dists[vids_t_cpu, :k_eff] = dist.to(device="cpu")

                del vids_t_cpu, vpos, cu_pos, vac_cells, morton_v, idx_in_sorted, start_idx, window_idx, cand_aids, cand_pos, v32, p32, delta, d2, d2_topk, pos_topk, idx_topk, dist

    def _initialize_topk_sparse(self):
        """
        Initialize topk for all vacancies using Adaptive expansion.
        Heavy distance work vectorized on GPU for all vacancies simultaneously.
        No PBC for distance or neighbor search.
        Batched to avoid OOM.
        """
        if self.M == 0 or self.N == 0:
            return

        self.topk_indices.fill_(-1)
        self.topk_dists.fill_(float("inf"))
        if self.M == 0:
            return
        batch_size = 65536
        for start in range(0, self.M, batch_size):
            end = min(start + batch_size, self.M)
            self._recalculate_topk_sparse_vids(list(range(start, end)))
            # if (end == self.M) or ((start // batch_size + 1) % 10 == 0):
            #     print(f"Initialized topk ({end}/{self.M})", flush=True)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


    def _gather_ring_cells(self, linear_cell: int, ring: int = 1) -> List[int]:
        if ring == 1:
            return list(self.neighbor_cells_ring1[int(linear_cell)])
        if not hasattr(self, '_ring_cells_cache'):
            self._ring_cells_cache = {}
        key = (int(linear_cell), int(ring))
        cached = self._ring_cells_cache.get(key)
        if cached is not None:
            return cached
        nx, ny, nz = self.cell_dim
        cx0, cy0, cz0 = self.linear_to_coords[int(linear_cell)]
        cells = []
        for dx in range(-ring, ring + 1):
            for dy in range(-ring, ring + 1):
                for dz in range(-ring, ring + 1):
                    nx_ = (int(cx0) + dx) % nx
                    ny_ = (int(cy0) + dy) % ny
                    nz_ = (int(cz0) + dz) % nz
                    linear = int(nx_) * (ny * nz) + int(ny_) * nz + int(nz_)
                    cells.append(int(linear))
        res = sorted(list(set(cells)), key=int)
        self._ring_cells_cache[key] = res
        return res

    # ---------------- incremental updates ----------------
    def update_vacancy(self, updated_vac: Dict[int, np.ndarray]) -> Set[int]:
        """
        支持两种输入格式：
        - {vid: new_pos}
        - {vid: (old_pos, new_pos)} 或 {vid: {"old": old_pos, "new": new_pos}}
        只需重算这些 vid 的 Top-K；其他 vacancy 不受影响。
        """
        if not updated_vac:
            return set()
        vids = sorted(updated_vac.keys())
        new_list = []
        old_list = []
        for v in vids:
            val = updated_vac[v]
            if isinstance(val, dict):
                npos = np.asarray(val.get("new", val.get("old", None)), dtype=np.float32).reshape(1, 3)
                opos = np.asarray(val.get("old", None), dtype=np.float32).reshape(1, 3) if (val.get("old", None) is not None) else None
            else:
                arr = np.asarray(val, dtype=np.float32)
                if arr.shape == (2, 3):
                    opos = arr[0:1]
                    npos = arr[1:2]
                else:
                    opos = None
                    npos = arr.reshape(1, 3)
            new_list.append(npos)
            old_list.append(opos)
        new_pos_np = np.vstack(new_list)
        new_pos_t = torch.tensor(new_pos_np, dtype=self.storage_dtype, device="cpu")
        vids_t = torch.tensor(vids, dtype=torch.long, device="cpu")
        self.P_vac[vids_t] = new_pos_t
        if getattr(self, "_use_cuda_cache", False):
            vids_t_dev = vids_t.to(device=self.device)
            self.P_vac_dev[vids_t_dev] = new_pos_t.to(device=self.device, dtype=self.P_vac_dev.dtype)
        # 更新 cell 映射
        new_vac_cells = self._pos_to_cell_index_np(new_pos_np)
        for i, vid in enumerate(vids):
            oldc = int(self.vac_cell[vid])
            newc = int(new_vac_cells[i])
            if old_list[i] is not None:
                oldc = int(self._pos_to_cell_index_np(old_list[i])[0])
            if oldc != newc:
                try:
                    self.cell_vac[oldc].remove(int(vid))
                except ValueError:
                    pass
                self.cell_vac[newc].append(int(vid))
                self.vac_cell[vid] = newc
        # 重算这些空位的 Top-K
        self._recalculate_topk_sparse_vids(vids)
        return set(vids)

    def update_cu(self, updated_cu: Dict[int, np.ndarray]) -> Set[int]:
        """
        支持两种输入格式：
        - {aid: new_pos}
        - {aid: (old_pos, new_pos)} 或 {aid: {"old": old_pos, "new": new_pos}}
        返回受影响的空位集合，并重算其 Top-K。
        """
        if not updated_cu:
            return set()
        aids = sorted(updated_cu.keys())
        new_list = []
        old_list = []
        for a in aids:
            val = updated_cu[a]
            if isinstance(val, dict):
                npos = np.asarray(val.get("new", val.get("old", None)), dtype=np.float32).reshape(1, 3)
                opos = np.asarray(val.get("old", None), dtype=np.float32).reshape(1, 3) if (val.get("old", None) is not None) else None
            else:
                arr = np.asarray(val, dtype=np.float32)
                if arr.shape == (2, 3):
                    opos = arr[0:1]
                    npos = arr[1:2]
                else:
                    opos = None
                    npos = arr.reshape(1, 3)
            new_list.append(npos)
            old_list.append(opos)
        new_pos_np = np.vstack(new_list)
        new_pos_t = torch.tensor(new_pos_np, dtype=self.storage_dtype, device="cpu")
        aids_t = torch.tensor(aids, dtype=torch.long, device="cpu")
        self.P_cu[aids_t] = new_pos_t
        if getattr(self, "_use_cuda_cache", False):
            aids_t_dev = aids_t.to(device=self.device)
            self.P_cu_dev[aids_t_dev] = new_pos_t.to(device=self.device, dtype=self.P_cu_dev.dtype)
        self._invalidate_cu_sorted_index()

        # 更新 cell 映射；收集 old/new 两侧影响的空位
        new_cu_cells = self._pos_to_cell_index_np(new_pos_np)
        affected_vacancies = set()
        for i, aid in enumerate(aids):
            oldc = int(self.cu_cell[aid])
            newc = int(new_cu_cells[i])
            if old_list[i] is not None:
                oldc = int(self._pos_to_cell_index_np(old_list[i])[0])
            self._cell_version[int(oldc)] = np.uint32(self._cell_version[int(oldc)] + np.uint32(1))
            self._cell_version[int(newc)] = np.uint32(self._cell_version[int(newc)] + np.uint32(1))
            if oldc != newc:
                try:
                    self.cell_cu[oldc].remove(int(aid))
                except ValueError:
                    pass
                self.cell_cu[newc].append(int(aid))
                self.cu_cell[aid] = newc
                lst_old = self.cell_cu.get(int(oldc), [])
                if lst_old:
                    self.cell_cu_np[int(oldc)] = np.asarray(lst_old, dtype=np.int64)
                else:
                    self.cell_cu_np.pop(int(oldc), None)
                lst_new = self.cell_cu.get(int(newc), [])
                if lst_new:
                    self.cell_cu_np[int(newc)] = np.asarray(lst_new, dtype=np.int64)
                else:
                    self.cell_cu_np.pop(int(newc), None)

            # 标记新旧两侧 1-ring 的空位
            for c in (oldc, newc):
                nx, ny, nz = self.cell_dim
                cx, cy, cz = self.linear_to_coords[c]
                ring_r = int(getattr(self, "max_extra_ring", 1))
                if ring_r < 1:
                    ring_r = 1
                for dx in range(-ring_r, ring_r + 1):
                    for dy in range(-ring_r, ring_r + 1):
                        for dz in range(-ring_r, ring_r + 1):
                            nx_ = (int(cx) + dx) % int(nx)
                            ny_ = (int(cy) + dy) % int(ny)
                            nz_ = (int(cz) + dz) % int(nz)
                            linear = int(nx_) * (int(ny) * int(nz)) + int(ny_) * int(nz) + int(nz_)
                            for vid in self.cell_vac.get(int(linear), []):
                                affected_vacancies.add(int(vid))

            # 任何当前 Top-K 中包含该 Cu 的空位也需要重算
            if int(self.M) * int(self.K) <= 2000000:
                mask = (self.topk_indices == int(aid))
                if mask.any():
                    vids_hit = torch.nonzero(mask.any(dim=1), as_tuple=False).view(-1).cpu().numpy().astype(int).tolist()
                    for vid in vids_hit:
                        affected_vacancies.add(int(vid))

        if affected_vacancies:
            self._recalculate_topk_sparse_vids(sorted(list(affected_vacancies)))
        return affected_vacancies

    def _recalculate_topk_sparse_vids(self, vids: List[int]):
        """
        Recompute topk for listed vacancy ids using adaptive expansion.
        Uses GPU for distance computations in batches per vacancy (vectorized per candidate set).
        (逻辑与 _initialize_topk_sparse 相同，但针对特定 vids)
        """
        if not vids:
            return
        if bool(getattr(self, "sorted_chunk_search", False)) or bool(getattr(self, "approximate_mode", False)):
            self._recalculate_topk_sparse_vids_searchsorted(vids)
            return
        use_device: torch.device = self.device
        vec_dtype = torch.float16 if str(use_device).startswith("cuda") else self.compute_dtype
        use_cuda_cache = bool(getattr(self, "_use_cuda_cache", False))
        vids = [int(v) for v in vids]
        vids_cpu = torch.tensor(vids, dtype=torch.long, device="cpu")
        prev_idx_np = self.topk_indices[vids_cpu].to(device="cpu").numpy().astype(np.int64, copy=False)

        vids_by_cell: Dict[int, List[tuple]] = {}
        for i, vid in enumerate(vids):
            vids_by_cell.setdefault(int(self.vac_cell[vid]), []).append((i, int(vid)))

        neighbor_cells_cache: Dict[int, tuple] = {}
        chunk_size = int(getattr(self, "recalc_cand_chunk", 32768))
        if chunk_size < 2048:
            chunk_size = 2048

        with torch.no_grad():
            for vid_cell, items in vids_by_cell.items():
                cand_cells_cached = neighbor_cells_cache.get(vid_cell, None)
                if cand_cells_cached is None:
                    cand_cells_cached = tuple(self._gather_ring_cells(vid_cell, ring=1))
                    neighbor_cells_cache[vid_cell] = cand_cells_cached
                candidate_cells = list(cand_cells_cached)

                cand_parts = []
                for c in candidate_cells:
                    arr = self.cell_cu_np.get(int(c), None)
                    if arr is not None and arr.size > 0:
                        cand_parts.append(arr)
                prev_parts = []
                for i, _vid in items:
                    prev_row = prev_idx_np[i]
                    if prev_row.size > 0:
                        prev_row = prev_row[prev_row >= 0]
                        if prev_row.size > 0:
                            prev_parts.append(prev_row)
                if prev_parts:
                    cand_parts.append(np.unique(np.concatenate(prev_parts, axis=0)))

                if not cand_parts:
                    for _i, _vid in items:
                        self.topk_indices[_vid, :] = -1
                        self.topk_dists[_vid, :] = self.fill_dist
                    continue

                cand_all = np.concatenate(cand_parts, axis=0)
                if cand_all.size == 0:
                    for _i, _vid in items:
                        self.topk_indices[_vid, :] = -1
                        self.topk_dists[_vid, :] = self.fill_dist
                    continue

                cand_unique = np.unique(cand_all)
                C = int(cand_unique.shape[0])
                num_k = min(int(self.K), int(C))
                if num_k <= 0:
                    for _i, _vid in items:
                        self.topk_indices[_vid, :] = -1
                        self.topk_dists[_vid, :] = self.fill_dist
                    continue

                cand_idx_cpu = torch.from_numpy(cand_unique.astype(np.int64, copy=False))

                vids_group = [vid for _, vid in items]
                if use_cuda_cache:
                    vids_group_t = torch.tensor(vids_group, dtype=torch.long, device=use_device)
                    vpos_dev = self.P_vac_dev[vids_group_t].to(dtype=vec_dtype)
                else:
                    vids_group_t_cpu = torch.tensor(vids_group, dtype=torch.long, device="cpu")
                    vpos_dev = self.P_vac[vids_group_t_cpu].to(device=use_device, dtype=vec_dtype)

                best_d = None
                best_pos = None
                for start in range(0, C, chunk_size):
                    end = min(start + chunk_size, C)
                    cand_chunk_cpu = cand_idx_cpu[start:end]
                    if use_cuda_cache:
                        P_chunk = self.P_cu_dev[cand_chunk_cpu.to(device=use_device)]
                    else:
                        P_chunk = self.P_cu[cand_chunk_cpu].to(device=use_device, dtype=vec_dtype)
                    D_sq_chunk = self._compute_pbc_sq_distance_matmul_MvN(vpos_dev, P_chunk)
                    k_chunk = min(num_k, int(D_sq_chunk.shape[1]))
                    d_sq_chunk, pos_chunk = torch.topk(D_sq_chunk, k=k_chunk, largest=False, dim=1)
                    pos_chunk = pos_chunk + int(start)
                    if best_d is None:
                        best_d = d_sq_chunk
                        best_pos = pos_chunk
                    else:
                        d_cat = torch.cat([best_d, d_sq_chunk], dim=1)
                        pos_cat = torch.cat([best_pos, pos_chunk], dim=1)
                        best_d, sel = torch.topk(d_cat, k=num_k, largest=False, dim=1)
                        best_pos = pos_cat.gather(1, sel)
                        del d_cat, pos_cat, sel
                    del P_chunk, D_sq_chunk, d_sq_chunk, pos_chunk

                if best_pos is None or best_d is None:
                    for _i, _vid in items:
                        self.topk_indices[_vid, :] = -1
                        self.topk_dists[_vid, :] = self.fill_dist
                    del cand_idx_cpu, vpos_dev
                    continue

                best_pos_cpu = best_pos.to(device="cpu")
                d_sq_topk_cpu_all = best_d.to(device="cpu", dtype=torch.float32)
                idx_topk_cpu_all = cand_idx_cpu[best_pos_cpu]
                dist_topk_cpu_all = torch.sqrt(d_sq_topk_cpu_all).to(dtype=self.dist_store_dtype)

                if (not self.approximate_mode) and (num_k == self.K):
                    current_cand_cells_set = set(candidate_cells)
                    try:
                        vids_group_t_cpu = torch.tensor(vids_group, dtype=torch.long, device="cpu")
                        vpos_np_group = self.P_vac[vids_group_t_cpu].to(dtype=torch.float32).numpy()
                    except Exception:
                        vpos_np_group = None
                    if vpos_np_group is not None:
                        extra_cells_list = []
                        for row in range(len(items)):
                            try:
                                r_max = float(dist_topk_cpu_all[row].max().item())
                            except Exception:
                                r_max = float("inf")
                            if (not np.isfinite(r_max)) or r_max <= 0.0:
                                continue
                            extra_cells = self._cells_overlapping_sphere_ring(
                                vpos_np_group[row], r_max, center_cell=vid_cell, ring=self.max_extra_ring, exclude_set=current_cand_cells_set
                            )
                            if extra_cells.size > 0:
                                extra_cells_list.append(extra_cells)
                        if extra_cells_list:
                            extra_cells_union = np.unique(np.concatenate(extra_cells_list, axis=0))
                            extra_parts = []
                            for c in extra_cells_union.tolist():
                                arr = self.cell_cu_np.get(int(c), None)
                                if arr is not None and arr.size > 0:
                                    extra_parts.append(arr)
                            if extra_parts:
                                extra_all = np.concatenate(extra_parts, axis=0)
                                if extra_all.size > 0:
                                    extra_unique = np.unique(extra_all)
                                    new_only = np.setdiff1d(extra_unique, cand_unique, assume_unique=True)
                                    if new_only.size > 0:
                                        cand_new_cpu = torch.from_numpy(new_only.astype(np.int64, copy=False))
                                        Cn = int(cand_new_cpu.numel())
                                        k_new = min(int(self.K), int(Cn))
                                        if k_new > 0:
                                            best_new_d = None
                                            best_new_pos = None
                                            for start in range(0, Cn, chunk_size):
                                                end = min(start + chunk_size, Cn)
                                                cand_chunk_cpu = cand_new_cpu[start:end]
                                                if use_cuda_cache:
                                                    P_chunk = self.P_cu_dev[cand_chunk_cpu.to(device=use_device)]
                                                else:
                                                    P_chunk = self.P_cu[cand_chunk_cpu].to(device=use_device, dtype=vec_dtype)
                                                D_sq_chunk = self._compute_pbc_sq_distance_matmul_MvN(vpos_dev, P_chunk)
                                                k_chunk = min(k_new, int(D_sq_chunk.shape[1]))
                                                d_sq_chunk, pos_chunk = torch.topk(D_sq_chunk, k=k_chunk, largest=False, dim=1)
                                                pos_chunk = pos_chunk + int(start)
                                                if best_new_d is None:
                                                    best_new_d = d_sq_chunk
                                                    best_new_pos = pos_chunk
                                                else:
                                                    d_cat = torch.cat([best_new_d, d_sq_chunk], dim=1)
                                                    pos_cat = torch.cat([best_new_pos, pos_chunk], dim=1)
                                                    best_new_d, sel = torch.topk(d_cat, k=k_new, largest=False, dim=1)
                                                    best_new_pos = pos_cat.gather(1, sel)
                                                    del d_cat, pos_cat, sel
                                                del P_chunk, D_sq_chunk, d_sq_chunk, pos_chunk
                                            if best_new_pos is not None and best_new_d is not None:
                                                best_new_pos_cpu = best_new_pos.to(device="cpu")
                                                d_sq_new_cpu_all = best_new_d.to(device="cpu", dtype=torch.float32)
                                                idx_new_cpu_all = cand_new_cpu[best_new_pos_cpu]
                                                d_sq_cat = torch.cat([d_sq_topk_cpu_all, d_sq_new_cpu_all], dim=1)
                                                idx_cat = torch.cat([idx_topk_cpu_all, idx_new_cpu_all], dim=1)
                                                d_sq_final, sel_final = torch.topk(d_sq_cat, k=int(self.K), largest=False, dim=1)
                                                idx_final = idx_cat.gather(1, sel_final)
                                                idx_topk_cpu_all = idx_final
                                                dist_topk_cpu_all = torch.sqrt(d_sq_final).to(dtype=self.dist_store_dtype)
                                                d_sq_topk_cpu_all = d_sq_final
                                                del best_new_pos_cpu, d_sq_new_cpu_all, idx_new_cpu_all, d_sq_cat, idx_cat, d_sq_final, sel_final, idx_final
                                            del best_new_d, best_new_pos
                                        del cand_new_cpu
                            del extra_parts

                for row, (_i, vid) in enumerate(items):
                    self.topk_indices[vid, :num_k] = idx_topk_cpu_all[row]
                    self.topk_dists[vid, :num_k] = dist_topk_cpu_all[row]
                    if num_k < self.K:
                        self.topk_indices[vid, num_k:] = -1
                        self.topk_dists[vid, num_k:] = self.fill_dist

                    mask_pad = (self.topk_indices[vid] == -1)
                    if mask_pad.any():
                        self.topk_dists[vid][mask_pad] = self.fill_dist

                del cand_idx_cpu, best_d, best_pos, best_pos_cpu, d_sq_topk_cpu_all, idx_topk_cpu_all, dist_topk_cpu_all, vpos_dev

    # ---------------- query APIs ----------------
    def update_system(self, updated_cu: Optional[Dict[int, np.ndarray]] = None, updated_vacancy: Optional[Dict[int, np.ndarray]] = None):
        """ (保持不变) """
        affected = set()
        if updated_cu:
            affected |= self.update_cu(updated_cu)
        if updated_vacancy:
            affected |= self.update_vacancy(updated_vacancy)
        vid_list = sorted(list(affected))
        # print(f"affected vid_list: {vid_list}")
        if not vid_list:
            empty_diff = torch.empty((0, self.K, 3), dtype=self.output_dtype, device=self.device)
            empty_dist = torch.empty((0, self.K), dtype=self.output_dtype, device=self.device)
            return {"vid_list": [], "diff_k": empty_diff, "dist_k": empty_dist}

        vids_t_cpu = torch.tensor(vid_list, dtype=torch.long, device="cpu")
        indices_k_cpu = self.topk_indices[vids_t_cpu]  # (B,K)
        # print(f"vids_t: {vids_t}")
        # print(f"indices_k: {indices_k}")

        
        mask_valid_cpu = (indices_k_cpu >= 0)
        idx_safe = indices_k_cpu.clone()
        idx_safe[~mask_valid_cpu] = 0
        mask_valid = mask_valid_cpu.to(device=self.device)
        idx_safe_cpu_flat = idx_safe.reshape(-1)
        P_cu_topk = self.P_cu[idx_safe_cpu_flat].reshape(len(vid_list), self.K, 3).to(device=self.device, dtype=self.compute_dtype)
        P_vac_affected = self.P_vac[vids_t_cpu].unsqueeze(1).to(device=self.device, dtype=self.compute_dtype)
        diff_k = P_cu_topk - P_vac_affected
        box = self.box.to(device=self.device, dtype=self.compute_dtype)
        diff_k = diff_k - torch.round(diff_k / box) * box
        diff_k[~mask_valid.unsqueeze(-1).expand_as(diff_k)] = 0.0
        diff_k_fp32 = diff_k.to(dtype=torch.float32)
        dist_k = torch.sqrt(torch.clamp(torch.sum(diff_k_fp32 * diff_k_fp32, dim=-1), min=0.0))
        dist_k[~mask_valid] = float(self.fill_dist)
        return {"vid_list": vid_list, "diff_k": diff_k_fp32, "dist_k": dist_k.to(dtype=torch.float32)}

    def get_all_topk_tensors(self):
        """ (保持不变) """
        indices_k_cpu = self.topk_indices
        mask_valid_cpu = (indices_k_cpu >= 0)
        idx_safe_cpu = indices_k_cpu.clone()
        idx_safe_cpu[~mask_valid_cpu] = 0

        diff_k = torch.empty((self.M, self.K, 3), dtype=self.output_dtype, device=self.device)
        dist_k = torch.empty((self.M, self.K), dtype=self.output_dtype, device=self.device)
        box = self.box.to(device=self.device, dtype=self.compute_dtype)

        chunk = 4096
        for start in range(0, self.M, chunk):
            end = min(start + chunk, self.M)
            idx_safe_flat = idx_safe_cpu[start:end].reshape(-1)
            mask_valid = mask_valid_cpu[start:end].to(device=self.device)

            P_cu_topk = self.P_cu[idx_safe_flat].reshape(end - start, self.K, 3).to(device=self.device, dtype=self.compute_dtype)
            P_vac_all = self.P_vac[start:end].unsqueeze(1).to(device=self.device, dtype=self.compute_dtype)
            diff = P_cu_topk - P_vac_all
            diff = diff - torch.round(diff / box) * box
            diff[~mask_valid.unsqueeze(-1).expand_as(diff)] = 0.0
            diff_fp32 = diff.to(dtype=torch.float32)
            dist = torch.sqrt(torch.clamp(torch.sum(diff_fp32 * diff_fp32, dim=-1), min=0.0))
            dist[~mask_valid] = float(self.fill_dist)

            diff_k[start:end] = diff_fp32
            dist_k[start:end] = dist.to(dtype=torch.float32)

        return {"vid_list": list(range(self.M)), "diff_k": diff_k, "dist_k": dist_k}

    # ---------------- verification (修正) ----------------
    def verify_update(self, updated_cu: Optional[Dict[int, np.ndarray]] = None, updated_vacancy: Optional[Dict[int, np.ndarray]] = None):
        """
        修正验证逻辑：
        1. 复制主系统更新后的所有位置。
        2. 在 CPU 副本上从头构建 Cell 映射和 Top-K（全量计算）作为黄金标准。
        3. 比较增量更新的结果和全量计算的结果。
        """
        # 1. 运行增量更新 (主系统已更新位置、Cell 映射、并计算了受影响的 TopK)
        # 这会返回受影响的 vids 及其增量更新后的结果
        result_update = self.update_system(updated_cu, updated_vacancy)
        vid_list = result_update["vid_list"]
        
        if not vid_list:
            print("✅ 验证通过: 无受影响 vacancy")
            return True

        # 2. 复制主系统更新后的所有位置 (用于黄金标准)
        current_cu_pos_np = self.P_cu.float().numpy().astype(np.float32)
        current_vac_pos_np = self.P_vac.float().numpy().astype(np.float32)

        # 3. 在 CPU 上创建黄金标准系统并进行全量 Top-K 计算
        # 警告：这里 AdaptiveVacancyTopK 的初始化会自动调用 _initialize_topk_sparse
        cpu_sys = AdaptiveVacancyTopK(
            current_cu_pos_np,
            current_vac_pos_np,
            self.K,
            tuple(self.box.numpy().astype(np.float32).tolist()),
            self.cell_size,
            device="cpu", # 强制 CPU
            storage_dtype="float32",
            max_extra_ring=self.max_extra_ring,
        )
        # cpu_sys 现已完成基于最新位置的全量 Top-K 计算

        # 4. 获取黄金标准的全量结果
        result_full = cpu_sys.get_all_topk_tensors()
        
        # 5. 比较：只比较受影响的 vids 的结果
        vids_np = np.array(vid_list, dtype=np.int64)
        
        diff_k_update = result_update["diff_k"].to(self.device)
        dist_k_update = result_update["dist_k"].to(self.device)
        
        diff_k_full = result_full["diff_k"][vids_np].to(self.device)
        dist_k_full = result_full["dist_k"][vids_np].to(self.device)
        
        # 检查 NaN 差异
        has_nan_update = torch.isnan(dist_k_update).any() or torch.isnan(diff_k_update).any()
        has_nan_full = torch.isnan(dist_k_full).any() or torch.isnan(diff_k_full).any()

        if has_nan_update != has_nan_full:
             print(f"❌ 验证失败 (NaN 差异). 主系统包含 NaN: {has_nan_update}, 黄金标准包含 NaN: {has_nan_full}")
             return False

        # 检查数值差异
        diff_ok = torch.allclose(diff_k_update.to(self.device), diff_k_full, atol=1e-5, equal_nan=True)
        dist_ok = torch.allclose(dist_k_update.to(self.device), dist_k_full, atol=1e-5, equal_nan=True)
        
        if diff_ok and dist_ok:
            print("✅ 验证通过 (增量 vs 全量)")
            return True
        else:
            print("❌ 验证失败 (增量 vs 全量)")
            try:
                m_dist = torch.max(torch.abs(dist_k_update.to(self.device) - dist_k_full)).item()
                m_diff = torch.max(torch.abs(diff_k_update.to(self.device) - diff_k_full)).item()
            except RuntimeError: 
                m_dist = float('nan')
                m_diff = float('nan')

            print(f"  dist_k 不一致. Max diff: {m_dist}")
            print(f"  diff_k 不一致. Max diff: {m_diff}")
            return False

    # ---------------- debug ----------------
    def debug_stats(self):
        """ (保持不变) """
        print(f"Cells: {self.cell_dim} total {self.ncells}, N={self.N}, M={self.M}")
        nonempty_cu = sum(1 for v in self.cell_cu.values() if v)
        nonempty_vac = sum(1 for v in self.cell_vac.values() if v)
        print(f"non-empty cu cells: {nonempty_cu}, vac cells: {nonempty_vac}")
        print(f"storage_dtype={self.storage_dtype}, compute_dtype={self.compute_dtype}, max_extra_ring={self.max_extra_ring}")
        # sample distribution
        lens = [len(v) for v in self.cell_cu.values()]
        if lens:
            import statistics as _st
            print(f"cu per non-empty cell: min={min(lens)}, median={int(_st.median(lens))}, max={max(lens)}")
        else:
            print("no cu in cells (weird)")

# End of class
