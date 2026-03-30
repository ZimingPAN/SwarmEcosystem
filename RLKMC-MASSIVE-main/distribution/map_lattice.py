import numpy as np
import time
import itertools
import math
import os

# ==============================================================================
# Part 1: HPC环境映射与工具函数
# ==============================================================================
def get_gpu_grid_dim_by_rule(gpus_per_node: int):
    # if gpus_per_node==1: return (1,1,1)
    # elif gpus_per_node==2: return (2,1,1)
    # elif gpus_per_node==4: return (2,2,1)
    # elif gpus_per_node==8: return (2,2,2)
    if gpus_per_node==1: return (1,1,1)
    elif gpus_per_node==2: return (1,1,2)
    elif gpus_per_node==4: return (1,2,2)
    elif gpus_per_node==8: return (2,2,2)
    else: raise ValueError(f"不支持 {gpus_per_node} 卡/节点的固定规则分解。")

def calculate_hierarchical_dims(proc_grid_dim, num_nodes, gpus_per_node):
    gpu_grid_dim = get_gpu_grid_dim_by_rule(gpus_per_node)
    px,py,pz = proc_grid_dim; gx,gy,gz = gpu_grid_dim
    if not (px%gx==0 and py%gy==0 and pz%gz==0): raise ValueError(f"PROC_GRID_DIM {proc_grid_dim} 无法被固定的GPU网格规则 {gpu_grid_dim} 整除。")
    node_grid_dim = (px//gx, py//gy, pz//gz)
    if np.prod(node_grid_dim) != num_nodes: raise ValueError(f"资源不匹配：需要 {np.prod(node_grid_dim)} 节点, 但提供 {num_nodes} 节点。")
    return node_grid_dim, gpu_grid_dim

def map_rank_to_proc_id_locality_aware(rank, gpus_per_node, node_grid_dim, gpu_grid_dim):
    node_id=rank//gpus_per_node; local_rank=rank%gpus_per_node
    nx,ny,nz=node_grid_dim; node_x,node_y,node_z=(node_id//(ny*nz)),((node_id%(ny*nz))//nz),(node_id%nz)
    gx,gy,gz=gpu_grid_dim; gpu_x,gpu_y,gpu_z=(local_rank//(gy*gz)),((local_rank%(gy*gz))//gz),(local_rank%gz)
    # nx,ny,nz=node_grid_dim; node_x,node_y,node_z=(node_id%nx),((node_id%(ny*nx))//nx),(node_id//(nx*ny))
    # gx,gy,gz=gpu_grid_dim; gpu_x,gpu_y,gpu_z=(local_rank%gx),((local_rank%(gx*gy))//gx),(local_rank//(gx*gy))
    return (node_x*gx+gpu_x), (node_y*gy+gpu_y), (node_z*gz+gpu_z)

def _calculate_box_intersection(box1_min, box1_max_ex, box2_min, box2_max_ex):
    """计算两个三维盒子（由最小点和最大点定义）的交集。"""
    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max_ex = np.minimum(box1_max_ex, box2_max_ex)
    
    if np.all(intersect_max_ex > intersect_min):
        return (tuple(intersect_min), tuple(intersect_max_ex))
    return None

def generate_final_communication_plan(processors, id_to_rank_map, halo_depth):
    """
    [修正版] 严格按照“六向邻居”规则生成通信方案，并正确处理周期性边界。
    """
    print("\n--- [功能] 正在生成带周期性边界的、正确的通信方案... ---")
    
    # 步骤 1: 构建从子块坐标到任务信息的映射，方便空间查找
    coord_to_task_map = {}
    all_max_coords = []
    for proc_id, proc in processors.items():
        rank = id_to_rank_map[proc_id]
        # 假设 proc 对象有 sub_block_boundaries 属性
        for sub_block_id, bounds in proc.sub_block_boundaries.items():
            task_info = {
                'rank': rank, 
                'sub_id': sub_block_id,
                'min': np.array(bounds['min']), 
                'max_ex': np.array(bounds['max_exclusive'])
            }
            # 使用子块的最小坐标作为键
            coord_to_task_map[tuple(bounds['min'])] = task_info
            all_max_coords.append(task_info['max_ex'])

    # 步骤 2: 确定全局边界，用于处理周期性
    global_box_dims = np.max(all_max_coords, axis=0)

    # 步骤 3: 初始化空的通信计划
    task_plan = {(b['rank'], b['sub_id']): {'pre_receive': {}, 'post_send': {}} 
                 for b in coord_to_task_map.values()}
    
    # 步骤 4: 遍历每个子块任务，为其确定所有通信伙伴和区域
    for source_task in coord_to_task_map.values():
        s_rank, s_sub_id = source_task['rank'], source_task['sub_id']
        s_min, s_max_ex = source_task['min'], source_task['max_ex']
        s_dims = s_max_ex - s_min
        
        # 定义6个方向 (维度, 方向乘数)
        directions = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]

        for dim, direction in directions:
            # --- 寻找邻居 (增加周期性处理) ---
            neighbor_min_coord = s_min.copy()
            neighbor_min_coord[dim] += direction * s_dims[dim]
            
            # 使用取模运算处理周期性边界
            neighbor_min_coord[dim] %= global_box_dims[dim]
            
            # --- 如果找到邻居，则计算通信区域 ---
            if tuple(neighbor_min_coord) in coord_to_task_map:
                neighbor_task = coord_to_task_map[tuple(neighbor_min_coord)]
                n_rank = neighbor_task['rank']

                # 只在跨越不同进程边界时才需要通信
                if s_rank != n_rank:
                    # --- 重新计算精确的 Halo 区域 ---
                    if direction == 1: # 正方向 (+x, +y, +z)
                        # S 发送自己右边界的内部数据
                        send_slab_min = s_max_ex.copy(); send_slab_min[dim] -= halo_depth
                        send_slab_max_ex = s_max_ex.copy()
                        
                        # S 接收邻居 N 左边界的外部数据
                        receive_slab_min = s_max_ex.copy()
                        receive_slab_max_ex = s_max_ex.copy(); receive_slab_max_ex[dim] += halo_depth
                    
                    else: # 负方向 (-x, -y, -z)
                        # S 发送自己左边界的内部数据
                        send_slab_min = s_min.copy()
                        send_slab_max_ex = s_min.copy(); send_slab_max_ex[dim] += halo_depth
                        
                        # S 接收邻居 N 右边界的外部数据
                        receive_slab_min = s_min.copy(); receive_slab_min[dim] -= halo_depth
                        receive_slab_max_ex = s_min.copy()

                    # 计算 S 需要从 N 接收的区域
                    # = (S需要的Halo区域) 与 (N实际拥有的数据区域) 的交集
                    receive_region = _calculate_box_intersection(
                        receive_slab_min, receive_slab_max_ex, 
                        neighbor_task['min'], neighbor_task['max_ex']
                    )
                    
                    # 计算 S 需要发送给 N 的区域
                    # = (S的发送区域) 与 (S实际拥有的数据区域) 的交集 (通常是其自身)
                    send_region = _calculate_box_intersection(
                        send_slab_min, send_slab_max_ex, 
                        s_min, s_max_ex
                    )
                    
                    # 将计算出的区域添加到通信计划中
                    key = (s_rank, s_sub_id)
                    if receive_region:
                        task_plan[key]['pre_receive'].setdefault(n_rank, []).append(receive_region)
                    if send_region:
                        task_plan[key]['post_send'].setdefault(n_rank, []).append(send_region)

    return task_plan

# def generate_final_communication_plan(processors, id_to_rank_map, halo_depth):
#     """
#     [最终实现] 严格按照“2D接触面六向扩展”规则生成通信方案。
#     """
#     print("\n--- [功能] 正在生成最终的、正确的通信方案... ---")
    
#     coord_to_task_map = {}
#     for proc_id, proc in processors.items():
#         rank = id_to_rank_map[proc_id]
#         for sub_block_id, bounds in proc.sub_block_boundaries.items():
#             task_info = {'rank': rank, 'sub_id': sub_block_id, 
#                          'min': np.array(bounds['min']), 'max_ex': np.array(bounds['max_exclusive'])}
#             coord_to_task_map[bounds['min']] = task_info

#     task_plan = {(b['rank'], b['sub_id']): {'pre_receive': {}, 'post_send': {}} for b in coord_to_task_map.values()}
    
#     for source_task in coord_to_task_map.values():
#         s_rank, s_sub_id = source_task['rank'], source_task['sub_id']
#         s_min, s_max_ex = source_task['min'], source_task['max_ex']
#         s_dims = s_max_ex - s_min
        
#         directions = {'right (+x)':(0,1), 'left (-x)':(0,-1), 'front (+y)':(1,1), 
#                       'back (-y)':(1,-1), 'top (+z)':(2,1), 'bottom (-z)':(2,-1)}

#         for name, (dim, direction) in directions.items():
#             neighbor_min_coord = s_min.copy()
#             neighbor_min_coord[dim] += direction * s_dims[dim]
            
#             if tuple(neighbor_min_coord) in coord_to_task_map:
#                 neighbor_task = coord_to_task_map[tuple(neighbor_min_coord)]
#                 n_rank = neighbor_task['rank']

#                 if s_rank != n_rank:
#                     contact_surface_min = s_min.copy()
#                     contact_surface_max_ex = s_max_ex.copy()
#                     if direction == 1:
#                         contact_surface_min[dim] = s_max_ex[dim]
#                         contact_surface_max_ex[dim] = s_max_ex[dim]
#                     else:
#                         contact_surface_min[dim] = s_min[dim]
#                         contact_surface_max_ex[dim] = s_min[dim]

#                     # print("contact_surface_min", contact_surface_min)
#                     # print("contact_surface_max_ex", contact_surface_max_ex)
#                     halo_box_min = contact_surface_min - halo_depth
#                     halo_box_max_ex = contact_surface_max_ex + halo_depth
#                     # print("halo_box_min, contact_surface_min", halo_box_min, contact_surface_min)
#                     # print("halo_box_max_ex, contact_surface_max_ex", halo_box_max_ex, contact_surface_max_ex)

#                     receive_region = (halo_box_min, halo_box_max_ex)
#                     send_region = (halo_box_min, halo_box_max_ex)
#                     # receive_region = _calculate_box_intersection(halo_box_min, halo_box_max_ex, neighbor_task['min'], neighbor_task['max_ex'])
#                     # send_region = _calculate_box_intersection(halo_box_min, halo_box_max_ex, s_min, s_max_ex)
#                     # send_region = _calculate_box_intersection(halo_box_min, halo_box_max_ex, s_min, s_max_ex)
                    
#                     if receive_region:
#                         task_plan[(s_rank, s_sub_id)]['pre_receive'].setdefault(n_rank, []).append(receive_region)
#                     if send_region:
#                         task_plan[(s_rank, s_sub_id)]['post_send'].setdefault(n_rank, []).append(send_region)
    
#     print("--- 最终通信方案生成完毕 ---")
#     return task_plan

def _check_box_overlap(min1, max_ex1, min2, max_ex2):
    for i in range(3):
        if min1[i] >= max_ex2[i] or max_ex1[i] <= min2[i]: return False
    return True

def _calculate_box_intersection(min1, max_ex1, min2, max_ex2):
    intersect_min = np.maximum(min1, min2)
    intersect_max_ex = np.minimum(max_ex1, max_ex2)
    if np.any(intersect_min >= intersect_max_ex): return None
    return (tuple(intersect_min), tuple(intersect_max_ex))

# ==============================================================================
# Part 2: 核心处理器模拟类
# ==============================================================================
class Processor:
    def __init__(self, proc_coords, global_grid, slices):
        self.coords = proc_coords
        self.slices = slices
        
        local_shape=tuple(s.stop-s.start for s in slices)
        self.padded_grid=np.zeros(tuple(s+2 for s in local_shape),dtype=int)
        self.core_slices=(slice(1,-1),slice(1,-1),slice(1,-1))
        # self.padded_grid[self.core_slices]=global_grid[self.slices]
        
        # --- [核心修正] ---
        # 必须先定义 global_offsets，然后才能调用依赖于它的 _calculate_sub_block_boundaries
        self.global_offsets = tuple(s.start for s in slices)
        
        self.sub_block_boundaries = {}
        self._calculate_sub_block_boundaries()
        # --- [修正结束] ---

        print(f"    -> 处理器 @{self.coords} 已创建, 负责全局区域 X:{slices[0].start}-{slices[0].stop-1}, Y:{slices[1].start}-{slices[1].stop-1}, Z:{slices[2].start}-{slices[2].stop-1}。")
        print("           --- 连续子块边界信息 [闭区间 , 开区间) ---")
        for block_id,bounds in self.sub_block_boundaries.items():
            min_c,max_c=bounds['min'],bounds['max_exclusive']
            bounds_str=f"[{min_c[0]},{min_c[1]},{min_c[2]} , {max_c[0]},{max_c[1]},{max_c[2]})"
            print(f"           -> 子块 {block_id}: {bounds_str}")
            
    def _calculate_sub_block_boundaries(self):
        proc_shape=np.array([s.stop-s.start for s in self.slices])
        num_blocks_per_dim=np.array([2,2,2])
        sub_block_dim=proc_shape//num_blocks_per_dim
        if not np.all(proc_shape%num_blocks_per_dim==0): print(f"警告: 处理器 {self.coords} 的尺寸 {tuple(proc_shape)} 不能被 {tuple(num_blocks_per_dim)} 整除。")
        for bx,by,bz in itertools.product(*map(range,num_blocks_per_dim)):
            block_id=(bx,by,bz)
            local_min=sub_block_dim*np.array(block_id) # 确保是numpy array
            global_min=self.global_offsets+local_min # 现在可以安全地使用了
            global_max_exclusive=global_min+sub_block_dim
            self.sub_block_boundaries[block_id]={'min':tuple(global_min),'max_exclusive':tuple(global_max_exclusive)}

class DomainDecomposer:
    """
    负责处理分布式环境的领域分解、进程映射和数据切片计算。
    """
    def __init__(self, proc_grid_dim: tuple, processor_dim: tuple):
        """
        初始化分解器。
        
        Args:
            proc_grid_dim (tuple): 全局的处理器网格维度, e.g., (2, 2, 4)。
            processor_dim (tuple): 单个处理器负责的数据维度, e.g., (100, 100, 100)。
        """
        self.proc_grid_dim = proc_grid_dim
        self.processor_dim = processor_dim
        
        # --- 步骤 1: 读取环境变量 ---
        self._read_environment_variables()
        
        # --- 步骤 2: 计算层级分解方案 ---
        self._calculate_hierarchical_decomposition()
        
        # --- 步骤 3: 为当前进程计算其专属信息 ---
        # 当前进程的3D逻辑ID
        self.proc_id_3d = self.map_rank_to_proc_id()
        # 当前进程负责的数据切片
        self.local_slice = self.get_slice_for_proc_id()
        self.local_offset = self.get_offset_for_proc_id()

    def _read_environment_variables(self):
        """读取环境变量，设置分布式信息。"""
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.num_nodes = int(os.environ.get("SLURM_NNODES", 1))
        self.gpus_per_node = int(os.environ.get("GPUS_PER_NODE", 1))

    def _calculate_hierarchical_decomposition(self):
        """计算节点网格和GPU网格。"""
        # 假设 calculate_hierarchical_dims 是一个可用的函数
        self.node_grid_dim, self.gpu_grid_dim = calculate_hierarchical_dims(
            self.proc_grid_dim, self.num_nodes, self.gpus_per_node
        )

    def map_rank_to_proc_id(self) -> tuple:
        """
        将任意 rank 映射到其对应的三维逻辑处理器ID。
        
        Args:
            rank (int): 进程的 rank。
            
        Returns:
            tuple: 三维逻辑ID, e.g., (px, py, pz)。
        """
        # 假设 map_rank_to_proc_id_locality_aware 是一个可用的函数
        return map_rank_to_proc_id_locality_aware(
            self.rank, self.gpus_per_node, self.node_grid_dim, self.gpu_grid_dim
        )

    def get_slice_for_proc_id(self) -> tuple:
        """
        根据三维逻辑处理器ID，计算其负责的数据切片。
        
        Args:
            proc_id (tuple): 三维逻辑ID (px, py, pz)。
            
        Returns:
            tuple: 三个 slice 对象的元组。
        """
        px, py, pz = self.proc_id_3d
        sx, sy, sz = self.processor_dim
        return (
            slice(px * sx, (px + 1) * sx),
            slice(py * sy, (py + 1) * sy),
            slice(pz * sz, (pz + 1) * sz)
        )

    def get_offset_for_proc_id(self) -> tuple:
        """
        根据三维逻辑处理器ID，计算其负责的数据切片。
        
        Args:
            proc_id (tuple): 三维逻辑ID (px, py, pz)。
            
        Returns:
            tuple: 三个 slice 对象的元组。
        """
        px, py, pz = self.proc_id_3d
        sx, sy, sz = self.processor_dim
        return (px * sx, py * sy, pz * sz)

# ==============================================================================
# Part 3: 主程序 (Orchestrator)
# ==============================================================================
def main():
    print("="*80)
    print("端到端模拟: 最终版通信方案")
    print("="*80)
    PROC_GRID_DIM=(2,2,2); PROCESSOR_DIM=(8,8,8); NUM_NODES=2; GPUS_PER_NODE=4; HALO_DEPTH = 2
    required_total_procs=np.prod(PROC_GRID_DIM)
    if NUM_NODES*GPUS_PER_NODE!=required_total_procs: raise ValueError("物理资源不匹配")
    GLOBAL_GRID_SIZE=tuple(d*g for d,g in zip(PROCESSOR_DIM,PROC_GRID_DIM))
    node_grid_dim,gpu_grid_dim=calculate_hierarchical_dims(PROC_GRID_DIM,NUM_NODES,GPUS_PER_NODE)
    print(f"逻辑网格:{PROC_GRID_DIM}, 单处理器尺寸:{PROCESSOR_DIM}, 全局尺寸:{GLOBAL_GRID_SIZE}")
    print(f"物理布局:{NUM_NODES}节点x{GPUS_PER_NODE}卡/节点, Halo深度:{HALO_DEPTH}, 层级分解:节点网格{node_grid_dim},GPU网格{gpu_grid_dim}\n")
    
    print("--- 步骤 4: 模拟进程映射 & 初始化所有逻辑处理器 ---\n")
    global_grid=np.zeros(GLOBAL_GRID_SIZE); processors={}; id_to_rank_map={}
    sx,sy,sz=PROCESSOR_DIM
    for rank in range(required_total_procs):
        proc_id_3d=map_rank_to_proc_id_locality_aware(rank,GPUS_PER_NODE,node_grid_dim,gpu_grid_dim)
        id_to_rank_map[proc_id_3d]=rank
        print(f"Rank {rank:2d} -> 映射到逻辑ID {proc_id_3d}")
        slices=(slice(proc_id_3d[0]*sx,(proc_id_3d[0]+1)*sx),slice(proc_id_3d[1]*sy,(proc_id_3d[1]+1)*sy),slice(proc_id_3d[2]*sz,(proc_id_3d[2]+1)*sz))
        processors[proc_id_3d]=Processor(proc_id_3d,global_grid,slices)

    # --- 步骤 5: 生成并打印最终的通信方案 ---
    task_dependency_plan = generate_final_communication_plan(processors, id_to_rank_map, HALO_DEPTH)
    
    print("\n--- 步骤 5: 最终通信方案 (部分示例) ---")
    
    rank_to_show = 0
    sub_block_to_show = (1, 1, 1) 
    key = (rank_to_show, sub_block_to_show)
    
    print(f"\n----------- 任务 [Rank {rank_to_show}, 子块 {sub_block_to_show}] 的通信方案 -----------")
    my_plan = task_dependency_plan.get(key, {})
    
    print("\n  1. PRE-RECEIVE (计算前, 需要从以下邻居接收数据):")
    pre_receive_tasks = my_plan.get('pre_receive', {})
    if not pre_receive_tasks: print("     (无前置接收依赖)")
    for partner_rank, regions in sorted(pre_receive_tasks.items()):
        print(f"    - 来自 [Rank {partner_rank}] 的数据:")
        for region in regions: min_c, max_ex_c = region; print(f"      - 全局坐标区域: {min_c} -> {max_ex_c}")

    print("\n  2. POST-SEND (计算后, 需要向以下邻居发送数据):")
    post_send_tasks = my_plan.get('post_send', {})
    if not post_send_tasks: print("     (无后置发送任务)")
    for partner_rank, regions in sorted(post_send_tasks.items()):
        print(f"    - 发往 [Rank {partner_rank}] 的数据:")
        for region in regions: min_c, max_ex_c = region; print(f"      - 全局坐标区域: {min_c} -> {max_ex_c}")

    print("\n" + "="*80 + "\n模拟结束\n" + "="*80)

if __name__ == "__main__":
    main()
