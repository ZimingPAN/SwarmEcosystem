from __future__ import annotations

"""分布式/分块 KMC 环境（DistributedKMCEnv）。

这个环境是 RL4KMC 训练/评测中“状态推进”的核心：

- 维护 vacancy / Cu 等粒子在晶格上的占位与邻域类型缓存（nn1/nn2 等）。
- 提供一次“快速跳跃”(fast jump) 的执行：选择事件 -> 执行迁移 -> 更新局部环境缓存 ->
    更新 TopK 系统（用于模型输入中的 diff_k/dist_k）-> 更新扩散速率缓存。
- 返回一个 obs 数据结构（KMCObs），包含：
    - `topk_update_info`: TopK 系统的增量更新信息（供 runner 的 TopK cache 增量刷新）
    - `changed_vids`: 本步受影响的 vacancy id 列表（供 runner 的 logits cache 增量刷新）

注意：
- 代码中存在大量“性能敏感”路径（例如 softmax/采样、torch/np 双缓存同步、局部更新），
    本文件的注释优先解释：张量形状约定、缓存一致性、为何要增量更新。
- 本环境同时继承 KMCEnv 与 DistributedMixin：后者提供 rank/分块相关工具；
    但本文件顶部也有“单环境模式”的参数改写逻辑（见 __init__）。
"""

from RL4KMC.envs.kmc import KMCEnv
from RL4KMC.envs.distributed_lattice import DistributedMixin
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple
import numpy as np
import time
import torch
import logging


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.addHandler(logging.NullHandler())


TopKUpdateInfo = Mapping[str, Any]
Coord3 = Tuple[int, ...]
CoordPair = Sequence[Coord3]
UpdateMap = Mapping[int, np.ndarray]


@dataclass(frozen=True)
class KMCObs:
    topk_update_info: TopKUpdateInfo | None
    updated_cu: UpdateMap | None
    updated_vacancy: UpdateMap | None
    cu_move_from: Coord3 | None
    cu_move_to: Coord3 | None
    cu_id: int | None
    cu_topk_id: int | None
    vac_id: int
    changed_vids: Sequence[int]
    dir_idx: int
    energy_change: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "topk_update_info": self.topk_update_info,
            "updated_cu": self.updated_cu,
            "updated_vacancy": self.updated_vacancy,
            "cu_move_from": self.cu_move_from,
            "cu_move_to": self.cu_move_to,
            "cu_id": self.cu_id,
            "cu_topk_id": self.cu_topk_id,
            "vac_id": self.vac_id,
            "changed_vids": self.changed_vids,
            "dir_idx": self.dir_idx,
            "energy_change": self.energy_change,
        }


def _safe_softmax(x, dim=-1):
    """一个更“保守”的 softmax 实现。

    用途：在某些平台/算子组合下，torch.softmax 的 kernel 可能触发数值/性能问题。
    这里用 `exp/sum` 自己实现，同时做以下防护：
    - 先减去 max 以避免 overflow。
    - clamp 到 [-50, 0] 避免 denorm/NaN 路径。

    注意：这不是严格等价的 softmax（因为 clamp），但通常用于概率采样场景更稳。
    """
    # Avoid torch.softmax kernel; use a stable exp/sum implementation.
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    # After subtracting max, values should be <= 0. Clamp to avoid denorm/NaN paths.
    x = x.clamp(min=-50.0, max=0.0)
    exp_x = torch.exp(x)
    denom = exp_x.sum(dim=dim, keepdim=True).clamp(min=1e-20)
    return exp_x / denom


def _stable_softmax(x, dim=-1):
    """对输入做裁剪预处理的 softmax（保留 torch.softmax）。"""
    # preprocess x to avoid extreme values
    x = x - x.max(dim=dim, keepdim=True).values
    x = x.clamp(min=-50.0, max=0.0)
    return torch.softmax(x, dim=dim)


class DistributedKMCEnv(KMCEnv, DistributedMixin):
    def __init__(self, args):
        """构建并初始化分布式 KMC 环境。

        这里会对 args 做一些“强制设置”，目的是让该环境以“按子块/局部更新”为主：
        - 跳过全局扩散率初始化/重置（由局部更新 + diffusion_rates_update 维护）。
        - 关闭一些 reset 时的重建开销（例如 precompute_subblock_rates）。

        环境内部维护多种缓存：
        - `v_pos_of_id` / `v_pos_to_id`：vacancy id <-> 坐标的双向映射（便于 O(1) 查找）。
        - `nn1_types`/`nn2_types`：每个 vacancy 的一阶/二阶邻域类型缓存（numpy）。
        - 对应的 torch 缓存 `nn1_types_t`/`nn2_types_t`/`nn_features_t`：用于快速构造模型输入与 mask。
        - `diffusion_rates`/`diffusion_rates_t`：扩散速率缓存（numpy/torch 双份），用于事件选择与 delta_t。
        - `_changed_vids_last`：最近一步受影响的 vacancy ids（供 runner 做增量推理）。

        末尾会调用 `reset()`，因此构造后环境已可直接 step/apply。
        """
        # 单环境模式：不再拆分 processor_dim 或按进程划分数量
        if hasattr(args, "processor_dim"):
            setattr(args, "lattice_size", tuple(getattr(args, "processor_dim")))
        # 跳过全局扩散计算，初始化采用子块速率
        setattr(args, "skip_global_diffusion_init", True)
        setattr(args, "skip_global_diffusion_reset", True)
        setattr(args, "compute_global_static_env_reset", False)
        if not hasattr(args, "self_generate_local"):
            setattr(args, "self_generate_local", False)
        setattr(args, "precompute_subblock_rates", False)
        setattr(args, "rebuild_sub_block_indices_on_reset", False)
        setattr(args, "skip_stats", True)
        if not hasattr(args, "validate_env_cache_each_step"):
            setattr(args, "validate_env_cache_each_step", False)
        KMCEnv.__init__(self, args)
        DistributedMixin.__init__(self, args)
        # if not hasattr(self, "E_a0_t"):
        self.E_a0_t = torch.as_tensor(self.E_a0, dtype=torch.float32, device=self.device)
        self.NN1_np = np.array(self.NN1, dtype=np.int32)
        self.NN2_np = np.array(self.NN2, dtype=np.int32)
        self.D_np = np.array(self.dims, dtype=np.int32)
        self.sample_apply_timing_once = bool(getattr(args, "timing_once", True))
        self._has_sampled_apply_timing = False
        self.prevent_backjump = args.prevent_backjump
        self.debug = True

        # self.debug = bool(getattr(args, "debug", False))
        # 为了排查分布式/多任务场景的行为差异，提供一个带 rank 前缀的 debug logger。
        def _log(msg):
            pass
            # if self.debug:
            #     _LOGGER.debug("[rank %s] %s", int(getattr(self, "rank", -1)), msg)

        self._log = _log
        try:
            # vacancy id <-> 坐标映射（构造时尽量缓存；后续 move_vacancy 会维护）。
            vac_local = np.array(self.get_vacancy_array(), dtype=np.int32)
            if (
                (not hasattr(self, "v_pos_of_id"))
                or (not isinstance(self.v_pos_of_id, dict))
                or (len(self.v_pos_of_id) == 0)
            ):
                self.v_pos_of_id = {
                    int(i): tuple(map(int, vac_local[i]))
                    for i in range(vac_local.shape[0])
                }
            if (
                (not hasattr(self, "v_pos_to_id"))
                or (not isinstance(self.v_pos_to_id, dict))
                or (len(self.v_pos_to_id) == 0)
            ):
                self.v_pos_to_id = {
                    tuple(map(int, vac_local[i])): int(i)
                    for i in range(vac_local.shape[0])
                }

        except Exception:
            pass
        self.reset()

    def reset(self):
        """重置环境并重建/校验本地缓存。

        reset 之后的重要副作用：
        - 重建 global linear index cache（用于更快的 vacancy 查找/更新）。
        - 计算 vacancy 的局部环境类型缓存（nn1/nn2 + 深层缓存）。
        - 初始化用于模型输入的 torch 特征缓存 `nn_features_t`。
        - 初始化 backjump mask 相关结构 `_opp_dir_idx` 与 `_backjump_forbid_dir`。
        - 计算初始扩散速率 `diffusion_rates`。
        """
        initial_obs, full_obs = KMCEnv.reset(self)
        self._rebuild_global_lin_cache()
        self._validate_global_lin_cache()
        self.nn1_types, self.nn2_types, self.nn1_nn1_types, self.nn1_nn2_types = (
            self._calculate_vacancy_local_environments_sparse_local()
        )

        # if torch.cuda.is_available():
        dev = self.device
        self.nn1_types_t = torch.as_tensor(self.nn1_types, dtype=torch.int8, device=dev)
        self.nn2_types_t = torch.as_tensor(self.nn2_types, dtype=torch.int8, device=dev)
        self.nn_features_t = torch.empty(
            (int(self.nn1_types_t.shape[0]), 14), dtype=torch.int8, device=dev
        )
        self.nn_features_t[:, :8] = self.nn1_types_t
        self.nn_features_t[:, 8:] = self.nn2_types_t
        if hasattr(self, "nn1_nn1_types"):
            self.nn1_nn1_types_t = torch.as_tensor(
                self.nn1_nn1_types, dtype=torch.int8, device=dev
            )
        if hasattr(self, "nn1_nn2_types"):
            self.nn1_nn2_types_t = torch.as_tensor(
                self.nn1_nn2_types, dtype=torch.int8, device=dev
            )

        # 预计算 NN1 的“反方向”索引：用于 backjump 禁止（走一步后禁止立即走回头路）。
        if not hasattr(self, "_opp_dir_idx"):
            nn1 = np.asarray(self.NN1_np, dtype=int)
            opp = np.full((8,), -1, dtype=int)
            for i in range(8):
                v = nn1[i]
                match = np.where(np.all(nn1 == (-v), axis=1))[0]
                opp[i] = int(match[0]) if int(match.size) > 0 else int(i)
            self._opp_dir_idx = opp
        Nv_local = int(np.array(self.get_vacancy_array(), dtype=np.int32).shape[0])
        # 每个 vacancy 一个禁止方向（-1 表示不禁止；这里历史上也用 0 作默认值）。
        # 该 mask 在事件选择与 apply 后都会用到。
        # self._backjump_forbid_dir = np.full((Nv_local,), -1, dtype=np.int16)
        self._backjump_forbid_dir = np.full((Nv_local,), 0, dtype=np.int16)

        self._validate_and_rebuild_env_cache()

        self.diffusion_rates = self.calculate_diffusion_rate()

        return initial_obs, full_obs

    # def _debug_apply_enter(self, vac_id, dir_idx, delta_t, method, episode):
    #     _LOGGER.debug(
    #         "Enter fast jump: vac_id=%s dir_idx=%s delta_t=%s method=%s episode=%s",
    #         vac_id,
    #         dir_idx,
    #         delta_t,
    #         method,
    #         episode,
    #     )

    # def _debug_apply_selection(self, vac_id, dir_idx, delta_t, moving_type, r_selected):
    #     _LOGGER.debug(
    #         "rank %s vac_id=%s dir_idx=%s delta_t=%s moving_type=%s r_selected=%s",
    #         int(getattr(self, "rank", -1)),
    #         vac_id,
    #         dir_idx,
    #         delta_t,
    #         moving_type,
    #         r_selected,
    #     )

    # def _debug_delta_val(self, delta_val):
    #     _LOGGER.debug(
    #         "rank %s delta_val=%s computed from rates",
    #         int(getattr(self, "rank", -1)),
    #         delta_val,
    #     )

    def _record_update_timing(
        self,
        step_idx,
        t_move_vacancy,
        t_move_cu,
        t_update_local_env,
        t_update_system,
        t_total,
        vac_id,
        moving_type,
        cu_id,
        cu_topk_id,
    ):
        try:
            step_limit = int(getattr(self, "_bench_step_limit", 0))
            logger = getattr(self, "_bench_update_logger", None)
            if (
                isinstance(step_idx, int)
                and step_idx >= 0
                and step_idx < step_limit
                and logger is not None
            ):
                logger.write(
                    int(step_idx),
                    {
                        "t_move_vacancy": float(t_move_vacancy),
                        "t_move_cu": float(t_move_cu),
                        "t_update_local_env": float(t_update_local_env),
                        "t_update_system": float(t_update_system),
                        "t_total": float(t_total),
                        "vac_id": int(vac_id),
                        "moving_type": int(moving_type),
                        "cu_id": "" if cu_id is None else int(cu_id),
                        "cu_topk_id": "" if cu_topk_id is None else int(cu_topk_id),
                    },
                )
        except Exception:
            pass

    def _record_rl_select_timing(
        self,
        step_idx,
        t_mask_sample,
        t_rate_sum,
        t_delta_t,
        t_total,
        total_rate,
        chosen_idx,
        masked_total,
        chosen_is_masked,
        vac_id,
        dir_idx,
        delta_t,
    ):
        try:
            step_limit = int(getattr(self, "_bench_step_limit", 0))
            logger = getattr(self, "_bench_rl_select_logger", None)
            if (
                isinstance(step_idx, int)
                and step_idx >= 0
                and step_idx < step_limit
                and logger is not None
            ):
                logger.write(
                    int(step_idx),
                    {
                        "t_mask_sample": float(t_mask_sample),
                        "t_rate_sum": float(t_rate_sum),
                        "t_delta_t": float(t_delta_t),
                        "t_total": float(t_total),
                        "total_rate": float(total_rate),
                        "chosen_idx": int(chosen_idx),
                        "masked_total": int(masked_total),
                        "chosen_is_masked": int(1 if chosen_is_masked else 0),
                        "vac_id": int(vac_id),
                        "dir_idx": int(dir_idx),
                        "delta_t": float(delta_t),
                    },
                )
        except Exception:
            pass

    def _record_apply_timing(
        self,
        step_idx,
        method,
        t_select,
        t_pos,
        t_energy,
        t_energy_pre,
        t_energy_post,
        t_energy_delta,
        t_move,
        t_topk,
        t_ratei,
        t_rateu,
        t_feat,
        t_total,
        vac_id,
        dir_idx,
        delta_t,
        delta_E,
    ):
        try:
            step_limit = int(getattr(self, "_bench_step_limit", 0))
            logger = getattr(self, "_bench_apply_logger", None)
            if (
                isinstance(step_idx, int)
                and step_idx >= 0
                and step_idx < step_limit
                and logger is not None
            ):
                logger.write(
                    int(step_idx),
                    {
                        "method": str(method),
                        "t_select": float(t_select),
                        "t_pos": float(t_pos),
                        "t_energy": float(t_energy),
                        "t_energy_pre": float(t_energy_pre),
                        "t_energy_post": float(t_energy_post),
                        "t_energy_delta": float(t_energy_delta),
                        "t_move": float(t_move),
                        "t_topk": float(t_topk),
                        "t_ratei": float(t_ratei),
                        "t_rateu": float(t_rateu),
                        "t_feat": float(t_feat),
                        "t_total": float(t_total),
                        "vac_id": int(vac_id),
                        "dir_idx": int(dir_idx),
                        "delta_t": float(delta_t),
                        "delta_E": float(delta_E),
                    },
                )
        except Exception:
            pass

    # def _log_diffusion_cache_state(self):
    #     dr = getattr(self, "diffusion_rates", None)
    #     dr_shape = str(getattr(dr, "shape", "N/A"))
    #     dr_dtype = str(getattr(dr, "dtype", "N/A"))
    #     dr_t = getattr(self, "diffusion_rates_t", None)
    #     dr_t_shape = str(getattr(dr_t, "shape", "N/A"))
    #     dr_t_dtype = str(getattr(dr_t, "dtype", "N/A"))
    #     dr_t_device = str(getattr(dr_t, "device", "N/A"))
    #     _LOGGER.debug(
    #         "diffusion_rates_update rank %s existing diffusion_rates shape: %s dtype: %s",
    #         int(getattr(self, "rank", -1)),
    #         dr_shape,
    #         dr_dtype,
    #     )
    #     _LOGGER.debug(
    #         "diffusion_rates_update rank %s existing diffusion_rates_t shape: %s dtype: %s device: %s",
    #         int(getattr(self, "rank", -1)),
    #         dr_t_shape,
    #         dr_t_dtype,
    #         dr_t_device,
    #     )

    def _get_local_affected_vac_ids(self, changed_positions):
        """给定发生变化的坐标（通常是 old_pos/new_pos），返回受影响的 vacancy ids。

        返回 (primary, secondary)：
        - primary: 变化点的 NN1 范围内能映射到 vacancy 的 id
        - secondary: primary 的 NN1 邻居 vacancy（更“深一层”的受影响集合）

        该集合用于 diffusion_rates_update / 局部环境更新，避免全量重算。
        """
        dims_np = self.D_np
        NN1_np = self.NN1_np
        changed = np.array(changed_positions, dtype=int)
        neigh1 = self._get_pbc_coord(
            changed[:, None, :], NN1_np[None, :, :], dims_np
        ).reshape(-1, 3)
        s1 = set()
        for p in neigh1:
            tp = tuple(map(int, p))
            lid = self.v_pos_to_id.get(tp)
            if lid is not None:
                s1.add(int(lid))
        s2 = set()
        if len(s1) > 0:
            vac_local_arr = np.array(self.get_vacancy_array(), dtype=np.int32)
            centers = np.array(
                [vac_local_arr[int(v)] for v in sorted(list(s1))], dtype=int
            )
            neigh_of_primary = self._get_pbc_coord(
                centers[:, None, :], NN1_np[None, :, :], dims_np
            ).reshape(-1, 3)
            for q in neigh_of_primary:
                tq = tuple(map(int, q))
                lid2 = self.v_pos_to_id.get(tq)
                if lid2 is not None:
                    s2.add(int(lid2))
        s2.difference_update(s1)
        return sorted(list(s1)), sorted(list(s2))

    def _get_offset_affected_vac_ids(self, changed_positions):
        """另一种受影响 vacancy 的枚举方式（用 offset 反推可能受影响的 vacancy center）。

        相比 `_get_local_affected_vac_ids` 更保守：同时考虑 NN1/NN2 与更深层组合。
        主要用于局部环境缓存（nn1/nn2/深层）更新。
        """
        dims_np = self.D_np
        NN1_np = self.NN1_np
        NN2_np = self.NN2_np
        s_env = set()
        s_deep = set()
        changed = np.array(changed_positions, dtype=int)
        for cp in changed:
            for d in NN1_np:
                center = (cp - d) % dims_np
                tp = tuple(map(int, center))
                lid = self.v_pos_to_id.get(tp)
                if lid is not None:
                    s_env.add(int(lid))
            for d in NN2_np:
                center = (cp - d) % dims_np
                tp = tuple(map(int, center))
                lid = self.v_pos_to_id.get(tp)
                if lid is not None:
                    s_env.add(int(lid))
            for d1 in NN1_np:
                for d2 in NN1_np:
                    center = (cp - d1 - d2) % dims_np
                    tp = tuple(map(int, center))
                    lid = self.v_pos_to_id.get(tp)
                    if lid is not None:
                        s_deep.add(int(lid))
                for d2 in NN2_np:
                    center = (cp - d1 - d2) % dims_np
                    tp = tuple(map(int, center))
                    lid = self.v_pos_to_id.get(tp)
                    if lid is not None:
                        s_deep.add(int(lid))
        return sorted(list(s_env)), sorted(list(s_deep))

    def _recompute_local_env_for_vac_ids(self, vids_arr):
        """对给定 vacancy ids 重算 nn1/nn2 与深层邻域类型缓存，并同步 torch 缓存。

        `vids_arr` 应为 1D int array/list。
        该函数是“局部重算”的核心：只更新受影响的 vacancy 行，避免 reset 式全量重建。
        """
        vac_local_arr = np.array(self.get_vacancy_array(), dtype=np.int32)
        if vac_local_arr.size == 0 or len(vids_arr) == 0:
            return
        dims_np = self.D_np
        centers = np.array([vac_local_arr[v] for v in vids_arr], dtype=int)
        V_nn1 = self._get_pbc_coord(
            centers[:, None, :], self.NN1_np[None, :, :], dims_np
        )
        V_nn2 = self._get_pbc_coord(
            centers[:, None, :], self.NN2_np[None, :, :], dims_np
        )
        stack_coords = np.vstack([V_nn1.reshape(-1, 3), V_nn2.reshape(-1, 3)])
        types_flat = self._batch_get_type_from_local_coords(stack_coords)
        m = int(centers.shape[0])
        nn1_types_batch = types_flat[: m * 8].reshape(m, 8)
        nn2_types_batch = types_flat[m * 8 :].reshape(m, 6)
        self.nn1_types[vids_arr, :] = nn1_types_batch
        self.nn2_types[vids_arr, :] = nn2_types_batch
        # if torch.cuda.is_available() and hasattr(self, 'nn1_types_t') and hasattr(self, 'nn2_types_t'):
        if hasattr(self, "nn1_types_t") and hasattr(self, "nn2_types_t"):
            dev = self.device
            vids_t = torch.as_tensor(vids_arr, dtype=torch.long, device=dev)
            self.nn1_types_t[vids_t, :] = torch.as_tensor(
                nn1_types_batch, dtype=torch.int8, device=dev
            )
            self.nn2_types_t[vids_t, :] = torch.as_tensor(
                nn2_types_batch, dtype=torch.int8, device=dev
            )
            if hasattr(self, "nn_features_t"):
                self.nn_features_t[vids_t, :8] = self.nn1_types_t[vids_t, :]
                self.nn_features_t[vids_t, 8:] = self.nn2_types_t[vids_t, :]
        A_nn1_nn1_coords = self._get_pbc_coord(
            V_nn1[:, :, None, :], self.NN1_np[None, None, :, :], dims_np
        )
        A_nn1_nn2_coords = self._get_pbc_coord(
            V_nn1[:, :, None, :], self.NN2_np[None, None, :, :], dims_np
        )
        stack_deep = np.vstack(
            [A_nn1_nn1_coords.reshape(-1, 3), A_nn1_nn2_coords.reshape(-1, 3)]
        )
        deep_types = self._batch_get_type_from_local_coords(stack_deep)
        nn1_nn1_batch = deep_types[: m * 8 * 8].reshape(m, 8, 8)
        nn1_nn2_batch = deep_types[m * 8 * 8 :].reshape(m, 8, 6)
        if hasattr(self, "nn1_nn1_types"):
            self.nn1_nn1_types[vids_arr, :, :] = nn1_nn1_batch
            # if torch.cuda.is_available() and hasattr(self, 'nn1_nn1_types_t'):
            if hasattr(self, "nn1_nn1_types_t"):
                dev = self.device
                vids_t = torch.as_tensor(vids_arr, dtype=torch.long, device=dev)
                self.nn1_nn1_types_t[vids_t, :, :] = torch.as_tensor(
                    nn1_nn1_batch, dtype=torch.int8, device=dev
                )
        if hasattr(self, "nn1_nn2_types"):
            self.nn1_nn2_types[vids_arr, :, :] = nn1_nn2_batch
            # if torch.cuda.is_available() and hasattr(self, 'nn1_nn2_types_t'):
            if hasattr(self, "nn1_nn2_types_t"):
                dev = self.device
                vids_t = torch.as_tensor(vids_arr, dtype=torch.long, device=dev)
                self.nn1_nn2_types_t[vids_t, :, :] = torch.as_tensor(
                    nn1_nn2_batch, dtype=torch.int8, device=dev
                )

    def _get_backjump_mask_t(self, device=None) -> torch.Tensor:
        """构造 backjump 禁止 mask（flatten 后与 Nv*8 对齐）。

        mask 的布局约定：按 vacancy 展开，每个 vacancy 对应 8 个 NN1 方向。
        True 表示该方向当前被禁止（例如刚从相反方向移动过来）。
        """
        dev = device if device is not None else self.device
        Nv = self._backjump_forbid_dir.shape[0]
        if Nv <= 0:
            return torch.zeros((0,), dtype=torch.bool, device=dev)
        forbid = self._backjump_forbid_dir
        if forbid is None:
            return torch.zeros((Nv * 8,), dtype=torch.bool, device=dev)
        forbid_t = torch.as_tensor(forbid, dtype=torch.long, device=dev)
        valid = forbid_t >= 0
        mask = torch.zeros((Nv, 8), dtype=torch.bool, device=dev)
        if bool(valid.any().item()):
            rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
            cols = forbid_t[rows]
            # assert (cols >= 0).all().item() and (cols < 8).all().item(), f"forbid_t[{rows}] = {cols}"
            # print(f"forbid_t[{rows}] = {cols}")
            mask[rows, cols] = True
        return mask.reshape(-1)

    def _set_backjump_forbid_after_move(self, vac_id: int, dir_idx: int):
        """在成功执行一次 vacancy move 后，设置该 vacancy 的“回头路禁止方向”。"""
        if not self.prevent_backjump:
            return
        if hasattr(self, "get_vacancy_array"):
            Nv = int(np.array(self.get_vacancy_array(), dtype=np.int32).shape[0])
        else:
            Nv = int(getattr(self, "V_nums", 0))
        if (
            (not hasattr(self, "_backjump_forbid_dir"))
            or (not isinstance(getattr(self, "_backjump_forbid_dir"), np.ndarray))
            or (int(self._backjump_forbid_dir.shape[0]) != int(Nv))
        ):
            # self._backjump_forbid_dir = np.full((int(Nv),), -1, dtype=np.int16)
            self._backjump_forbid_dir = np.full((int(Nv),), 0, dtype=np.int16)
        if int(vac_id) < 0 or int(vac_id) >= int(self._backjump_forbid_dir.shape[0]):
            return
        opp = getattr(self, "_opp_dir_idx", None)
        if opp is None:
            return
        # prev = int(self._backjump_forbid_dir[int(vac_id)])
        self._backjump_forbid_dir[int(vac_id)] = int(opp[int(dir_idx)])

    def update_local_environments(self, vac_idx: int, old_pos, new_pos):
        """在 vacancy 从 old_pos -> new_pos 后，局部更新环境类型缓存。

        做的事情：
        - 先为 vac_idx（中心 vacancy）重算其邻域类型（nn1/nn2/深层）。
        - 再找出受 old/new 影响的其它 vacancy ids，局部重算它们。
        - 最后写入 `_changed_vids_last`：该列表会被 runner 用于 logits 的增量更新。
        """
        dev = self.device

        NN1 = self.NN1
        NN2 = self.NN2
        local_old = tuple(map(int, old_pos))
        local_new = tuple(map(int, new_pos))

        old_pos_1x3 = np.array(local_old, dtype=int).reshape(1, 3)
        new_pos_1x3 = np.array(local_new, dtype=int).reshape(1, 3)
        new_V_nn1_coords = self._get_pbc_coord(
            new_pos_1x3, self.NN1_np[None, :, :], self.D_np
        ).reshape(-1, 3)
        new_V_nn2_coords = self._get_pbc_coord(
            new_pos_1x3, self.NN2_np[None, :, :], self.D_np
        ).reshape(-1, 3)
        A_nn1_nn1_coords = self._get_pbc_coord(
            new_V_nn1_coords[:, None, :], self.NN1_np[None, :, :], self.D_np
        ).reshape(-1, 3)
        A_nn1_nn2_coords = self._get_pbc_coord(
            new_V_nn1_coords[:, None, :], self.NN2_np[None, :, :], self.D_np
        ).reshape(-1, 3)
        stack_coords = np.vstack(
            [new_V_nn1_coords, new_V_nn2_coords, A_nn1_nn1_coords, A_nn1_nn2_coords]
        )
        types_flat = self._batch_get_type_from_local_coords(stack_coords)
        t0 = 8
        t1 = 6
        t2 = 8 * 8
        new_V_nn1_types = types_flat[:t0]
        new_V_nn2_types = types_flat[t0 : t0 + t1]
        new_nn1_nn1_types = types_flat[t0 + t1 : t0 + t1 + t2].reshape(8, 8)
        new_nn1_nn2_types = types_flat[t0 + t1 + t2 :].reshape(8, 6)

        self.nn1_types[int(vac_idx), :] = new_V_nn1_types.reshape(8)
        self.nn2_types[int(vac_idx), :] = new_V_nn2_types.reshape(6)
        self.nn1_types_t[int(vac_idx), :] = torch.as_tensor(
            new_V_nn1_types.reshape(8), dtype=torch.int8, device=dev
        )
        self.nn2_types_t[int(vac_idx), :] = torch.as_tensor(
            new_V_nn2_types.reshape(6), dtype=torch.int8, device=dev
        )
        self.nn1_nn1_types[int(vac_idx), :, :] = new_nn1_nn1_types
        # if torch.cuda.is_available() and hasattr(self, 'nn1_nn1_types_t'):
        if hasattr(self, "nn1_nn1_types_t"):
            self.nn1_nn1_types_t[int(vac_idx), :, :] = torch.as_tensor(
                new_nn1_nn1_types, dtype=torch.int8, device=dev
            )
        self.nn1_nn2_types[int(vac_idx), :, :] = new_nn1_nn2_types
        # if torch.cuda.is_available() and hasattr(self, 'nn1_nn2_types_t'):
        if hasattr(self, "nn1_nn2_types_t"):
            self.nn1_nn2_types_t[int(vac_idx), :, :] = torch.as_tensor(
                new_nn1_nn2_types, dtype=torch.int8, device=dev
            )

        if hasattr(self, "nn_features_t"):
            # debug(zrg) need?
            self.nn_features_t[int(vac_idx), :8] = self.nn1_types_t[int(vac_idx), :]
            self.nn_features_t[int(vac_idx), 8:] = self.nn2_types_t[int(vac_idx), :]

        aff_env, aff_deep = self._get_offset_affected_vac_ids([local_old, local_new])
        aff_env = sorted([i for i in set(aff_env) if int(i) != int(vac_idx)])
        aff_deep = sorted([i for i in set(aff_deep) if int(i) != int(vac_idx)])
        if len(aff_env) > 0:
            self._recompute_local_env_for_vac_ids(np.array(aff_env, dtype=int))
        if len(aff_deep) > 0:
            self._recompute_local_deep_for_vac_ids(np.array(aff_deep, dtype=int))
        try:
            self._changed_vids_last = sorted(
                list({int(vac_idx)} | set(map(int, aff_env)) | set(map(int, aff_deep)))
            )
        except Exception:
            self._changed_vids_last = [int(vac_idx)]
        return None

    def _update_pipeline(
        self, vac_idx: int, old_pos: tuple, new_pos: tuple, moving_type: int
    ):
        """一次 move 的“更新流水线”。

        顺序非常关键：
        1) move_vacancy: 更新 vacancy 坐标与映射表，并维护 global linear cache。
        2) 若 moving_type==1（通常表示 Cu 迁移）：move_cu 更新 Cu 位置。
        3) update_local_environments: 局部重算 nn1/nn2/深层缓存，并设置 changed_vids。
        4) topk_sys.update_system: 用 updated_{cu,vacancy} 增量更新 topk 系统。

        返回值包含 updated_cu/updated_vacancy 以及 topk_update_info，供上层记录/缓存刷新。
        """
        local_old = tuple(map(int, old_pos))
        local_new = tuple(map(int, new_pos))
        updated_cu = None
        cu_move_from = None
        cu_move_to = None
        cu_id = None
        cu_topk_id = None
        t0 = time.perf_counter()
        self.move_vacancy(local_old, local_new)
        t1 = time.perf_counter()
        updated_vacancy = {
            int(vac_idx): np.vstack(
                [
                    np.array(local_old, dtype=np.float32),
                    np.array(local_new, dtype=np.float32),
                ]
            )
        }
        t_move_cu = 0.0
        if int(moving_type) == 1:
            t2 = time.perf_counter()
            self.move_cu(local_new, local_old)
            t3 = time.perf_counter()
            t_move_cu = float(t3 - t2)
            cu_move_from = local_new
            cu_move_to = local_old
            cu_pos_index = getattr(self, "cu_pos_index", None)
            if isinstance(cu_pos_index, dict):
                cu_id = cu_pos_index.get(cu_move_to)
            else:
                cu_id = None
            if cu_id is not None:
                cu_topk_id = int(cu_id) - int(self.V_nums)
                updated_cu = {
                    int(cu_topk_id): np.vstack(
                        [
                            np.array(cu_move_from, dtype=np.float32),
                            np.array(cu_move_to, dtype=np.float32),
                        ]
                    )
                }

        t4 = time.perf_counter()
        self.update_local_environments(int(vac_idx), local_old, local_new)
        t5 = time.perf_counter()
        topk_update_info = self.topk_sys.update_system(
            updated_cu=updated_cu, updated_vacancy=updated_vacancy
        )
        t6 = time.perf_counter()

        try:
            step_idx = getattr(self, "_bench_step_idx", None)
            self._record_update_timing(
                step_idx=step_idx,
                t_move_vacancy=float(t1 - t0),
                t_move_cu=float(t_move_cu),
                t_update_local_env=float(t5 - t4),
                t_update_system=float(t6 - t5),
                t_total=float(t6 - t0),
                vac_id=int(vac_idx),
                moving_type=int(moving_type),
                cu_id=cu_id,
                cu_topk_id=cu_topk_id,
            )
        except Exception:
            pass

        return (
            updated_cu,
            updated_vacancy,
            cu_move_from,
            cu_move_to,
            cu_id,
            cu_topk_id,
            topk_update_info,
        )

    def _compute_occ_illegal_mask(self):
        D = self.D_np
        vac_local_arr = np.array(self.get_vacancy_array(), dtype=np.int32)
        if vac_local_arr.size == 0:
            return np.zeros((0, 8), dtype=bool)
        nbrs = self._get_pbc_coord(
            vac_local_arr[:, None, :], self.NN1_np[None, :, :], D
        )
        mask = np.zeros((int(vac_local_arr.shape[0]), 8), dtype=bool)
        v_map = getattr(self, "v_pos_to_id", {})
        if not isinstance(v_map, dict):
            v_map = {}
        for i in range(int(mask.shape[0])):
            for j in range(8):
                tp = tuple(map(int, nbrs[i, j]))
                if v_map.get(tp) is not None:
                    mask[i, j] = True
        return mask

    def _traditional_select(self, features):
        """传统 KMC 事件选择：用 diffusion_rates 作为权重，从 Nv*8 的事件空间采样。"""
        rates = getattr(self, "diffusion_rates", None)
        t_ratei = 0.0
        rates_np = (
            np.asarray(rates) if rates is not None else np.empty((0,), dtype=float)
        )
        if rates is None or rates_np.size == 0:
            _t0 = time.time()
            rates = np.asarray(self.calculate_diffusion_rate(), dtype=float)
            _t1 = time.time()
            t_ratei = float(_t1 - _t0)
            self.diffusion_rates = rates
        dev = self.device
        rates_t = torch.as_tensor(rates, dtype=torch.float32, device=dev)
        # if hasattr(self, 'nn1_types_t') and isinstance(self.nn1_types_t, torch.Tensor):
        illegal_mask_t = self.nn1_types_t.to(device=dev) == 2
        rates_t = rates_t.masked_fill(illegal_mask_t, 0.0)
        # else:
        # if features is not None:
        #     if isinstance(features, torch.Tensor):
        #         illegal_mask_t = (features[:, 0:8] == 2).to(dtype=torch.bool, device=dev)
        #     else:
        #         illegal_mask_t = torch.as_tensor((np.asarray(features)[:, 0:8] == 2), dtype=torch.bool, device=dev)
        #     rates_t = rates_t.masked_fill(illegal_mask_t, 0.0)
        # else:
        #     occ_illegal = self._compute_occ_illegal_mask()
        #     rates_t = rates_t.masked_fill(torch.as_tensor(occ_illegal, dtype=torch.bool, device=dev), 0.0)
        flat_t = rates_t.reshape(-1)
        total_rate_t = torch.sum(flat_t)
        if bool((total_rate_t <= 0).item()):
            return 0, 0, 0.0, rates, t_ratei
        idx = int(torch.multinomial(flat_t / total_rate_t, 1).item())
        vac_local, d_idx = divmod(idx, 8)
        vac_id = int(vac_local)
        dir_idx = int(d_idx)
        delta_t = float((-torch.log(torch.rand((), device=dev)) / total_rate_t).item())
        return vac_id, dir_idx, delta_t, rates, t_ratei

    def _rl_softmax_select(self, logits, features):
        """RL 事件选择（softmax + multinomial）。

        约定：
        - 输入 logits 形状通常与 Nv*8 对齐（可被 reshape(-1)）。
        - 会用 illegal mask 将不可执行事件概率置 0：
            - `nn1_types_t == 2` 表示该方向目标是 vacancy（vac->vac 不合法）。
            - `prevent_backjump` 时额外禁止回头路。
        - `delta_t` 的采样仍基于 diffusion_rates 的总率（而非 logits），保持 SSA 物理时间推进。
        """
        # if _LOGGER.isEnabledFor(logging.DEBUG):
        #     _LOGGER.debug(
        #         "_rl_select_and_time called. logits type=%s logits shape=%s",
        #         type(logits),
        #         getattr(logits, "shape", "N/A"),
        #     )
        #     _LOGGER.debug("logits=%s", logits)
        dev = self.device
        _t0 = time.time()
        logits_t = (
            logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
        )
        # logits_t = logits_t.to(device=dev)
        #     device = self.device
        # assert not (torch.isinf(logits_t).any() or torch.isnan(logits_t).any() or (logits_t < 0).any()), \
        #     f"logits_t before softmax contains inf/nan/negative: inf={torch.isinf(logits_t).any().item()} " \
        #     f"nan={torch.isnan(logits_t).any().item()} neg={(logits_t < 0).any().item()}"
        # print(f"logits_t before softmax: {logits_t}")
        logits_t = torch.softmax(logits_t.reshape(-1), dim=-1).to(dev).contiguous()
        # logits_t = _stable_softmax(logits_t.reshape(-1), dim=-1).to(dev).contiguous()
        # assert not (torch.isinf(logits_t).any() or torch.isnan(logits_t).any() or (logits_t < 0).any()), \
        #     f"logits_t after softmax contains inf/nan/negative: inf={torch.isinf(logits_t).any().item()} " \
        #     f"nan={torch.isnan(logits_t).any().item()} neg={(logits_t < 0).any().item()}"

        #     w = torch.clamp_min(probs, 1e-12)
        #     mask_t = torch.zeros((w.numel(),), dtype=torch.bool, device=device)
        #     if bool(getattr(self.args, "prevent_backjump", False)):
        #         back_mask_t = self._get_backjump_mask_t(device=device)
        #         if isinstance(back_mask_t, torch.Tensor) and back_mask_t.numel() == w.numel():
        #             mask_t = back_mask_t
        #     w = w.masked_fill(mask_t, 0.0)
        #     # sum_val = w.sum()
        #     # if bool((sum_val <= 0).item()):
        #     #     fallback = (~mask_t).float()
        #     #     w = fallback
        #     return idx, w
        # logits_t = torch.softmax(logits_t, dim=-1)
        # if _LOGGER.isEnabledFor(logging.DEBUG):
        #     _LOGGER.debug("logits_t shape=%s", tuple(logits_t.shape))
        #     _LOGGER.debug("logits_t dtype=%s", str(logits_t.dtype))
        #     _LOGGER.debug("logits_t=%s", logits_t)

        flat_logits = logits_t.reshape(-1)
        # illegal_flat = torch.zeros((flat_logits.numel(),), dtype=torch.bool, device=dev)
        # illegal_flat: True 表示该 vacancy-direction 不可选。
        # 这里 `2` 是占位类型编码（通常 vacancy type==2）。
        illegal_flat = (self.nn1_types_t == 2).reshape(-1)
        if self.prevent_backjump:
            back_mask_flat = self._get_backjump_mask_t(device=dev)
            illegal_flat = illegal_flat | back_mask_flat

        # masked_logits = flat_logits.masked_fill(illegal_flat, torch.tensor(-1e30, device=dev, dtype=flat_logits.dtype))
        # chosen_idx = int(torch.argmax(masked_logits).item())

        # 由于 flat_logits 已经经过 softmax，理论上不会出现 inf/nan/负值；若出现则直接跳过断言
        has_inf = bool(torch.isinf(flat_logits).any().item())
        has_nan = bool(torch.isnan(flat_logits).any().item())
        has_neg = bool((flat_logits < 0).any().item())
        if has_inf or has_nan or has_neg:
            _LOGGER.error(
                "flat_logits contains inf/nan/negative: inf=%s nan=%s neg=%s",
                has_inf,
                has_nan,
                has_neg,
            )
            # print(f"flat_logits  contains inf/nan/negative: {flat_logits}")
            # pass
        # else:
        # print(f"flat_logits softmax ")
        # print(f"flat_logits no inf/nan/negative : {flat_logits}")

        # 注意：这里 flat_logits 已是 softmax 后的概率；被 mask 的位置置 0 即可。
        masked_logits = flat_logits.masked_fill(
            illegal_flat, torch.tensor(0, device=dev, dtype=flat_logits.dtype)
        )

        assert not (
            torch.isinf(masked_logits).any()
            or torch.isnan(masked_logits).any()
            or (masked_logits < 0).any()
        ), (
            f"masked_logits contains inf/nan/negative: inf={torch.isinf(masked_logits).any().item()} "
            f"nan={torch.isnan(masked_logits).any().item()} neg={(masked_logits < 0).any().item()}"
        )

        chosen_idx = int(torch.multinomial(masked_logits, 1).item())
        # chosen_idx = 1

        if True:
            vac_local, d_idx = divmod(chosen_idx, 8)
            vac_id = int(vac_local)
            dir_idx = int(d_idx)
            forbid = -1
            mask8 = None
            if (
                hasattr(self, "_backjump_forbid_dir")
                and isinstance(self._backjump_forbid_dir, np.ndarray)
                and 0 <= vac_id < int(self._backjump_forbid_dir.shape[0])
            ):
                forbid = int(self._backjump_forbid_dir[vac_id])
                mask8 = [0] * 8
                if 0 <= forbid < 8:
                    mask8[forbid] = 1
            masked_total = (
                int(illegal_flat.sum().item())
                if isinstance(illegal_flat, torch.Tensor)
                else -1
            )
            chosen_is_masked = (
                bool(illegal_flat[int(chosen_idx)].item())
                if isinstance(illegal_flat, torch.Tensor)
                and illegal_flat.numel() > int(chosen_idx)
                else False
            )
            # print(f"[rank {self.rank}] backjump_mask_select idx={int(chosen_idx)} vac={vac_id} dir={dir_idx} forbid={forbid} chosen_masked={chosen_is_masked} masked_total={masked_total} mask8={mask8}", flush=True)

        _t1 = time.time()
        t_ratei = float(_t1 - _t0)
        vac_local, d_idx = divmod(chosen_idx, 8)
        vac_id = int(vac_local)
        dir_idx = int(d_idx)
        # print(f"[rank {self.rank}] t_ratei={t_ratei:.8f}")

        _t2 = time.time()
        rates_t = torch.as_tensor(self.diffusion_rates, dtype=torch.float32, device=dev)
        if self.prevent_backjump:
            back_mask_flat = self._get_backjump_mask_t(device=dev)
            if (
                isinstance(back_mask_flat, torch.Tensor)
                and back_mask_flat.numel() == rates_t.numel()
            ):
                rates_t = rates_t.masked_fill(back_mask_flat.view_as(rates_t), 0.0)
        total_rate_t = torch.sum(rates_t.reshape(-1))
        _t3 = time.time()
        # _t4 = time.time()
        if bool((total_rate_t <= 0).item()):
            delta_t = 0.0
        else:
            delta_t = float(
                (-torch.log(torch.rand((), device=dev)) / total_rate_t).item()
            )
        _t4 = time.time()

        step_idx = getattr(self, "_bench_step_idx", None)
        self._record_rl_select_timing(
            step_idx=step_idx,
            t_mask_sample=float(t_ratei),
            t_rate_sum=float(_t3 - _t2),
            t_delta_t=float(_t4 - _t3),
            t_total=float(_t4 - _t0),
            total_rate=(
                float(total_rate_t.item())
                if isinstance(total_rate_t, torch.Tensor)
                else float(total_rate_t)
            ),
            chosen_idx=int(chosen_idx),
            masked_total=int(masked_total),
            chosen_is_masked=bool(chosen_is_masked),
            vac_id=int(vac_id),
            dir_idx=int(dir_idx),
            delta_t=float(delta_t),
        )

        # probs = probs.masked_fill(mask2_t, 0.0)
        # probs = torch.clamp_min(probs, 1e-12)
        # # true = probs.clone()
        # # true = true.masked_fill(mask2_t, 0.0)
        # chosen_idx = min(int(chosen_idx), int(probs.numel()) - 1)
        # diffusion_rates_t = torch.as_tensor(self.diffusion_rates, dtype=torch.float32, device=dev).reshape(-1)
        # # print("self.diffusion_rates_t.shape, probs.shape:", diffusion_rates_t.shape, probs.shape)

        # tmp = (diffusion_rates_t / probs)
        # tmp = tmp.masked_fill(mask2_t, 0.0)
        # r0 = torch.max(tmp)

        # # if bool((r0 <= 0).item()):
        # #     r0 = torch.tensor(1.0, device=dev)
        # sum_ratio = torch.sum(tmp / r0)
        # null_prob = torch.clamp(1.0 - sum_ratio, 1e-12, 1.0 - 1e-12)
        # denom = torch.log(null_prob / ((diffusion_rates_t[chosen_idx] / r0) + null_prob))
        # denom = torch.where(denom == 0, torch.tensor(-1e-9, device=dev), denom)
        # u_rand = torch.rand((), device=dev)
        # n_trials = torch.ceil(torch.log(u_rand) / denom).to(torch.int64)
        # n_trials = torch.clamp(n_trials, 1)
        # step_delta = torch.distributions.Gamma(concentration=n_trials.float(), rate=torch.tensor(1.0, device=dev)).sample()
        # delta_t = float((step_delta / r0).item())

        return vac_id, dir_idx, delta_t, t_ratei

    def _rl_max_select(self, logits: torch.Tensor):
        dev = self.device
        flat_logits = logits.reshape(-1)
        illegal_flat = (self.nn1_types_t == 2).reshape(-1)
        if self.prevent_backjump:
            back_mask_flat = self._get_backjump_mask_t(device=dev)
            illegal_flat = illegal_flat | back_mask_flat
        masked_logits = flat_logits.masked_fill(
            illegal_flat, torch.tensor(-1e30, device=dev, dtype=flat_logits.dtype)
        )
        chosen_idx = int(torch.argmax(masked_logits).item())
        vac_local, d_idx = divmod(chosen_idx, 8)
        vac_id = int(vac_local)
        dir_idx = int(d_idx)
        return vac_id, dir_idx

    def _recompute_local_deep_for_vac_ids(self, vids_arr):
        vac_local_arr = np.array(self.get_vacancy_array(), dtype=np.int32)
        if vac_local_arr.size == 0 or len(vids_arr) == 0:
            return
        dims_np = self.D_np
        centers = np.array([vac_local_arr[v] for v in vids_arr], dtype=int)
        V_nn1 = self._get_pbc_coord(
            centers[:, None, :], self.NN1_np[None, :, :], dims_np
        )
        V_nn2 = self._get_pbc_coord(
            centers[:, None, :], self.NN2_np[None, :, :], dims_np
        )
        nn1_dir = self._batch_get_type_from_local_coords(V_nn1.reshape(-1, 3)).reshape(
            int(centers.shape[0]), 8
        )
        nn2_dir = self._batch_get_type_from_local_coords(V_nn2.reshape(-1, 3)).reshape(
            int(centers.shape[0]), 6
        )
        self.nn1_types[vids_arr, :] = nn1_dir
        self.nn2_types[vids_arr, :] = nn2_dir
        # if torch.cuda.is_available() and hasattr(self, 'nn1_types_t') and hasattr(self, 'nn2_types_t'):
        if hasattr(self, "nn1_types_t") and hasattr(self, "nn2_types_t"):
            dev = self.device
            vids_t = torch.as_tensor(vids_arr, dtype=torch.long, device=dev)
            self.nn1_types_t[vids_t, :] = torch.as_tensor(
                nn1_dir, dtype=torch.int8, device=dev
            )
            self.nn2_types_t[vids_t, :] = torch.as_tensor(
                nn2_dir, dtype=torch.int8, device=dev
            )
            if hasattr(self, "nn_features_t"):
                self.nn_features_t[vids_t, :8] = self.nn1_types_t[vids_t, :]
                self.nn_features_t[vids_t, 8:] = self.nn2_types_t[vids_t, :]
        A_nn1_nn1_coords = self._get_pbc_coord(
            V_nn1[:, :, None, :], self.NN1_np[None, None, :, :], dims_np
        )
        A_nn1_nn2_coords = self._get_pbc_coord(
            V_nn1[:, :, None, :], self.NN2_np[None, None, :, :], dims_np
        )
        stack_deep = np.vstack(
            [A_nn1_nn1_coords.reshape(-1, 3), A_nn1_nn2_coords.reshape(-1, 3)]
        )
        deep_types = self._batch_get_type_from_local_coords(stack_deep)
        m = int(centers.shape[0])
        nn1_nn1_batch = deep_types[: m * 8 * 8].reshape(m, 8, 8)
        nn1_nn2_batch = deep_types[m * 8 * 8 :].reshape(m, 8, 6)
        if hasattr(self, "nn1_nn1_types"):
            self.nn1_nn1_types[vids_arr, :, :] = nn1_nn1_batch
            # if torch.cuda.is_available() and hasattr(self, 'nn1_nn1_types_t'):
            if hasattr(self, "nn1_nn1_types_t"):
                dev = self.device
                vids_t = torch.as_tensor(vids_arr, dtype=torch.long, device=dev)
                self.nn1_nn1_types_t[vids_t, :, :] = torch.as_tensor(
                    nn1_nn1_batch, dtype=torch.int8, device=dev
                )
        if hasattr(self, "nn1_nn2_types"):
            self.nn1_nn2_types[vids_arr, :, :] = nn1_nn2_batch
            # if torch.cuda.is_available() and hasattr(self, 'nn1_nn2_types_t'):
            if hasattr(self, "nn1_nn2_types_t"):
                dev = self.device
                vids_t = torch.as_tensor(vids_arr, dtype=torch.long, device=dev)
                self.nn1_nn2_types_t[vids_t, :, :] = torch.as_tensor(
                    nn1_nn2_batch, dtype=torch.int8, device=dev
                )

    def _validate_and_rebuild_env_cache(self):
        vac_arr = np.array(self.get_vacancy_array(), dtype=np.int32)
        Nv = int(vac_arr.shape[0])
        if Nv == 0:
            return
        NN1 = self.NN1_np
        NN2 = self.NN2_np
        V_nn1 = self._get_pbc_coord(vac_arr[:, None, :], NN1[None, :, :], self.D_np)
        V_nn2 = self._get_pbc_coord(vac_arr[:, None, :], NN2[None, :, :], self.D_np)
        nn1_dir = self._batch_get_type_from_local_coords(V_nn1.reshape(-1, 3)).reshape(
            Nv, 8
        )
        nn2_dir = self._batch_get_type_from_local_coords(V_nn2.reshape(-1, 3)).reshape(
            Nv, 6
        )
        has_buf = hasattr(self, "nn1_types") and hasattr(self, "nn2_types")
        assert has_buf
        ok_shape = (nn1_dir.shape == self.nn1_types.shape) and (
            nn2_dir.shape == self.nn2_types.shape
        )
        ok_equal_n1 = np.array_equal(nn1_dir, self.nn1_types) if ok_shape else False
        ok_equal_n2 = np.array_equal(nn2_dir, self.nn2_types) if ok_shape else False
        if not (ok_equal_n1 and ok_equal_n2):
            try:
                diff1 = (
                    np.argwhere(self.nn1_types != nn1_dir)
                    if not ok_equal_n1
                    else np.empty((0, 2), dtype=int)
                )
                diff2 = (
                    np.argwhere(self.nn2_types != nn2_dir)
                    if not ok_equal_n2
                    else np.empty((0, 2), dtype=int)
                )
                self._log(
                    f"env_cache_mismatch nn1_equal={ok_equal_n1} nn2_equal={ok_equal_n2} nn1_shape={self.nn1_types.shape} recomputed_nn1={nn1_dir.shape} nn2_shape={self.nn2_types.shape} recomputed_nn2={nn2_dir.shape} nn1_diff_count={diff1.shape[0]} nn2_diff_count={diff2.shape[0]}"
                )
                if diff1.shape[0] > 0:
                    head = diff1[:10]
                    for i, j in head:
                        vi, vj, vk = tuple(map(int, vac_arr[int(i)]))
                        di, dj, dk = NN1[int(j)]
                        D = self.D_np
                        ni, nj, nk = (
                            (vi + di) % D[0],
                            (vj + dj) % D[1],
                            (vk + dk) % D[2],
                        )
                        t_cached = int(self.nn1_types[int(i), int(j)])
                        t_recomp = int(nn1_dir[int(i), int(j)])
                        t_now = int(
                            self._get_type_from_coord(
                                np.asarray((int(ni), int(nj), int(nk)), dtype=int)
                            )
                        )
                        self._log(
                            f"nn1_diff vac={int(i)} dir={int(j)} center={(vi, vj, vk)} nn1_coord={(int(ni), int(nj), int(nk))} cached={t_cached} recomputed={t_recomp} current={t_now}"
                        )
                if diff2.shape[0] > 0:
                    head2 = diff2[:10]
                    for i, j in head2:
                        vi, vj, vk = tuple(map(int, vac_arr[int(i)]))
                        di, dj, dk = NN2[int(j)]
                        D = self.D_np
                        ni, nj, nk = (
                            (vi + di) % D[0],
                            (vj + dj) % D[1],
                            (vk + dk) % D[2],
                        )
                        t_cached = int(self.nn2_types[int(i), int(j)])
                        t_recomp = int(nn2_dir[int(i), int(j)])
                        t_now = int(
                            self._get_type_from_coord(
                                np.asarray((int(ni), int(nj), int(nk)), dtype=int)
                            )
                        )
                        self._log(
                            f"nn2_diff vac={int(i)} dir={int(j)} center={(vi, vj, vk)} nn2_coord={(int(ni), int(nj), int(nk))} cached={t_cached} recomputed={t_recomp} current={t_now}"
                        )
            except Exception:
                pass
            try:
                self.nn1_types = nn1_dir
                self.nn2_types = nn2_dir
                # if torch.cuda.is_available():
                dev = self.device
                if hasattr(self, "nn1_types_t"):
                    self.nn1_types_t = torch.as_tensor(
                        self.nn1_types, dtype=torch.int8, device=dev
                    )
                if hasattr(self, "nn2_types_t"):
                    self.nn2_types_t = torch.as_tensor(
                        self.nn2_types, dtype=torch.int8, device=dev
                    )
            except Exception:
                pass
        try:
            A_nn1_nn1_coords = self._get_pbc_coord(
                V_nn1[:, :, None, :], NN1[None, None, :, :], self.D_np
            )
            A_nn1_nn2_coords = self._get_pbc_coord(
                V_nn1[:, :, None, :], NN2[None, None, :, :], self.D_np
            )
            stack_deep = np.vstack(
                [A_nn1_nn1_coords.reshape(-1, 3), A_nn1_nn2_coords.reshape(-1, 3)]
            )
            deep_types = self._batch_get_type_from_local_coords(stack_deep)
            nn1_nn1_dir = deep_types[: Nv * 8 * 8].reshape(Nv, 8, 8)
            nn1_nn2_dir = deep_types[Nv * 8 * 8 :].reshape(Nv, 8, 6)
            if hasattr(self, "nn1_nn1_types"):
                ok_11_shape = self.nn1_nn1_types.shape == nn1_nn1_dir.shape
                ok_11_equal = (
                    np.array_equal(self.nn1_nn1_types, nn1_nn1_dir)
                    if ok_11_shape
                    else False
                )
                if not ok_11_equal:
                    try:
                        diff11 = np.count_nonzero(self.nn1_nn1_types != nn1_nn1_dir)
                        self._log(
                            f"deep_cache_mismatch nn1_nn1_equal={ok_11_equal} shape_cached={self.nn1_nn1_types.shape} shape_recomputed={nn1_nn1_dir.shape} diff_count={int(diff11)}"
                        )
                    except Exception:
                        pass
                    try:
                        self.nn1_nn1_types = nn1_nn1_dir
                        # if torch.cuda.is_available() and hasattr(self, 'nn1_nn1_types_t'):
                        if hasattr(self, "nn1_nn1_types_t"):
                            dev = self.device
                            self.nn1_nn1_types_t = torch.as_tensor(
                                self.nn1_nn1_types, dtype=torch.int8, device=dev
                            )
                    except Exception:
                        pass
            if hasattr(self, "nn1_nn2_types"):
                ok_12_shape = self.nn1_nn2_types.shape == nn1_nn2_dir.shape
                ok_12_equal = (
                    np.array_equal(self.nn1_nn2_types, nn1_nn2_dir)
                    if ok_12_shape
                    else False
                )
                if not ok_12_equal:
                    try:
                        diff12 = np.count_nonzero(self.nn1_nn2_types != nn1_nn2_dir)
                        self._log(
                            f"deep_cache_mismatch nn1_nn2_equal={ok_12_equal} shape_cached={self.nn1_nn2_types.shape} shape_recomputed={nn1_nn2_dir.shape} diff_count={int(diff12)}"
                        )
                    except Exception:
                        pass
                    try:
                        self.nn1_nn2_types = nn1_nn2_dir
                        # if torch.cuda.is_available() and hasattr(self, 'nn1_nn2_types_t'):
                        if hasattr(self, "nn1_nn2_types_t"):
                            dev = self.device
                            self.nn1_nn2_types_t = torch.as_tensor(
                                self.nn1_nn2_types, dtype=torch.int8, device=dev
                            )
                    except Exception:
                        pass
        except Exception:
            pass
        pass
        assert (self.nn1_types.shape == nn1_dir.shape) and (
            self.nn2_types.shape == nn2_dir.shape
        )

    def _ensure_rates_cache(self):
        dev = self.device
        rates_np = getattr(self, "diffusion_rates", None)
        rates_t = getattr(self, "diffusion_rates_t", None)
        computed = False
        if (rates_np is None) or (np.asarray(rates_np).size == 0):
            t0 = time.time()
            rates_np = np.asarray(self.calculate_diffusion_rate(), dtype=float)
            t_ratei = float(time.time() - t0)
            self.diffusion_rates = rates_np
            computed = True
        else:
            t_ratei = 0.0
        if (
            (rates_t is None)
            or (not isinstance(rates_t, torch.Tensor))
            or (rates_t.numel() == 0)
        ):
            try:
                self.diffusion_rates_t = torch.as_tensor(
                    rates_np, dtype=torch.float32, device=dev
                )
            except Exception:
                self.diffusion_rates_t = None
        return self.diffusion_rates_t, t_ratei

    def step_local(self, vac_local_id: int, dir_idx: int, episode: int):
        action = int(vac_local_id) * 8 + int(dir_idx)
        return self.step_with_stats(action, episode)

    def step_fast_local(self, vac_local_id: int, dir_idx: int, episode: int):
        action = int(vac_local_id) * 8 + int(dir_idx)
        return self.step_fast(action, episode)

    def apply_fast_jump(
        self,
        vac_id: int | None = None,
        dir_idx: int | None = None,
        delta_t: float | None = None,
        method: str = "traditional",
        logits=None,
        features=None,
    ):
        """执行一次 fast jump，并返回本步的观测信息（KMCObs）。

        参数语义：
        - method == "traditional": 走传统 KMC 选择（diffusion_rates 权重）。
        - method == "rl": 走 RL 选择（logits -> softmax -> multinomial）。
        - vac_id/dir_idx/delta_t: 若未提供，会在内部按 method 自动选择并采样 delta_t。

        返回的 obs（关键字段）：
        - `topk_update_info`: 来自 topk_sys.update_system 的增量更新信息
            （runner 会用它更新 diff_k/dist_k 的缓存）。
        - `changed_vids`: 本步受影响的 vacancy ids（runner 用它做 logits_cache 增量更新）。
        - `vac_id`/`dir_idx`: 实际执行的事件。
        - `energy_change`: 本步能量变化/代理指标（当前实现基于 r_selected 与活化能）。

        注意：该函数内部还会写计时日志（若 bench logger 被 runner 配置）。
        """
        t_total0 = time.time()
        t_select0 = time.time()
        t_ratei = 0.0
        t_rateu = 0.0
        # self._validate_and_rebuild_env_cache()

        # --- 1) 选择事件（vac_id, dir_idx）并采样 delta_t ---
        if method == "traditional":
            vac_id, dir_idx, delta_t, rates, t_ratei = self._traditional_select(
                features
            )
        elif method == "rl":
            vac_id, dir_idx, delta_t, t_ratei = self._rl_softmax_select(
                logits, features
            )
        else:
            raise ValueError(f"unknown apply_fast_jump method: {method!r}")

        if vac_id is None or dir_idx is None or delta_t is None:
            raise ValueError(
                f"apply_fast_jump selection failed: vac_id={vac_id} dir_idx={dir_idx} delta_t={delta_t}"
            )

        vac_id_i = int(vac_id)
        dir_idx_i = int(dir_idx)
        delta_t_f = float(delta_t)
        t_select1 = time.time()
        lid = int(vac_id_i)
        # try:
        #     if (bool(getattr(self.args, "enable_backjump_mask_log", False)) or bool(getattr(self.args, "debug", True))) and bool(getattr(self.args, "prevent_backjump", False)):
        #         forbid = -1
        #         mask8 = None
        #         if hasattr(self, "_backjump_forbid_dir") and isinstance(self._backjump_forbid_dir, np.ndarray) and 0 <= int(lid) < int(self._backjump_forbid_dir.shape[0]):
        #             forbid = int(self._backjump_forbid_dir[int(lid)])
        #             mask8 = [0] * 8
        #             if 0 <= forbid < 8:
        #                 mask8[forbid] = 1
        #         chosen_is_forbidden = (int(dir_idx) == int(forbid)) if forbid >= 0 else False
        #         print(f"[rank {self.rank}] backjump_mask_apply vac={int(lid)} chosen_dir={int(dir_idx)} forbid={forbid} chosen_forbidden={chosen_is_forbidden} mask8={mask8}", flush=True)
        # except Exception:
        #     pass
        t_pos0 = time.time()

        # 在这里验证nn1是否正确
        # nn1_type = self.nn1_types[int(lid)]
        # assert nn1_type == 1 or nn1_type == 2, f"rank {self.rank} nn1_type={nn1_type} lid={lid}"
        # _LOGGER.debug(
        #     "rank %s applying jump vac_id=%s dir_idx=%s delta_t=%s",
        #     int(getattr(self, "rank", -1)),
        #     vac_id,
        #     dir_idx,
        #     delta_t,
        # )
        # --- 2) 计算 vacancy 的迁移目标坐标 (old_pos -> new_pos) ---
        pos_raw = self.get_vacancy_pos_by_id(int(lid))
        if pos_raw is None:
            raise ValueError(f"invalid vacancy id {int(lid)}: position not found")
        pos = tuple(map(int, pos_raw))
        vi, vj, vk = pos
        di, dj, dk = self.NN1[int(dir_idx_i)]
        D = self.D_np
        ni, nj, nk = (vi + di) % D[0], (vj + dj) % D[1], (vk + dk) % D[2]

        t_pos1 = time.time()
        delta_val = float("nan")
        t_energy_pre = 0.0
        t_energy_post = 0.0
        t_energy_delta = 0.0
        # moving_type 是目标格点的占位类型：
        # - 若为 vacancy(type==2)，表示 vac->vac 的非法事件（此处直接抛错）。
        # - 其它类型则允许，进入更新流水线。
        moving_type = int(
            self._get_type_from_coord(
                np.asarray((int(ni), int(nj), int(nk)), dtype=int)
            )
        )
        updated_cu = None
        topk_update_info = None
        cu_move_from = None
        cu_move_to = None
        cu_id = None
        cu_topk_id = None
        updated_vacancy = None
        vac_local_topk_id = int(lid)
        t_move = 0.0
        t_topk = 0.0

        # r_selected：本事件的扩散速率（来自 diffusion_rates 缓存），用于 delta_t 与能量代理。
        r_selected = float(self.diffusion_rates[int(lid), int(dir_idx_i)])
        # self._debug_apply_selection(
        #     vac_id_i, dir_idx_i, delta_t_f, moving_type, r_selected
        # )
        if moving_type != 2:
            t_move0 = time.time()
            old_pos = (vi, vj, vk)
            new_pos = (ni, nj, nk)
            t_ep0 = time.time()
            # e_old_before = float(self._center_bond_energy_sum(old_pos))
            # e_new_before = float(self._center_bond_energy_sum(new_pos))
            t_ep1 = time.time()
            t_energy_pre = float(t_ep1 - t_ep0)

            # --- 3) 执行迁移 + 局部缓存更新 + TopK 系统增量更新 ---
            (
                updated_cu,
                updated_vacancy,
                cu_move_from,
                cu_move_to,
                cu_id,
                cu_topk_id,
                topk_update_info,
            ) = self._update_pipeline(int(lid), old_pos, new_pos, int(moving_type))
            self._set_backjump_forbid_after_move(int(lid), int(dir_idx))
            # self._validate_global_lin_cache()
            # if int(moving_type) == int(self.CU_TYPE):
            #     try:
            #         if hasattr(self, "_validate_global_lin_cache"):
            #             self._validate_global_lin_cache()
            #     except Exception:
            #         try:
            #             self._rebuild_global_lin_cache()
            #         except Exception:
            #             pass
            t_move1 = time.time()
            t_move = float(t_move1 - t_move0)
            # _LOGGER.debug(
            #     "moved cu_id=%s from %s to %s",
            #     cu_id,
            #     old_pos,
            #     new_pos,
            # )
            # 扩散率更新是最重的路径之一：支持只在前几步采样计时（便于 profiling）。
            if (not self._has_sampled_apply_timing) and self.sample_apply_timing_once:
                t_rateu0 = time.time()
                self.diffusion_rates_update([old_pos, new_pos])
                t_rateu1 = time.time()
                t_rateu = float(t_rateu1 - t_rateu0)
            else:
                self.diffusion_rates_update([old_pos, new_pos])
            t_ea0 = time.time()
            e_old_after = float(self._center_bond_energy_sum(old_pos))
            e_new_after = float(self._center_bond_energy_sum(new_pos))
            t_ea1 = time.time()
            t_energy_post = float(t_ea1 - t_ea0)
            # _LOGGER.debug(
            #     "rank %s e_old_after=%s e_new_after=%s",
            #     int(getattr(self, "rank", -1)),
            #     e_old_after,
            #     e_new_after,
            # )
            t_ed0 = time.time()
            # if r_selected is not None and r_selected > 0.0:
            # energy_change 的当前实现：用速率反推“等效能垒变化”。
            # 这更像一个诊断/代理指标，并非严格的系统总能量差。
            kB = 8.617e-5
            T = self.args.temperature
            nu = 1e13
            delta_val = (
                float(-kB * T * np.log(r_selected / nu)) - self.E_a0_t[moving_type]
            ) * 2
            # else:
            # delta_val = float(((e_old_after + e_new_after) - (e_old_before + e_new_before)) / 2.0)
            # print(f"rank {self.rank} moving type={moving_type}")
            # print(f"rank {self.rank} delata_val={delta_val} e_old_after={e_old_after} e_new_after={e_new_after} e_old_before={e_old_before} e_new_before={e_new_before}")

            t_ed1 = time.time()
            t_energy_delta = float(t_ed1 - t_ed0)
            # self._debug_delta_val(delta_val)
        else:
            raise ValueError(
                f"rank {self.rank} moving_type={moving_type} lid={lid} pos={pos} dir_idx={dir_idx_i} delta_t={delta_t_f}"
            )

        # --- 4) 推进物理时间（SSA 时间增量） ---
        self.time += float(delta_t_f)
        t_feat0 = time.time()
        # type_obs = self.get_vacancy_neighbor_features()
        t_feat1 = time.time()
        t_select = float(t_select1 - t_select0)
        t_pos = float(t_pos1 - t_pos0)
        t_energy = float(t_energy_pre + t_energy_post + t_energy_delta)
        t_feat = float(t_feat1 - t_feat0)
        t_total = float(time.time() - t_total0)
        step_idx = getattr(self, "_bench_step_idx", None)
        self._record_apply_timing(
            step_idx=step_idx,
            method=method,
            t_select=float(t_select),
            t_pos=float(t_pos),
            t_energy=float(t_energy),
            t_energy_pre=float(t_energy_pre),
            t_energy_post=float(t_energy_post),
            t_energy_delta=float(t_energy_delta),
            t_move=float(t_move),
            t_topk=float(t_topk),
            t_ratei=float(t_ratei),
            t_rateu=float(t_rateu),
            t_feat=float(t_feat),
            t_total=float(t_total),
            vac_id=int(vac_id_i),
            dir_idx=int(dir_idx_i),
            delta_t=float(delta_t_f),
            delta_E=float(delta_val),
        )
        return KMCObs(
            topk_update_info=topk_update_info,
            updated_cu=updated_cu,
            updated_vacancy=updated_vacancy,
            cu_move_from=cu_move_from,
            cu_move_to=cu_move_to,
            cu_id=cu_id,
            cu_topk_id=cu_topk_id,
            vac_id=(
                int(vac_local_topk_id)
                if isinstance(vac_local_topk_id, (int, np.integer))
                else vac_local_topk_id
            ),
            changed_vids=self._changed_vids_last,
            dir_idx=int(dir_idx_i),
            energy_change=float(delta_val),
        )

    def bench_fast_jump(
        self,
        logits: torch.Tensor,
    ):
        # 固定步数模式下的 fast jump，只做选择与更新，不做时间推进
        vac_id, dir_idx = self._rl_max_select(logits=logits)
        if vac_id is None or dir_idx is None:
            raise ValueError(
                f"bench_fast_jump selection failed: vac_id={vac_id} dir_idx={dir_idx}"
            )

        vac_id_i = int(vac_id)
        dir_idx_i = int(dir_idx)
        lid = int(vac_id_i)

        pos_raw = self.get_vacancy_pos_by_id(int(lid))
        if pos_raw is None:
            raise ValueError(f"invalid vacancy id {int(lid)}: position not found")
        pos = tuple(map(int, pos_raw))
        vi, vj, vk = pos
        di, dj, dk = self.NN1[int(dir_idx_i)]
        D = self.D_np
        ni, nj, nk = (vi + di) % D[0], (vj + dj) % D[1], (vk + dk) % D[2]

        moving_type = int(
            self._get_type_from_coord(
                np.asarray((int(ni), int(nj), int(nk)), dtype=int)
            )
        )

        if moving_type == 2:
            raise ValueError(
                f"bench_fast_jump moving_type={moving_type} lid={lid} pos={pos} dir_idx={dir_idx_i}"
            )

        old_pos = (vi, vj, vk)
        new_pos = (ni, nj, nk)
        r_selected = float(self.diffusion_rates[int(lid), int(dir_idx_i)])

        (
            updated_cu,
            updated_vacancy,
            cu_move_from,
            cu_move_to,
            cu_id,
            cu_topk_id,
            topk_update_info,
        ) = self._update_pipeline(int(lid), old_pos, new_pos, int(moving_type))
        self._set_backjump_forbid_after_move(int(lid), int(dir_idx_i))
        self.diffusion_rates_update([old_pos, new_pos])

        kB = 8.617e-5
        T = self.args.temperature
        nu = 1e13
        delta_val = (
            float(-kB * T * np.log(r_selected / nu)) - self.E_a0_t[moving_type]
        ) * 2

        return KMCObs(
            topk_update_info=topk_update_info,
            updated_cu=updated_cu,
            updated_vacancy=updated_vacancy,
            cu_move_from=cu_move_from,
            cu_move_to=cu_move_to,
            cu_id=cu_id,
            cu_topk_id=cu_topk_id,
            vac_id=int(lid),
            changed_vids=self._changed_vids_last,
            dir_idx=int(dir_idx_i),
            energy_change=float(delta_val),
        )

    def _center_bond_energy_sum(self, coord: tuple) -> float:
        dims = self.D_np
        NN1 = self.NN1_np
        NN2 = self.NN2_np
        pe = self.pair_energies
        pair1 = pe[0]
        pair2 = pe[1]
        c = np.array(coord, dtype=int).reshape(1, 3)
        nn1_coords = self._get_pbc_coord(c[:, None, :], NN1[None, :, :], dims).reshape(
            8, 3
        )
        nn2_coords = self._get_pbc_coord(c[:, None, :], NN2[None, :, :], dims).reshape(
            6, 3
        )
        stack = np.vstack([nn1_coords, nn2_coords])
        types_flat = self._batch_get_type_from_local_coords(stack)
        nn1_types = types_flat[:8]
        nn2_types = types_flat[8:]
        center_type = int(
            self._get_type_from_coord(np.asarray(tuple(map(int, coord)), dtype=int))
        )
        E1 = np.sum(pair1[center_type, nn1_types])
        E2 = np.sum(pair2[center_type, nn2_types])
        return float(E1 + E2)

    def move_vacancy(self, old_pos: tuple, new_pos: tuple, used_torch=True):
        """更新 vacancy 位置并维护 vacancy 映射表与 global linear cache。

        global linear cache 维护两份：
        - numpy: `_global_vac_lin_sorted`
        - torch: `_global_vac_lin_sorted_t`
        这里优先尝试 torch 增量更新（便于后续 GPU 上 searchsorted），失败再回退到 numpy。
        """
        local_old = tuple(map(int, old_pos))
        local_new = tuple(map(int, new_pos))

        self.vac_pos_set.discard(local_old)
        self.vac_pos_set.add(local_new)
        idx = self.v_pos_to_id.pop(local_old)
        self.v_pos_to_id[local_new] = idx
        self.v_pos_of_id[idx] = local_new
        try:
            D = self.D_np
            ov0, ov1, ov2 = local_old
            nv0, nv1, nv2 = local_new
            old_lin = int(((int(ov0) * int(D[1]) + int(ov1)) * int(D[2]) + int(ov2)))
            new_lin = int(((int(nv0) * int(D[1]) + int(nv1)) * int(D[2]) + int(nv2)))

            prefer_torch = bool(used_torch)
            did_torch_update = False
            if prefer_torch:
                arr_t = getattr(self, "_global_vac_lin_sorted_t", None)
                if isinstance(arr_t, torch.Tensor):
                    try:
                        arr_t = arr_t.view(-1)
                        dev = arr_t.device
                        old_val_t = torch.as_tensor(
                            old_lin, dtype=torch.int64, device=dev
                        )
                        new_val_t = torch.as_tensor(
                            (new_lin,), dtype=torch.int64, device=dev
                        )
                        n = int(arr_t.numel())
                        if n > 0:
                            i = int(torch.searchsorted(arr_t, old_val_t).item())
                            if 0 <= i < n and int(arr_t[i].item()) == int(old_lin):
                                arr_t = torch.cat((arr_t[:i], arr_t[i + 1 :]), dim=0)
                        j = int(torch.searchsorted(arr_t, new_val_t[0]).item())
                        arr_t = torch.cat((arr_t[:j], new_val_t, arr_t[j:]), dim=0)
                        self._global_vac_lin_sorted_t = arr_t
                        did_torch_update = True
                    except Exception:
                        did_torch_update = False

            if did_torch_update and hasattr(self, "_global_vac_lin_sorted"):
                arr = self._global_vac_lin_sorted
                if isinstance(arr, np.ndarray):
                    if arr.size > 0:
                        i = int(np.searchsorted(arr, old_lin))
                        if i < arr.size and int(arr[i]) == int(old_lin):
                            arr = np.delete(arr, i)
                    j = int(np.searchsorted(arr, new_lin))
                    self._global_vac_lin_sorted = np.insert(arr, j, new_lin).astype(
                        np.int64
                    )

            if (not did_torch_update) and hasattr(self, "_global_vac_lin_sorted"):
                arr = self._global_vac_lin_sorted
                if arr.size > 0:
                    i = int(np.searchsorted(arr, old_lin))
                    if i < arr.size and arr[i] == old_lin:
                        arr = np.delete(arr, i)
                j = int(np.searchsorted(arr, new_lin))
                self._global_vac_lin_sorted = np.insert(arr, j, new_lin).astype(
                    np.int64
                )
                # if torch.cuda.is_available():
                self._global_vac_lin_sorted_t = torch.as_tensor(
                    self._global_vac_lin_sorted, dtype=torch.int64, device=self.device
                )
        except Exception:
            try:
                self._rebuild_global_lin_cache()
            except Exception:
                pass

    # 兼容旧名
    def step_only_jump(self, action: int, episode: int, verbose: bool = False):
        vac_id, dir_idx = divmod(int(action), 8)
        return self.step_fast_local(int(vac_id), int(dir_idx), int(episode))

    def calculate_diffusion_rate(self):
        """全量计算 diffusion_rates（numpy）。

        输出形状约定：`(Nv, 8)`，Nv 为 vacancy 数量。
        值为尝试频率 * exp(-E/(kB*T))。
        """
        # Nv_local = int(self.nn1_types.shape[0]) if hasattr(self, 'nn1_types') and isinstance(self.nn1_types, np.ndarray) else 0
        num_vacancies = self.nn1_types.shape[0]
        if num_vacancies == 0:
            return np.empty((0, 8), dtype=float)
        energies = self._batch_vacancy_diffusion_energy(None)
        kB = 8.617e-5
        T = self.args.temperature
        rates = 1e13 * np.exp(-energies / (kB * T))
        return rates

    def _batch_vacancy_diffusion_energy(self, vacancies_index=None):
        return KMCEnv._batch_vacancy_diffusion_energy(self, vacancies_index)

    def diffusion_rates_update(self, changed_positions: list[tuple[int, int, int]]):
        """对发生变化的坐标集合做扩散率增量更新。

        输入 `changed_positions` 通常包含 old_pos/new_pos。
        该函数会：
        - 找到受影响的 vacancy ids（primary/secondary 并集）。
        - 批量重算这些 vacancy 的扩散能垒与速率。
        - 同步更新 numpy 与 torch 两份 diffusion rate 缓存。
        """
        # t7 = time.time()

        # vacancies = self.get_vacancy_array()
        # t8 = time.time()
        # print(f"[rank {self.rank}] diffusion_rates_update t8-t7: {t8 - t7}")

        if (
            len(self.vac_pos_set) == 0
            or changed_positions is None
            or len(changed_positions) == 0
        ):
            return

        # try:
        affected_primary, affected_secondary = self._get_local_affected_vac_ids(
            changed_positions
        )
        affected_all = sorted(list(set(affected_primary) | set(affected_secondary)))
        # except Exception:
        # affected_all = []
        if len(affected_all) == 0:
            return
        idx_arr = np.array(affected_all, dtype=int)
        # t_batch0 = time.time()
        new_energies = self._batch_vacancy_diffusion_energy(idx_arr)
        # t_batch1 = time.time()
        # _LOGGER.debug(
        #     "diffusion_rates_update rank %s computed batch energies for %s vacancies in %s seconds",
        #     int(getattr(self, "rank", -1)),
        #     int(len(idx_arr)),
        #     float(t_batch1 - t_batch0),
        # )
        kB = 8.617e-5
        T = self.args.temperature
        nu = 1e13
        new_rates = nu * np.exp(-new_energies / (kB * T))

        # 打印 self diffusion_rates 的信息
        # self._log_diffusion_cache_state()

        # 如果已有 numpy 缓存且 torch 缓存有效，则增量更新；否则整体重新计算并拷贝
        if (
            hasattr(self, "diffusion_rates")
            and isinstance(self.diffusion_rates, np.ndarray)
            and hasattr(self, "diffusion_rates_t")
            and isinstance(self.diffusion_rates_t, torch.Tensor)
        ):
            # _LOGGER.debug(
            #     "has both numpy and torch diffusion rate caches, updating incrementally"
            # )
            # 更新 numpy 缓存
            self.diffusion_rates[idx_arr, :] = new_rates
            # 同步 torch 缓存
            idx_t = torch.as_tensor(idx_arr, dtype=torch.long, device=self.device)
            self.diffusion_rates_t[idx_t, :] = torch.as_tensor(
                new_rates, dtype=torch.float32, device=self.device
            )
        else:
            # _LOGGER.debug("recomputing full diffusion rate caches")
            time_start = time.time()
            self.diffusion_rates = self.calculate_diffusion_rate()
            self.diffusion_rates_t = torch.as_tensor(
                self.diffusion_rates, dtype=torch.float32, device=self.device
            )
            time_end = time.time()
            # _LOGGER.debug(
            #     "recomputed full diffusion rate caches in %s seconds",
            #     float(time_end - time_start),
            # )
            # except Exception:
            # self.diffusion_rates_t = None

    def compute_local_total_energy(self) -> float:
        dims = self.D_np
        nx, ny, nz = dims.tolist()
        nx = nx // 2
        ny = ny // 2
        nz = nz // 2
        # print(nx, ny, nz)

        if nx <= 0 or ny <= 0 or nz <= 0:
            return 0.0
        NN1 = self.NN1_np
        NN2 = self.NN2_np
        pe = self.pair_energies
        try:
            pair1 = pe[0]
            pair2 = pe[1]
        except Exception:
            pair1, pair2 = pe
        total = 0.0
        Z_CHUNK = max(1, int(min(nz, 64)))
        # for z0 in range(0, nz, Z_CHUNK):
        #     z1 = min(z0 + Z_CHUNK, nz)
        #     xs = np.arange(nx, dtype=int)
        #     ys = np.arange(ny, dtype=int)
        #     zs = np.arange(z0, z1, dtype=int)
        #     X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        #     coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(int)
        #     if coords.size == 0:
        #         continue
        for z0 in range(0, nz, Z_CHUNK):
            z1 = min(z0 + Z_CHUNK, nz)

            # 1. 生成当前块的基础索引 (类似 np.indices 的分块版本)
            xs = np.arange(nx, dtype=int)
            ys = np.arange(ny, dtype=int)
            zs = np.arange(z0, z1, dtype=int)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

            # 将索引展平为 (N, 3)
            base_indices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(int)

            # 2. 应用你的逻辑：生成 Corner 和 Body 坐标
            corner = base_indices * 2
            body = corner + 1

            # 3. 合并两组坐标
            coords = np.vstack((corner, body)).astype(int)

            # 后续计算逻辑不变...
            if coords.size == 0:
                continue
            # ... 进行能量计算 ...
            center_types = self._batch_get_type_from_coords(coords)
            V_nn1_coords = self._get_pbc_coord(
                coords[:, None, :], NN1[None, :, :], dims
            )
            V_nn2_coords = self._get_pbc_coord(
                coords[:, None, :], NN2[None, :, :], dims
            )
            stack_neighbors = np.vstack(
                [V_nn1_coords.reshape(-1, 3), V_nn2_coords.reshape(-1, 3)]
            )
            types_neigh = self._batch_get_type_from_coords(stack_neighbors)
            n = int(coords.shape[0])
            nn1_types = types_neigh[: n * 8].reshape(n, 8)
            nn2_types = types_neigh[n * 8 :].reshape(n, 6)
            E1 = pair1[center_types[:, None], nn1_types].sum(axis=1)
            E2 = pair2[center_types[:, None], nn2_types].sum(axis=1)
            total += float(np.sum(E1 + E2))
        return float(total / 2.0)
