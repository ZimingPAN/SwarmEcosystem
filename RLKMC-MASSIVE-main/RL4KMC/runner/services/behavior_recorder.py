from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from RL4KMC.envs.distributed_kmc import KMCObs
from RL4KMC.utils.util import SSAEvalFileLogger


class BehaviorRecorder:
    """Behavior/SSA recording helper with injected env access."""

    def __init__(
        self,
        rank: int,
        enable_behavior_log: bool,
        energy_series_interval_steps: int,
        log_global_cb=None,
    ) -> None:
        self.rank = int(rank)
        self.enable_behavior_log = bool(enable_behavior_log)
        self.energy_series_interval_steps = int(energy_series_interval_steps)
        self._log_global_cb = log_global_cb
        self._ssa_file_logger: Optional[SSAEvalFileLogger] = None
        self._cluster_fp: Optional[str] = None
        self._cluster_series: list[dict] = []

    def update_rank(self, rank: int) -> None:
        self.rank = int(rank)

    def update_enable_behavior_log(self, enable_behavior_log: bool) -> None:
        self.enable_behavior_log = bool(enable_behavior_log)

    def reset_cluster_series(self) -> None:
        self._cluster_series = []

    def init_logger(self, env: Any, output_paths, sim_dir: Optional[str]) -> None:
        self.reset_cluster_series()
        if not self.enable_behavior_log or output_paths is None:
            self._ssa_file_logger = None
            return
        num_v = int(env.get_vacancy_array().shape[0])
        num_c = int(env.get_cu_array().shape[0])
        self._cluster_fp = os.path.join(
            os.path.dirname(output_paths.cu_moves_fp),
            f"cluster_V{int(num_v)}_C{int(num_c)}.csv",
        )
        self._ssa_file_logger = SSAEvalFileLogger(
            sim_dir=sim_dir,
            worker_id=int(self.rank),
            temperature=float(env.args.temperature),
            num_v=int(num_v),
            num_c=int(num_c),
            cluster_fp=self._cluster_fp,
            cu_moves_fp=output_paths.cu_moves_fp,
            energy_drops_fp=output_paths.energy_drops_fp,
            energy_series_fp=output_paths.energy_series_fp,
        )
        self._ssa_file_logger.init_files()

    def record_cluster_sample(self, env: Any, t: float) -> None:
        try:
            stats = env.get_system_stats()
            cu_cv = (
                float(stats[6])
                if (isinstance(stats, np.ndarray) and stats.size > 6)
                else float("nan")
            )
            iso_frac = float(env.get_cu_isolated_fraction())
        except Exception:
            cu_cv = float("nan")
            iso_frac = float("nan")
        if callable(self._log_global_cb):
            try:
                self._log_global_cb(
                    {
                        "event": "cu_cluster_sample",
                        "t": float(t),
                        "cu_cv": float(cu_cv),
                        "temperature": float(env.args.temperature),
                        "worker_id": int(self.rank),
                    }
                )
            except Exception:
                pass
        if self._ssa_file_logger is not None:
            self._ssa_file_logger.write_cluster_row(
                float(t), float(cu_cv), float(iso_frac)
            )
        self._cluster_series.append(
            {"t": float(t), "cu_cv": float(cu_cv), "iso_frac": float(iso_frac)}
        )

    def record_energy_series_if_needed(
        self, env: Any, step_id: int, total_energy: float
    ) -> None:
        if (
            self._ssa_file_logger is not None
            and (self.energy_series_interval_steps > 0)
            and (int(step_id) % int(self.energy_series_interval_steps) == 0)
        ):
            self._ssa_file_logger.write_energy_series_row(
                int(step_id), float(env.time), float(total_energy)
            )

    def write_initial_energy_series(self, env: Any, total_energy: float) -> None:
        if self._ssa_file_logger is not None:
            self._ssa_file_logger.write_energy_series_row(
                0, float(env.time), float(total_energy)
            )

    def record_move_if_needed(self, env: Any, obs: KMCObs) -> None:
        if self._ssa_file_logger is None:
            return
        if not isinstance(obs, KMCObs):
            raise TypeError(f"expected KMCObs, got {type(obs)}")
        vac_id_obj = obs.vac_id
        dir_idx_obj = obs.dir_idx
        energy_obj = obs.energy_change
        local_id = int(vac_id_obj) if vac_id_obj is not None else -1
        dir_idx = int(dir_idx_obj) if dir_idx_obj is not None else -1
        delta_val = float(energy_obj) if energy_obj is not None else float("nan")
        cf = obs.cu_move_from
        ct = obs.cu_move_to
        if isinstance(cf, (list, tuple)) and isinstance(ct, (list, tuple)):
            self._ssa_file_logger.write_energy_drop_row(
                float(env.time),
                int(local_id),
                int(dir_idx),
                float(delta_val),
                ct,
                cf,
            )
            gid = obs.cu_id
            tid = obs.cu_topk_id
            tnow = float(env.time)
            self._ssa_file_logger.write_cu_move_row(
                tnow, gid, tid, cf, ct, int(local_id), int(dir_idx)
            )

    def write_advancement(self) -> None:
        if self._ssa_file_logger is not None:
            self._ssa_file_logger.write_advancement(self._cluster_series)

    def plot_energy_series(self) -> None:
        if self._ssa_file_logger is not None:
            self._ssa_file_logger.plot_energy_series()

    @property
    def cluster_series(self) -> list[dict]:
        return self._cluster_series
