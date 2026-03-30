from __future__ import annotations

import logging
import os
import random
import time
import math
import json
from typing import Any, Dict, List, Optional, Tuple
from RL4KMC.utils.json_sanitize import sanitize_for_json
from RL4KMC.envs.distributed_kmc import DistributedKMCEnv
import numpy as np


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.addHandler(logging.NullHandler())


def calculate_vacancy_concentration(
    temperature_k: float, energy_formation_ev: float = 1.6
) -> float:
    """Compute equilibrium vacancy concentration Cv,real.

    k is Boltzmann constant in eV/K.
    """

    k = 8.617333262145e-5
    exponent = -float(energy_formation_ev) / (k * float(temperature_k))
    return float(math.exp(exponent))


def calculate_num_tasks(n_radial: int, n_axial: int) -> int:
    return n_radial * n_axial


def generate_kmc_tasks(
    n_radial: int,
    n_axial: int,
    rescaled_sim_time: float,
    cu_density: float,
    v_density: float,
) -> List[Dict[str, float]]:
    """Generate per-(radial, axial) KMC tasks.

    This function is intentionally pure (no MPI, no torch, no filesystem).
    """

    def generate_aging_temps(
        n_r: int,
        n_z: int,
        base_temp: float = 290.0,
        gradient: float = 20.0,
    ) -> Tuple[np.ndarray, List[float], List[float]]:
        axial_variation = 5.0

        r_fractions = [(i + 0.5) / n_r for i in range(n_r)]
        z_fractions = [(j + 0.5) / n_z for j in range(n_z)]

        results = []
        for j in range(n_z):
            z_frac = (j + 0.5) / n_z
            t_current_z = base_temp + (z_frac * axial_variation)

            row = []
            for i in range(n_r):
                r_frac = (i + 0.5) / n_r
                t_point = t_current_z - (r_frac * gradient)
                row.append(round(t_point, 2))
            results.append(row)
        return np.array(results), r_fractions, z_fractions

    temp_matrix, _, _ = generate_aging_temps(int(n_radial), int(n_axial))

    temp_num = int(n_radial) * int(n_axial)
    tasks: List[Dict[str, float]] = []

    temperatures = [273.0 + float(temp) for temp in temp_matrix.flatten()]
    cu_densities = [float(cu_density)] * int(temp_num)

    real_vacancy_densities = [calculate_vacancy_concentration(t) for t in temperatures]
    real_sim_times = [float(rescaled_sim_time)] * int(temp_num)
    vacancy_densities = [float(v_density)] * int(temp_num)

    simulation_times = [
        real_sim_times[i] * real_vacancy_densities[i] / vacancy_densities[i]
        for i in range(int(temp_num))
    ]

    # order = sorted(
    #     range(int(temp_num)), key=lambda i: float(simulation_times[i]), reverse=True
    # )

    # for idx in order:
    #     tasks.append(
    #         {
    #             "temp": float(temperatures[idx]),
    #             "cu_density": float(cu_densities[idx]),
    #             "v_density": float(vacancy_densities[idx]),
    #             "time": float(simulation_times[idx]),
    #         }
    #     )

    # for i in range(int(temp_num)):
    #     tasks.append(
    #         {
    #             "temp": float(temperatures[i]),
    #             "cu_density": float(cu_densities[i]),
    #             "v_density": float(vacancy_densities[i]),
    #             "time": float(simulation_times[i]),
    #         }
    #     )
    rng = random.Random(42)
    idx = list(range(int(temp_num)))
    rng.shuffle(idx)
    for i in idx:
        tasks.append(
            {
                "temp": float(temperatures[i]),
                "cu_density": float(cu_densities[i]),
                "v_density": float(vacancy_densities[i]),
                "time": float(simulation_times[i]),
            }
        )


    return tasks


def reinit_env_for_task(
    task: Dict[str, float],
    args: Any,
    topk_device: Any,
) -> Tuple[Any, Any, Any]:
    # task 字典来自任务分发：正常情况下必须包含 temp/cu_density/v_density。
    # 这里用局部变量避免重复 task.get(...) 造成的类型收窄失效，并在缺关键字段时给出明确错误。
    temp_raw = task.get("temp")
    if temp_raw is None:
        raise ValueError(f"Task missing required field 'temp': {task}")
    assigned_temp = float(temp_raw)

    cu_raw = task.get("cu_density")
    assigned_cu_density = float(cu_raw) if (cu_raw is not None) else None

    v_raw = task.get("v_density")
    assigned_v_density = float(v_raw) if (v_raw is not None) else None
    args.temperature = float(assigned_temp)
    nx, ny, nz = tuple(getattr(args, "lattice_size", (0, 0, 0)))
    total_half_sites = int(nx) * int(ny) * int(nz) * 2
    cu_nums = (
        int(round(float(assigned_cu_density) * total_half_sites))
        if (assigned_cu_density is not None)
        else int(getattr(args, "lattice_cu_nums", 0))
    )
    v_nums = (
        int(round(float(assigned_v_density) * total_half_sites))
        if (assigned_v_density is not None)
        else int(getattr(args, "lattice_v_nums", 0))
    )
    if assigned_cu_density is not None:
        args.cu_density = float(assigned_cu_density)
        args.lattice_cu_nums = int(cu_nums)
    if assigned_v_density is not None:
        args.v_density = float(assigned_v_density)
        args.lattice_v_nums = int(v_nums)
    env = DistributedKMCEnv(args)
    topk_all = env.topk_sys.get_all_topk_tensors()
    # cache_dev_raw = env_str(EnvKeys.TOPK_CACHE_DEVICE, None)
    # if cache_dev_raw is not None and str(cache_dev_raw).strip() != "":
    #     cache_device = cache_dev_raw.strip()
    # else:
    #     cache_device = topk_device
    # diff_k_cache = topk_all["diff_k"].to(cache_device)
    # dist_k_cache = topk_all["dist_k"].to(cache_device)
    diff_k_cache = topk_all["diff_k"].to(topk_device)
    dist_k_cache = topk_all["dist_k"].to(topk_device)
    return env, diff_k_cache, dist_k_cache


def record_task_time(
    output_dir: str,
    rank_dir_name: str,
    assigned_temp: Optional[float],
    assigned_cu: Optional[float],
    assigned_vac: Optional[float],
    sim_time: float,
    start_ts: float,
    end_ts: float,
) -> None:
    rank_dir = os.path.join(output_dir, "rank-detail", str(rank_dir_name))
    os.makedirs(rank_dir, exist_ok=True)
    fp = os.path.join(rank_dir, "task_times.csv")
    if not os.path.isfile(fp):
        with open(fp, "w") as f:
            f.write(
                "worker,temp,cu_density,v_density,sim_time,start_ts,end_ts,duration\n"
            )
    with open(fp, "a") as f:
        f.write(
            f"{str(rank_dir_name)},{'' if assigned_temp is None else float(assigned_temp)},{'' if assigned_cu is None else float(assigned_cu)},{'' if assigned_vac is None else float(assigned_vac)},{float(sim_time)},{float(start_ts)},{float(end_ts)},{float(end_ts - start_ts)}\n"
        )


def aggregate_task_times(
    output_dir: str,
    start_global_ts: float,
    end_global_ts: float,
) -> None:

    agg_fp = os.path.join(output_dir, "task_times.csv")

    rank_detail_dir = os.path.join(output_dir, "rank-detail")
    try:
        worker_names = sorted(
            [
                d
                for d in os.listdir(rank_detail_dir)
                if os.path.isdir(os.path.join(rank_detail_dir, d))
            ]
        )
    except Exception:
        worker_names = []

    worker_sum = {str(name): 0.0 for name in worker_names}
    worker_cnt = {str(name): 0 for name in worker_names}
    with open(agg_fp, "w") as f:
        f.write("worker,temp,cu_density,v_density,sim_time,start_ts,end_ts,duration\n")
        for name in worker_names:
            rfp = os.path.join(rank_detail_dir, str(name), "task_times.csv")
            if os.path.isfile(rfp):
                with open(rfp, "r") as rf:
                    next(rf, None)
                    for line in rf:
                        f.write(line)
                        parts = line.strip().split(",")
                        if len(parts) >= 8:
                            try:
                                worker = str(parts[0])
                                worker_sum[worker] = float(
                                    worker_sum.get(worker, 0.0)
                                ) + float(parts[7])
                                worker_cnt[worker] = int(worker_cnt.get(worker, 0)) + 1
                            except Exception:
                                pass
        f.write(f"total_wall_time_seconds,{float(end_global_ts - start_global_ts)}\n")

    # load balance report
    totals_fp = os.path.join(output_dir, "load_balance_report.csv")
    details = {}
    with open(totals_fp, "w") as f:
        f.write(
            "worker,task_count,total_duration,mean_duration,min_duration,max_duration,var_duration\n"
        )
        for name in worker_names:
            durations = []
            rfp = os.path.join(rank_detail_dir, str(name), "task_times.csv")
            if os.path.isfile(rfp):
                try:
                    with open(rfp, "r") as rf:
                        next(rf, None)
                        for line in rf:
                            parts = line.strip().split(",")
                            if len(parts) >= 8:
                                try:
                                    durations.append(float(parts[7]))
                                except Exception:
                                    pass
                except Exception:
                    pass
            if durations:
                cnt = len(durations)
                total = float(sum(durations))
                mean = float(total / cnt)
                vmin = float(min(durations))
                vmax = float(max(durations))
                var = float(sum((x - mean) ** 2 for x in durations) / cnt)
            else:
                cnt = 0
                total = 0.0
                mean = float("nan")
                vmin = float("nan")
                vmax = float("nan")
                var = float("nan")
            details[str(name)] = {
                "task_count": cnt,
                "total_duration": total,
                "mean_duration": mean,
                "min_duration": vmin,
                "max_duration": vmax,
                "var_duration": var,
            }
            f.write(f"{str(name)},{cnt},{total},{mean},{vmin},{vmax},{var}\n")

    try:
        json_fp = os.path.join(output_dir, "load_balance_report.json")
        with open(json_fp, "w") as f:
            json.dump(sanitize_for_json(details), f, indent=2, allow_nan=False)
    except Exception:
        pass


def write_load_balance_report(
    output_dir: str, details: dict[str, dict[str, Any]]
) -> None:
    """Write load balance report files to output_dir.

    This is a lightweight alternative to aggregate_task_times() when per-task
    timing CSVs are not produced (e.g. output_level < 2).

    Expected details schema per worker key:
      - task_count
      - total_duration
      - mean_duration
      - min_duration
      - max_duration
      - var_duration
    """

    if output_dir is None or str(output_dir).strip() == "":
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        return

    # CSV
    try:
        totals_fp = os.path.join(output_dir, "load_balance_report.csv")
        with open(totals_fp, "w") as f:
            f.write(
                "worker,task_count,total_duration,mean_duration,min_duration,max_duration,var_duration\n"
            )
            for worker in sorted(details.keys(), key=lambda x: str(x)):
                d = details.get(worker, {}) or {}
                f.write(
                    f"{str(worker)},"
                    f"{int(d.get('task_count', 0) or 0)},"
                    f"{float(d.get('total_duration', 0.0) or 0.0)},"
                    f"{float(d.get('mean_duration', float('nan')))},"
                    f"{float(d.get('min_duration', float('nan')))},"
                    f"{float(d.get('max_duration', float('nan')))},"
                    f"{float(d.get('var_duration', float('nan')))}\n"
                )
    except Exception:
        pass

    # JSON
    try:
        json_fp = os.path.join(output_dir, "load_balance_report.json")
        with open(json_fp, "w") as f:
            json.dump(sanitize_for_json(details), f, indent=2, allow_nan=False)
    except Exception:
        pass


def _weighted_median(pairs: list[tuple[float, float]]) -> float | None:
    clean: list[tuple[float, float]] = []
    for v, w in pairs or []:
        try:
            fv = float(v)
            fw = float(w)
        except Exception:
            continue
        if not (fw > 0.0):
            continue
        if not math.isfinite(fv):
            continue
        clean.append((fv, fw))
    if not clean:
        return None
    clean.sort(key=lambda x: x[0])
    total = float(sum(w for _, w in clean))
    if not (total > 0.0):
        return None
    half = 0.5 * total
    cum = 0.0
    for v, w in clean:
        cum += float(w)
        if cum >= half:
            return float(v)
    return float(clean[-1][0])


def merge_load_balance_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-worker/per-node load-balance stats into a single dict.

    Expected keys (best-effort):
      - task_count
      - total_duration
      - mean_duration
      - min_duration
      - max_duration
      - m2_duration (preferred) OR var_duration
      - median_duration (approx)
    """

    acc_count = 0
    acc_mean = 0.0
    acc_m2 = 0.0
    acc_total = 0.0
    acc_min = None
    acc_max = None
    med_pairs: list[tuple[float, float]] = []

    for d in items or []:
        if not isinstance(d, dict):
            continue
        try:
            c = int(d.get("task_count", 0) or 0)
        except Exception:
            c = 0
        if c <= 0:
            continue

        try:
            total = float(d.get("total_duration", 0.0) or 0.0)
        except Exception:
            total = 0.0
        try:
            mean = float(d.get("mean_duration", float("nan")))
        except Exception:
            mean = float("nan")

        # Prefer exact M2 if provided.
        m2 = None
        try:
            m2_raw = d.get("m2_duration", None)
            if m2_raw is not None:
                m2 = float(m2_raw)
        except Exception:
            m2 = None
        if m2 is None:
            try:
                var = float(d.get("var_duration", float("nan")))
                m2 = float(var) * float(c) if math.isfinite(var) else 0.0
            except Exception:
                m2 = 0.0

        try:
            vmin = d.get("min_duration", None)
            vmin = float(vmin) if vmin is not None else None
        except Exception:
            vmin = None
        try:
            vmax = d.get("max_duration", None)
            vmax = float(vmax) if vmax is not None else None
        except Exception:
            vmax = None

        # Median: approximate weighted median of medians.
        try:
            med = d.get("median_duration", None)
            med = float(med) if med is not None else None
            if med is not None and math.isfinite(med):
                med_pairs.append((float(med), float(c)))
        except Exception:
            pass

        if acc_count <= 0:
            acc_count = int(c)
            acc_total = float(total)
            acc_mean = float(mean) if math.isfinite(mean) else (float(total) / float(c))
            acc_m2 = float(m2)
            acc_min = vmin
            acc_max = vmax
        else:
            # Parallel merge of (count, mean, m2).
            n_a = float(acc_count)
            n_b = float(c)
            mean_a = float(acc_mean)
            mean_b = float(mean) if math.isfinite(mean) else (float(total) / float(c))
            delta = float(mean_b - mean_a)
            n = float(n_a + n_b)
            if n > 0:
                acc_mean = float(mean_a + delta * (n_b / n))
                acc_m2 = float(acc_m2 + float(m2) + delta * delta * (n_a * n_b / n))
            acc_count = int(acc_count + int(c))
            acc_total = float(acc_total + float(total))
            if vmin is not None:
                acc_min = (
                    vmin if acc_min is None else float(min(float(acc_min), float(vmin)))
                )
            if vmax is not None:
                acc_max = (
                    vmax if acc_max is None else float(max(float(acc_max), float(vmax)))
                )

    if acc_count <= 0:
        return {
            "task_count": 0,
            "total_duration": 0.0,
            "mean_duration": float("nan"),
            "min_duration": float("nan"),
            "max_duration": float("nan"),
            "median_duration": float("nan"),
            "var_duration": float("nan"),
            "m2_duration": 0.0,
        }

    var = float(acc_m2) / float(acc_count) if acc_count > 0 else float("nan")
    med = _weighted_median(med_pairs)
    return {
        "task_count": int(acc_count),
        "total_duration": float(acc_total),
        "mean_duration": (
            float(acc_total / float(acc_count)) if acc_count > 0 else float("nan")
        ),
        "min_duration": float(acc_min) if acc_min is not None else float("nan"),
        "max_duration": float(acc_max) if acc_max is not None else float("nan"),
        "median_duration": float(med) if med is not None else float("nan"),
        "var_duration": float(var),
        "m2_duration": float(acc_m2),
    }


def write_node_load_balance_report(
    *,
    rank_dir: str,
    node_summary: dict[str, Any],
    worker_details: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write a node-local load-balance report under a rank-detail directory."""

    if rank_dir is None or str(rank_dir).strip() == "":
        return
    try:
        os.makedirs(rank_dir, exist_ok=True)
    except Exception:
        return

    try:
        json_fp = os.path.join(rank_dir, "load_balance_report.json")
        payload = {
            "scope": "node",
            "node": dict(node_summary or {}),
        }
        if worker_details is not None:
            payload["workers"] = dict(worker_details)
        with open(json_fp, "w") as f:
            json.dump(sanitize_for_json(payload), f, indent=2, allow_nan=False)
    except Exception:
        pass

    try:
        csv_fp = os.path.join(rank_dir, "load_balance_report.csv")
        with open(csv_fp, "w") as f:
            f.write(
                "rank,host,rank_dir_name,task_count,total_duration,mean_duration,min_duration,max_duration,median_duration,var_duration\n"
            )
            f.write(
                f"{node_summary.get('rank','')},{node_summary.get('host','')},{node_summary.get('rank_dir_name','')},"
                f"{node_summary.get('task_count','')},{node_summary.get('total_duration','')},{node_summary.get('mean_duration','')},"
                f"{node_summary.get('min_duration','')},{node_summary.get('max_duration','')},{node_summary.get('median_duration','')},{node_summary.get('var_duration','')}\n"
            )
    except Exception:
        pass


def write_global_load_balance_report(
    *,
    output_dir: str,
    nodes: list[dict[str, Any]],
) -> None:
    """Write a single global load-balance report at output_dir (rank0 only)."""

    if output_dir is None or str(output_dir).strip() == "":
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        return

    # Global aggregate across nodes.
    try:
        global_stats = merge_load_balance_stats(
            [n for n in (nodes or []) if isinstance(n, dict)]
        )
    except Exception:
        global_stats = {}

    try:
        json_fp = os.path.join(output_dir, "load_balance_report.json")
        payload = {
            "scope": "global",
            "global": dict(global_stats or {}),
            "nodes": list(nodes or []),
        }
        with open(json_fp, "w") as f:
            json.dump(sanitize_for_json(payload), f, indent=2, allow_nan=False)
    except Exception:
        pass

    try:
        csv_fp = os.path.join(output_dir, "load_balance_report.csv")
        with open(csv_fp, "w") as f:
            f.write(
                "rank,host,rank_dir_name,task_count,total_duration,mean_duration,min_duration,max_duration,median_duration,var_duration\n"
            )
            for n in sorted(
                list(nodes or []),
                key=lambda x: int(x.get("rank", 0) or 0) if isinstance(x, dict) else 0,
            ):
                if not isinstance(n, dict):
                    continue
                f.write(
                    f"{n.get('rank','')},{n.get('host','')},{n.get('rank_dir_name','')},"
                    f"{n.get('task_count','')},{n.get('total_duration','')},{n.get('mean_duration','')},"
                    f"{n.get('min_duration','')},{n.get('max_duration','')},{n.get('median_duration','')},{n.get('var_duration','')}\n"
                )
    except Exception:
        pass
