from __future__ import annotations

import logging
import json
import os
import socket
import time
import numpy as np
import math
from typing import Any

from RL4KMC.utils.env import EnvKeys, env_float, env_has, env_int, env_str
from RL4KMC.utils.json_sanitize import sanitize_for_json


from RL4KMC.runner.engine import aggregate_task_times
from RL4KMC.diagnostics.timing_report import (
    write_overall_time_report,
    write_timing_reports,
)


_LOGGER = logging.getLogger(__name__)


_MPI_MODULE: Any | None = None


def _get_mpi() -> Any:
    global _MPI_MODULE
    if _MPI_MODULE is not None:
        return _MPI_MODULE

    import mpi4py  # type: ignore

    try:
        mpi4py.rc.initialize = False  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        mpi4py.rc.finalize = False  # type: ignore[attr-defined]
    except Exception:
        pass

    from mpi4py import MPI  # type: ignore

    _MPI_MODULE = MPI
    return _MPI_MODULE


class TimingReporter:
    def __init__(
        self,
        *,
        output_dir: str | None,
        rank_dir: str | None = None,
        rank_dir_name: str | None = None,
        timing_stats: Any,
    ) -> None:
        self.output_dir = output_dir
        self.rank_dir = rank_dir
        self.rank_dir_name = str(rank_dir_name) if rank_dir_name is not None else None
        self.timing_stats = timing_stats

    def _get_meta(self, key: str, default: Any = None) -> Any:
        try:
            return self.timing_stats.get_metadata(key, default)
        except Exception:
            pass
        try:
            meta = self.timing_stats.metadata
        except Exception:
            meta = {}
        try:
            return meta.get(str(key), default)
        except Exception:
            return default

    def _compute_series_stats(self, values):
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        # Keep a merge-friendly representation for hierarchical aggregation.
        return {
            "count": int(arr.size),
            "sum": float(arr.sum()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            # Median is only used for an approximate weighted-median merge.
            "median": float(np.median(arr)),
        }

    def _weighted_median(self, pairs):
        """Compute weighted median from (value, weight) pairs.

        Note: this is an approximation strategy when merging per-rank medians.
        """

        clean = []
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

    def _merge_step_stats(self, payloads, *, output_level: int):
        """Merge per-label timing stats across payloads.

        Each payload may contain a 'step_timing_stats' dict where each label has:
        {count,sum,min,max,median}. We merge sum/count/min/max exactly, and merge
        median as a weighted median of medians (approx).
        """

        if int(output_level) < 1:
            return None
        acc: dict[str, dict[str, Any]] = {}
        for p in payloads or []:
            step = (p or {}).get("step_timing_stats") or {}
            if not isinstance(step, dict):
                continue
            for label, st in step.items():
                if not isinstance(st, dict):
                    continue
                try:
                    c = int(st.get("count", 0) or 0)
                except Exception:
                    c = 0
                if c <= 0:
                    continue
                try:
                    s = float(st.get("sum", 0.0) or 0.0)
                except Exception:
                    s = 0.0
                try:
                    vmin = st.get("min", None)
                    vmin = float(vmin) if vmin is not None else None
                except Exception:
                    vmin = None
                try:
                    vmax = st.get("max", None)
                    vmax = float(vmax) if vmax is not None else None
                except Exception:
                    vmax = None
                try:
                    med = st.get("median", None)
                    med = float(med) if med is not None else None
                except Exception:
                    med = None

                bucket = acc.setdefault(
                    str(label),
                    {"count": 0, "sum": 0.0, "min": None, "max": None, "_med": []},
                )
                bucket["count"] = int(bucket.get("count", 0) or 0) + int(c)
                bucket["sum"] = float(bucket.get("sum", 0.0) or 0.0) + float(s)
                if vmin is not None:
                    cur = bucket.get("min", None)
                    bucket["min"] = (
                        vmin if cur is None else float(min(float(cur), float(vmin)))
                    )
                if vmax is not None:
                    cur = bucket.get("max", None)
                    bucket["max"] = (
                        vmax if cur is None else float(max(float(cur), float(vmax)))
                    )
                if med is not None:
                    try:
                        bucket["_med"].append((float(med), float(c)))
                    except Exception:
                        pass

        if not acc:
            return None

        out: dict[str, dict[str, Any]] = {}
        for label, b in acc.items():
            try:
                c = int(b.get("count", 0) or 0)
            except Exception:
                c = 0
            if c <= 0:
                continue
            s = float(b.get("sum", 0.0) or 0.0)
            mean = float(s / float(c)) if c > 0 else float("nan")
            med = self._weighted_median(b.get("_med", []))
            out[str(label)] = {
                "count": int(c),
                "sum": float(s),
                "mean": float(mean),
                "min": b.get("min", None),
                "max": b.get("max", None),
                "median": med,
            }
        return out or None

    def _reduce_payloads(self, payloads, *, output_level: int) -> dict[str, Any] | None:
        gathered_payloads = [p for p in (payloads or []) if p is not None]
        if not gathered_payloads:
            return None

        total_compute = float(
            sum(float(p.get("compute_time", 0.0) or 0.0) for p in gathered_payloads)
        )
        total_steps = int(
            sum(int(p.get("total_steps", 0) or 0) for p in gathered_payloads)
        )
        total_tasks = int(
            sum(int(p.get("total_tasks", 0) or 0) for p in gathered_payloads)
        )
        total_jumps = int(
            sum(int(p.get("total_jumps", 0) or 0) for p in gathered_payloads)
        )

        all_starts = []
        all_ends = []
        for p in gathered_payloads:
            try:
                s = p.get("first_task_start_ts")
                e = p.get("last_task_end_ts")
                if s is not None:
                    all_starts.append(float(s))
                if e is not None:
                    all_ends.append(float(e))
            except Exception:
                pass
        if all_starts and all_ends:
            global_first = float(min(all_starts))
            global_last = float(max(all_ends))
            task_compute_window_seconds = float(max(0.0, global_last - global_first))
        else:
            global_first = None
            global_last = None
            task_compute_window_seconds = None

        step_stats = self._merge_step_stats(
            gathered_payloads, output_level=int(output_level)
        )

        return {
            "total_compute_time_seconds": float(total_compute),
            "total_steps": int(total_steps),
            "total_tasks": int(total_tasks),
            "total_jumps": int(total_jumps),
            "first_task_start_ts": global_first,
            "last_task_end_ts": global_last,
            "task_compute_window_seconds": task_compute_window_seconds,
            "step_timing_stats": step_stats,
        }

    def _print_timing_summary_table(self, summary: dict | None) -> None:
        if summary is None:
            return
        header = "Timing Summary"
        lines = [header, "=" * len(header)]
        lines.append(f"output_level: {summary.get('output_level')}")
        lines.append(f"Total ranks: {summary.get('ranks')}")
        lines.append(f"Nodes: {summary.get('nodes')}")
        if summary.get("workers_total") is not None:
            lines.append(f"Workers per rank: {summary.get('workers_total')}")
        lines.append(
            "aggregate_compute_time_seconds: "
            f"{summary.get('total_compute_time_seconds'):.6f}"
        )
        if summary.get("task_compute_window_seconds") is not None:
            lines.append(
                "task_compute_window_seconds (first_task_start->last_task_end): "
                f"{summary.get('task_compute_window_seconds'):.6f}"
            )
        lines.append(f"duration_seconds: {summary.get('duration_seconds'):.6f}")
        lines.append(f"total_tasks: {summary.get('total_tasks')}")
        lines.append(f"total_jumps: {summary.get('total_jumps')}")
        lines.append(f"task_grid: {summary.get('task_grid')}")
        if summary.get("total_grid_points") is not None:
            lines.append(f"G (grid_points_total): {summary.get('total_grid_points')}")
        if summary.get("node_throughput") is not None:
            lines.append(
                "node_throughput (grid_points/s/node): "
                f"{summary.get('node_throughput'):.6f}"
            )
        if summary.get("rank_throughput") is not None:
            lines.append(
                "rank_throughput (grid_points/s/rank): "
                f"{summary.get('rank_throughput'):.6f}"
            )
        lines.append(f"seconds_per_task: {summary.get('seconds_per_task'):.6f}")
        lines.append(f"seconds_per_jump: {summary.get('seconds_per_jump'):.6f}")
        if summary.get("step_timing_stats"):
            lines.append("")
            lines.append("Per-stage timing stats:")
            lines.append("stage | count | mean | min | max | median")
            lines.append("----- | ----- | ---- | --- | --- | ------")
            for label, stats in summary["step_timing_stats"].items():
                lines.append(
                    f"{label} | {stats.get('count', 0)} | "
                    f"{stats.get('mean', float('nan')):.6f} | "
                    f"{stats.get('min', float('nan')):.6f} | "
                    f"{stats.get('max', float('nan')):.6f} | "
                    f"{stats.get('median', float('nan')):.6f}"
                )
        try:
            print("\n".join(lines), flush=True)
        except Exception:
            pass

    def timing_summary(self, start_ts: float, end_ts: float) -> dict | None:
        mpi = _get_mpi()
        rank = mpi.COMM_WORLD.Get_rank()
        world_size = mpi.COMM_WORLD.Get_size()
        is_root = bool(int(rank) == 0)

        output_level = int(self._get_meta("output_level", 0) or 0)

        step_stats = None
        if output_level >= 1:
            try:
                series = getattr(self.timing_stats, "series", {}) or {}
            except Exception:
                series = {}
            if isinstance(series, dict) and series:
                merged = {}
                for label, values in series.items():
                    stats = self._compute_series_stats(values or [])
                    if stats is not None:
                        merged[str(label)] = stats
                step_stats = merged or None

        payload = {
            "rank": int(rank),
            "compute_time": float(self.timing_stats.total_compute_time),
            "total_steps": int(self.timing_stats.total_steps),
            "total_tasks": int(self.timing_stats.total_tasks),
            "total_jumps": int(self.timing_stats.total_jumps),
            # Do NOT ship raw per-step series values to rank0 for large jobs.
            # Instead ship merge-friendly summary stats to enable hierarchical aggregation.
            "step_timing_stats": step_stats,
            "first_task_start_ts": getattr(
                self.timing_stats, "first_task_start_ts", None
            ),
            "last_task_end_ts": getattr(self.timing_stats, "last_task_end_ts", None),
        }
        # Hierarchical aggregation for scalability.
        try:
            default_group_size = 256
            group_size = int(
                env_int(
                    EnvKeys.TIMING_SUMMARY_GROUP_SIZE,
                    int(default_group_size),
                    min_value=1,
                    max_value=max(1, int(world_size)),
                )
            )
        except Exception:
            group_size = 256
        group_size = int(max(1, min(int(group_size), int(world_size))))

        final_reduced = None
        hierarchical = bool(int(world_size) > int(group_size))

        _LOGGER.info(
            f"Timing summary payload prepared on rank {rank}, group_size={group_size}, hierarchical={hierarchical}"
        )
        if not hierarchical:
            if int(world_size) > 1:
                gathered = mpi.COMM_WORLD.gather(payload, root=0)
            else:
                gathered = [payload]
            if is_root:
                final_reduced = self._reduce_payloads(
                    gathered, output_level=int(output_level)
                )
            else:
                final_reduced = {}
        else:
            group_id = int(int(rank) // int(group_size))
            group_comm = mpi.COMM_WORLD.Split(color=int(group_id), key=int(rank))
            group_rank = int(group_comm.Get_rank())
            group_n = int(group_comm.Get_size())
            is_leader = bool(group_rank == 0)
            group_gathered = group_comm.gather(payload, root=0)

            if is_leader:
                reduced = self._reduce_payloads(
                    group_gathered, output_level=int(output_level)
                )
                group_summary = {
                    "group_id": int(group_id),
                    "group_size": int(group_n),
                    "compute_time": float(
                        (reduced or {}).get("total_compute_time_seconds", 0.0) or 0.0
                    ),
                    "total_steps": int((reduced or {}).get("total_steps", 0) or 0),
                    "total_tasks": int((reduced or {}).get("total_tasks", 0) or 0),
                    "total_jumps": int((reduced or {}).get("total_jumps", 0) or 0),
                    "first_task_start_ts": (reduced or {}).get(
                        "first_task_start_ts", None
                    ),
                    "last_task_end_ts": (reduced or {}).get("last_task_end_ts", None),
                    "task_compute_window_seconds": (reduced or {}).get(
                        "task_compute_window_seconds", None
                    ),
                    "step_timing_stats": (reduced or {}).get("step_timing_stats", None),
                }
            else:
                group_summary = None
            _LOGGER.info(
                f"rank {rank} gathered and reduced group summary, is_leader={is_leader}"
            )
            leaders_comm = mpi.COMM_WORLD.Split(
                color=(0 if is_leader else mpi.UNDEFINED),
                key=int(rank),
            )
            if is_leader:
                leaders_gathered = leaders_comm.gather(group_summary, root=0)
            else:
                leaders_gathered = None

            if is_root:
                final_reduced = self._reduce_payloads(
                    leaders_gathered, output_level=int(output_level)
                )
            else:
                final_reduced = {}

        if final_reduced is None:
            return None

        total_compute = float(
            final_reduced.get("total_compute_time_seconds", 0.0) or 0.0
        )
        total_steps = int(final_reduced.get("total_steps", 0) or 0)
        total_tasks = int(final_reduced.get("total_tasks", 0) or 0)
        total_jumps = int(final_reduced.get("total_jumps", 0) or 0)
        global_first = final_reduced.get("first_task_start_ts", None)
        global_last = final_reduced.get("last_task_end_ts", None)
        task_compute_window_seconds = final_reduced.get(
            "task_compute_window_seconds", None
        )

        try:
            gx, gy, gz = tuple(self._get_meta("lattice_size", (0, 0, 0)))
        except Exception:
            gx, gy, gz = (0, 0, 0)
        try:
            total_grid_points = int(int(gx) * int(gy) * int(gz) * int(total_tasks))
        except Exception:
            total_grid_points = 0

        try:
            nodes = int(self._get_meta("nodes", 1) or 1)
        except Exception:
            nodes = 1
        duration_seconds = float(end_ts - start_ts)
        if duration_seconds > 0:
            node_throughput = float(total_grid_points) / (
                float(nodes) * duration_seconds
            )
            rank_throughput = float(total_grid_points) / (
                float(world_size) * duration_seconds
            )
        else:
            node_throughput = float("nan")
            rank_throughput = float("nan")

        workers_total = self._get_meta("workers_total", None)
        if workers_total is None:
            workers_total = self._get_meta("workers", None)
        if workers_total is None:
            workers_total = int(world_size)
        try:
            workers_total = int(workers_total)
        except Exception:
            workers_total = int(world_size)

        summary = {
            "event": "timing_summary",
            "start_ts": float(start_ts),
            "end_ts": float(end_ts),
            "duration_seconds": float(end_ts - start_ts),
            "output_level": output_level,
            "workers_total": int(workers_total),
            "total_compute_time_seconds": total_compute,
            "first_task_start_ts": global_first,
            "last_task_end_ts": global_last,
            "task_compute_window_seconds": task_compute_window_seconds,
            "total_tasks": int(total_tasks),
            "total_tasks_raw": int(total_tasks),
            "total_jumps": int(total_jumps),
            "total_jumps_raw": int(total_jumps),
            "task_grid": f"{gx}x{gy}x{gz}",
            "total_grid_points": total_grid_points,
            "nodes": int(nodes),
            "ranks": int(world_size),
            "node_throughput": node_throughput,
            "rank_throughput": rank_throughput,
            "seconds_per_task": float(
                (total_compute / int(total_tasks))
                if int(total_tasks) > 0
                else float("nan")
            ),
            "seconds_per_jump": float(
                (total_compute / int(total_jumps))
                if int(total_jumps) > 0
                else float("nan")
            ),
            "timing_summary_aggregation": {
                "mode": "hierarchical" if hierarchical else "flat",
                "group_size": int(group_size),
                "median_merge": "weighted_median_of_medians_approx",
            },
        }

        if output_level >= 1 and final_reduced.get("step_timing_stats") is not None:
            # Already merged in a hierarchical/merge-friendly way.
            # NOTE: median is approximate for scalability.
            timing_stats = final_reduced.get("step_timing_stats") or {}
            if isinstance(timing_stats, dict):
                # Ensure mean exists even if upstream provided only sum/count.
                for _, st in timing_stats.items():
                    if not isinstance(st, dict):
                        continue
                    if "mean" in st:
                        continue
                    try:
                        c = int(st.get("count", 0) or 0)
                        s = float(st.get("sum", 0.0) or 0.0)
                        st["mean"] = float(s / float(c)) if c > 0 else float("nan")
                    except Exception:
                        st["mean"] = float("nan")
            summary["step_timing_stats"] = timing_stats

        if is_root:
            self._print_timing_summary_table(summary)

        if self.output_dir is None:
            return summary if is_root else None

        if is_root:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                current_time = time.time()
                timestamp_str = time.strftime(
                    "%Y%m%d-%H%M%S", time.localtime(current_time)
                )
                json_name = f"timing_summary_rank{int(rank)}_{timestamp_str}.json"
                json_fp = os.path.join(self.output_dir, json_name)
                with open(json_fp, "w") as f:
                    json.dump(sanitize_for_json(summary), f, indent=2, allow_nan=False)
            except Exception:
                pass

        # Load-balance reporting:
        # - output_level>=1: write a single global report at rank0 (output_dir)
        # - output_level>=2: additionally write node-local report under rank-detail
        if int(output_level) >= 1:
            try:
                from RL4KMC.runner.engine import (
                    write_global_load_balance_report,
                    write_node_load_balance_report,
                )

                node_lb = self._get_meta("node_load_balance", None)
                node_payload = dict(node_lb) if isinstance(node_lb, dict) else {}
                node_payload.setdefault("rank", int(rank))
                node_payload.setdefault("host", str(socket.gethostname()))
                if self.rank_dir_name is not None:
                    node_payload.setdefault("rank_dir_name", str(self.rank_dir_name))

                # Node-local summary into rank-detail only when output_level>=2.
                if (
                    int(output_level) >= 2
                    and self.rank_dir is not None
                    and str(self.rank_dir).strip() != ""
                ):
                    try:
                        node_json_fp = os.path.join(
                            str(self.rank_dir), "load_balance_report.json"
                        )
                        node_csv_fp = os.path.join(
                            str(self.rank_dir), "load_balance_report.csv"
                        )
                        # Leader may already have written a richer node-local
                        # report with worker details. Keep it if present.
                        if not (
                            os.path.exists(node_json_fp) and os.path.exists(node_csv_fp)
                        ):
                            write_node_load_balance_report(
                                rank_dir=str(self.rank_dir),
                                node_summary=dict(node_payload),
                                worker_details=None,
                            )
                    except Exception:
                        pass

                # 2-level gather to rank0 for a single global report.
                if not hierarchical:
                    if int(world_size) > 1:
                        gathered_nodes = mpi.COMM_WORLD.gather(node_payload, root=0)
                    else:
                        gathered_nodes = [node_payload]
                    if int(rank) == 0:
                        nodes = [
                            n for n in (gathered_nodes or []) if isinstance(n, dict)
                        ]
                        write_global_load_balance_report(
                            output_dir=str(self.output_dir), nodes=list(nodes)
                        )
                else:
                    group_id = int(int(rank) // int(group_size))
                    group_comm = mpi.COMM_WORLD.Split(
                        color=int(group_id), key=int(rank)
                    )
                    group_rank = int(group_comm.Get_rank())
                    is_leader = bool(group_rank == 0)
                    group_gathered = group_comm.gather(node_payload, root=0)
                    group_pack = list(group_gathered or []) if is_leader else None

                    leaders_comm = mpi.COMM_WORLD.Split(
                        color=(0 if is_leader else mpi.UNDEFINED),
                        key=int(rank),
                    )
                    if is_leader:
                        leaders_gathered = leaders_comm.gather(group_pack, root=0)
                    else:
                        leaders_gathered = None

                    if int(rank) == 0:
                        flat: list[dict[str, Any]] = []
                        for grp in leaders_gathered or []:
                            if not isinstance(grp, list):
                                continue
                            for n in grp:
                                if isinstance(n, dict):
                                    flat.append(n)
                        write_global_load_balance_report(
                            output_dir=str(self.output_dir), nodes=list(flat)
                        )
            except Exception:
                pass
        return summary if is_root else None


class RunFinalizer:
    def __init__(
        self,
        *,
        output_dir: str | None,
        rank_dir: str | None,
        rank_dir_name: str,
        rank: int,
        world_size: int,
        timing_stats: Any,
        enable_rank_detail: bool,
        enable_timing_io: bool,
        enable_timing_report: bool,
        enable_timing_log: bool,
    ) -> None:
        self.output_dir = output_dir
        self.rank_dir = rank_dir
        self.rank_dir_name = str(rank_dir_name)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.timing_stats = timing_stats
        self.enable_rank_detail = bool(enable_rank_detail)
        self.enable_timing_io = bool(enable_timing_io)
        self.enable_timing_report = bool(enable_timing_report)
        self.enable_timing_log = bool(enable_timing_log)

    def finalize(self, start_global_ts: float, end_global_ts: float) -> float:
        # End-of-run synchronization: keep the original blocking barrier by default.
        # Some HPC failures manifest as a "hang" here when one rank never reaches finalize.
        # Opt-in envs allow a bounded wait for better debuggability.
        _LOGGER.info(
            "RunFinalizer entering MPI Barrier rank=%s/%s",
            int(self.rank),
            int(self.world_size),
        )
        _get_mpi().COMM_WORLD.Barrier()

        reporter = TimingReporter(
            output_dir=self.output_dir,
            rank_dir=self.rank_dir,
            rank_dir_name=str(self.rank_dir_name),
            timing_stats=self.timing_stats,
        )
        reporter.timing_summary(start_global_ts, end_global_ts)

        _LOGGER.debug("finished report")

        if self.output_dir is not None and self.enable_rank_detail:
            aggregate_task_times(self.output_dir, start_global_ts, end_global_ts)

        if (
            self.enable_timing_io
            and self.output_dir is not None
            and self.rank_dir is not None
        ):
            write_timing_reports(
                self.output_dir,
                self.rank_dir,
                self.rank_dir_name,
                bool(int(self.rank) == 0),
                self.enable_timing_report,
                self.enable_timing_log,
            )
            write_overall_time_report(
                self.output_dir,
                self.rank,
                start_global_ts,
                end_global_ts,
            )

        return float(end_global_ts)
