import csv
import json
import os
import time
from typing import Dict, List, Optional


def _compute_stats(values: List[float]) -> Optional[Dict[str, float]]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = float(sum(vals) / n)
    vmin = float(min(vals))
    vmax = float(max(vals))
    var = float(sum((x - mean) ** 2 for x in vals) / n)
    return {"count": n, "mean": mean, "min": vmin, "max": vmax, "var": var}


def _collect_timing_files(dirs: List[str]) -> Dict[str, List[str]]:
    groups = {
        "task_step_timing": "task_step_timing",
        "task_env_apply_timing": "task_env_apply_timing",
        "task_env_rl_select_timing": "task_env_rl_select_timing",
        "task_env_update_pipeline_timing": "task_env_update_pipeline_timing",
    }
    grouped = {k: [] for k in groups.keys()}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".csv"):
                    continue
                for key, token in groups.items():
                    if token in fn:
                        grouped[key].append(os.path.join(root, fn))
                        break
    return grouped


def aggregate_timing_stats(dirs: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped_files = _collect_timing_files(dirs)
    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group, files in grouped_files.items():
        col_values: Dict[str, List[float]] = {}
        for fp in files:
            try:
                with open(fp, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if not header or len(header) <= 1:
                        continue
                    col_names = [str(h).strip() for h in header[1:]]
                    for row in reader:
                        if len(row) < 2:
                            continue
                        for idx, col in enumerate(col_names, start=1):
                            if idx >= len(row):
                                continue
                            try:
                                v = float(row[idx])
                            except Exception:
                                continue
                            col_values.setdefault(col, []).append(v)
            except Exception:
                continue
        group_stats: Dict[str, Dict[str, float]] = {}
        for col, vals in col_values.items():
            stats = _compute_stats(vals)
            if stats is not None:
                group_stats[col] = stats
        report[group] = group_stats
    return report


def write_timing_report_files(
    report: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: str,
    name_prefix: str,
) -> None:
    if not report:
        return
    os.makedirs(out_dir, exist_ok=True)
    json_fp = os.path.join(out_dir, f"{name_prefix}.json")
    csv_fp = os.path.join(out_dir, f"{name_prefix}.csv")
    try:
        with open(json_fp, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass
    try:
        with open(csv_fp, "w") as f:
            f.write("group,metric,count,mean,min,max,var\n")
            for group, stats_map in report.items():
                for metric, stats in stats_map.items():
                    f.write(
                        f"{group},{metric},{stats.get('count', 0)},"
                        f"{stats.get('mean', float('nan'))},"
                        f"{stats.get('min', float('nan'))},"
                        f"{stats.get('max', float('nan'))},"
                        f"{stats.get('var', float('nan'))}\n"
                    )
    except Exception:
        pass


def write_overall_time_report(
    output_dir: str,
    rank: int,
    start_ts: float,
    end_ts: float,
) -> None:
    if int(rank) != 0:
        return
    duration = float(end_ts - start_ts)

    rank_detail_dir = os.path.join(output_dir, "rank-detail")
    timing_stats = aggregate_timing_stats([rank_detail_dir])
    step_stats = timing_stats.get("task_step_timing", {})
    inference_stats = step_stats.get("inference")
    kmc_stats = step_stats.get("kmc_step")

    per_worker_totals: Dict[str, float] = {}
    slowest_worker: str | None = None
    slowest_time: float | None = None

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

    for name in worker_names:
        rfp = os.path.join(rank_detail_dir, str(name), "task_times.csv")
        total = 0.0
        if os.path.isfile(rfp):
            try:
                with open(rfp, "r") as rf:
                    next(rf, None)
                    for line in rf:
                        parts = line.strip().split(",")
                        if len(parts) >= 8:
                            try:
                                total += float(parts[7])
                            except Exception:
                                pass
            except Exception:
                pass
        per_worker_totals[str(name)] = float(total)
        if slowest_time is None or total > slowest_time:
            slowest_time = float(total)
            slowest_worker = str(name)

    report = {
        "event": "overall_time_report",
        "start_ts": float(start_ts),
        "end_ts": float(end_ts),
        "duration_seconds": duration,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts)),
        "model_inference_stats": inference_stats,
        "kmc_step_stats": kmc_stats,
        "per_worker_total_task_time_seconds": per_worker_totals,
        "slowest_worker": slowest_worker,
        "slowest_worker_total_task_time_seconds": slowest_time,
    }
    try:
        json_fp = os.path.join(output_dir, "overall_time_report.json")
        with open(json_fp, "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass
    try:
        csv_fp = os.path.join(output_dir, "overall_time_report.csv")
        with open(csv_fp, "w") as f:
            f.write(
                "start_ts,end_ts,duration_seconds,start_time,end_time,"
                "slowest_worker,slowest_worker_total_task_time_seconds\n"
            )
            f.write(
                f"{report['start_ts']},{report['end_ts']},{report['duration_seconds']},"
                f"{report['start_time']},{report['end_time']},"
                f"{report['slowest_worker']},{report['slowest_worker_total_task_time_seconds']}\n"
            )
    except Exception:
        pass


def write_timing_reports(
    output_dir: str,
    rank_dir: str,
    rank_dir_name: str,
    is_leader: bool,
    enable_timing_report: bool,
    enable_timing_log: bool,
) -> None:
    if not enable_timing_report or not enable_timing_log:
        return
    safe_name = str(rank_dir_name).replace(os.sep, "_")
    rank_report = aggregate_timing_stats([rank_dir])
    write_timing_report_files(rank_report, rank_dir, f"timing_report_{safe_name}")
    if bool(is_leader):
        rank_detail_dir = os.path.join(output_dir, "rank-detail")
        run_report = aggregate_timing_stats([rank_detail_dir])
        write_timing_report_files(run_report, output_dir, "timing_report_run")
