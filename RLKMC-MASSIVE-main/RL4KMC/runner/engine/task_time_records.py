from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TaskTimeRecord:
    worker: str
    temp: Optional[float]
    cu_density: Optional[float]
    v_density: Optional[float]
    sim_time: float
    start_ts: float
    end_ts: float

    @property
    def duration(self) -> float:
        try:
            return float(self.end_ts - self.start_ts)
        except Exception:
            return float("nan")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["duration"] = float(self.duration)
        return d


def task_time_record_from_dict(d: Dict[str, Any]) -> TaskTimeRecord:
    return TaskTimeRecord(
        worker=str(d.get("worker", "")),
        temp=(float(d["temp"]) if d.get("temp", None) is not None and str(d.get("temp")).strip() != "" else None),
        cu_density=(
            float(d["cu_density"])
            if d.get("cu_density", None) is not None and str(d.get("cu_density")).strip() != ""
            else None
        ),
        v_density=(
            float(d["v_density"])
            if d.get("v_density", None) is not None and str(d.get("v_density")).strip() != ""
            else None
        ),
        sim_time=float(d.get("sim_time", float("nan"))),
        start_ts=float(d.get("start_ts", float("nan"))),
        end_ts=float(d.get("end_ts", float("nan"))),
    )
