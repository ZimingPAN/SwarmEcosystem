import numpy as np
from typing import Optional, Tuple, List, Dict
import ast
import os

class Plotter:
    def __init__(self):
        self.colors = ['red', 'yellow']  # Only Cu and Vacancy colors
        self.labels = ['Cu', 'Vacancy']
        
    def plot_lattice(self, lattice: np.ndarray, 
                    constant: float,
                    title: Optional[str] = None,
                    save_path: Optional[str] = None) -> Tuple[None, None]:
        return None, None
    
    def plot_energy_evolution(self, energies: list, 
                            times: list,
                            save_path: Optional[str] = None):
        return None

    def parse_comm_exchange_time(self, comm_log_path: str) -> List[Dict]:
        records: List[Dict] = []
        if not os.path.isfile(comm_log_path):
            return records
        with open(comm_log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = ast.literal_eval(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get('event') != 'exchange':
                    continue
                dt = obj.get('dt', None)
                if dt is None:
                    continue
                records.append({
                    'dt': float(dt),
                    'rank': obj.get('rank', None),
                    'sub_block': obj.get('sub_block', None),
                    'has_meta': ('recv_from' in obj) or ('send_to' in obj)
                })
        return records

    def plot_comm_exchange_time(self, comm_log_path: str, save_path: Optional[str] = None, only_inner: bool = True):
        return None

    def summarize_comm_exchange(self, comm_log_path: str, only_inner: bool = True, by_rank: bool = False) -> Dict:
        recs = self.parse_comm_exchange_time(comm_log_path)
        if only_inner:
            recs = [r for r in recs if r.get('has_meta')]
        if not recs:
            return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "by_rank": {}}
        dt = np.array([r['dt'] for r in recs], dtype=np.float64)
        summary = {
            "count": int(dt.size),
            "mean": float(np.mean(dt)),
            "median": float(np.median(dt)),
            "p95": float(np.percentile(dt, 95)),
            "max": float(np.max(dt)),
        }
        if by_rank:
            rank_map: Dict[int, List[float]] = {}
            for r in recs:
                rk = r.get('rank', None)
                if rk is None:
                    continue
                rank_map.setdefault(int(rk), []).append(float(r['dt']))
            summary["by_rank"] = {rk: {
                "count": len(v),
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "max": float(np.max(v)),
            } for rk, v in rank_map.items()}
        else:
            summary["by_rank"] = {}
        return summary

    def plot_scalability_exchange(self, experiments: List[Dict], save_path: Optional[str] = None, metric: str = "mean", only_inner: bool = True):
        return None
