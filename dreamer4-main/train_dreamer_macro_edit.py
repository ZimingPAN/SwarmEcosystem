from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[0]
RLKMC = ROOT.parent / "RLKMC-MASSIVE-main"
LIGHTZERO = ROOT.parent / "LightZero-main"
for path in [str(ROOT), str(RLKMC), str(LIGHTZERO)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from RL4KMC.envs.kmc import KMCEnv
from RL4KMC.parser.parser import get_config
from RL4KMC.world_models import DefectGraphObservationShape, build_defect_graph_observation
from dreamer4.macro_edit import (
    MacroDreamerEditModel,
    NUM_SITE_TYPES,
    kl_divergence_diag_gaussian,
    lognormal_nll,
    macro_duration_baseline_log_tau,
    project_types_by_inventory,
    teacher_path_summary_dim,
)


FE_TYPE = 0
CU_TYPE = 1
V_TYPE = 2


@dataclass
class MacroSegmentSample:
    start_obs: np.ndarray
    next_obs: np.ndarray
    start_vacancy_positions: np.ndarray
    start_cu_positions: np.ndarray
    global_summary: np.ndarray
    teacher_path_summary: np.ndarray
    candidate_positions: np.ndarray
    nearest_vacancy_offset: np.ndarray
    reach_depth: np.ndarray
    is_start_vacancy: np.ndarray
    current_types: np.ndarray
    target_types: np.ndarray
    candidate_mask: np.ndarray
    changed_mask: np.ndarray
    tau_exp: float
    tau_real: float
    reward_sum: float
    horizon_k: int
    box_dims: np.ndarray


class MacroSegmentDataset(Dataset):
    def __init__(self, samples: list[MacroSegmentSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MacroSegmentSample:
        return self.samples[idx]


def _build_args(cfg: dict):
    parser = get_config()
    args = parser.parse_known_args([])[0]
    total = int(np.prod(cfg["lattice_size"]) * 2)
    args.lattice_size = list(cfg["lattice_size"])
    args.temperature = cfg.get("temperature", 300.0)
    args.reward_scale = cfg.get("reward_scale", 1.0)
    args.topk = cfg.get("rlkmc_topk", 16)
    args.device = "cpu"
    args.cu_density = cfg["cu_density"]
    args.v_density = cfg["v_density"]
    args.lattice_cu_nums = int(round(cfg["cu_density"] * total))
    args.lattice_v_nums = max(int(round(cfg["v_density"] * total)), 1)
    args.compute_global_static_env_reset = True
    args.skip_stats = True
    args.skip_global_diffusion_reset = False
    args.max_ssa_rounds = cfg["max_episode_steps"]
    args.neighbor_order = cfg.get("neighbor_order", "2NN")
    return args


class MacroKMCEnv:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.shape = DefectGraphObservationShape(
            max_vacancies=cfg["max_vacancies"],
            max_defects=cfg["max_defects"],
            max_shells=cfg["max_shells"],
            node_feat_dim=4,
            stats_dim=cfg.get("stats_dim", 10),
        )
        self.env = KMCEnv(_build_args(cfg))
        self.timestep = 0
        self.max_steps = cfg["max_episode_steps"]

    def reset(self) -> np.ndarray:
        self.env.reset()
        self.timestep = 0
        return self.obs()

    def current_total_rate(self) -> float:
        self.env._ensure_diffusion_rates()
        flat = [rate for vac_rates in self.env.diffusion_rates for rate in vac_rates if rate > 0]
        return float(np.sum(flat)) if flat else 0.0

    def obs(self) -> np.ndarray:
        share_obs = np.zeros(self.shape.stats_dim, dtype=np.float32)
        share_obs[0] = self.cfg.get("temperature", 300.0) / 1000.0
        share_obs[1] = self.cfg.get("cu_density", 0.0134)
        share_obs[2] = self.cfg.get("v_density", 0.0002)
        return build_defect_graph_observation(self.env, shape=self.shape, share_obs=share_obs).astype(np.float32)

    def action_mask(self) -> np.ndarray:
        self.env._ensure_diffusion_rates()
        masks = []
        for vac_rates in self.env.diffusion_rates[: self.shape.max_vacancies]:
            masks.extend([1.0 if rate > 0 else 0.0 for rate in vac_rates])
        masks.extend([0.0] * max(0, self.shape.max_vacancies * 8 - len(masks)))
        return np.asarray(masks[: self.shape.max_vacancies * 8], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        total_rate = self.current_total_rate()
        expected_delta_t = 1.0 / total_rate if total_rate > 0 else 0.0
        vac_idx, dir_idx, old_pos, new_pos, moving_type = self.env._decode_action(int(action))
        self.env.step_fast(int(action), self.timestep)
        delta_t = -np.log(np.random.rand()) / total_rate if total_rate > 0 else 0.0
        self.env.time += delta_t
        self.env.time_history.append(self.env.time)
        energy_after = self.env.calculate_system_energy()
        delta_E = self.env.energy_last - energy_after
        reward = float(delta_E * self.env.args.reward_scale)
        self.env.energy_last = energy_after
        self.env.energy_history.append(energy_after)
        self.timestep += 1
        done = self.timestep >= self.max_steps
        return self.obs(), reward, done, {
            "delta_t": float(delta_t),
            "expected_delta_t": float(expected_delta_t),
            "total_rate": float(total_rate),
            "delta_E": float(delta_E),
            "dir_idx": int(dir_idx),
            "vac_idx": int(vac_idx),
            "moving_type": int(moving_type),
            "old_pos": np.asarray(old_pos, dtype=np.int32),
            "new_pos": np.asarray(new_pos, dtype=np.int32),
        }


def _periodic_offset(src: np.ndarray, dst: np.ndarray, box: np.ndarray) -> np.ndarray:
    delta = src - dst
    return delta - np.round(delta / box) * box


def _positions_to_type_lookup(vacancies: np.ndarray, cu_atoms: np.ndarray) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    vac_set = {tuple(map(int, pos)) for pos in vacancies.tolist()}
    cu_set = {tuple(map(int, pos)) for pos in cu_atoms.tolist()}
    return vac_set, cu_set


def _type_from_lookup(pos: tuple[int, int, int], vac_set: set[tuple[int, int, int]], cu_set: set[tuple[int, int, int]]) -> int:
    if pos in vac_set:
        return V_TYPE
    if pos in cu_set:
        return CU_TYPE
    return FE_TYPE


def _one_hop_neighbors(pos: tuple[int, int, int], nn1: np.ndarray, box: np.ndarray) -> list[tuple[int, int, int]]:
    base = np.asarray(pos, dtype=np.int32)
    out = []
    for step in nn1:
        nxt = tuple(((base + step) % box).tolist())
        out.append(nxt)
    return out


def _vacancy_rate_sums(env: MacroKMCEnv) -> np.ndarray:
    env.env._ensure_diffusion_rates()
    rate_sums = [float(np.sum([rate for rate in vac_rates if rate > 0])) for vac_rates in env.env.diffusion_rates]
    return np.asarray(rate_sums, dtype=np.float32)


def _sample_teacher_action(env: MacroKMCEnv, rng: np.random.Generator) -> Optional[int]:
    env.env._ensure_diffusion_rates()
    actions: list[int] = []
    rates: list[float] = []
    for vac_idx, vac_rates in enumerate(env.env.diffusion_rates):
        for dir_idx, rate in enumerate(vac_rates):
            if rate > 0:
                actions.append(vac_idx * 8 + dir_idx)
                rates.append(float(rate))
    if not actions:
        return None
    probs = np.asarray(rates, dtype=np.float64)
    probs = probs / probs.sum()
    return int(actions[int(rng.choice(len(actions), p=probs))])


def _global_summary(env: MacroKMCEnv) -> np.ndarray:
    stats = env.env.get_system_stats().astype(np.float32)
    env.env._ensure_diffusion_rates()
    positive_rates = np.asarray([rate for vac_rates in env.env.diffusion_rates for rate in vac_rates if rate > 0], dtype=np.float32)
    rate_sums = _vacancy_rate_sums(env)
    top_rates = np.sort(positive_rates)[-8:] if positive_rates.size > 0 else np.zeros((0,), dtype=np.float32)
    summary = np.zeros((16,), dtype=np.float32)
    summary[: min(10, stats.size)] = stats[:10]
    total_rate = float(positive_rates.sum()) if positive_rates.size > 0 else 0.0
    summary[10] = math.log(total_rate + 1e-12)
    summary[11] = math.log(float(top_rates.mean()) + 1e-12) if top_rates.size > 0 else -27.0
    summary[12] = math.log(float(top_rates.max()) + 1e-12) if top_rates.size > 0 else -27.0
    summary[13] = math.log(float(top_rates.std()) + 1e-12) if top_rates.size > 1 else -27.0
    summary[14] = float((positive_rates.size / max(env.shape.max_vacancies * 8, 1)))
    summary[15] = float((rate_sums > 0).mean()) if rate_sums.size > 0 else 0.0
    return summary
def _teacher_path_summary(
    path_infos: list[dict],
    max_candidate_sites: int,
    horizon_k: int,
    *,
    include_stepwise_features: bool = True,
) -> np.ndarray:
    direction_hist = np.zeros((8,), dtype=np.float32)
    moving_hist = np.zeros((NUM_SITE_TYPES,), dtype=np.float32)
    log_rates = []
    delta_es = []
    step_log_expected_dt = np.full((horizon_k,), -27.0, dtype=np.float32)
    step_delta_es = np.zeros((horizon_k,), dtype=np.float32)
    touched = set()
    vacancy_ids = set()
    for step_idx, info in enumerate(path_infos[:horizon_k]):
        direction_hist[int(info["dir_idx"])] += 1.0
        moving_type = int(info["moving_type"])
        if 0 <= moving_type < NUM_SITE_TYPES:
            moving_hist[moving_type] += 1.0
        log_rates.append(math.log(float(info["total_rate"]) + 1e-12))
        delta_es.append(float(info["delta_E"]))
        step_log_expected_dt[step_idx] = math.log(float(info["expected_delta_t"]) + 1e-12)
        step_delta_es[step_idx] = float(info["delta_E"])
        touched.add(tuple(map(int, info["old_pos"].tolist())))
        touched.add(tuple(map(int, info["new_pos"].tolist())))
        vacancy_ids.add(int(info["vac_idx"]))
    if path_infos:
        direction_hist /= len(path_infos)
        moving_hist /= len(path_infos)
    summary = np.zeros((teacher_path_summary_dim(horizon_k, include_stepwise_features=include_stepwise_features),), dtype=np.float32)
    summary[:8] = direction_hist
    summary[8:11] = moving_hist
    summary[11] = float(np.mean(log_rates)) if log_rates else -27.0
    summary[12] = float(np.std(log_rates)) if len(log_rates) > 1 else 0.0
    summary[13] = float(np.mean(delta_es)) if delta_es else 0.0
    summary[14] = float(np.mean([de > 0 for de in delta_es])) if delta_es else 0.0
    summary[15] = float(len(touched) / max(max_candidate_sites, 1))
    summary[16] = float(len(vacancy_ids) / max(len(path_infos), 1)) if path_infos else 0.0
    summary[17] = float(len(path_infos) / max(horizon_k, 1))
    if include_stepwise_features:
        summary[18 : 18 + horizon_k] = step_log_expected_dt
        summary[18 + horizon_k : 18 + 2 * horizon_k] = step_delta_es
    return summary


def _select_seed_vacancies(env: MacroKMCEnv, max_seed_vacancies: int) -> np.ndarray:
    vacancy_positions = env.env.get_vacancy_array().astype(np.int32)
    if vacancy_positions.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    rate_sums = _vacancy_rate_sums(env)
    order = np.argsort(rate_sums)[::-1][: max(1, min(max_seed_vacancies, len(vacancy_positions)))]
    return vacancy_positions[order]


def _build_candidate_positions(env: MacroKMCEnv, horizon_k: int, max_seed_vacancies: int, max_candidate_sites: int) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int, int], int], np.ndarray]:
    box = np.asarray(env.env.dims, dtype=np.int32)
    nn1 = np.asarray(env.env.NN1, dtype=np.int32)
    seeds = _select_seed_vacancies(env, max_seed_vacancies)
    if seeds.size == 0:
        return [], {}, seeds
    depth_map: dict[tuple[int, int, int], int] = {}
    frontier = {tuple(map(int, pos.tolist())) for pos in seeds}
    for pos in frontier:
        depth_map[pos] = 0
    for depth in range(1, horizon_k + 1):
        next_frontier: set[tuple[int, int, int]] = set()
        for pos in frontier:
            for nxt in _one_hop_neighbors(pos, nn1, box):
                if nxt not in depth_map:
                    depth_map[nxt] = depth
                    next_frontier.add(nxt)
        frontier = next_frontier
        if not frontier:
            break

    def rank_key(pos: tuple[int, int, int]) -> tuple[int, float]:
        pos_arr = np.asarray(pos, dtype=np.float32)
        min_dist = min(np.linalg.norm(_periodic_offset(pos_arr, seed.astype(np.float32), box.astype(np.float32))) for seed in seeds)
        return depth_map[pos], float(min_dist)

    ranked = sorted(depth_map.keys(), key=rank_key)
    return ranked[:max_candidate_sites], depth_map, seeds


def _build_patch_features(
    *,
    candidate_positions: list[tuple[int, int, int]],
    depth_map: dict[tuple[int, int, int], int],
    seeds: np.ndarray,
    start_vac_set: set[tuple[int, int, int]],
    start_cu_set: set[tuple[int, int, int]],
    end_vac_set: set[tuple[int, int, int]],
    end_cu_set: set[tuple[int, int, int]],
    max_candidate_sites: int,
    box: np.ndarray,
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.zeros((max_candidate_sites, 3), dtype=np.float32)
    nearest_offsets = np.zeros((max_candidate_sites, 3), dtype=np.float32)
    reach_depth = np.zeros((max_candidate_sites,), dtype=np.float32)
    is_start_vacancy = np.zeros((max_candidate_sites,), dtype=np.float32)
    current_types = np.zeros((max_candidate_sites,), dtype=np.int64)
    target_types = np.zeros((max_candidate_sites,), dtype=np.int64)
    mask = np.zeros((max_candidate_sites,), dtype=np.float32)
    for idx, pos in enumerate(candidate_positions[:max_candidate_sites]):
        pos_arr = np.asarray(pos, dtype=np.float32)
        positions[idx] = pos_arr
        if len(seeds) > 0:
            offsets = [_periodic_offset(pos_arr, seed.astype(np.float32), box.astype(np.float32)) for seed in seeds]
            nearest = min(offsets, key=lambda item: float(np.linalg.norm(item)))
            nearest_offsets[idx] = nearest.astype(np.float32)
        reach_depth[idx] = float(depth_map.get(pos, horizon_k)) / max(horizon_k, 1)
        is_start_vacancy[idx] = 1.0 if pos in start_vac_set else 0.0
        current_types[idx] = _type_from_lookup(pos, start_vac_set, start_cu_set)
        target_types[idx] = _type_from_lookup(pos, end_vac_set, end_cu_set)
        mask[idx] = 1.0
    changed_mask = (current_types != target_types).astype(np.float32) * mask
    return positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, target_types, changed_mask


def _collect_segments(
    *,
    env: MacroKMCEnv,
    num_segments: int,
    horizon_k: int,
    max_seed_vacancies: int,
    max_candidate_sites: int,
    rng: np.random.Generator,
    max_attempt_multiplier: int = 20,
    include_stepwise_path_summary: bool = True,
) -> tuple[list[MacroSegmentSample], dict[str, float]]:
    def restart_env(current_env: MacroKMCEnv) -> tuple[MacroKMCEnv, np.ndarray]:
        new_env = MacroKMCEnv(copy.deepcopy(current_env.cfg))
        return new_env, new_env.reset()

    samples: list[MacroSegmentSample] = []
    stats = {
        "attempts": 0,
        "skipped_uncovered": 0,
        "skipped_terminal": 0,
        "skipped_noop": 0,
        "candidate_size_sum": 0.0,
    }
    progress_every = max(50, num_segments // 10)
    max_segments_per_rollout = 50
    max_stall_attempts = 16
    obs = env.reset()
    segments_since_reset = 0
    stall_attempts = 0
    attempts_limit = num_segments * max_attempt_multiplier
    while len(samples) < num_segments and stats["attempts"] < attempts_limit:
        stats["attempts"] += 1
        start_obs = obs.copy()
        start_vacancies = env.env.get_vacancy_array().astype(np.int32)
        start_cu = env.env.get_cu_array().astype(np.int32)
        start_vac_set, start_cu_set = _positions_to_type_lookup(start_vacancies, start_cu)
        candidate_positions, depth_map, seeds = _build_candidate_positions(
            env, horizon_k, max_seed_vacancies=max_seed_vacancies, max_candidate_sites=max_candidate_sites
        )
        if not candidate_positions:
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue

        global_summary = _global_summary(env)
        tau_exp = 0.0
        tau_real = 0.0
        reward_sum = 0.0
        touched_positions: set[tuple[int, int, int]] = set()
        path_infos: list[dict] = []
        done = False
        next_obs = start_obs
        for _ in range(horizon_k):
            action = _sample_teacher_action(env, rng)
            if action is None:
                done = True
                break
            next_obs, reward, done, info = env.step(action)
            tau_exp += float(info["expected_delta_t"])
            tau_real += float(info["delta_t"])
            reward_sum += float(reward)
            path_infos.append(info)
            touched_positions.add(tuple(map(int, info["old_pos"].tolist())))
            touched_positions.add(tuple(map(int, info["new_pos"].tolist())))
            if done:
                break
        if done:
            stats["skipped_terminal"] += 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        candidate_set = set(candidate_positions)
        if not touched_positions.issubset(candidate_set):
            stats["skipped_uncovered"] += 1
            stall_attempts += 1
            if stall_attempts >= max_stall_attempts:
                env, obs = restart_env(env)
                segments_since_reset = 0
                stall_attempts = 0
            else:
                obs = next_obs
            continue

        end_vacancies = env.env.get_vacancy_array().astype(np.int32)
        end_cu = env.env.get_cu_array().astype(np.int32)
        end_vac_set, end_cu_set = _positions_to_type_lookup(end_vacancies, end_cu)
        positions, nearest_offsets, reach_depth, is_start_vacancy, current_types, target_types, changed_mask = _build_patch_features(
            candidate_positions=candidate_positions,
            depth_map=depth_map,
            seeds=seeds,
            start_vac_set=start_vac_set,
            start_cu_set=start_cu_set,
            end_vac_set=end_vac_set,
            end_cu_set=end_cu_set,
            max_candidate_sites=max_candidate_sites,
            box=np.asarray(env.env.dims, dtype=np.int32),
            horizon_k=horizon_k,
        )
        if float(changed_mask.sum()) <= 0.0:
            stats["skipped_noop"] += 1
            env, obs = restart_env(env)
            segments_since_reset = 0
            stall_attempts = 0
            continue
        teacher_summary = _teacher_path_summary(
            path_infos,
            max_candidate_sites=max_candidate_sites,
            horizon_k=horizon_k,
            include_stepwise_features=include_stepwise_path_summary,
        )
        mask = np.zeros((max_candidate_sites,), dtype=np.float32)
        mask[: len(candidate_positions)] = 1.0
        stats["candidate_size_sum"] += float(mask.sum())
        samples.append(
            MacroSegmentSample(
                start_obs=start_obs,
                next_obs=next_obs.copy(),
                start_vacancy_positions=start_vacancies.copy(),
                start_cu_positions=start_cu.copy(),
                global_summary=global_summary,
                teacher_path_summary=teacher_summary,
                candidate_positions=positions,
                nearest_vacancy_offset=nearest_offsets,
                reach_depth=reach_depth,
                is_start_vacancy=is_start_vacancy,
                current_types=current_types,
                target_types=target_types,
                candidate_mask=mask,
                changed_mask=changed_mask,
                tau_exp=float(tau_exp),
                tau_real=float(tau_real),
                reward_sum=float(reward_sum),
                horizon_k=int(horizon_k),
                box_dims=np.asarray(env.env.dims, dtype=np.float32),
            )
        )
        if len(samples) % progress_every == 0 or len(samples) == num_segments:
            coverage = float(len(samples) / max(stats["attempts"], 1))
            print(
                json.dumps(
                    {
                        "collect_progress": {
                            "samples": len(samples),
                            "target": num_segments,
                            "attempts": stats["attempts"],
                            "coverage": coverage,
                        }
                    },
                    ensure_ascii=False,
                )
            )
        stall_attempts = 0
        segments_since_reset += 1
        if segments_since_reset >= max_segments_per_rollout:
            env, obs = restart_env(env)
            segments_since_reset = 0
        else:
            obs = next_obs
    denom = max(len(samples), 1)
    stats["coverage"] = float(len(samples) / max(stats["attempts"], 1))
    stats["avg_candidate_size"] = float(stats["candidate_size_sum"] / denom)
    return samples, stats


def _save_samples(samples: list[MacroSegmentSample], path: Path) -> None:
    payload = [asdict(sample) for sample in samples]
    torch.save(payload, path)


def _load_samples(path: Path) -> list[MacroSegmentSample]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [MacroSegmentSample(**item) for item in payload]


def _batch_to_device(batch: list[MacroSegmentSample], device: str) -> dict[str, torch.Tensor]:
    return {
        "start_obs": torch.tensor(np.stack([sample.start_obs for sample in batch]), dtype=torch.float32, device=device),
        "next_obs": torch.tensor(np.stack([sample.next_obs for sample in batch]), dtype=torch.float32, device=device),
        "global_summary": torch.tensor(np.stack([sample.global_summary for sample in batch]), dtype=torch.float32, device=device),
        "teacher_path_summary": torch.tensor(np.stack([sample.teacher_path_summary for sample in batch]), dtype=torch.float32, device=device),
        "candidate_positions": torch.tensor(np.stack([sample.candidate_positions for sample in batch]), dtype=torch.float32, device=device),
        "nearest_vacancy_offset": torch.tensor(np.stack([sample.nearest_vacancy_offset for sample in batch]), dtype=torch.float32, device=device),
        "reach_depth": torch.tensor(np.stack([sample.reach_depth for sample in batch]), dtype=torch.float32, device=device),
        "is_start_vacancy": torch.tensor(np.stack([sample.is_start_vacancy for sample in batch]), dtype=torch.float32, device=device),
        "current_types": torch.tensor(np.stack([sample.current_types for sample in batch]), dtype=torch.long, device=device),
        "target_types": torch.tensor(np.stack([sample.target_types for sample in batch]), dtype=torch.long, device=device),
        "candidate_mask": torch.tensor(np.stack([sample.candidate_mask for sample in batch]), dtype=torch.float32, device=device),
        "changed_mask": torch.tensor(np.stack([sample.changed_mask for sample in batch]), dtype=torch.float32, device=device),
        "tau_exp": torch.tensor([sample.tau_exp for sample in batch], dtype=torch.float32, device=device),
        "tau_real": torch.tensor([sample.tau_real for sample in batch], dtype=torch.float32, device=device),
        "reward_sum": torch.tensor([sample.reward_sum for sample in batch], dtype=torch.float32, device=device),
        "horizon_k": torch.tensor([sample.horizon_k for sample in batch], dtype=torch.long, device=device),
        "box_dims": torch.tensor(np.stack([sample.box_dims for sample in batch]), dtype=torch.float32, device=device),
    }


class _ProjectedStateEnv:
    def __init__(self, vacancies: np.ndarray, cu_positions: np.ndarray, dims: np.ndarray):
        self._vacancies = np.asarray(vacancies, dtype=np.int32)
        self._cu_positions = np.asarray(cu_positions, dtype=np.int32)
        self.dims = tuple(int(x) for x in np.asarray(dims, dtype=np.int32).tolist())
        self.V_TYPE = V_TYPE
        self.CU_TYPE = CU_TYPE

    def get_vacancy_array(self) -> np.ndarray:
        return self._vacancies

    def get_cu_array(self) -> np.ndarray:
        return self._cu_positions


def _apply_projected_types(sample: MacroSegmentSample, projected_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vac_set = {tuple(map(int, pos)) for pos in sample.start_vacancy_positions.tolist()}
    cu_set = {tuple(map(int, pos)) for pos in sample.start_cu_positions.tolist()}
    valid_indices = np.flatnonzero(sample.candidate_mask > 0)
    for idx in valid_indices.tolist():
        pos = tuple(map(int, sample.candidate_positions[idx].astype(np.int32).tolist()))
        vac_set.discard(pos)
        cu_set.discard(pos)
        new_type = int(projected_types[idx])
        if new_type == V_TYPE:
            vac_set.add(pos)
        elif new_type == CU_TYPE:
            cu_set.add(pos)
    vacancies = np.asarray(sorted(vac_set), dtype=np.int32) if vac_set else np.empty((0, 3), dtype=np.int32)
    cu_positions = np.asarray(sorted(cu_set), dtype=np.int32) if cu_set else np.empty((0, 3), dtype=np.int32)
    return vacancies, cu_positions


def _projected_global_latent_batch(
    *,
    batch: list[MacroSegmentSample],
    projected_types: torch.Tensor,
    model: MacroDreamerEditModel,
    device: str,
) -> torch.Tensor:
    shape = model.global_encoder.shape
    types_np = projected_types.detach().cpu().numpy()

    def _build_one(args: tuple[MacroSegmentSample, np.ndarray]) -> np.ndarray:
        sample, types = args
        vacancies, cu_positions = _apply_projected_types(sample, types)
        proxy_env = _ProjectedStateEnv(vacancies, cu_positions, sample.box_dims)
        share_obs = sample.start_obs[-shape.stats_dim:]
        return build_defect_graph_observation(proxy_env, shape=shape, share_obs=share_obs).astype(np.float32)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
        projected_obs = list(executor.map(_build_one, zip(batch, types_np)))

    projected_obs_t = torch.tensor(np.stack(projected_obs), dtype=torch.float32, device=device)
    return model.encode_global(projected_obs_t)


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    if pred.size > 1 and np.std(pred) > 0 and np.std(target) > 0:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = 0.0
    return {"mae": mae, "rmse": rmse, "corr": corr}


def _focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * ((1.0 - pt).clamp(min=1e-6) ** gamma) * ce).mean()


def _log_target_tau(tau: torch.Tensor) -> torch.Tensor:
    return torch.log(tau.clamp(min=1e-10))


def _scheduled_aux_scale(epoch: int, total_epochs: int, start_fraction: float = 0.55, end_scale: float = 0.1) -> float:
    if total_epochs <= 1:
        return 1.0
    progress = float(max(epoch - 1, 0)) / float(max(total_epochs - 1, 1))
    if progress <= start_fraction:
        return 1.0
    tail_progress = (progress - start_fraction) / max(1.0 - start_fraction, 1e-6)
    return float(1.0 - (1.0 - end_scale) * min(max(tail_progress, 0.0), 1.0))


def _scheduled_posterior_tau_scale(epoch: int, total_epochs: int, start_fraction: float = 0.2, end_scale: float = 0.25) -> float:
    if total_epochs <= 1:
        return 1.0
    progress = float(max(epoch - 1, 0)) / float(max(total_epochs - 1, 1))
    if progress <= start_fraction:
        return 1.0
    tail_progress = (progress - start_fraction) / max(1.0 - start_fraction, 1e-6)
    return float(1.0 - (1.0 - end_scale) * min(max(tail_progress, 0.0), 1.0))


def _selection_score(val_metrics: dict[str, float], dataset_stats: dict[str, object], *, proj_l1_score_weight: float = 80.0) -> float:
    coverage_penalty = max(0.0, 0.9 - float(dataset_stats.get("val", {}).get("coverage", 0.0)))
    projected_global_l1 = float(val_metrics.get("projected_global_l1", 0.0))
    unchanged_vacancy_copy_acc = float(val_metrics.get("unchanged_vacancy_copy_acc", 1.0))
    return (
        val_metrics["tau_log_mae"]
        + 0.5 * val_metrics["reward_mae"]
        + 0.5 * max(0.0, 1.0 - val_metrics["change_topk_f1"])
        + 0.5 * max(0.0, 1.0 - val_metrics["projected_change_f1"])
        + max(0.0, 1.0 - val_metrics["projected_changed_type_acc"])
        + 0.5 * max(0.0, 1.0 - unchanged_vacancy_copy_acc)
        + proj_l1_score_weight * projected_global_l1
        + 2.0 * val_metrics["reachability_violation_rate"]
        + coverage_penalty
    )


def _soft_typed_change_count(
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    type_probs = F.softmax(type_logits, dim=-1)
    current_copy_prob = type_probs.gather(-1, current_types.unsqueeze(-1)).squeeze(-1)
    typed_change_mass = torch.sigmoid(change_logits) * (1.0 - current_copy_prob)
    return (typed_change_mass * candidate_mask).sum(dim=-1)


def _soft_directional_transition_counts(
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    change_prob = torch.sigmoid(change_logits) * candidate_mask
    type_probs = F.softmax(type_logits, dim=-1)
    vacancy_mask = (current_types == V_TYPE).float() * candidate_mask
    fe_mask = (current_types == FE_TYPE).float() * candidate_mask
    cu_mask = (current_types == CU_TYPE).float() * candidate_mask
    return {
        "vac_to_fe": (change_prob * type_probs[..., FE_TYPE] * vacancy_mask).sum(dim=-1),
        "vac_to_cu": (change_prob * type_probs[..., CU_TYPE] * vacancy_mask).sum(dim=-1),
        "fe_to_vac": (change_prob * type_probs[..., V_TYPE] * fe_mask).sum(dim=-1),
        "cu_to_vac": (change_prob * type_probs[..., V_TYPE] * cu_mask).sum(dim=-1),
    }


def _target_directional_transition_counts(
    current_types: torch.Tensor,
    target_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    valid = candidate_mask > 0
    return {
        "vac_to_fe": ((current_types == V_TYPE) & (target_types == FE_TYPE) & valid).float().sum(dim=-1),
        "vac_to_cu": ((current_types == V_TYPE) & (target_types == CU_TYPE) & valid).float().sum(dim=-1),
        "fe_to_vac": ((current_types == FE_TYPE) & (target_types == V_TYPE) & valid).float().sum(dim=-1),
        "cu_to_vac": ((current_types == CU_TYPE) & (target_types == V_TYPE) & valid).float().sum(dim=-1),
    }


def _matched_pair_count_loss(
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    target_types: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    pred_counts = _soft_directional_transition_counts(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )
    target_counts = _target_directional_transition_counts(
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )
    vac_to_atom_loss = 0.5 * (
        F.smooth_l1_loss(pred_counts["vac_to_fe"], target_counts["vac_to_fe"])
        + F.smooth_l1_loss(pred_counts["vac_to_cu"], target_counts["vac_to_cu"])
    )
    atom_to_vac_loss = 0.5 * (
        F.smooth_l1_loss(pred_counts["fe_to_vac"], target_counts["fe_to_vac"])
        + F.smooth_l1_loss(pred_counts["cu_to_vac"], target_counts["cu_to_vac"])
    )
    pred_pair_count = torch.minimum(pred_counts["vac_to_fe"], pred_counts["fe_to_vac"]) + torch.minimum(
        pred_counts["vac_to_cu"], pred_counts["cu_to_vac"]
    )
    target_pair_count = torch.minimum(target_counts["vac_to_fe"], target_counts["fe_to_vac"]) + torch.minimum(
        target_counts["vac_to_cu"], target_counts["cu_to_vac"]
    )
    matched_pair_loss = F.smooth_l1_loss(pred_pair_count, target_pair_count)
    return 0.35 * vac_to_atom_loss + 0.65 * atom_to_vac_loss + 0.5 * matched_pair_loss


def _masked_type_cross_entropy(type_logits: torch.Tensor, target_types: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return F.cross_entropy(type_logits[mask], target_types[mask])
    return torch.zeros((), device=type_logits.device)


def _edit_supervision_losses(
    *,
    change_logits: torch.Tensor,
    type_logits: torch.Tensor,
    current_types: torch.Tensor,
    target_types: torch.Tensor,
    changed_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    aux_scale: float,
    sparsity_weight: float = 0.0,
    count_loss_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    device = change_logits.device
    valid = candidate_mask > 0
    changed_valid = valid & (changed_mask > 0)
    unchanged_valid = valid & (changed_mask <= 0)
    changed_atom_valid = changed_valid & (current_types != V_TYPE)
    changed_vacancy_valid = changed_valid & (current_types == V_TYPE)
    unchanged_atom_valid = unchanged_valid & (current_types != V_TYPE)
    unchanged_vacancy_valid = unchanged_valid & (current_types == V_TYPE)
    atom_to_vac_valid = changed_valid & (current_types != V_TYPE) & (target_types == V_TYPE)
    vac_to_atom_valid = changed_valid & (current_types == V_TYPE) & (target_types != V_TYPE)

    pos_count = changed_mask[valid].sum().clamp(min=1.0)
    neg_count = valid.float().sum().clamp(min=1.0) - pos_count + 1e-6
    pos_weight = (neg_count / pos_count).detach()
    mask_bce = F.binary_cross_entropy_with_logits(change_logits[valid], changed_mask[valid], pos_weight=pos_weight)
    mask_focal = _focal_bce_with_logits(change_logits[valid], changed_mask[valid])

    if changed_atom_valid.any():
        atom_change_loss = F.binary_cross_entropy_with_logits(
            change_logits[changed_atom_valid],
            torch.ones_like(change_logits[changed_atom_valid]),
        )
    else:
        atom_change_loss = torch.zeros((), device=device)
    if changed_vacancy_valid.any():
        vacancy_change_loss = F.binary_cross_entropy_with_logits(
            change_logits[changed_vacancy_valid],
            torch.ones_like(change_logits[changed_vacancy_valid]),
        )
    else:
        vacancy_change_loss = torch.zeros((), device=device)
    if unchanged_vacancy_valid.any():
        vacancy_static_loss = F.binary_cross_entropy_with_logits(
            change_logits[unchanged_vacancy_valid],
            torch.zeros_like(change_logits[unchanged_vacancy_valid]),
        )
    else:
        vacancy_static_loss = torch.zeros((), device=device)

    predicted_change_count = _soft_typed_change_count(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )
    target_change_count = changed_mask.sum(dim=-1)
    count_loss = F.smooth_l1_loss(predicted_change_count, target_change_count)
    pair_count_loss = _matched_pair_count_loss(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )
    if sparsity_weight > 0 and unchanged_valid.any():
        sparsity_loss = torch.sigmoid(change_logits[unchanged_valid]).mean()
    else:
        sparsity_loss = torch.zeros((), device=device)
    mask_loss = (
        mask_bce
        + 0.25 * mask_focal
        + aux_scale * (0.4 * atom_change_loss + 0.4 * vacancy_change_loss + 0.2 * vacancy_static_loss)
        + count_loss_weight * count_loss
        + sparsity_weight * sparsity_loss
    )

    changed_type_loss = _masked_type_cross_entropy(type_logits, target_types, changed_valid)
    atom_to_vac_type_loss = _masked_type_cross_entropy(type_logits, target_types, atom_to_vac_valid)
    vac_to_atom_type_loss = _masked_type_cross_entropy(type_logits, target_types, vac_to_atom_valid)
    unchanged_copy_loss = _masked_type_cross_entropy(type_logits, current_types, unchanged_atom_valid)
    vacancy_type_static_loss = _masked_type_cross_entropy(type_logits, current_types, unchanged_vacancy_valid)
    type_loss = (
        changed_type_loss
        + 0.5 * atom_to_vac_type_loss
        + 0.5 * vac_to_atom_type_loss
        + 0.05 * unchanged_copy_loss
        + 0.25 * vacancy_type_static_loss
    )

    return {
        "mask": mask_loss,
        "count": count_loss,
        "pair": pair_count_loss,
        "type": type_loss,
        "atom_to_vac_type": atom_to_vac_type_loss,
        "vac_to_atom_type": vac_to_atom_type_loss,
    }


def _projected_mask_distill_loss(
    change_logits: torch.Tensor,
    projected_changed_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    reachability_violation: torch.Tensor,
) -> torch.Tensor:
    supervision_mask = valid_mask & (projected_changed_mask > 0) & (reachability_violation.unsqueeze(-1) <= 0)
    if supervision_mask.any():
        return F.binary_cross_entropy_with_logits(change_logits[supervision_mask], projected_changed_mask[supervision_mask])
    return torch.zeros((), device=change_logits.device)


def _projected_state_alignment_loss(
    projected_patch_latent: torch.Tensor,
    target_patch_latent: torch.Tensor,
    projected_global: torch.Tensor,
    next_global: torch.Tensor,
    next_pred: torch.Tensor,
    projected_changed_mask: torch.Tensor,
    reachability_violation: torch.Tensor,
) -> torch.Tensor:
    has_projected_edit = projected_changed_mask.sum(dim=-1) > 0
    success_mask = (reachability_violation <= 0) & has_projected_edit
    if success_mask.any():
        return (
            F.smooth_l1_loss(projected_patch_latent[success_mask], target_patch_latent[success_mask])
            + 0.5 * F.smooth_l1_loss(projected_global[success_mask], next_global[success_mask])
            + 0.5 * F.smooth_l1_loss(projected_global[success_mask], next_pred[success_mask])
        )
    return torch.zeros((), device=projected_patch_latent.device)


def _validate_resume_args(args: argparse.Namespace, ckpt_args: dict[str, object]) -> None:
    resume_segment_k = ckpt_args.get("segment_k")
    if resume_segment_k is not None and int(resume_segment_k) != int(args.segment_k):
        raise ValueError(
            f"Resume checkpoint segment_k={int(resume_segment_k)} does not match current segment_k={int(args.segment_k)}"
        )
    resume_summary_mode = ckpt_args.get("teacher_path_summary_mode")
    if resume_summary_mode is not None and str(resume_summary_mode) != str(args.teacher_path_summary_mode):
        raise ValueError(
            "Resume checkpoint teacher_path_summary_mode="
            f"{resume_summary_mode} does not match current teacher_path_summary_mode={args.teacher_path_summary_mode}"
        )


def _initialize_best_score_from_saved_best(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_changed_sites: int,
    dataset_stats: dict[str, object],
    save_dir: Path,
    checkpoint_best_score: Optional[float] = None,
    allow_checkpoint_best_score_fallback: bool = True,
    proj_l1_score_weight: float = 80.0,
) -> tuple[float, str]:
    current_state = copy.deepcopy(model.state_dict())
    source = "resume checkpoint"
    best_model_path = save_dir / "best_model.pt"
    if best_model_path.exists():
        best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(best_ckpt["model"])
            source = "saved best model"
        except RuntimeError:
            source = "resume checkpoint (skipped incompatible saved best model)"
    best_metrics = _evaluate(model, loader, device, max_changed_sites)
    best_score = _selection_score(best_metrics, dataset_stats, proj_l1_score_weight=proj_l1_score_weight)
    model.load_state_dict(current_state)
    if source.startswith("resume checkpoint") and checkpoint_best_score is not None and allow_checkpoint_best_score_fallback:
        best_score = min(best_score, float(checkpoint_best_score))
        source = f"{source} + stored best_score"
    return best_score, source


def _evaluate(model: MacroDreamerEditModel, loader: DataLoader, device: str, max_changed_sites: int) -> dict[str, float]:
    model.eval()
    reward_pred = []
    reward_true = []
    tau_pred = []
    tau_true = []
    changed_f1_scores = []
    change_topk_f1_scores = []
    changed_type_acc_scores = []
    projected_change_f1_scores = []
    projected_changed_type_acc_scores = []
    unchanged_copy_acc_scores = []
    unchanged_atom_copy_acc_scores = []
    unchanged_vacancy_copy_acc_scores = []
    raw_vac_to_fe_counts = []
    raw_fe_to_vac_counts = []
    raw_vac_to_cu_counts = []
    raw_cu_to_vac_counts = []
    raw_matched_pair_counts = []
    latent_losses = []
    projected_global_losses = []
    reachability_violations = []
    transport_costs = []
    with torch.no_grad():
        for batch in loader:
            tensors = _batch_to_device(batch, device)
            global_latent = model.encode_global(tensors["start_obs"])
            next_global = model.encode_global(tensors["next_obs"])
            site_latent, patch_latent = model.encode_patch(
                positions=tensors["candidate_positions"],
                nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                reach_depth=tensors["reach_depth"],
                is_start_vacancy=tensors["is_start_vacancy"],
                type_ids=tensors["current_types"],
                node_mask=tensors["candidate_mask"],
                global_summary=tensors["global_summary"],
                box_dims=tensors["box_dims"],
            )
            prior_mu, prior_logvar = model.prior_stats(global_latent, tensors["global_summary"], tensors["horizon_k"])
            path_latent = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
            next_pred = model.predict_next_global(global_latent, path_latent, tensors["horizon_k"])
            change_logits, raw_type_logits = model.decode_edit(
                site_latent=site_latent,
                patch_latent=patch_latent,
                predicted_next_global=next_pred,
                path_latent=path_latent,
                horizon_k=tensors["horizon_k"],
                current_types=tensors["current_types"],
            )
            reward_hat, tau_mu, _tau_log_sigma = model.predict_reward_and_duration(
                global_latent, next_pred, path_latent, tensors["global_summary"], tensors["horizon_k"]
            )
            reward_pred.extend(reward_hat.cpu().numpy().tolist())
            reward_true.extend(tensors["reward_sum"].cpu().numpy().tolist())
            tau_pred.extend(torch.exp(tau_mu).cpu().numpy().tolist())
            tau_true.extend(tensors["tau_exp"].cpu().numpy().tolist())

            raw_change = (torch.sigmoid(change_logits) > 0.5).float() * tensors["candidate_mask"]
            target_change = tensors["changed_mask"]
            inter = (raw_change * target_change).sum(dim=-1)
            precision = inter / raw_change.sum(dim=-1).clamp(min=1.0)
            recall = inter / target_change.sum(dim=-1).clamp(min=1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-6)
            changed_f1_scores.extend(f1.cpu().numpy().tolist())

            valid = tensors["candidate_mask"] > 0
            raw_types = raw_type_logits.argmax(dim=-1)
            changed_valid = valid & (tensors["changed_mask"] > 0)
            unchanged_valid = valid & (tensors["changed_mask"] <= 0)
            unchanged_atom_valid = unchanged_valid & (tensors["current_types"] != V_TYPE)
            unchanged_vacancy_valid = unchanged_valid & (tensors["current_types"] == V_TYPE)
            changed_type_acc = (raw_types[changed_valid] == tensors["target_types"][changed_valid]).float().mean().item() if changed_valid.any() else 1.0
            unchanged_copy_acc = (raw_types[unchanged_valid] == tensors["current_types"][unchanged_valid]).float().mean().item() if unchanged_valid.any() else 1.0
            unchanged_atom_copy_acc = (
                (raw_types[unchanged_atom_valid] == tensors["current_types"][unchanged_atom_valid]).float().mean().item()
                if unchanged_atom_valid.any()
                else 1.0
            )
            unchanged_vacancy_copy_acc = (
                (raw_types[unchanged_vacancy_valid] == tensors["current_types"][unchanged_vacancy_valid]).float().mean().item()
                if unchanged_vacancy_valid.any()
                else 1.0
            )
            changed_type_acc_scores.append(changed_type_acc)
            unchanged_copy_acc_scores.append(unchanged_copy_acc)
            unchanged_atom_copy_acc_scores.append(unchanged_atom_copy_acc)
            unchanged_vacancy_copy_acc_scores.append(unchanged_vacancy_copy_acc)

            for sample_idx in range(tensors["current_types"].shape[0]):
                sample_valid = valid[sample_idx]
                sample_current = tensors["current_types"][sample_idx, sample_valid]
                sample_pred = raw_types[sample_idx, sample_valid]
                vac_to_fe = float(((sample_current == V_TYPE) & (sample_pred == FE_TYPE)).sum().item())
                fe_to_vac = float(((sample_current == FE_TYPE) & (sample_pred == V_TYPE)).sum().item())
                vac_to_cu = float(((sample_current == V_TYPE) & (sample_pred == CU_TYPE)).sum().item())
                cu_to_vac = float(((sample_current == CU_TYPE) & (sample_pred == V_TYPE)).sum().item())
                raw_vac_to_fe_counts.append(vac_to_fe)
                raw_fe_to_vac_counts.append(fe_to_vac)
                raw_vac_to_cu_counts.append(vac_to_cu)
                raw_cu_to_vac_counts.append(cu_to_vac)
                raw_matched_pair_counts.append(min(vac_to_fe, fe_to_vac) + min(vac_to_cu, cu_to_vac))

            projected_types, _, proj_transport_cost, proj_violation = project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=change_logits,
                type_logits=raw_type_logits,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=max_changed_sites,
            )
            projected_changed_mask = ((projected_types != tensors["current_types"]).float() * tensors["candidate_mask"])
            proj_changed_acc = (projected_types[changed_valid] == tensors["target_types"][changed_valid]).float().mean().item() if changed_valid.any() else 1.0
            projected_changed_type_acc_scores.append(proj_changed_acc)
            reachability_violations.extend(proj_violation.cpu().numpy().tolist())
            transport_costs.extend(proj_transport_cost.cpu().numpy().tolist())
            latent_losses.append(F.smooth_l1_loss(next_pred, next_global).item())
            projected_global = _projected_global_latent_batch(batch=batch, projected_types=projected_types, model=model, device=device)
            projected_global_losses.append(F.smooth_l1_loss(projected_global, next_global).item())

            change_probs = torch.sigmoid(change_logits)
            type_probs = torch.softmax(raw_type_logits, dim=-1)
            current_conf = type_probs.gather(-1, tensors["current_types"].unsqueeze(-1)).squeeze(-1)
            type_change_score = 1.0 - current_conf
            combined_scores = 0.5 * change_probs + 0.5 * type_change_score
            for sample_idx in range(tensors["current_types"].shape[0]):
                valid_idx = torch.nonzero(valid[sample_idx], as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    change_topk_f1_scores.append(1.0)
                    projected_change_f1_scores.append(1.0)
                    continue
                target_local = target_change[sample_idx, valid_idx]
                target_count = int(target_local.sum().item())
                if target_count <= 0:
                    change_topk_f1_scores.append(1.0)
                else:
                    ranked_local = torch.argsort(combined_scores[sample_idx, valid_idx], descending=True)[:target_count]
                    topk_pred = torch.zeros_like(target_local)
                    topk_pred[ranked_local] = 1.0
                    topk_inter = float((topk_pred * target_local).sum().item())
                    topk_precision = topk_inter / max(float(topk_pred.sum().item()), 1.0)
                    topk_recall = topk_inter / max(float(target_local.sum().item()), 1.0)
                    topk_f1 = 2.0 * topk_precision * topk_recall / max(topk_precision + topk_recall, 1e-6)
                    change_topk_f1_scores.append(float(topk_f1))

                proj_local = projected_changed_mask[sample_idx, valid_idx]
                proj_inter = float((proj_local * target_local).sum().item())
                proj_precision = proj_inter / max(float(proj_local.sum().item()), 1.0)
                proj_recall = proj_inter / max(float(target_local.sum().item()), 1.0)
                proj_f1 = 2.0 * proj_precision * proj_recall / max(proj_precision + proj_recall, 1e-6)
                projected_change_f1_scores.append(float(proj_f1))

    reward_metrics = _compute_metrics(np.asarray(reward_pred), np.asarray(reward_true))
    tau_metrics = _compute_metrics(np.log(np.asarray(tau_pred) + 1e-12), np.log(np.asarray(tau_true) + 1e-12))
    tau_scale = float(np.mean(np.asarray(tau_pred) / np.maximum(np.asarray(tau_true), 1e-12)))
    return {
        "reward_mae": reward_metrics["mae"],
        "reward_rmse": reward_metrics["rmse"],
        "reward_corr": reward_metrics["corr"],
        "tau_log_mae": tau_metrics["mae"],
        "tau_log_rmse": tau_metrics["rmse"],
        "tau_log_corr": tau_metrics["corr"],
        "tau_scale_ratio": tau_scale,
        "change_f1": float(np.mean(changed_f1_scores)) if changed_f1_scores else 0.0,
        "change_topk_f1": float(np.mean(change_topk_f1_scores)) if change_topk_f1_scores else 0.0,
        "changed_type_acc": float(np.mean(changed_type_acc_scores)) if changed_type_acc_scores else 0.0,
        "projected_change_f1": float(np.mean(projected_change_f1_scores)) if projected_change_f1_scores else 0.0,
        "projected_changed_type_acc": float(np.mean(projected_changed_type_acc_scores)) if projected_changed_type_acc_scores else 0.0,
        "unchanged_copy_acc": float(np.mean(unchanged_copy_acc_scores)) if unchanged_copy_acc_scores else 0.0,
        "unchanged_atom_copy_acc": float(np.mean(unchanged_atom_copy_acc_scores)) if unchanged_atom_copy_acc_scores else 0.0,
        "unchanged_vacancy_copy_acc": float(np.mean(unchanged_vacancy_copy_acc_scores)) if unchanged_vacancy_copy_acc_scores else 0.0,
        "raw_vac_to_fe_count": float(np.mean(raw_vac_to_fe_counts)) if raw_vac_to_fe_counts else 0.0,
        "raw_fe_to_vac_count": float(np.mean(raw_fe_to_vac_counts)) if raw_fe_to_vac_counts else 0.0,
        "raw_vac_to_cu_count": float(np.mean(raw_vac_to_cu_counts)) if raw_vac_to_cu_counts else 0.0,
        "raw_cu_to_vac_count": float(np.mean(raw_cu_to_vac_counts)) if raw_cu_to_vac_counts else 0.0,
        "raw_matched_pair_count": float(np.mean(raw_matched_pair_counts)) if raw_matched_pair_counts else 0.0,
        "latent_l1": float(np.mean(latent_losses)) if latent_losses else 0.0,
        "projected_global_l1": float(np.mean(projected_global_losses)) if projected_global_losses else 0.0,
        "reachability_violation_rate": float(np.mean(reachability_violations)) if reachability_violations else 0.0,
        "mean_vacancy_transport_cost": float(np.mean(transport_costs)) if transport_costs else 0.0,
    }


def _train_epoch(
    model: MacroDreamerEditModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_changed_sites: int,
    weights: dict[str, float],
    *,
    epoch: int = 1,
    total_epochs: int = 1,
    tau_supervision_mode: str = "prior_main",
    proj_every_n_batches: int = 1,
    aux_anneal: bool = True,
    mask_sparsity_weight: float = 0.0,
    count_loss_weight: float = 0.1,
    detach_proj_encoder: bool = False,
) -> dict[str, float]:
    model.train()
    aux_scale = _scheduled_aux_scale(epoch, total_epochs) if aux_anneal else 1.0
    logs = {
        "loss": 0.0,
        "mask": 0.0,
        "count": 0.0,
        "pair": 0.0,
        "proj_mask": 0.0,
        "type": 0.0,
        "atom_to_vac_type": 0.0,
        "vac_to_atom_type": 0.0,
        "tau": 0.0,
        "tau_post": 0.0,
        "tau_prior": 0.0,
        "tau_post_scale": 0.0,
        "reward": 0.0,
        "latent": 0.0,
        "proj": 0.0,
        "path": 0.0,
        "prior_edit": 0.0,
        "prior_latent": 0.0,
        "mask_aux_scale": 0.0,
    }
    count = 0
    proj_scale = float(max(proj_every_n_batches, 1))
    for batch_idx, batch in enumerate(loader):
        compute_proj = (proj_every_n_batches <= 1) or (batch_idx % proj_every_n_batches == 0)
        tensors = _batch_to_device(batch, device)
        global_latent = model.encode_global(tensors["start_obs"])
        next_global = model.encode_global(tensors["next_obs"]).detach()
        site_latent, patch_latent = model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=tensors["current_types"],
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        target_site_latent, target_patch_latent = model.encode_patch(
            positions=tensors["candidate_positions"],
            nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
            reach_depth=tensors["reach_depth"],
            is_start_vacancy=tensors["is_start_vacancy"],
            type_ids=tensors["target_types"],
            node_mask=tensors["candidate_mask"],
            global_summary=tensors["global_summary"],
            box_dims=tensors["box_dims"],
        )
        prior_mu, prior_logvar = model.prior_stats(global_latent, tensors["global_summary"], tensors["horizon_k"])
        post_mu, post_logvar = model.posterior_stats(global_latent, next_global, tensors["teacher_path_summary"], tensors["horizon_k"])
        post_c = model.sample_path_latent(post_mu, post_logvar)
        prior_c = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
        next_pred = model.predict_next_global(global_latent, post_c, tensors["horizon_k"])
        next_pred_prior = model.predict_next_global(global_latent, prior_c, tensors["horizon_k"])
        change_logits, raw_type_logits = model.decode_edit(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred,
            path_latent=post_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        change_logits_prior, raw_type_logits_prior = model.decode_edit(
            site_latent=site_latent,
            patch_latent=patch_latent,
            predicted_next_global=next_pred_prior,
            path_latent=prior_c,
            horizon_k=tensors["horizon_k"],
            current_types=tensors["current_types"],
        )
        reward_hat, tau_mu, tau_log_sigma = model.predict_reward_and_duration(
            global_latent,
            next_pred,
            post_c,
            tensors["global_summary"],
            tensors["horizon_k"],
            detach_duration_inputs=True,
        )
        reward_hat_prior, tau_mu_prior, tau_log_sigma_prior = model.predict_reward_and_duration(
            global_latent,
            next_pred_prior,
            prior_c,
            tensors["global_summary"],
            tensors["horizon_k"],
            detach_duration_inputs=True,
        )
        if compute_proj:
            projected_types, _, transport_cost, reachability_violation = project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=change_logits,
                type_logits=raw_type_logits,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=max_changed_sites,
            )
            projected_changed_mask = ((projected_types != tensors["current_types"]).float() * tensors["candidate_mask"]).detach()
            _, projected_patch_latent = model.encode_patch(
                positions=tensors["candidate_positions"],
                nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                reach_depth=tensors["reach_depth"],
                is_start_vacancy=tensors["is_start_vacancy"],
                type_ids=projected_types,
                node_mask=tensors["candidate_mask"],
                global_summary=tensors["global_summary"],
                box_dims=tensors["box_dims"],
            )
            projected_global = _projected_global_latent_batch(batch=batch, projected_types=projected_types, model=model, device=device)
            projected_types_prior, _, _prior_transport_cost, prior_reachability_violation = project_types_by_inventory(
                current_types=tensors["current_types"],
                change_logits=change_logits_prior,
                type_logits=raw_type_logits_prior,
                node_mask=tensors["candidate_mask"],
                positions=tensors["candidate_positions"],
                box_dims=tensors["box_dims"],
                horizon_k=tensors["horizon_k"],
                max_changed_sites=max_changed_sites,
            )
            projected_changed_mask_prior = ((projected_types_prior != tensors["current_types"]).float() * tensors["candidate_mask"]).detach()
            _, projected_patch_latent_prior = model.encode_patch(
                positions=tensors["candidate_positions"],
                nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
                reach_depth=tensors["reach_depth"],
                is_start_vacancy=tensors["is_start_vacancy"],
                type_ids=projected_types_prior,
                node_mask=tensors["candidate_mask"],
                global_summary=tensors["global_summary"],
                box_dims=tensors["box_dims"],
            )
            projected_global_prior = _projected_global_latent_batch(
                batch=batch,
                projected_types=projected_types_prior,
                model=model,
                device=device,
            )

        valid = tensors["candidate_mask"] > 0
        posterior_edit = _edit_supervision_losses(
            change_logits=change_logits,
            type_logits=raw_type_logits,
            current_types=tensors["current_types"],
            target_types=tensors["target_types"],
            changed_mask=tensors["changed_mask"],
            candidate_mask=tensors["candidate_mask"],
            aux_scale=aux_scale,
            sparsity_weight=mask_sparsity_weight,
            count_loss_weight=count_loss_weight,
        )
        prior_edit = _edit_supervision_losses(
            change_logits=change_logits_prior,
            type_logits=raw_type_logits_prior,
            current_types=tensors["current_types"],
            target_types=tensors["target_types"],
            changed_mask=tensors["changed_mask"],
            candidate_mask=tensors["candidate_mask"],
            aux_scale=aux_scale,
            sparsity_weight=mask_sparsity_weight,
            count_loss_weight=count_loss_weight,
        )
        count_loss = posterior_edit["count"]
        pair_count_loss = posterior_edit["pair"]
        if compute_proj:
            proj_mask_loss = _projected_mask_distill_loss(
                change_logits=change_logits,
                projected_changed_mask=projected_changed_mask,
                valid_mask=valid,
                reachability_violation=reachability_violation,
            ) * proj_scale
            prior_proj_mask_loss = _projected_mask_distill_loss(
                change_logits=change_logits_prior,
                projected_changed_mask=projected_changed_mask_prior,
                valid_mask=valid,
                reachability_violation=prior_reachability_violation,
            ) * proj_scale
        else:
            proj_mask_loss = torch.tensor(0.0, device=device)
            prior_proj_mask_loss = torch.tensor(0.0, device=device)
        mask_loss = posterior_edit["mask"] + 0.5 * proj_mask_loss

        type_loss = posterior_edit["type"]
        atom_to_vac_type_loss = posterior_edit["atom_to_vac_type"]
        vac_to_atom_type_loss = posterior_edit["vac_to_atom_type"]
        prior_edit_loss = prior_edit["mask"] + 0.1 * prior_edit["count"] + (0.5 * prior_proj_mask_loss if compute_proj else 0.0) + prior_edit["type"]

        tau_loss = lognormal_nll(tensors["tau_exp"], tau_mu, tau_log_sigma).mean()
        reward_loss = F.smooth_l1_loss(reward_hat, tensors["reward_sum"])
        latent_loss = F.smooth_l1_loss(next_pred, next_global)
        if compute_proj:
            proj_state_loss = _projected_state_alignment_loss(
                projected_patch_latent=projected_patch_latent.detach() if detach_proj_encoder else projected_patch_latent,
                target_patch_latent=target_patch_latent.detach(),
                projected_global=projected_global.detach() if detach_proj_encoder else projected_global,
                next_global=next_global,
                next_pred=next_pred.detach(),
                projected_changed_mask=projected_changed_mask,
                reachability_violation=reachability_violation,
            ) * proj_scale
            prior_proj_state_loss = _projected_state_alignment_loss(
                projected_patch_latent=projected_patch_latent_prior.detach() if detach_proj_encoder else projected_patch_latent_prior,
                target_patch_latent=target_patch_latent.detach(),
                projected_global=projected_global_prior.detach() if detach_proj_encoder else projected_global_prior,
                next_global=next_global,
                next_pred=next_pred_prior.detach(),
                projected_changed_mask=projected_changed_mask_prior,
                reachability_violation=prior_reachability_violation,
            ) * proj_scale
        else:
            proj_state_loss = torch.tensor(0.0, device=device)
            prior_proj_state_loss = torch.tensor(0.0, device=device)
        prior_latent_loss = F.smooth_l1_loss(next_pred_prior, next_global) + 0.5 * prior_proj_state_loss
        path_loss = kl_divergence_diag_gaussian(post_mu, post_logvar, prior_mu, prior_logvar).mean()
        prior_tau_loss = lognormal_nll(tensors["tau_exp"], tau_mu_prior, tau_log_sigma_prior).mean()
        prior_reward_loss = F.smooth_l1_loss(reward_hat_prior, tensors["reward_sum"])
        if tau_supervision_mode == "posterior_only":
            combined_tau_loss = tau_loss
            effective_tau_post_scale = 1.0
        else:
            combined_tau_loss = prior_tau_loss
            effective_tau_post_scale = 0.0

        main_loss = (
            weights["mask"] * mask_loss
            + weights["type"] * type_loss
            + weights["pair"] * pair_count_loss
            + weights["reward"] * reward_loss
            + weights["latent"] * latent_loss
            + weights["proj"] * proj_state_loss
            + weights["path"] * path_loss
            + weights["prior_edit"] * prior_edit_loss
            + weights["prior_latent"] * prior_latent_loss
            + 0.5 * weights["reward"] * prior_reward_loss
        )
        tau_total_loss = weights["tau"] * combined_tau_loss
        loss = main_loss + tau_total_loss
        optimizer.zero_grad()
        main_loss.backward()
        tau_total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = len(batch)
        count += batch_size
        logs["loss"] += float(loss.item()) * batch_size
        logs["mask"] += float(mask_loss.item()) * batch_size
        logs["count"] += float(count_loss.item()) * batch_size
        logs["pair"] += float(pair_count_loss.item()) * batch_size
        logs["proj_mask"] += float(proj_mask_loss.item()) * batch_size
        logs["type"] += float(type_loss.item()) * batch_size
        logs["atom_to_vac_type"] += float(atom_to_vac_type_loss.item()) * batch_size
        logs["vac_to_atom_type"] += float(vac_to_atom_type_loss.item()) * batch_size
        logs["tau"] += float(combined_tau_loss.item()) * batch_size
        logs["tau_post"] += float(tau_loss.item()) * batch_size
        logs["tau_prior"] += float(prior_tau_loss.item()) * batch_size
        logs["tau_post_scale"] += float(effective_tau_post_scale) * batch_size
        logs["reward"] += float(reward_loss.item()) * batch_size
        logs["latent"] += float(latent_loss.item()) * batch_size
        logs["proj"] += float(proj_state_loss.item()) * batch_size
        logs["path"] += float(path_loss.item()) * batch_size
        logs["prior_edit"] += float(prior_edit_loss.item()) * batch_size
        logs["prior_latent"] += float(prior_latent_loss.item()) * batch_size
        logs["mask_aux_scale"] += float(aux_scale) * batch_size

    if count == 0:
        return logs
    return {key: value / count for key, value in logs.items()}


def _build_loader(samples: list[MacroSegmentSample], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(MacroSegmentDataset(samples), batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: batch)


def _initialize_output_heads(model: MacroDreamerEditModel, train_samples: list[MacroSegmentSample]) -> None:
    if not train_samples:
        return
    reward_mean = float(np.mean([sample.reward_sum for sample in train_samples]))
    tau_residuals = []
    for sample in train_samples:
        baseline_log_tau = math.log(max(int(sample.horizon_k), 1)) - float(sample.global_summary[10])
        tau_residuals.append(math.log(sample.tau_exp + 1e-12) - baseline_log_tau)
    tau_residual_mean = float(np.mean(tau_residuals))
    tau_residual_std = max(float(np.std(tau_residuals)), 0.2)
    valid_site_count = float(np.sum([sample.candidate_mask.sum() for sample in train_samples]))
    changed_site_count = float(np.sum([sample.changed_mask.sum() for sample in train_samples]))
    changed_rate = changed_site_count / max(valid_site_count, 1.0)
    sparse_prior = min(max(changed_rate, 1e-4), 5e-2)
    change_bias = math.log(sparse_prior) - math.log(1.0 - sparse_prior)
    with torch.no_grad():
        model.change_head.weight.zero_()
        model.change_head.bias.fill_(change_bias)
        model.type_head.weight.zero_()
        model.type_head.bias.zero_()
        model.reward_head[-1].weight.zero_()
        model.reward_head[-1].bias.fill_(reward_mean)
        model.duration_head[-1].weight.zero_()
        model.duration_head[-1].bias[0] = tau_residual_mean
        model.duration_head[-1].bias[1] = float(np.log(tau_residual_std))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dreamer fixed-k macro edit training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="results/dreamer_macro_edit_v1")
    parser.add_argument("--dataset_cache", type=str, default="results/dreamer_macro_edit_v1/segments.pt")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    parser.add_argument("--cu_density", type=float, default=0.0134)
    parser.add_argument("--v_density", type=float, default=0.0002)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--max_vacancies", type=int, default=32)
    parser.add_argument("--max_defects", type=int, default=64)
    parser.add_argument("--max_shells", type=int, default=16)
    parser.add_argument("--stats_dim", type=int, default=10)
    parser.add_argument("--segment_k", type=int, default=4)
    parser.add_argument("--train_segments", type=int, default=2000)
    parser.add_argument("--val_segments", type=int, default=400)
    parser.add_argument("--max_seed_vacancies", type=int, default=8)
    parser.add_argument("--max_candidate_sites", type=int, default=128)
    parser.add_argument("--dim_latent", type=int, default=16)
    parser.add_argument("--graph_hidden_size", type=int, default=32)
    parser.add_argument("--patch_hidden_size", type=int, default=96)
    parser.add_argument("--patch_latent_dim", type=int, default=64)
    parser.add_argument("--path_latent_dim", type=int, default=32)
    parser.add_argument("--teacher_path_summary_mode", type=str, default="stepwise", choices=["stepwise", "legacy"])
    parser.add_argument("--tau_supervision_mode", type=str, default="prior_main", choices=["prior_main", "posterior_only"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj_every_n_batches", type=int, default=1,
                        help="Compute projection losses every N batches (1=every batch). Higher values reduce CPU overhead from projected-global re-encoding.")
    parser.add_argument("--no_aux_anneal", action="store_true",
                        help="Disable aux_scale annealing; keep auxiliary mask losses at full strength throughout training.")
    parser.add_argument("--mask_sparsity_weight", type=float, default=0.0,
                        help="Weight for L1 sparsity penalty on change probabilities of unchanged sites.")
    parser.add_argument("--count_loss_weight", type=float, default=0.1,
                        help="Weight for count_loss within mask_loss (default: 0.1).")
    parser.add_argument("--detach_proj_encoder", action="store_true",
                        help="Detach encoder outputs (projected_global, projected_patch_latent) in proj_state_loss, preventing projection consistency gradients from flowing into the encoder.")
    parser.add_argument("--proj_weight", type=float, default=0.5,
                        help="Training loss weight for proj_state_loss (default: 0.5). Higher values strengthen encoder projection consistency.")
    parser.add_argument("--proj_l1_score_weight", type=float, default=80.0,
                        help="Weight for proj_global_l1 in the selection score formula (default: 80.0). Lower values reduce proj_l1 dominance in model selection.")
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def _dataset_signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_version": 7,
        "seed": int(args.seed),
        "lattice_size": list(args.lattice_size),
        "cu_density": float(args.cu_density),
        "v_density": float(args.v_density),
        "segment_k": int(args.segment_k),
        "max_seed_vacancies": int(args.max_seed_vacancies),
        "max_candidate_sites": int(args.max_candidate_sites),
        "max_episode_steps": int(args.max_episode_steps),
        "max_vacancies": int(args.max_vacancies),
        "max_defects": int(args.max_defects),
        "max_shells": int(args.max_shells),
        "neighbor_order": str(args.neighbor_order),
        "reward_scale": float(args.reward_scale),
        "temperature": float(args.temperature),
        "stats_dim": int(args.stats_dim),
        "train_segments": int(args.train_segments),
        "val_segments": int(args.val_segments),
        "teacher_path_summary_mode": str(args.teacher_path_summary_mode),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache = Path(args.dataset_cache)
    dataset_cache.parent.mkdir(parents=True, exist_ok=True)
    dataset_signature = _dataset_signature(args)

    env_cfg = {
        "lattice_size": tuple(args.lattice_size),
        "max_episode_steps": args.max_episode_steps,
        "max_vacancies": args.max_vacancies,
        "max_defects": args.max_defects,
        "max_shells": args.max_shells,
        "stats_dim": args.stats_dim,
        "temperature": args.temperature,
        "reward_scale": args.reward_scale,
        "cu_density": args.cu_density,
        "v_density": args.v_density,
        "rlkmc_topk": 16,
        "neighbor_order": args.neighbor_order,
    }
    include_stepwise_path_summary = args.teacher_path_summary_mode == "stepwise"

    if dataset_cache.exists():
        payload = torch.load(dataset_cache, map_location="cpu", weights_only=False)
        cached_signature = payload.get("signature")
        if cached_signature != dataset_signature:
            print("Cached dataset signature mismatch; regenerating dataset.")
            dataset_cache.unlink()
            payload = None
        else:
            train_samples = [MacroSegmentSample(**item) for item in payload["train"]]
            val_samples = [MacroSegmentSample(**item) for item in payload["val"]]
            dataset_stats = payload.get("stats", {})
            print(f"Loaded cached dataset from {dataset_cache}")
    else:
        payload = None
    if payload is None:
        train_env = MacroKMCEnv(env_cfg)
        val_env = MacroKMCEnv(env_cfg)
        train_rng = np.random.default_rng(args.seed)
        val_rng = np.random.default_rng(args.seed + 1)
        train_samples, train_stats = _collect_segments(
            env=train_env,
            num_segments=args.train_segments,
            horizon_k=args.segment_k,
            max_seed_vacancies=args.max_seed_vacancies,
            max_candidate_sites=args.max_candidate_sites,
            rng=train_rng,
            include_stepwise_path_summary=include_stepwise_path_summary,
        )
        val_samples, val_stats = _collect_segments(
            env=val_env,
            num_segments=args.val_segments,
            horizon_k=args.segment_k,
            max_seed_vacancies=args.max_seed_vacancies,
            max_candidate_sites=args.max_candidate_sites,
            rng=val_rng,
            include_stepwise_path_summary=include_stepwise_path_summary,
        )
        dataset_stats = {"train": train_stats, "val": val_stats}
        torch.save(
            {
                "train": [asdict(sample) for sample in train_samples],
                "val": [asdict(sample) for sample in val_samples],
                "stats": dataset_stats,
                "signature": dataset_signature,
            },
            dataset_cache,
        )
        print(json.dumps({"dataset_stats": dataset_stats}, ensure_ascii=False))

    train_loader = _build_loader(train_samples, args.batch_size, shuffle=True)
    val_loader = _build_loader(val_samples, args.batch_size, shuffle=False)
    model = MacroDreamerEditModel(
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=args.stats_dim,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        dim_latent=args.dim_latent,
        graph_hidden_size=args.graph_hidden_size,
        patch_hidden_size=args.patch_hidden_size,
        patch_latent_dim=args.patch_latent_dim,
        path_latent_dim=args.path_latent_dim,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(args.segment_k, include_stepwise_features=include_stepwise_path_summary),
        max_macro_k=max(args.segment_k, 16),
    ).to(args.device)
    if not args.resume:
        _initialize_output_heads(model, train_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    weights = {
        "mask": 1.0,
        "type": 1.0,
        "pair": 0.0,
        "tau": 1.0,
        "reward": 0.5,
        "latent": 0.5,
        "proj": args.proj_weight,
        "path": 0.05,
        "prior_edit": 0.25,
        "prior_latent": 0.25,
    }
    max_changed_sites = 2 * args.segment_k
    best_score = float("inf")
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        _validate_resume_args(args, ckpt.get("args", {}))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        if not args.eval_only:
            allow_checkpoint_best_score_fallback = Path(args.resume).resolve().parent == save_dir.resolve()
            best_score, score_source = _initialize_best_score_from_saved_best(
                model=model,
                loader=val_loader,
                device=args.device,
                max_changed_sites=max_changed_sites,
                dataset_stats=dataset_stats,
                save_dir=save_dir,
                checkpoint_best_score=ckpt.get("best_score"),
                allow_checkpoint_best_score_fallback=allow_checkpoint_best_score_fallback,
                proj_l1_score_weight=args.proj_l1_score_weight,
            )
            print(f"Initialized best_score from {score_source} under current selection metric: {best_score:.4f}")
        else:
            best_score = float(ckpt.get("best_score", best_score))

    if args.eval_only:
        metrics = _evaluate(model, val_loader, args.device, max_changed_sites)
        print(json.dumps({"val": metrics, "dataset": dataset_stats}, ensure_ascii=False, indent=2))
        return

    log_path = save_dir / "training_log.txt"
    metrics_path = save_dir / "metrics.json"
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            max_changed_sites,
            weights,
            epoch=epoch,
            total_epochs=args.epochs,
            tau_supervision_mode=args.tau_supervision_mode,
            proj_every_n_batches=args.proj_every_n_batches,
            aux_anneal=not args.no_aux_anneal,
            mask_sparsity_weight=args.mask_sparsity_weight,
            count_loss_weight=args.count_loss_weight,
            detach_proj_encoder=args.detach_proj_encoder,
        )
        elapsed = time.time() - t0
        train_msg = (
            f"[Epoch {epoch:03d}/{args.epochs}] loss={train_metrics['loss']:.4f} "
            f"mask={train_metrics['mask']:.4f} count={train_metrics['count']:.4f} pair={train_metrics['pair']:.4f} proj_mask={train_metrics['proj_mask']:.4f} type={train_metrics['type']:.4f} "
            f"prior_edit={train_metrics['prior_edit']:.4f} "
            f"tau={train_metrics['tau']:.4f} tau_prior={train_metrics['tau_prior']:.4f} "
            f"tau_post={train_metrics['tau_post']:.4f} tau_post_scale={train_metrics['tau_post_scale']:.2f} reward={train_metrics['reward']:.4f} "
            f"latent={train_metrics['latent']:.4f} proj={train_metrics['proj']:.4f} "
            f"path={train_metrics['path']:.4f} prior_latent={train_metrics['prior_latent']:.4f} "
            f"mask_aux={train_metrics['mask_aux_scale']:.2f} time={elapsed:.1f}s"
        )
        print(train_msg)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(train_msg + "\n")

        if epoch % args.eval_freq == 0 or epoch == 1:
            val_metrics = _evaluate(model, val_loader, args.device, max_changed_sites)
            val_msg = (
                f"  >>> VAL reward_mae={val_metrics['reward_mae']:.4f} reward_corr={val_metrics['reward_corr']:.4f} "
                f"tau_log_mae={val_metrics['tau_log_mae']:.4f} tau_log_corr={val_metrics['tau_log_corr']:.4f} "
                f"tau_scale={val_metrics['tau_scale_ratio']:.2f} change_f1={val_metrics['change_f1']:.4f} "
                f"change_topk_f1={val_metrics['change_topk_f1']:.4f} proj_change_f1={val_metrics['projected_change_f1']:.4f} "
                f"chg_type_acc={val_metrics['changed_type_acc']:.4f} proj_chg_type_acc={val_metrics['projected_changed_type_acc']:.4f} "
                f"pair_fe={val_metrics['raw_vac_to_fe_count']:.2f}/{val_metrics['raw_fe_to_vac_count']:.2f} "
                f"pair_cu={val_metrics['raw_vac_to_cu_count']:.2f}/{val_metrics['raw_cu_to_vac_count']:.2f} "
                f"matched_pair={val_metrics['raw_matched_pair_count']:.2f} "
                f"unchg_copy_acc={val_metrics['unchanged_copy_acc']:.4f} vac_copy_acc={val_metrics['unchanged_vacancy_copy_acc']:.4f} "
                f"reach_violation={val_metrics['reachability_violation_rate']:.4f} proj_global_l1={val_metrics['projected_global_l1']:.4f}"
            )
            print(val_msg)
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(val_msg + "\n")
            metrics_payload = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "dataset": dataset_stats,
            }
            score = _selection_score(val_metrics, dataset_stats, proj_l1_score_weight=args.proj_l1_score_weight)
            metrics_payload["selection_score"] = score
            metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if score < best_score:
                best_score = score
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "args": vars(args),
                        "dataset": dataset_stats,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  >>> New best model: score={best_score:.4f}")

        if epoch % args.save_freq == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "args": vars(args),
                    "dataset": dataset_stats,
                },
                save_dir / f"checkpoint_{epoch}.pt",
            )

    final_metrics = _evaluate(model, val_loader, args.device, max_changed_sites)
    print(json.dumps({"final_val": final_metrics, "dataset": dataset_stats}, ensure_ascii=False, indent=2))
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "best_score": best_score,
            "args": vars(args),
            "dataset": dataset_stats,
            "final_val": final_metrics,
        },
        save_dir / "final_model.pt",
    )


if __name__ == "__main__":
    main()