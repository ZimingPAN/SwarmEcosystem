from __future__ import annotations

import numpy as np

import train_dreamer_macro_edit as mod


def _stats(name: str, data: list[mod.MacroSegmentSample]) -> None:
    changed = np.asarray([float(np.sum(sample.changed_mask)) for sample in data], dtype=np.float64)
    tau = np.asarray([float(sample.tau_exp) for sample in data], dtype=np.float64)
    reward = np.asarray([float(sample.reward_sum) for sample in data], dtype=np.float64)
    print(
        f"{name} n={len(data)} changed_mean={changed.mean():.4f} "
        f"zero_frac={np.mean(changed == 0.0):.4f} tau_mean={tau.mean():.6e} "
        f"tau_std={tau.std():.6e} reward_mean={reward.mean():.6f} reward_std={reward.std():.6f}"
    )


def main() -> None:
    env_cfg = {
        "lattice_size": (40, 40, 40),
        "max_episode_steps": 200,
        "max_vacancies": 32,
        "max_defects": 64,
        "max_shells": 16,
        "stats_dim": 10,
        "temperature": 300.0,
        "reward_scale": 10.0,
        "cu_density": 0.0134,
        "v_density": 0.0002,
        "rlkmc_topk": 16,
        "neighbor_order": "2NN",
    }

    rng = np.random.default_rng(0)
    shared_env = mod.MacroKMCEnv(env_cfg)
    train_samples, train_stats = mod._collect_segments(
        env=shared_env,
        num_segments=50,
        horizon_k=4,
        max_seed_vacancies=32,
        max_candidate_sites=384,
        rng=rng,
    )
    val_same_env, val_same_stats = mod._collect_segments(
        env=shared_env,
        num_segments=20,
        horizon_k=4,
        max_seed_vacancies=32,
        max_candidate_sites=384,
        rng=rng,
    )

    fresh_env = mod.MacroKMCEnv(env_cfg)
    val_fresh_env, val_fresh_stats = mod._collect_segments(
        env=fresh_env,
        num_segments=20,
        horizon_k=4,
        max_seed_vacancies=32,
        max_candidate_sites=384,
        rng=np.random.default_rng(1),
    )

    _stats("train", train_samples)
    _stats("val_same_env", val_same_env)
    _stats("val_fresh_env", val_fresh_env)
    print(f"stats_same_env={val_same_stats}")
    print(f"stats_fresh_env={val_fresh_stats}")


if __name__ == "__main__":
    main()