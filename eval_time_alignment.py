#!/usr/bin/env python3
"""
Time alignment evaluation for world models vs traditional KMC.

Core rule: a deterministic state-only time head should be evaluated against the
state-identifiable target E[Δt | s] = 1 / Γ_tot(s), not against one realized
Poisson sample -log(u) / Γ_tot(s).
"""
import sys, os, argparse, random, json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "RLKMC-MASSIVE-main"))
sys.path.insert(0, os.path.join(ROOT, "LightZero-main"))
sys.path.insert(0, os.path.join(ROOT, "dreamer4-main"))
pydeps = os.path.expanduser("/home/likun/panziming/pydeps")
if os.path.isdir(pydeps):
    sys.path.insert(0, pydeps)


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def infer_dreamer_feature_flags(state_dict):
    return {
        "use_topology_head": any(key.startswith("topology_head.") for key in state_dict),
        "use_shortcut_forcing": "horizon_embed.weight" in state_dict,
    }


def total_rate_from_rates(rates):
    flat_rates = []
    vac_indices = []
    dir_indices = []
    for vac_idx, vac_rates in enumerate(rates):
        for dir_idx, rate in enumerate(vac_rates):
            if rate > 0:
                flat_rates.append(rate)
                vac_indices.append(vac_idx)
                dir_indices.append(dir_idx)
    return flat_rates, vac_indices, dir_indices


def expected_delta_t_from_rate(total_rate):
    return 1.0 / total_rate if total_rate > 0 else 0.0


def compute_alignment_summary(trajs):
    eps = 1e-12
    true_expected = np.clip(np.concatenate([t['true_expected_dts'] for t in trajs]), eps, None)
    pred_expected = np.clip(np.concatenate([t['pred_expected_dts'] for t in trajs]), eps, None)
    realized = np.clip(np.concatenate([t['realized_dts'] for t in trajs]), eps, None)

    ss_res = np.sum((pred_expected - true_expected) ** 2)
    ss_tot = np.sum((true_expected - np.mean(true_expected)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(pred_expected - true_expected))
    rmse = np.sqrt(np.mean((pred_expected - true_expected) ** 2))

    log_true = np.log(true_expected)
    log_pred = np.log(pred_expected)
    log_mae = np.mean(np.abs(log_pred - log_true))
    if len(log_true) > 1 and np.std(log_true) > 0 and np.std(log_pred) > 0:
        log_corr = float(np.corrcoef(log_true, log_pred)[0, 1])
    else:
        log_corr = 0.0

    cum_true = np.array([t['cum_true_expected_time'] for t in trajs])
    cum_pred = np.array([t['cum_pred_expected_time'] for t in trajs])
    cum_real = np.array([t['cum_real_time'] for t in trajs])
    cum_expected_err_pct = np.mean(np.abs(cum_pred - cum_true) / np.clip(cum_true, eps, None) * 100.0)
    realized_noise_pct = np.mean(np.abs(cum_real - cum_true) / np.clip(cum_true, eps, None) * 100.0)

    return {
        'per_step_r2': float(r2),
        'per_step_mae': float(mae),
        'per_step_rmse': float(rmse),
        'log_mae': float(log_mae),
        'log_corr': float(log_corr),
        'true_expected_mean_dt': float(np.mean(true_expected)),
        'pred_expected_mean_dt': float(np.mean(pred_expected)),
        'realized_mean_dt': float(np.mean(realized)),
        'mean_abs_cum_expected_time_error_pct': float(cum_expected_err_pct),
        'realized_vs_expected_noise_pct': float(realized_noise_pct),
    }


def traditional_kmc_step(env):
    """Perform one traditional KMC step and return both expected and realized time."""
    rates = env.calculate_diffusion_rate()
    flat_rates, vac_indices, dir_indices = total_rate_from_rates(rates)
    if not flat_rates:
        return {"realized_dt": 0.0, "expected_dt": 0.0, "total_rate": 0.0}

    total_rate = np.sum(flat_rates)
    expected_dt = expected_delta_t_from_rate(total_rate)
    r = np.random.rand() * total_rate
    chosen_idx = np.searchsorted(np.cumsum(flat_rates), r)
    vac_idx = vac_indices[chosen_idx]
    dir_idx = dir_indices[chosen_idx]

    # Execute jump via step_fast (action = vac_idx * 8 + dir_idx)
    action = int(vac_idx) * 8 + int(dir_idx)
    env.step_fast(action, 0)

    # Sample Poisson time
    delta_t = -np.log(np.random.rand()) / total_rate
    env.time += delta_t
    env.time_history.append(env.time)
    energy = env.calculate_system_energy()
    env.energy_last = energy
    env.energy_history.append(energy)
    return {"realized_dt": delta_t, "expected_dt": expected_dt, "total_rate": float(total_rate)}


def run_traditional_kmc(env_cfg, n_episodes, max_steps):
    """Run traditional KMC and collect expected / realized time statistics."""
    from RL4KMC.envs.kmc import KMCEnv
    trajectories = []
    for ep in range(n_episodes):
        class Args: pass
        args = Args()
        args.lattice_size = list(env_cfg["lattice_size"])
        total = args.lattice_size[0] * args.lattice_size[1] * args.lattice_size[2] * 2
        args.lattice_cu_nums = max(int(round(env_cfg["cu_density"] * total)), 1)
        args.lattice_v_nums = max(int(round(env_cfg["v_density"] * total)), 1)
        args.compute_global_static_env_reset = True
        args.skip_stats = True
        args.skip_global_diffusion_reset = False
        args.max_ssa_rounds = max_steps
        args.neighbor_order = env_cfg.get("neighbor_order", "2NN")
        args.temperature = env_cfg.get("temperature", 300.0)
        args.reward_scale = env_cfg.get("reward_scale", 10.0)

        env = KMCEnv(args)
        env.reset()

        real_times = [0.0]
        expected_times = [0.0]
        realized_dts = []
        expected_dts = []
        energies = [env.calculate_system_energy()]

        for s in range(max_steps):
            step_info = traditional_kmc_step(env)
            realized_dts.append(step_info["realized_dt"])
            expected_dts.append(step_info["expected_dt"])
            real_times.append(env.time)
            expected_times.append(expected_times[-1] + step_info["expected_dt"])
            energies.append(env.energy_history[-1])

        trajectories.append({
            "realized_dts": realized_dts,
            "expected_dts": expected_dts,
            "cum_real_time": real_times[-1],
            "cum_expected_time": expected_times[-1],
            "real_times": real_times,
            "expected_times": expected_times,
            "energies": energies,
        })
        print(f"  [Traditional KMC] Episode {ep+1}/{n_episodes}: "
              f"E[T]={expected_times[-1]:.6e}, real_T={real_times[-1]:.6e}, "
              f"energy_drop={energies[0]-energies[-1]:.4f}", flush=True)
    return trajectories


def run_muzero_with_time(env_cfg, model_path, device, n_episodes, max_steps, mcts_sims):
    """Run MuZero and collect state-expected time targets + predictions."""
    sys.path.insert(0, os.path.join(ROOT, "LightZero-main"))
    from lzero.model.kmc_graph_muzero_model import KMCGraphMuZeroModel
    from zoo.kmc.train_muzero_standalone import SimpleMCTS, KMCEnvWrapper

    env = KMCEnvWrapper(env_cfg)
    obs, mask = env.reset()
    obs_dim = obs.shape[0]
    action_dim = mask.shape[0]

    model = KMCGraphMuZeroModel(
        observation_shape=obs_dim, action_space_size=action_dim,
        max_vacancies=env_cfg["max_vacancies"], max_defects=env_cfg["max_defects"],
        max_shells=env_cfg["max_shells"], latent_state_dim=128, graph_hidden_size=32,
        per_vacancy_latent_dim=8, lattice_size=env_cfg["lattice_size"],
        neighbor_order=env_cfg["neighbor_order"], categorical_distribution=False,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    mcts = SimpleMCTS(model, num_simulations=mcts_sims, discount=0.997,
                      c_puct=1.25, device=device,
                      use_physics_discount=True, time_scale_tau=1.0)

    trajectories = []
    for ep in range(n_episodes):
        env = KMCEnvWrapper(env_cfg)
        obs, mask = env.reset()
        true_expected_dts, pred_expected_dts, realized_dts = [], [], []
        energies = [env.env.calculate_system_energy()]
        cum_real_time = 0.0
        cum_true_expected_time = 0.0
        cum_pred_expected_time = 0.0
        done = False
        while not done:
            true_total_rate = env.current_total_rate()
            true_expected_dt = expected_delta_t_from_rate(true_total_rate)

            policy = mcts.search(obs, mask)
            action = int(np.argmax(policy))

            # Predict the state time scale E[Δt | s] before taking the action.
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                init_out = model.initial_inference(obs_t)
                latent = init_out.latent_state
                pred_dt = max(model.predict_time_delta(latent).item(), 0.0)

            obs, mask, reward, done, info = env.step(action)
            real_dt = info["delta_t"]
            cum_real_time += real_dt
            cum_true_expected_time += true_expected_dt
            cum_pred_expected_time += pred_dt
            true_expected_dts.append(true_expected_dt)
            pred_expected_dts.append(pred_dt)
            realized_dts.append(real_dt)
            energies.append(env.env.calculate_system_energy())

        trajectories.append({
            "true_expected_dts": true_expected_dts,
            "pred_expected_dts": pred_expected_dts,
            "realized_dts": realized_dts,
            "energies": energies,
            "cum_true_expected_time": cum_true_expected_time,
            "cum_pred_expected_time": cum_pred_expected_time,
            "cum_real_time": cum_real_time,
        })
        print(
            f"  [MuZero] Episode {ep+1}/{n_episodes}: "
            f"E[T]={cum_true_expected_time:.6e}, pred_E[T]={cum_pred_expected_time:.6e}, "
            f"real_T={cum_real_time:.6e}, energy_drop={energies[0]-energies[-1]:.4f}",
            flush=True,
        )
    return trajectories


def run_dreamer_with_time(env_cfg, model_path, device, n_episodes, max_steps):
    """Run Dreamer and collect state-expected time targets + predictions."""
    sys.path.insert(0, os.path.join(ROOT, "dreamer4-main"))
    from train_dreamer_standalone import DreamerKMCAgent, KMCEnvWrapper

    env = KMCEnvWrapper(env_cfg)
    obs, mask = env.reset()
    action_dim = mask.shape[0]

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = extract_model_state_dict(ckpt)
    feature_flags = infer_dreamer_feature_flags(state_dict)

    agent = DreamerKMCAgent(
        dim_latent=16, max_vacancies=env_cfg["max_vacancies"],
        max_defects=env_cfg["max_defects"], max_shells=env_cfg["max_shells"],
        stats_dim=10, lattice_size=env_cfg["lattice_size"],
        neighbor_order=env_cfg["neighbor_order"], action_space_size=action_dim,
        graph_hidden_size=32,
        use_topology_head=feature_flags["use_topology_head"],
        use_shortcut_forcing=feature_flags["use_shortcut_forcing"],
    ).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    trajectories = []
    for ep in range(n_episodes):
        env = KMCEnvWrapper(env_cfg)
        obs, mask = env.reset()
        true_expected_dts, pred_expected_dts, realized_dts = [], [], []
        energies = [env.env.calculate_system_energy()]
        cum_real_time = 0.0
        cum_true_expected_time = 0.0
        cum_pred_expected_time = 0.0
        done = False
        while not done:
            true_total_rate = env.current_total_rate()
            true_expected_dt = expected_delta_t_from_rate(true_total_rate)

            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
                latent = agent.encode(obs_t)
                logits = agent.forward_policy(latent, mask_t)
                action = int(logits[0].argmax().item())
                pred_dt = max(agent.forward_time(latent.view(1, -1)).item(), 0.0)

            obs, mask, reward, done, info = env.step(action)
            real_dt = info["delta_t"]
            cum_real_time += real_dt
            cum_true_expected_time += true_expected_dt
            cum_pred_expected_time += pred_dt
            true_expected_dts.append(true_expected_dt)
            pred_expected_dts.append(pred_dt)
            realized_dts.append(real_dt)
            energies.append(env.env.calculate_system_energy())

        trajectories.append({
            "true_expected_dts": true_expected_dts,
            "pred_expected_dts": pred_expected_dts,
            "realized_dts": realized_dts,
            "energies": energies,
            "cum_true_expected_time": cum_true_expected_time,
            "cum_pred_expected_time": cum_pred_expected_time,
            "cum_real_time": cum_real_time,
        })
        print(
            f"  [Dreamer] Episode {ep+1}/{n_episodes}: "
            f"E[T]={cum_true_expected_time:.6e}, pred_E[T]={cum_pred_expected_time:.6e}, "
            f"real_T={cum_real_time:.6e}, energy_drop={energies[0]-energies[-1]:.4f}",
            flush=True,
        )
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Time alignment evaluation")
    parser.add_argument("--muzero_ckpt", type=str, default="muzero_v9_results/best_model.pt")
    parser.add_argument("--dreamer_ckpt", type=str, default="dreamer_v9_results/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--mcts_sims", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="time_eval_results")
    # Env
    parser.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    parser.add_argument("--cu_density", type=float, default=0.005)
    parser.add_argument("--v_density", type=float, default=0.0002)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_vacancies", type=int, default=32)
    parser.add_argument("--max_defects", type=int, default=64)
    parser.add_argument("--max_shells", type=int, default=16)
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    env_cfg = {
        "lattice_size": tuple(args.lattice_size),
        "max_episode_steps": args.max_steps,
        "max_vacancies": args.max_vacancies,
        "max_defects": args.max_defects,
        "max_shells": args.max_shells,
        "stats_dim": 10,
        "temperature": args.temperature,
        "reward_scale": args.reward_scale,
        "cu_density": args.cu_density,
        "v_density": args.v_density,
        "rlkmc_topk": 16,
        "neighbor_order": args.neighbor_order,
    }

    print("=" * 60)
    print("Expected Time Alignment Evaluation")
    print(f"cu={args.cu_density}, v={args.v_density}, {args.n_episodes} episodes × {args.max_steps} steps")
    print("=" * 60, flush=True)

    # ===== Run all models =====
    print("\n[1/4] Running Traditional KMC...", flush=True)
    trad_trajs = run_traditional_kmc(env_cfg, args.n_episodes, args.max_steps)

    print("\n[2/4] Running MuZero...", flush=True)
    muzero_trajs = run_muzero_with_time(
        env_cfg, args.muzero_ckpt, args.device, args.n_episodes, args.max_steps, args.mcts_sims
    )

    dreamer_trajs = None
    if os.path.exists(args.dreamer_ckpt):
        print("\n[3/4] Running Dreamer...", flush=True)
        dreamer_trajs = run_dreamer_with_time(env_cfg, args.dreamer_ckpt, args.device, args.n_episodes, args.max_steps)
    else:
        print(f"\n[3/4] Skipping Dreamer (checkpoint not found: {args.dreamer_ckpt})", flush=True)

    # ===== Save raw data =====
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    muzero_summary = compute_alignment_summary(muzero_trajs)
    dreamer_summary = compute_alignment_summary(dreamer_trajs) if dreamer_trajs else {}

    save_data = {
        "traditional": [{k: convert(v) if not isinstance(v, list) else [convert(x) for x in v] for k, v in t.items()} for t in trad_trajs],
        "muzero": [{k: convert(v) if not isinstance(v, list) else [convert(x) for x in v] for k, v in t.items()} for t in muzero_trajs],
        "summary": {"muzero": muzero_summary},
    }
    if dreamer_trajs:
        save_data["dreamer"] = [{k: convert(v) if not isinstance(v, list) else [convert(x) for x in v] for k, v in t.items()} for t in dreamer_trajs]
        save_data["summary"]["dreamer"] = dreamer_summary

    with open(os.path.join(args.output_dir, "time_eval_data.json"), "w") as f:
        json.dump(save_data, f)

    print("\n[4/4] Generating analysis plots...", flush=True)

    # ===== Analysis & Plotting =====
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    colors = {'Traditional KMC': '#9E9E9E', 'MuZero': '#FF5722', 'Dreamer': '#4CAF50'}

    model_list = [('MuZero', muzero_trajs, colors['MuZero'], muzero_summary)]
    if dreamer_trajs:
        model_list.append(('Dreamer', dreamer_trajs, colors['Dreamer'], dreamer_summary))

    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    for idx, (name, trajs, color, summary) in enumerate(model_list):
        ax = fig.add_subplot(gs[0, idx])
        true_expected = np.concatenate([t['true_expected_dts'] for t in trajs])
        pred_expected = np.concatenate([t['pred_expected_dts'] for t in trajs])
        eps = 1e-12
        lo = min(true_expected.min(), pred_expected.min())
        hi = max(np.percentile(true_expected, 99), np.percentile(pred_expected, 99))
        lo = max(lo, eps)

        ax.scatter(true_expected, pred_expected, alpha=0.25, s=10, color=color)
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y=x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('True E[Δt | s] = 1 / Γ_tot', fontsize=11)
        ax.set_ylabel('Predicted E[Δt | s]', fontsize=11)
        ax.set_title(
            f'Task 1: {name} Expected-Δt Alignment\n'
            f'R²={summary["per_step_r2"]:.4f}, log-corr={summary["log_corr"]:.4f}',
            fontsize=12,
            fontweight='bold',
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    plot_models = [('MuZero', muzero_trajs, colors['MuZero'])]
    if dreamer_trajs:
        plot_models.append(('Dreamer', dreamer_trajs, colors['Dreamer']))
    for name, trajs, color in plot_models:
        true_times = [t['cum_true_expected_time'] for t in trajs]
        pred_times = [t['cum_pred_expected_time'] for t in trajs]
        real_times = [t['cum_real_time'] for t in trajs]
        episodes = np.arange(1, len(trajs) + 1)

        ax.scatter(episodes, true_times, marker='o', s=36, color=color, alpha=0.8, label=f'{name} true E[T]')
        ax.scatter(episodes, pred_times, marker='x', s=36, color=color, alpha=0.8, label=f'{name} pred E[T]')
        ax.scatter(episodes, real_times, marker='^', s=24, color=color, alpha=0.35, label=f'{name} realized T')
        for episode, true_time, pred_time in zip(episodes, true_times, pred_times):
            ax.plot([episode, episode], [true_time, pred_time], color=color, alpha=0.25, linewidth=1)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Time', fontsize=12)
    ax.set_title('Task 2: Per-Trajectory Expected Time vs Predicted Time\n(realized T shown as noisy reference)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    dist_models = [('MuZero', muzero_trajs, colors['MuZero'])]
    if dreamer_trajs:
        dist_models.append(('Dreamer', dreamer_trajs, colors['Dreamer']))
    for idx, (name, trajs, color) in enumerate(dist_models):
        ax = fig.add_subplot(gs[2, idx])
        true_expected = np.concatenate([t['true_expected_dts'] for t in trajs])
        pred_expected = np.concatenate([t['pred_expected_dts'] for t in trajs])
        lo = max(min(true_expected.min(), pred_expected.min()), 1e-12)
        hi = max(np.percentile(true_expected, 99.5), np.percentile(pred_expected, 99.5))
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)

        ax.hist(true_expected, bins=bins, alpha=0.5, density=True, color='gray', label='True E[Δt | s]')
        ax.hist(pred_expected, bins=bins, alpha=0.5, density=True, color=color, label='Predicted E[Δt | s]')
        ax.set_xscale('log')
        ax.set_xlabel('Expected Δt', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Task 3: {name} Expected-Δt Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, :])
    max_steps_trad = min(len(t['expected_times']) for t in trad_trajs)
    avg_trad_times = np.mean([t['expected_times'][:max_steps_trad] for t in trad_trajs], axis=0)
    avg_trad_energies = np.mean([t['energies'][:max_steps_trad] for t in trad_trajs], axis=0)
    ax.plot(avg_trad_times, avg_trad_energies, color=colors['Traditional KMC'], linewidth=3, label='Traditional KMC (mean expected time)')

    energy_models = [('MuZero', muzero_trajs, colors['MuZero'])]
    if dreamer_trajs:
        energy_models.append(('Dreamer', dreamer_trajs, colors['Dreamer']))
    for name, trajs, color in energy_models:
        n_steps = min(len(t['true_expected_dts']) for t in trajs)
        avg_true_times = np.mean([[0.0] + list(np.cumsum(t['true_expected_dts'][:n_steps])) for t in trajs], axis=0)
        avg_pred_times = np.mean([[0.0] + list(np.cumsum(t['pred_expected_dts'][:n_steps])) for t in trajs], axis=0)
        avg_energies = np.mean([t['energies'][:n_steps + 1] for t in trajs], axis=0)
        ax.plot(avg_true_times, avg_energies, color=color, linewidth=3, label=f'{name} vs true E[T]')
        ax.plot(avg_pred_times, avg_energies, color=color, linewidth=2, linestyle='--', label=f'{name} vs pred E[T]')

    ax.set_xlabel('Expected Physical Time', fontsize=13)
    ax.set_ylabel('System Energy (eV)', fontsize=13)
    ax.set_title('Task 4: Energy vs Expected Physical Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(args.output_dir, 'time_alignment_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"\n✅ All plots saved to {args.output_dir}/time_alignment_analysis.png")

    # ===== Print final summary =====
    print(f"\n{'='*60}")
    print("TIME ALIGNMENT SUMMARY")
    print(f"{'='*60}")
    summary_list = [('MuZero', muzero_summary)]
    if dreamer_trajs:
        summary_list.append(('Dreamer', dreamer_summary))
    for name, summary in summary_list:
        print(f"\n  {name}:")
        print(f"    Per-step R²:    {summary['per_step_r2']:.4f}")
        print(f"    Per-step MAE:   {summary['per_step_mae']:.2e}")
        print(f"    Per-step RMSE:  {summary['per_step_rmse']:.2e}")
        print(f"    Log-space corr: {summary['log_corr']:.4f}")
        print(f"    Log-space MAE:  {summary['log_mae']:.4f}")
        print(f"    Pred mean E[Δt]: {summary['pred_expected_mean_dt']:.2e} (true: {summary['true_expected_mean_dt']:.2e})")
        print(f"    Mean abs cum expected-time error: {summary['mean_abs_cum_expected_time_error_pct']:.1f}%")
        print(f"    Realized-time noise vs expected:  {summary['realized_vs_expected_noise_pct']:.1f}%")


if __name__ == "__main__":
    main()
