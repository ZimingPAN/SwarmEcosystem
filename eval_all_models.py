#!/usr/bin/env python3
"""
Eval-only script for all three v9 models (PPO, MuZero, Dreamer).
Loads best_model.pt checkpoint and runs N eval rounds on low-density config.
"""
import sys, os, argparse, random, json
import numpy as np
import torch

# Paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "RLKMC-MASSIVE-main"))
sys.path.insert(0, os.path.join(ROOT, "LightZero-main"))
sys.path.insert(0, os.path.join(ROOT, "dreamer4-main"))
pydeps = os.path.expanduser("/home/likun/panziming/pydeps")
if os.path.isdir(pydeps):
    sys.path.insert(0, pydeps)

# KMCEnvWrapper is defined inside each training script, import from the appropriate one
# We'll import lazily per model to avoid conflicts


def build_eval_cfg(args):
    return {
        "lattice_size": tuple(args.lattice_size),
        "max_episode_steps": args.max_episode_steps,
        "max_vacancies": args.max_vacancies,
        "max_defects": args.max_defects,
        "max_shells": args.max_shells,
        "stats_dim": 10,
        "temperature": args.temperature,
        "reward_scale": args.reward_scale,
        "cu_density": args.eval_cu_density,
        "v_density": args.eval_v_density,
        "rlkmc_topk": 16,
        "neighbor_order": args.neighbor_order,
    }


def eval_ppo(args, eval_cfg, device):
    sys.path.insert(0, os.path.join(ROOT, "RLKMC-MASSIVE-main"))
    from train_ppo_standalone import PPOGNNAgent, KMCEnvWrapper
    # Need one env to get dims
    env = KMCEnvWrapper(eval_cfg)
    obs, mask = env.reset()
    obs_dim = obs.shape[0]
    action_dim = mask.shape[0]

    agent = PPOGNNAgent(
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=10,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        action_space_size=action_dim,
        graph_hidden_size=32,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt)
    agent.eval()

    all_results = []
    for r in range(args.num_rounds):
        rewards = []
        for _ in range(args.episodes_per_round):
            env = KMCEnvWrapper(eval_cfg)
            obs, mask = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_t, mask_t, deterministic=True)
                obs, mask, reward, done, _ = env.step(action.item())
                ep_reward += reward
            rewards.append(ep_reward)
        mean_r = np.mean(rewards)
        all_results.append({"round": r+1, "mean_reward": mean_r,
                           "rewards": [f"{x:+.4f}" for x in rewards]})
        print(f"  [PPO] Round {r+1}/{args.num_rounds}: mean={mean_r:+.4f}  rewards={[f'{x:+.3f}' for x in rewards]}", flush=True)
    return all_results


def eval_muzero(args, eval_cfg, device):
    sys.path.insert(0, os.path.join(ROOT, "LightZero-main"))
    from lzero.model.kmc_graph_muzero_model import KMCGraphMuZeroModel
    from zoo.kmc.train_muzero_standalone import SimpleMCTS, KMCEnvWrapper

    env = KMCEnvWrapper(eval_cfg)
    obs, mask = env.reset()
    obs_dim = obs.shape[0]
    action_dim = mask.shape[0]

    model = KMCGraphMuZeroModel(
        observation_shape=obs_dim,
        action_space_size=action_dim,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        latent_state_dim=128,
        graph_hidden_size=32,
        per_vacancy_latent_dim=8,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        categorical_distribution=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()

    mcts = SimpleMCTS(model, num_simulations=args.mcts_sims, discount=0.997,
                      c_puct=1.25, device=device,
                      use_physics_discount=True, time_scale_tau=1.0)

    all_results = []
    for r in range(args.num_rounds):
        rewards = []
        for _ in range(args.episodes_per_round):
            env = KMCEnvWrapper(eval_cfg)
            obs, mask = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                policy = mcts.search(obs, mask)
                action = int(np.argmax(policy))
                obs, mask, reward, done, info = env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
        mean_r = np.mean(rewards)
        all_results.append({"round": r+1, "mean_reward": mean_r,
                           "rewards": [f"{x:+.4f}" for x in rewards]})
        print(f"  [MuZero] Round {r+1}/{args.num_rounds}: mean={mean_r:+.4f}  rewards={[f'{x:+.3f}' for x in rewards]}", flush=True)
    return all_results


def eval_dreamer(args, eval_cfg, device):
    sys.path.insert(0, os.path.join(ROOT, "dreamer4-main"))
    from train_dreamer_standalone import DreamerKMCAgent, KMCEnvWrapper

    env = KMCEnvWrapper(eval_cfg)
    obs, mask = env.reset()
    obs_dim = obs.shape[0]
    action_dim = mask.shape[0]

    agent = DreamerKMCAgent(
        dim_latent=16,
        max_vacancies=args.max_vacancies,
        max_defects=args.max_defects,
        max_shells=args.max_shells,
        stats_dim=10,
        lattice_size=tuple(args.lattice_size),
        neighbor_order=args.neighbor_order,
        action_space_size=action_dim,
        graph_hidden_size=32,
        use_topology_head=True,
        use_shortcut_forcing=True,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # best_model.pt saves agent state_dict directly; checkpoint_*.pt has 'model' key
    if isinstance(ckpt, dict) and "model" in ckpt:
        agent.load_state_dict(ckpt["model"])
    else:
        agent.load_state_dict(ckpt)
    agent.eval()

    all_results = []
    for r in range(args.num_rounds):
        rewards = []
        for _ in range(args.episodes_per_round):
            env = KMCEnvWrapper(eval_cfg)
            obs, mask = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
                    latent = agent.encode(obs_t)
                    logits = agent.forward_policy(latent, mask_t)
                    action = int(logits[0].argmax().item())
                obs, mask, reward, done, _ = env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)
        mean_r = np.mean(rewards)
        all_results.append({"round": r+1, "mean_reward": mean_r,
                           "rewards": [f"{x:+.4f}" for x in rewards]})
        print(f"  [Dreamer] Round {r+1}/{args.num_rounds}: mean={mean_r:+.4f}  rewards={[f'{x:+.3f}' for x in rewards]}", flush=True)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Eval-only for v9 models")
    parser.add_argument("--model", type=str, required=True, choices=["ppo", "muzero", "dreamer"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--episodes_per_round", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    # Env config
    parser.add_argument("--lattice_size", type=int, nargs=3, default=[40, 40, 40])
    parser.add_argument("--eval_cu_density", type=float, default=0.005)
    parser.add_argument("--eval_v_density", type=float, default=0.0002)
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--reward_scale", type=float, default=10.0)
    parser.add_argument("--neighbor_order", type=str, default="2NN")
    parser.add_argument("--max_vacancies", type=int, default=32)
    parser.add_argument("--max_defects", type=int, default=64)
    parser.add_argument("--max_shells", type=int, default=16)
    # MuZero specific
    parser.add_argument("--mcts_sims", type=int, default=50)
    # Output
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    eval_cfg = build_eval_cfg(args)

    print(f"{'='*60}")
    print(f"Eval-only: {args.model.upper()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Rounds: {args.num_rounds} x {args.episodes_per_round} episodes")
    print(f"Eval density: cu={args.eval_cu_density}, v={args.eval_v_density}")
    print(f"{'='*60}", flush=True)

    if args.model == "ppo":
        results = eval_ppo(args, eval_cfg, device)
    elif args.model == "muzero":
        results = eval_muzero(args, eval_cfg, device)
    elif args.model == "dreamer":
        results = eval_dreamer(args, eval_cfg, device)

    # Summary
    means = [r["mean_reward"] for r in results]
    pos_count = sum(1 for m in means if m > 0.01)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model.upper()}")
    print(f"  Total rounds: {len(means)}")
    print(f"  Overall mean: {np.mean(means):+.4f}")
    print(f"  Std: {np.std(means):.4f}")
    print(f"  Max: {np.max(means):+.4f}")
    print(f"  Positive rate: {pos_count}/{len(means)} ({pos_count/len(means)*100:.1f}%)")
    print(f"  Last 10 mean: {np.mean(means[-10:]):+.4f}")
    print(f"  Last 20 mean: {np.mean(means[-20:]):+.4f}")
    print(f"{'='*60}", flush=True)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"model": args.model, "results": results,
                       "summary": {"mean": float(np.mean(means)), "std": float(np.std(means)),
                                   "max": float(np.max(means)), "positive_rate": pos_count/len(means)}}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
