from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_dreamer_macro_edit as mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate macro-edit Dreamer time alignment against traditional KMC teacher segments"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print_samples", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    if pred.size > 1 and np.std(pred) > 0 and np.std(target) > 0:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = 0.0
    return {"mae": mae, "rmse": rmse, "corr": corr}


def _compute_log_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    pred = np.clip(np.asarray(pred, dtype=np.float64), eps, None)
    target = np.clip(np.asarray(target, dtype=np.float64), eps, None)
    log_pred = np.log(pred)
    log_target = np.log(target)
    log_mae = float(np.mean(np.abs(log_pred - log_target)))
    log_rmse = float(np.sqrt(np.mean((log_pred - log_target) ** 2)))
    if pred.size > 1 and np.std(log_pred) > 0 and np.std(log_target) > 0:
        log_corr = float(np.corrcoef(log_pred, log_target)[0, 1])
    else:
        log_corr = 0.0
    scale_ratio = float(np.mean(pred / target))
    return {
        "log_mae": log_mae,
        "log_rmse": log_rmse,
        "log_corr": log_corr,
        "scale_ratio": scale_ratio,
    }


def _build_model(ckpt: dict[str, object], device: str) -> mod.MacroDreamerEditModel:
    args = ckpt["args"]
    include_stepwise_path_summary = args.get("teacher_path_summary_mode", "stepwise") == "stepwise"
    model = mod.MacroDreamerEditModel(
        max_vacancies=args["max_vacancies"],
        max_defects=args["max_defects"],
        max_shells=args["max_shells"],
        stats_dim=args["stats_dim"],
        lattice_size=tuple(args["lattice_size"]),
        neighbor_order=args["neighbor_order"],
        dim_latent=args["dim_latent"],
        graph_hidden_size=args["graph_hidden_size"],
        patch_hidden_size=args["patch_hidden_size"],
        patch_latent_dim=args["patch_latent_dim"],
        path_latent_dim=args["path_latent_dim"],
        global_summary_dim=16,
        teacher_path_summary_dim=mod.teacher_path_summary_dim(int(args["segment_k"]), include_stepwise_features=include_stepwise_path_summary),
        max_macro_k=max(int(args["segment_k"]), 16),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _load_samples(cache_path: Path, split: str, limit: int, expected_segment_k: int) -> tuple[list[mod.MacroSegmentSample], dict[str, object], dict[str, object]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("Dataset cache is missing signature metadata; refusing to run time alignment without segment_k validation")
    cache_segment_k = signature.get("segment_k")
    if cache_segment_k is None or int(cache_segment_k) != int(expected_segment_k):
        raise ValueError(
            f"Dataset cache segment_k={cache_segment_k} does not match checkpoint segment_k={expected_segment_k}"
        )
    samples = [mod.MacroSegmentSample(**item) for item in payload[split]]
    mismatched_sample = next((sample for sample in samples if int(sample.horizon_k) != int(expected_segment_k)), None)
    if mismatched_sample is not None:
        raise ValueError(
            f"Found sample with horizon_k={int(mismatched_sample.horizon_k)} in cache split {split}, expected {expected_segment_k}"
        )
    if limit > 0:
        samples = samples[:limit]
    return samples, payload.get("stats", {}), signature


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    cache_path = Path(args.cache)

    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model = _build_model(ckpt, args.device)
    reward_scale = float(ckpt["args"].get("reward_scale", 1.0))
    segment_k = int(ckpt["args"]["segment_k"])

    samples, dataset_stats, cache_signature = _load_samples(cache_path, args.split, args.limit, segment_k)
    loader = mod._build_loader(samples, batch_size=args.batch_size, shuffle=False)

    pred_reward_sum = []
    true_reward_sum = []
    pred_tau = []
    true_tau_exp = []
    true_tau_real = []
    sample_rows = []
    sample_index = 0

    with torch.no_grad():
        for batch in loader:
            tensors = mod._batch_to_device(batch, args.device)
            global_latent = model.encode_global(tensors["start_obs"])
            prior_mu, prior_logvar = model.prior_stats(
                global_latent,
                tensors["global_summary"],
                tensors["horizon_k"],
            )
            path_latent = model.sample_path_latent(prior_mu, prior_logvar, deterministic=True)
            next_pred = model.predict_next_global(global_latent, path_latent, tensors["horizon_k"])
            reward_hat, tau_mu, _tau_log_sigma = model.predict_reward_and_duration(
                global_latent,
                next_pred,
                path_latent,
                tensors["global_summary"],
                tensors["horizon_k"],
            )
            batch_pred_reward = reward_hat.detach().cpu().numpy()
            batch_pred_tau = torch.exp(tau_mu).detach().cpu().numpy()

            for sample, item_pred_reward, item_pred_tau in zip(batch, batch_pred_reward, batch_pred_tau):
                pred_reward_sum.append(float(item_pred_reward))
                true_reward_sum.append(float(sample.reward_sum))
                pred_tau.append(float(item_pred_tau))
                true_tau_exp.append(float(sample.tau_exp))
                true_tau_real.append(float(sample.tau_real))
                sample_rows.append(
                    {
                        "index": sample_index,
                        "segment_k": int(sample.horizon_k),
                        "traditional_kmc_reward_sum": float(sample.reward_sum),
                        "traditional_kmc_delta_e": float(sample.reward_sum / reward_scale),
                        "traditional_kmc_expected_tau": float(sample.tau_exp),
                        "traditional_kmc_realized_tau": float(sample.tau_real),
                        "predicted_reward_sum": float(item_pred_reward),
                        "predicted_delta_e": float(item_pred_reward / reward_scale),
                        "predicted_tau": float(item_pred_tau),
                    }
                )
                sample_index += 1

    pred_reward_sum_np = np.asarray(pred_reward_sum, dtype=np.float64)
    true_reward_sum_np = np.asarray(true_reward_sum, dtype=np.float64)
    pred_delta_e_np = pred_reward_sum_np / reward_scale
    true_delta_e_np = true_reward_sum_np / reward_scale
    pred_tau_np = np.asarray(pred_tau, dtype=np.float64)
    true_tau_exp_np = np.asarray(true_tau_exp, dtype=np.float64)
    true_tau_real_np = np.asarray(true_tau_real, dtype=np.float64)

    summary = {
        "checkpoint": str(checkpoint_path),
        "cache": str(cache_path),
        "split": args.split,
        "num_samples": int(len(samples)),
        "segment_k": segment_k,
        "cache_signature": cache_signature,
        "dataset_stats": dataset_stats.get(args.split, {}),
        "teacher_source": "traditional_kmc_segment_cache",
        "reward_sum": _compute_metrics(pred_reward_sum_np, true_reward_sum_np),
        "delta_e": _compute_metrics(pred_delta_e_np, true_delta_e_np),
        "tau_expected": {
            **_compute_metrics(pred_tau_np, true_tau_exp_np),
            **_compute_log_metrics(pred_tau_np, true_tau_exp_np),
            "traditional_mean": float(np.mean(true_tau_exp_np)),
            "predicted_mean": float(np.mean(pred_tau_np)),
        },
        "tau_realized": {
            **_compute_metrics(pred_tau_np, true_tau_real_np),
            **_compute_log_metrics(pred_tau_np, true_tau_real_np),
            "traditional_mean": float(np.mean(true_tau_real_np)),
            "predicted_mean": float(np.mean(pred_tau_np)),
        },
        "traditional_energy": {
            "reward_sum_mean": float(np.mean(true_reward_sum_np)),
            "delta_e_mean": float(np.mean(true_delta_e_np)),
        },
        "predicted_energy": {
            "reward_sum_mean": float(np.mean(pred_reward_sum_np)),
            "delta_e_mean": float(np.mean(pred_delta_e_np)),
        },
        "sample_preview": sample_rows[: max(args.print_samples, 0)],
    }

    print("=" * 60)
    print("Macro-Edit Dreamer vs Traditional KMC Teacher")
    print(f"samples={len(samples)}, split={args.split}, segment_k={segment_k}")
    print("=" * 60)
    print(
        "Traditional KMC energy/time means: "
        f"reward_sum={summary['traditional_energy']['reward_sum_mean']:.6f}, "
        f"delta_E={summary['traditional_energy']['delta_e_mean']:.6f}, "
        f"E[tau]={summary['tau_expected']['traditional_mean']:.6e}, "
        f"real_tau={summary['tau_realized']['traditional_mean']:.6e}"
    )
    print(
        "Model prediction means: "
        f"reward_sum={summary['predicted_energy']['reward_sum_mean']:.6f}, "
        f"delta_E={summary['predicted_energy']['delta_e_mean']:.6f}, "
        f"pred_tau={summary['tau_expected']['predicted_mean']:.6e}"
    )
    print(
        "Reward alignment: "
        f"mae={summary['reward_sum']['mae']:.6f}, rmse={summary['reward_sum']['rmse']:.6f}, corr={summary['reward_sum']['corr']:.4f}"
    )
    print(
        "Expected-time alignment: "
        f"mae={summary['tau_expected']['mae']:.6e}, log_mae={summary['tau_expected']['log_mae']:.4f}, "
        f"log_corr={summary['tau_expected']['log_corr']:.4f}, scale_ratio={summary['tau_expected']['scale_ratio']:.2f}"
    )
    print(
        "Realized-time reference: "
        f"mae={summary['tau_realized']['mae']:.6e}, log_mae={summary['tau_realized']['log_mae']:.4f}, "
        f"log_corr={summary['tau_realized']['log_corr']:.4f}, scale_ratio={summary['tau_realized']['scale_ratio']:.2f}"
    )
    if summary["sample_preview"]:
        print("Sample preview:")
        for row in summary["sample_preview"]:
            print(json.dumps(row, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()