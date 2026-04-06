from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import train_dreamer_macro_edit as mod
from dreamer4.macro_edit import project_types_by_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect macro-edit projection behavior on cached samples")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=8)
    return parser.parse_args()


def _valid_indices(mask: torch.Tensor) -> list[int]:
    return torch.nonzero(mask > 0, as_tuple=False).squeeze(-1).cpu().tolist()


def main() -> None:
    args = parse_args()
    payload = torch.load(Path(args.cache), map_location="cpu", weights_only=False)
    val_samples = [mod.MacroSegmentSample(**item) for item in payload["val"]]
    ckpt = torch.load(Path(args.checkpoint), map_location=args.device, weights_only=False)

    ckpt_args = ckpt["args"]
    include_stepwise_path_summary = ckpt_args.get("teacher_path_summary_mode", "stepwise") == "stepwise"
    model = mod.MacroDreamerEditModel(
        max_vacancies=ckpt_args["max_vacancies"],
        max_defects=ckpt_args["max_defects"],
        max_shells=ckpt_args["max_shells"],
        stats_dim=ckpt_args["stats_dim"],
        lattice_size=tuple(ckpt_args["lattice_size"]),
        neighbor_order=ckpt_args["neighbor_order"],
        dim_latent=ckpt_args["dim_latent"],
        graph_hidden_size=ckpt_args["graph_hidden_size"],
        patch_hidden_size=ckpt_args["patch_hidden_size"],
        patch_latent_dim=ckpt_args["patch_latent_dim"],
        path_latent_dim=ckpt_args["path_latent_dim"],
        global_summary_dim=16,
        teacher_path_summary_dim=mod.teacher_path_summary_dim(int(ckpt_args["segment_k"]), include_stepwise_features=include_stepwise_path_summary),
        max_macro_k=max(ckpt_args["segment_k"], 16),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = mod._build_loader(val_samples[: args.limit], batch_size=min(len(val_samples), args.limit), shuffle=False)
    batch = next(iter(loader))
    tensors = mod._batch_to_device(batch, args.device)

    with torch.no_grad():
        global_latent = model.encode_global(tensors["start_obs"])
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
        type_logits = model.apply_type_copy_bias(raw_type_logits, tensors["current_types"])
        projected_types, projected_changed_mask, transport_costs, violations = project_types_by_inventory(
            current_types=tensors["current_types"],
            change_logits=change_logits,
            type_logits=type_logits,
            node_mask=tensors["candidate_mask"],
            positions=tensors["candidate_positions"],
            box_dims=tensors["box_dims"],
            horizon_k=tensors["horizon_k"],
            max_changed_sites=2 * ckpt_args["segment_k"],
        )

    for sample_idx, sample in enumerate(batch):
        valid_idx = _valid_indices(tensors["candidate_mask"][sample_idx])
        target_changed = [idx for idx in valid_idx if int(tensors["target_types"][sample_idx, idx].item()) != int(tensors["current_types"][sample_idx, idx].item())]
        projected_changed = [idx for idx in valid_idx if int(projected_types[sample_idx, idx].item()) != int(tensors["current_types"][sample_idx, idx].item())]
        raw_changed = [idx for idx in valid_idx if float(torch.sigmoid(change_logits[sample_idx, idx]).item()) > 0.5]
        type_probs = torch.softmax(type_logits[sample_idx, valid_idx], dim=-1)
        current_types = tensors["current_types"][sample_idx, valid_idx]
        predicted_types = type_logits[sample_idx, valid_idx].argmax(dim=-1)
        current_conf = type_probs.gather(1, current_types.unsqueeze(-1)).squeeze(-1)
        type_change_score = 1.0 - current_conf
        change_probs = torch.sigmoid(change_logits[sample_idx, valid_idx])
        combined = 0.5 * change_probs + 0.5 * type_change_score
        transition_counts = {
            "vacancy_to_fe": int(((current_types == 2) & (predicted_types == 0)).sum().item()),
            "vacancy_to_cu": int(((current_types == 2) & (predicted_types == 1)).sum().item()),
            "fe_to_vacancy": int(((current_types == 0) & (predicted_types == 2)).sum().item()),
            "cu_to_vacancy": int(((current_types == 1) & (predicted_types == 2)).sum().item()),
        }
        ranking = torch.argsort(combined, descending=True).cpu().tolist()[:12]
        top_items = []
        for local_idx in ranking:
            global_idx = valid_idx[local_idx]
            top_items.append(
                {
                    "idx": global_idx,
                    "pos": sample.candidate_positions[global_idx].astype(int).tolist(),
                    "current": int(tensors["current_types"][sample_idx, global_idx].item()),
                    "target": int(tensors["target_types"][sample_idx, global_idx].item()),
                    "pred_type": int(type_logits[sample_idx, global_idx].argmax().item()),
                    "change_prob": float(torch.sigmoid(change_logits[sample_idx, global_idx]).item()),
                    "type_change_score": float(type_change_score[local_idx].item()),
                    "combined_score": float(combined[local_idx].item()),
                }
            )
        atom_swap_candidates = []
        atom_candidate_mask = ((current_types == 0) | (current_types == 1)) & (predicted_types == 2)
        atom_candidate_idx = torch.nonzero(atom_candidate_mask, as_tuple=False).squeeze(-1).cpu().tolist()[:8]
        for local_idx in atom_candidate_idx:
            global_idx = valid_idx[local_idx]
            atom_swap_candidates.append(
                {
                    "idx": global_idx,
                    "pos": sample.candidate_positions[global_idx].astype(int).tolist(),
                    "current": int(tensors["current_types"][sample_idx, global_idx].item()),
                    "target": int(tensors["target_types"][sample_idx, global_idx].item()),
                    "pred_type": int(type_logits[sample_idx, global_idx].argmax().item()),
                    "change_prob": float(torch.sigmoid(change_logits[sample_idx, global_idx]).item()),
                    "type_change_score": float(type_change_score[local_idx].item()),
                    "combined_score": float(combined[local_idx].item()),
                }
            )
        print(
            json.dumps(
                {
                    "sample": sample_idx,
                    "tau_exp": float(sample.tau_exp),
                    "reward_sum": float(sample.reward_sum),
                    "transition_counts": transition_counts,
                    "target_changed": target_changed,
                    "raw_changed": raw_changed,
                    "projected_changed": projected_changed,
                    "projected_changed_count": len(projected_changed),
                    "transport_cost": float(transport_costs[sample_idx].item()),
                    "violation": float(violations[sample_idx].item()),
                    "atom_swap_candidates": atom_swap_candidates,
                    "top_ranked": top_items,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()