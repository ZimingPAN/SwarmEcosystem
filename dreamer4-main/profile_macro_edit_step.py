from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import train_dreamer_macro_edit as mod


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one macro-edit training step")
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = Path(args.cache)
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    include_stepwise_path_summary = payload.get("signature", {}).get("teacher_path_summary_mode", "stepwise") == "stepwise"
    train_samples = [mod.MacroSegmentSample(**item) for item in payload["train"]]
    loader = mod._build_loader(train_samples[: args.num_samples], batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))

    model = mod.MacroDreamerEditModel(
        max_vacancies=32,
        max_defects=64,
        max_shells=16,
        stats_dim=10,
        lattice_size=(40, 40, 40),
        neighbor_order="2NN",
        dim_latent=16,
        graph_hidden_size=32,
        patch_hidden_size=96,
        patch_latent_dim=64,
        path_latent_dim=32,
        global_summary_dim=16,
        teacher_path_summary_dim=mod.teacher_path_summary_dim(4, include_stepwise_features=include_stepwise_path_summary),
        max_macro_k=16,
    ).to(args.device)
    mod._initialize_output_heads(model, train_samples)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    weights = {
        "mask": 1.0,
        "type": 1.0,
        "tau": 1.0,
        "reward": 0.5,
        "latent": 0.5,
        "proj": 0.5,
        "path": 0.05,
        "prior_latent": 0.25,
    }
    max_changed_sites = 8

    _sync()
    t0 = time.time()
    tensors = mod._batch_to_device(batch, args.device)
    _sync()
    print(f"batch_to_device {time.time() - t0:.4f}")

    t0 = time.time()
    global_latent = model.encode_global(tensors["start_obs"])
    next_global = model.encode_global(tensors["next_obs"]).detach()
    _sync()
    print(f"encode_global {time.time() - t0:.4f}")

    t0 = time.time()
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
    _, target_patch_latent = model.encode_patch(
        positions=tensors["candidate_positions"],
        nearest_vacancy_offset=tensors["nearest_vacancy_offset"],
        reach_depth=tensors["reach_depth"],
        is_start_vacancy=tensors["is_start_vacancy"],
        type_ids=tensors["target_types"],
        node_mask=tensors["candidate_mask"],
        global_summary=tensors["global_summary"],
        box_dims=tensors["box_dims"],
    )
    _sync()
    print(f"encode_patch_twice {time.time() - t0:.4f}")

    t0 = time.time()
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
    type_logits = model.apply_type_copy_bias(raw_type_logits, tensors["current_types"])
    reward_hat, tau_mu, tau_log_sigma = model.predict_reward_and_duration(
        global_latent, next_pred, post_c, tensors["global_summary"], tensors["horizon_k"]
    )
    reward_hat_prior, tau_mu_prior, tau_log_sigma_prior = model.predict_reward_and_duration(
        global_latent, next_pred_prior, prior_c, tensors["global_summary"], tensors["horizon_k"]
    )
    _sync()
    print(f"forward_heads {time.time() - t0:.4f}")

    t0 = time.time()
    projected_types, _, _, _ = mod.project_types_by_inventory(
        current_types=tensors["current_types"],
        change_logits=change_logits,
        type_logits=raw_type_logits,
        node_mask=tensors["candidate_mask"],
        positions=tensors["candidate_positions"],
        box_dims=tensors["box_dims"],
        horizon_k=tensors["horizon_k"],
        max_changed_sites=max_changed_sites,
    )
    _sync()
    print(f"projection {time.time() - t0:.4f}")

    t0 = time.time()
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
    _sync()
    print(f"encode_projected_patch {time.time() - t0:.4f}")

    t0 = time.time()
    projected_global = mod._projected_global_latent_batch(batch=batch, projected_types=projected_types, model=model, device=args.device)
    _sync()
    print(f"projected_global_latent_batch {time.time() - t0:.4f}")

    t0 = time.time()
    valid = tensors["candidate_mask"] > 0
    pos_count = tensors["changed_mask"][valid].sum().clamp(min=1.0)
    neg_count = valid.float().sum().clamp(min=1.0) - pos_count + 1e-6
    pos_weight = (neg_count / pos_count).detach()
    mask_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        change_logits[valid], tensors["changed_mask"][valid], pos_weight=pos_weight
    )
    mask_focal = mod._focal_bce_with_logits(change_logits[valid], tensors["changed_mask"][valid])
    aux_scale = mod._scheduled_aux_scale(epoch=1, total_epochs=10)
    changed_atom_valid = valid & (tensors["changed_mask"] > 0) & (tensors["current_types"] != mod.V_TYPE)
    unchanged_vacancy_valid = valid & (tensors["changed_mask"] <= 0) & (tensors["current_types"] == mod.V_TYPE)
    if changed_atom_valid.any():
        atom_change_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            change_logits[changed_atom_valid],
            torch.ones_like(change_logits[changed_atom_valid]),
        )
    else:
        atom_change_loss = torch.zeros((), device=args.device)
    if unchanged_vacancy_valid.any():
        vacancy_static_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            change_logits[unchanged_vacancy_valid],
            torch.zeros_like(change_logits[unchanged_vacancy_valid]),
        )
    else:
        vacancy_static_loss = torch.zeros((), device=args.device)
    predicted_change_count = mod._soft_typed_change_count(
        change_logits=change_logits,
        type_logits=raw_type_logits,
        current_types=tensors["current_types"],
        candidate_mask=tensors["candidate_mask"],
    )
    target_change_count = tensors["changed_mask"].sum(dim=-1)
    count_loss = torch.nn.functional.smooth_l1_loss(predicted_change_count, target_change_count)
    mask_loss = (
        mask_bce
        + 0.25 * mask_focal
        + aux_scale * (0.5 * atom_change_loss + 0.5 * vacancy_static_loss)
        + 0.1 * count_loss
    )
    changed_valid = valid & (tensors["changed_mask"] > 0)
    unchanged_valid = valid & (tensors["changed_mask"] <= 0)
    changed_type_loss = mod._masked_type_cross_entropy(raw_type_logits, tensors["target_types"], changed_valid)
    unchanged_copy_loss = (
        torch.nn.functional.cross_entropy(raw_type_logits[unchanged_valid], tensors["current_types"][unchanged_valid])
        if unchanged_valid.any()
        else torch.zeros((), device=args.device)
    )
    type_loss = changed_type_loss + 0.05 * unchanged_copy_loss
    tau_loss = mod.lognormal_nll(tensors["tau_exp"], tau_mu, tau_log_sigma).mean()
    reward_loss = torch.nn.functional.smooth_l1_loss(reward_hat, tensors["reward_sum"])
    latent_loss = torch.nn.functional.smooth_l1_loss(next_pred, next_global)
    prior_latent_loss = torch.nn.functional.smooth_l1_loss(next_pred_prior, next_global)
    proj_state_loss = (
        torch.nn.functional.smooth_l1_loss(projected_patch_latent, target_patch_latent.detach())
        + 0.5 * torch.nn.functional.smooth_l1_loss(projected_global, next_global)
        + 0.5 * torch.nn.functional.smooth_l1_loss(projected_global, next_pred.detach())
    )
    path_loss = mod.kl_divergence_diag_gaussian(post_mu, post_logvar, prior_mu, prior_logvar).mean()
    prior_tau_loss = mod.lognormal_nll(tensors["tau_exp"], tau_mu_prior, tau_log_sigma_prior).mean()
    prior_reward_loss = torch.nn.functional.smooth_l1_loss(reward_hat_prior, tensors["reward_sum"])
    loss = (
        weights["mask"] * mask_loss
        + weights["type"] * type_loss
        + weights["tau"] * tau_loss
        + weights["reward"] * reward_loss
        + weights["latent"] * latent_loss
        + weights["proj"] * proj_state_loss
        + weights["path"] * path_loss
        + weights["prior_latent"] * prior_latent_loss
        + 0.5 * weights["tau"] * prior_tau_loss
        + 0.5 * weights["reward"] * prior_reward_loss
    )
    _sync()
    print(f"loss_build {time.time() - t0:.4f}")

    t0 = time.time()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    _sync()
    print(f"backward_step {time.time() - t0:.4f}")


if __name__ == "__main__":
    main()