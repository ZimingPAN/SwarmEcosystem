from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_dreamer_macro_edit as mod
import eval_macro_time_alignment as eval_mod
from dreamer4.macro_edit import MacroDreamerEditModel, macro_duration_baseline_log_tau, project_types_by_inventory, teacher_path_summary_dim
from train_dreamer_macro_edit import (
    MacroKMCEnv,
    _build_loader,
    _collect_segments,
    _edit_supervision_losses,
    _evaluate,
    _initialize_best_score_from_saved_best,
    _matched_pair_count_loss,
    _projected_mask_distill_loss,
    _projected_state_alignment_loss,
    _selection_score,
    _soft_directional_transition_counts,
    _soft_typed_change_count,
    _train_epoch,
    _validate_resume_args,
)


def test_inventory_projection_preserves_patch_counts():
    current_types = torch.tensor([[0, 1, 2, 0, 1]], dtype=torch.long)
    change_logits = torch.tensor([[8.0, 6.0, 5.0, -4.0, -5.0]])
    type_logits = torch.tensor(
        [
            [
                [0.1, 3.0, 0.2],
                [2.5, 0.1, 0.3],
                [0.2, 0.3, 2.7],
                [1.5, 0.1, 0.1],
                [0.2, 1.6, 0.2],
            ]
        ],
        dtype=torch.float32,
    )
    node_mask = torch.ones_like(change_logits)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
    box_dims = torch.tensor([[20.0, 20.0, 20.0]])
    horizon_k = torch.tensor([3])
    final_types, _, transport_cost, violations = project_types_by_inventory(
        current_types=current_types,
        change_logits=change_logits,
        type_logits=type_logits,
        node_mask=node_mask,
        positions=positions,
        box_dims=box_dims,
        horizon_k=horizon_k,
        max_changed_sites=3,
    )
    current_counts = torch.bincount(current_types[0], minlength=3)
    final_counts = torch.bincount(final_types[0], minlength=3)
    assert torch.equal(current_counts, final_counts)
    assert float(violations[0].item()) in {0.0, 1.0}
    assert float(transport_cost[0].item()) >= 0.0


def test_projection_skips_violation_when_change_mass_has_no_typed_swap_support():
    current_types = torch.tensor([[2, 0, 2, 1, 0, 0, 1, 0]], dtype=torch.long)
    change_logits = torch.zeros((1, 8), dtype=torch.float32)
    type_logits = torch.full((1, 8, 3), -4.0, dtype=torch.float32)
    for idx, type_id in enumerate(current_types[0].tolist()):
        type_logits[0, idx, type_id] = 4.0
    node_mask = torch.ones_like(change_logits)
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 1.0], [4.0, 0.0, 0.0], [5.0, 1.0, 1.0], [6.0, 0.0, 0.0], [7.0, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    box_dims = torch.tensor([[20.0, 20.0, 20.0]], dtype=torch.float32)
    horizon_k = torch.tensor([4], dtype=torch.long)

    final_types, changed_mask, _transport_cost, violations = project_types_by_inventory(
        current_types=current_types,
        change_logits=change_logits,
        type_logits=type_logits,
        node_mask=node_mask,
        positions=positions,
        box_dims=box_dims,
        horizon_k=horizon_k,
        max_changed_sites=8,
    )

    assert torch.equal(final_types, current_types)
    assert torch.equal(changed_mask, torch.zeros_like(node_mask))
    assert float(violations[0].item()) == 0.0


def test_soft_typed_change_count_requires_noncopy_type_support():
    change_logits = torch.zeros((1, 4), dtype=torch.float32)
    current_types = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    candidate_mask = torch.ones((1, 4), dtype=torch.float32)
    copy_type_logits = torch.full((1, 4, 3), -6.0, dtype=torch.float32)
    for idx, type_id in enumerate(current_types[0].tolist()):
        copy_type_logits[0, idx, type_id] = 6.0

    count = _soft_typed_change_count(
        change_logits=change_logits,
        type_logits=copy_type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )

    assert float(count.item()) < 1e-3


def test_soft_directional_transition_counts_track_both_swap_halves():
    change_logits = torch.full((1, 4), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0, 2, 1]], dtype=torch.long)
    candidate_mask = torch.ones((1, 4), dtype=torch.float32)
    type_logits = torch.full((1, 4, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 0] = 6.0
    type_logits[0, 1, 2] = 6.0
    type_logits[0, 2, 1] = 6.0
    type_logits[0, 3, 2] = 6.0

    counts = _soft_directional_transition_counts(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        candidate_mask=candidate_mask,
    )

    assert float(counts["vac_to_fe"].item()) > 0.95
    assert float(counts["fe_to_vac"].item()) > 0.95
    assert float(counts["vac_to_cu"].item()) > 0.95
    assert float(counts["cu_to_vac"].item()) > 0.95


def test_matched_pair_count_loss_is_low_when_directional_counts_match_targets():
    change_logits = torch.full((1, 2), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0]], dtype=torch.long)
    target_types = torch.tensor([[0, 2]], dtype=torch.long)
    candidate_mask = torch.ones((1, 2), dtype=torch.float32)
    type_logits = torch.full((1, 2, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 0] = 6.0
    type_logits[0, 1, 2] = 6.0

    loss = _matched_pair_count_loss(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )

    assert float(loss.item()) < 1e-2


def test_matched_pair_count_loss_penalizes_single_sided_prediction():
    change_logits = torch.full((1, 2), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0]], dtype=torch.long)
    target_types = torch.tensor([[0, 2]], dtype=torch.long)
    candidate_mask = torch.ones((1, 2), dtype=torch.float32)
    type_logits = torch.full((1, 2, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 0] = 6.0
    type_logits[0, 1, 0] = 6.0

    loss = _matched_pair_count_loss(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        candidate_mask=candidate_mask,
    )

    assert float(loss.item()) > 0.3


def test_macro_edit_smoke_train_eval():
    cfg = {
        "lattice_size": (10, 10, 10),
        "max_episode_steps": 40,
        "max_vacancies": 8,
        "max_defects": 32,
        "max_shells": 8,
        "stats_dim": 10,
        "temperature": 300.0,
        "reward_scale": 1.0,
        "cu_density": 0.01,
        "v_density": 0.001,
        "rlkmc_topk": 8,
        "neighbor_order": "2NN",
    }
    env = MacroKMCEnv(cfg)
    env.reset()
    rng = np.random.default_rng(0)
    samples, stats = _collect_segments(
        env=env,
        num_segments=6,
        horizon_k=2,
        max_seed_vacancies=8,
        max_candidate_sites=48,
        rng=rng,
        max_attempt_multiplier=20,
    )
    assert len(samples) >= 2
    assert stats["coverage"] > 0.0

    train_samples = samples[:-1]
    val_samples = samples[-1:]
    train_loader = _build_loader(train_samples, batch_size=2, shuffle=False)
    val_loader = _build_loader(val_samples, batch_size=1, shuffle=False)
    model = MacroDreamerEditModel(
        max_vacancies=cfg["max_vacancies"],
        max_defects=cfg["max_defects"],
        max_shells=cfg["max_shells"],
        stats_dim=cfg["stats_dim"],
        lattice_size=cfg["lattice_size"],
        neighbor_order=cfg["neighbor_order"],
        dim_latent=8,
        graph_hidden_size=16,
        patch_hidden_size=32,
        patch_latent_dim=24,
        path_latent_dim=12,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = {
        "mask": 1.0,
        "type": 1.0,
        "pair": 0.0,
        "tau": 1.0,
        "reward": 0.5,
        "latent": 0.5,
        "proj": 0.5,
        "path": 0.05,
        "prior_edit": 0.25,
        "prior_latent": 0.25,
    }
    train_metrics = _train_epoch(model, train_loader, optimizer, "cpu", max_changed_sites=4, weights=weights)
    eval_metrics = _evaluate(model, val_loader, "cpu", max_changed_sites=4)
    assert np.isfinite(train_metrics["loss"])
    assert np.isfinite(eval_metrics["tau_log_mae"])
    assert np.isfinite(eval_metrics["reward_mae"])
    assert "projected_changed_type_acc" in eval_metrics
    assert "unchanged_vacancy_copy_acc" in eval_metrics
    assert "reachability_violation_rate" in eval_metrics
    assert "raw_vac_to_fe_count" in eval_metrics
    assert "raw_fe_to_vac_count" in eval_metrics
    assert "raw_matched_pair_count" in eval_metrics
    assert "pair" in train_metrics
    assert "tau_prior" in train_metrics
    assert "tau_post" in train_metrics


def test_teacher_path_summary_keeps_stepwise_time_sketch():
    path_infos = [
        {
            "dir_idx": 0,
            "moving_type": 1,
            "total_rate": 10.0,
            "expected_delta_t": 0.1,
            "delta_E": 0.5,
            "old_pos": np.asarray([0, 0, 0], dtype=np.int32),
            "new_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "vac_idx": 0,
        },
        {
            "dir_idx": 3,
            "moving_type": 0,
            "total_rate": 4.0,
            "expected_delta_t": 0.25,
            "delta_E": -0.2,
            "old_pos": np.asarray([1, 1, 1], dtype=np.int32),
            "new_pos": np.asarray([2, 2, 2], dtype=np.int32),
            "vac_idx": 1,
        },
    ]

    summary = mod._teacher_path_summary(path_infos, max_candidate_sites=32, horizon_k=4)

    assert summary.shape == (teacher_path_summary_dim(4),)
    assert np.isclose(summary[18], math.log(0.1 + 1e-12), atol=1e-6)
    assert np.isclose(summary[19], math.log(0.25 + 1e-12), atol=1e-6)
    assert np.isclose(summary[22], 0.5, atol=1e-6)
    assert np.isclose(summary[23], -0.2, atol=1e-6)


def test_teacher_path_summary_legacy_mode_keeps_base_dim_only():
    summary = mod._teacher_path_summary([], max_candidate_sites=32, horizon_k=4, include_stepwise_features=False)

    assert summary.shape == (teacher_path_summary_dim(4, include_stepwise_features=False),)
    assert summary.shape == (18,)


def test_macro_duration_baseline_matches_k_over_total_rate():
    global_summary = torch.zeros((2, 16), dtype=torch.float32)
    global_summary[:, 10] = torch.log(torch.tensor([100.0, 25.0], dtype=torch.float32))
    horizon_k = torch.tensor([4, 2], dtype=torch.long)

    baseline = macro_duration_baseline_log_tau(global_summary, horizon_k)
    expected = torch.log(torch.tensor([4.0 / 100.0, 2.0 / 25.0], dtype=torch.float32))

    assert torch.allclose(baseline, expected, atol=1e-6)


def test_predict_reward_and_duration_uses_physical_baseline_when_residual_is_zero():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    with torch.no_grad():
        model.duration_head[-1].weight.zero_()
        model.duration_head[-1].bias.zero_()

    global_summary = torch.zeros((1, 16), dtype=torch.float32)
    global_summary[0, 10] = math.log(100.0)
    _reward, tau_mu, _tau_log_sigma = model.predict_reward_and_duration(
        global_latent=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        predicted_next_global=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        path_latent=torch.zeros((1, model.path_latent_dim), dtype=torch.float32),
        global_summary=global_summary,
        horizon_k=torch.tensor([4], dtype=torch.long),
    )

    assert torch.allclose(tau_mu, torch.tensor([math.log(4.0 / 100.0)], dtype=torch.float32), atol=1e-6)


def test_detached_duration_inputs_block_backbone_gradients():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    global_summary = torch.zeros((1, 16), dtype=torch.float32)
    global_summary[0, 10] = math.log(80.0)

    global_latent = torch.randn((1, model.global_latent_dim), dtype=torch.float32, requires_grad=True)
    next_global = torch.randn((1, model.global_latent_dim), dtype=torch.float32, requires_grad=True)
    path_latent = torch.randn((1, model.path_latent_dim), dtype=torch.float32, requires_grad=True)
    _reward, tau_mu, tau_log_sigma = model.predict_reward_and_duration(
        global_latent=global_latent,
        predicted_next_global=next_global,
        path_latent=path_latent,
        global_summary=global_summary,
        horizon_k=torch.tensor([4], dtype=torch.long),
        detach_duration_inputs=True,
    )
    (tau_mu.sum() + tau_log_sigma.sum()).backward()

    assert global_latent.grad is None or torch.allclose(global_latent.grad, torch.zeros_like(global_latent.grad))
    assert next_global.grad is None or torch.allclose(next_global.grad, torch.zeros_like(next_global.grad))
    assert path_latent.grad is None or torch.allclose(path_latent.grad, torch.zeros_like(path_latent.grad))
    assert model.duration_head[-1].weight.grad is not None
    assert float(model.duration_head[-1].weight.grad.abs().sum().item()) > 0.0


def test_apply_projected_types_handles_non_prefix_origin_candidate():
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[1, 1, 1]], dtype=np.int32),
        start_cu_positions=np.asarray([[0, 0, 0]], dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(2),), dtype=np.float32),
        candidate_positions=np.asarray([[2, 2, 2], [9, 9, 9], [0, 0, 0]], dtype=np.float32),
        nearest_vacancy_offset=np.zeros((3, 3), dtype=np.float32),
        reach_depth=np.zeros((3,), dtype=np.float32),
        is_start_vacancy=np.zeros((3,), dtype=np.float32),
        current_types=np.asarray([0, 0, 1], dtype=np.int64),
        target_types=np.asarray([0, 0, 2], dtype=np.int64),
        candidate_mask=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        changed_mask=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=0.0,
        horizon_k=2,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )

    vacancies, cu_positions = mod._apply_projected_types(sample, np.asarray([0, 0, 2], dtype=np.int64))

    assert [0, 0, 0] in vacancies.tolist()
    assert [0, 0, 0] not in cu_positions.tolist()


def test_projected_mask_distill_loss_skips_violation_samples():
    change_logits = torch.zeros((2, 2), dtype=torch.float32)
    projected_changed_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    valid_mask = torch.ones((2, 2), dtype=torch.bool)
    reachability_violation = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = _projected_mask_distill_loss(
        change_logits=change_logits,
        projected_changed_mask=projected_changed_mask,
        valid_mask=valid_mask,
        reachability_violation=reachability_violation,
    )

    expected = F.binary_cross_entropy_with_logits(change_logits[1, :1], projected_changed_mask[1, :1])
    assert torch.isclose(loss, expected)


def test_projected_state_alignment_loss_skips_violation_samples():
    projected_patch_latent = torch.tensor([[9.0, 9.0], [1.0, 1.0]], dtype=torch.float32)
    target_patch_latent = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    projected_global = torch.tensor([[9.0, 9.0], [1.0, 1.0]], dtype=torch.float32)
    next_global = torch.tensor([[0.0, 0.0], [3.0, 3.0]], dtype=torch.float32)
    next_pred = torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    projected_changed_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    reachability_violation = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = _projected_state_alignment_loss(
        projected_patch_latent=projected_patch_latent,
        target_patch_latent=target_patch_latent,
        projected_global=projected_global,
        next_global=next_global,
        next_pred=next_pred,
        projected_changed_mask=projected_changed_mask,
        reachability_violation=reachability_violation,
    )

    expected = (
        F.smooth_l1_loss(projected_patch_latent[1:], target_patch_latent[1:])
        + 0.5 * F.smooth_l1_loss(projected_global[1:], next_global[1:])
        + 0.5 * F.smooth_l1_loss(projected_global[1:], next_pred[1:])
    )
    assert torch.isclose(loss, expected)


def test_projected_state_alignment_loss_skips_empty_projected_edit():
    loss = _projected_state_alignment_loss(
        projected_patch_latent=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        target_patch_latent=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        projected_global=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        next_global=torch.tensor([[3.0, 3.0]], dtype=torch.float32),
        next_pred=torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        projected_changed_mask=torch.zeros((1, 2), dtype=torch.float32),
        reachability_violation=torch.zeros((1,), dtype=torch.float32),
    )

    assert torch.isclose(loss, torch.zeros((), dtype=torch.float32))


def test_decode_edit_returns_raw_type_logits_without_copy_prior():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    with torch.no_grad():
        model.type_head.weight.zero_()
        model.type_head.bias.zero_()

    current_types = torch.tensor([[0, 1, 2]], dtype=torch.long)
    _change_logits, raw_type_logits = model.decode_edit(
        site_latent=torch.zeros((1, 3, 8), dtype=torch.float32),
        patch_latent=torch.zeros((1, 8), dtype=torch.float32),
        predicted_next_global=torch.zeros((1, model.global_latent_dim), dtype=torch.float32),
        path_latent=torch.zeros((1, 4), dtype=torch.float32),
        horizon_k=torch.tensor([2], dtype=torch.long),
        current_types=current_types,
    )

    assert torch.allclose(raw_type_logits, torch.zeros_like(raw_type_logits))


def test_apply_type_copy_bias_prefers_copy_when_type_head_is_neutral():
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    current_types = torch.tensor([[0, 1, 2]], dtype=torch.long)
    raw_type_logits = torch.zeros((1, 3, 3), dtype=torch.float32)

    type_logits = model.apply_type_copy_bias(raw_type_logits, current_types)

    assert torch.equal(type_logits.argmax(dim=-1), current_types)


def test_masked_type_cross_entropy_ignores_copy_prior_offset():
    current_types = torch.tensor([[0, 2]], dtype=torch.long)
    target_types = torch.tensor([[2, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True]])
    raw_type_logits = torch.zeros((1, 2, 3), dtype=torch.float32)
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )
    biased_type_logits = model.apply_type_copy_bias(raw_type_logits, current_types)

    raw_loss = mod._masked_type_cross_entropy(raw_type_logits, target_types, mask)
    biased_loss = mod._masked_type_cross_entropy(biased_type_logits, target_types, mask)

    assert torch.isclose(raw_loss, torch.tensor(math.log(3.0), dtype=torch.float32), atol=1e-6)
    assert float(biased_loss.item()) > float(raw_loss.item()) + 1.0


def test_edit_supervision_adds_vacancy_to_atom_type_term():
    change_logits = torch.full((1, 2), 8.0, dtype=torch.float32)
    current_types = torch.tensor([[2, 0]], dtype=torch.long)
    target_types = torch.tensor([[0, 2]], dtype=torch.long)
    changed_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    candidate_mask = torch.ones((1, 2), dtype=torch.float32)
    type_logits = torch.full((1, 2, 3), -6.0, dtype=torch.float32)
    type_logits[0, 0, 2] = 6.0
    type_logits[0, 1, 2] = 6.0

    losses = _edit_supervision_losses(
        change_logits=change_logits,
        type_logits=type_logits,
        current_types=current_types,
        target_types=target_types,
        changed_mask=changed_mask,
        candidate_mask=candidate_mask,
        aux_scale=1.0,
    )

    assert float(losses["vac_to_atom_type"].item()) > 5.0
    assert float(losses["atom_to_vac_type"].item()) < 1e-2


def test_evaluate_uses_raw_type_logits_for_type_metrics(monkeypatch):
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[1, 1, 1]], dtype=np.int32),
        start_cu_positions=np.empty((0, 3), dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(2),), dtype=np.float32),
        candidate_positions=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        nearest_vacancy_offset=np.zeros((2, 3), dtype=np.float32),
        reach_depth=np.zeros((2,), dtype=np.float32),
        is_start_vacancy=np.asarray([0.0, 1.0], dtype=np.float32),
        current_types=np.asarray([0, 2], dtype=np.int64),
        target_types=np.asarray([2, 2], dtype=np.int64),
        candidate_mask=np.asarray([1.0, 1.0], dtype=np.float32),
        changed_mask=np.asarray([1.0, 0.0], dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=0.0,
        horizon_k=2,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )

    class FakeModel:
        def eval(self):
            return self

        def encode_global(self, obs):
            return torch.zeros((obs.shape[0], 2), dtype=torch.float32, device=obs.device)

        def encode_patch(self, *, positions, nearest_vacancy_offset, reach_depth, is_start_vacancy, type_ids, node_mask, global_summary, box_dims):
            batch, sites = positions.shape[:2]
            return (
                torch.zeros((batch, sites, 2), dtype=torch.float32, device=positions.device),
                torch.zeros((batch, 2), dtype=torch.float32, device=positions.device),
            )

        def prior_stats(self, global_latent, global_summary, horizon_k):
            batch = global_latent.shape[0]
            return (
                torch.zeros((batch, 1), dtype=torch.float32, device=global_latent.device),
                torch.zeros((batch, 1), dtype=torch.float32, device=global_latent.device),
            )

        def sample_path_latent(self, mu, logvar, deterministic=False):
            return mu

        def predict_next_global(self, global_latent, path_latent, horizon_k):
            return global_latent

        def decode_edit(self, *, site_latent, patch_latent, predicted_next_global, path_latent, horizon_k, current_types):
            batch = current_types.shape[0]
            change_logits = torch.tensor([[8.0, -8.0]], dtype=torch.float32, device=current_types.device).expand(batch, -1).clone()
            raw_type_logits = torch.tensor(
                [[[1.5, -5.0, 2.0], [-5.0, -5.0, 4.0]]],
                dtype=torch.float32,
                device=current_types.device,
            ).expand(batch, -1, -1).clone()
            return change_logits, raw_type_logits

        def apply_type_copy_bias(self, raw_type_logits, current_types):
            copy_prior = F.one_hot(current_types, num_classes=3).float() * 2.0
            return raw_type_logits + copy_prior

        def predict_reward_and_duration(self, global_latent, predicted_next_global, path_latent, global_summary, horizon_k):
            batch = global_latent.shape[0]
            zeros = torch.zeros((batch,), dtype=torch.float32, device=global_latent.device)
            return zeros, zeros, zeros

    def fake_project_types_by_inventory(*, current_types, change_logits, type_logits, node_mask, positions, box_dims, horizon_k, max_changed_sites):
        batch = current_types.shape[0]
        return current_types.clone(), torch.zeros_like(node_mask), torch.zeros((batch,), dtype=torch.float32), torch.zeros((batch,), dtype=torch.float32)

    def fake_projected_global_latent_batch(*, batch, projected_types, model, device):
        return torch.zeros((len(batch), 2), dtype=torch.float32, device=device)

    monkeypatch.setattr(mod, "project_types_by_inventory", fake_project_types_by_inventory)
    monkeypatch.setattr(mod, "_projected_global_latent_batch", fake_projected_global_latent_batch)

    metrics = mod._evaluate(FakeModel(), _build_loader([sample], batch_size=1, shuffle=False), "cpu", max_changed_sites=2)

    assert metrics["changed_type_acc"] == pytest.approx(1.0)
    assert metrics["unchanged_vacancy_copy_acc"] == pytest.approx(1.0)
    assert metrics["raw_fe_to_vac_count"] == pytest.approx(1.0)


def test_initialize_output_heads_tracks_empirical_changed_rate_without_extra_sparsification():
    sample = mod.MacroSegmentSample(
        start_obs=np.zeros((4,), dtype=np.float32),
        next_obs=np.zeros((4,), dtype=np.float32),
        start_vacancy_positions=np.asarray([[0, 0, 0]], dtype=np.int32),
        start_cu_positions=np.asarray([[1, 1, 1]], dtype=np.int32),
        global_summary=np.zeros((16,), dtype=np.float32),
        teacher_path_summary=np.zeros((teacher_path_summary_dim(2),), dtype=np.float32),
        candidate_positions=np.zeros((50, 3), dtype=np.float32),
        nearest_vacancy_offset=np.zeros((50, 3), dtype=np.float32),
        reach_depth=np.zeros((50,), dtype=np.float32),
        is_start_vacancy=np.zeros((50,), dtype=np.float32),
        current_types=np.zeros((50,), dtype=np.int64),
        target_types=np.zeros((50,), dtype=np.int64),
        candidate_mask=np.ones((50,), dtype=np.float32),
        changed_mask=np.asarray([1.0] + [0.0] * 49, dtype=np.float32),
        tau_exp=1.0,
        tau_real=1.0,
        reward_sum=0.0,
        horizon_k=2,
        box_dims=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
    )
    model = MacroDreamerEditModel(
        max_vacancies=4,
        max_defects=8,
        max_shells=4,
        stats_dim=10,
        lattice_size=(10, 10, 10),
        neighbor_order="2NN",
        dim_latent=4,
        graph_hidden_size=8,
        patch_hidden_size=16,
        patch_latent_dim=8,
        path_latent_dim=4,
        global_summary_dim=16,
        teacher_path_summary_dim=teacher_path_summary_dim(2),
        max_macro_k=4,
    )

    mod._initialize_output_heads(model, [sample])

    expected_changed_rate = 1.0 / 50.0
    assert math.isclose(torch.sigmoid(model.change_head.bias).item(), expected_changed_rate, rel_tol=1e-6)


def test_validate_resume_args_rejects_segment_k_mismatch():
    with pytest.raises(ValueError, match="segment_k"):
        _validate_resume_args(SimpleNamespace(segment_k=4), {"segment_k": 2})


def test_validate_resume_args_rejects_path_summary_mode_mismatch():
    with pytest.raises(ValueError, match="teacher_path_summary_mode"):
        _validate_resume_args(
            SimpleNamespace(segment_k=4, teacher_path_summary_mode="stepwise"),
            {"segment_k": 4, "teacher_path_summary_mode": "legacy"},
        )


def test_initialize_best_score_from_saved_best_uses_best_checkpoint(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    best_model = torch.nn.Linear(1, 1, bias=False)
    best_model.weight.data.fill_(0.1)
    torch.save({"model": best_model.state_dict()}, tmp_path / "best_model.pt")

    def fake_evaluate(eval_model, loader, device, max_changed_sites):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
    )

    assert math.isclose(score, 0.1, rel_tol=1e-6)
    assert source == "saved best model"
    assert math.isclose(float(model.weight.detach().cpu().item()), 0.9, rel_tol=1e-6)


def test_initialize_best_score_from_checkpoint_fallback_uses_stored_best_score(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    def fake_evaluate(eval_model, loader, device, max_changed_sites):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
        checkpoint_best_score=0.2,
    )

    assert math.isclose(score, 0.2, rel_tol=1e-6)
    assert source == "resume checkpoint + stored best_score"
    assert math.isclose(float(model.weight.detach().cpu().item()), 0.9, rel_tol=1e-6)


def test_initialize_best_score_from_checkpoint_skips_stored_best_score_for_new_save_dir(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    def fake_evaluate(eval_model, loader, device, max_changed_sites):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
        checkpoint_best_score=0.2,
        allow_checkpoint_best_score_fallback=False,
    )

    assert math.isclose(score, 0.9, rel_tol=1e-6)
    assert source == "resume checkpoint"
    assert math.isclose(float(model.weight.detach().cpu().item()), 0.9, rel_tol=1e-6)


def test_initialize_best_score_skips_incompatible_saved_best_model(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.9)

    incompatible_best_model = torch.nn.Linear(2, 1, bias=False)
    torch.save({"model": incompatible_best_model.state_dict()}, tmp_path / "best_model.pt")

    def fake_evaluate(eval_model, loader, device, max_changed_sites):
        tau_log_mae = float(eval_model.weight.detach().cpu().item())
        return {
            "tau_log_mae": tau_log_mae,
            "reward_mae": 0.0,
            "change_topk_f1": 1.0,
            "projected_change_f1": 1.0,
            "projected_changed_type_acc": 1.0,
            "reachability_violation_rate": 0.0,
        }

    monkeypatch.setattr(mod, "_evaluate", fake_evaluate)
    score, source = _initialize_best_score_from_saved_best(
        model=model,
        loader=None,
        device="cpu",
        max_changed_sites=4,
        dataset_stats={"val": {"coverage": 1.0}},
        save_dir=tmp_path,
    )

    assert math.isclose(score, 0.9, rel_tol=1e-6)
    assert "skipped incompatible saved best model" in source


def test_eval_cache_validation_rejects_segment_k_mismatch(tmp_path):
    payload = {
        "train": [],
        "val": [],
        "stats": {},
        "signature": {"dataset_version": 7, "segment_k": 2},
    }
    cache_path = tmp_path / "segments.pt"
    torch.save(payload, cache_path)

    with pytest.raises(ValueError, match="segment_k"):
        eval_mod._load_samples(cache_path, "val", 0, expected_segment_k=4)


def test_selection_score_penalizes_projected_global_l1():
    base_metrics = {
        "tau_log_mae": 2.0,
        "reward_mae": 1.0,
        "change_topk_f1": 0.2,
        "projected_change_f1": 0.1,
        "projected_changed_type_acc": 0.1,
        "projected_global_l1": 0.002,
        "unchanged_vacancy_copy_acc": 0.95,
        "reachability_violation_rate": 0.0,
    }
    worse_metrics = dict(base_metrics)
    worse_metrics["projected_global_l1"] = 0.008

    base_score = _selection_score(base_metrics, {"val": {"coverage": 1.0}})
    worse_score = _selection_score(worse_metrics, {"val": {"coverage": 1.0}})

    assert worse_score > base_score


def test_selection_score_penalizes_unchanged_vacancy_copy_acc():
    base_metrics = {
        "tau_log_mae": 2.0,
        "reward_mae": 1.0,
        "change_topk_f1": 0.2,
        "projected_change_f1": 0.1,
        "projected_changed_type_acc": 0.1,
        "projected_global_l1": 0.002,
        "unchanged_vacancy_copy_acc": 0.95,
        "reachability_violation_rate": 0.0,
    }
    worse_metrics = dict(base_metrics)
    worse_metrics["unchanged_vacancy_copy_acc"] = 0.60

    base_score = _selection_score(base_metrics, {"val": {"coverage": 1.0}})
    worse_score = _selection_score(worse_metrics, {"val": {"coverage": 1.0}})

    assert worse_score > base_score