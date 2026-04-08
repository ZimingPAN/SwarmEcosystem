"""
Plot time alignment analysis and eval comparison for macro-edit Dreamer v22/v23.

Baseline: Traditional KMC (teacher segments).
Generates two figures:
1. macro_edit_time_alignment.png -- τ alignment with KMC baseline
2. macro_edit_eval_comparison.png -- training curves + final metrics
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent

# ── Load full-sample eval data ──────────────────────────────

def load_eval(version_dir: str) -> dict:
    p = ROOT / "dreamer4-main" / "results" / version_dir / "eval_full_samples.json"
    with open(p) as f:
        return json.load(f)

v22_eval = load_eval("dreamer_macro_edit_v22_extreme")
v23_eval = load_eval("dreamer_macro_edit_v23_ultraextreme")

def extract_arrays(eval_data: dict):
    samples = eval_data["all_samples"]
    pred_tau = np.array([s["predicted_tau"] for s in samples])
    true_tau_exp = np.array([s["traditional_kmc_expected_tau"] for s in samples])
    true_tau_real = np.array([s["traditional_kmc_realized_tau"] for s in samples])
    pred_reward = np.array([s["predicted_reward_sum"] for s in samples])
    true_reward = np.array([s["traditional_kmc_reward_sum"] for s in samples])
    pred_de = np.array([s["predicted_delta_e"] for s in samples])
    true_de = np.array([s["traditional_kmc_delta_e"] for s in samples])
    return pred_tau, true_tau_exp, true_tau_real, pred_reward, true_reward, pred_de, true_de

v22_arrays = extract_arrays(v22_eval)
v23_arrays = extract_arrays(v23_eval)

# ── Parse training VAL logs ─────────────────────────────────

def parse_val_log(version_dir: str) -> dict:
    p = ROOT / "dreamer4-main" / "results" / version_dir / "train.log"
    metrics = {k: [] for k in [
        "reward_mae", "reward_corr", "tau_log_mae", "tau_log_corr",
        "tau_scale", "change_f1", "proj_change_f1", "chg_type_acc",
        "proj_chg_type_acc"
    ]}
    with open(p) as f:
        for line in f:
            if ">>> VAL" not in line:
                continue
            for key in metrics:
                m = re.search(rf"{key}=([\d.e+-]+)", line)
                if m:
                    metrics[key].append(float(m.group(1)))
    return metrics

def parse_train_loss(version_dir: str) -> list:
    p = ROOT / "dreamer4-main" / "results" / version_dir / "train.log"
    losses = []
    with open(p) as f:
        for line in f:
            m = re.search(r"\[Epoch\s+\d+/\d+\]\s+loss=([\d.]+)", line)
            if m:
                losses.append(float(m.group(1)))
    return losses

v22_val = parse_val_log("dreamer_macro_edit_v22_extreme")
v23_val = parse_val_log("dreamer_macro_edit_v23_ultraextreme")
v22_loss = parse_train_loss("dreamer_macro_edit_v22_extreme")
v23_loss = parse_train_loss("dreamer_macro_edit_v23_ultraextreme")

# ═══════════════════════════════════════════════════════════
# FIGURE 1: Time & Energy Alignment (baseline = Traditional KMC)
# ═══════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 32))
fig.suptitle(
    "Macro-Edit Dreamer: Time & Energy Alignment\n"
    "Baseline = Traditional KMC  |  400 val segments, k=4, cu=1.34%, v=0.02%",
    fontsize=16, fontweight="bold", y=0.99
)

gs = fig.add_gridspec(6, 2, hspace=0.42, wspace=0.3,
                       top=0.95, bottom=0.03, left=0.08, right=0.95)

# ── Row 1: τ Scatter (log-log) — Predicted vs KMC Baseline ──

for col, (name, arrays, eval_data, clr) in enumerate([
    ("v22", v22_arrays, v22_eval, "tab:orange"),
    ("v23", v23_arrays, v23_eval, "tab:green"),
]):
    ax = fig.add_subplot(gs[0, col])
    pred_tau, true_tau_exp, *_ = arrays
    tau_metrics = eval_data["tau_expected"]

    ax.scatter(true_tau_exp, pred_tau, alpha=0.4, s=15, c=clr, edgecolors="none")
    lo = min(true_tau_exp.min(), pred_tau.min()) * 0.5
    hi = max(true_tau_exp.max(), pred_tau.max()) * 2
    ax.plot([lo, hi], [lo, hi], "--", color="black", alpha=0.6, linewidth=2,
            label="y=x (perfect match with KMC)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Traditional KMC: E[tau]", fontsize=10)
    ax.set_ylabel("World Model: Predicted tau", fontsize=10)
    ax.set_title(
        f"{name} -- Predicted tau vs KMC Baseline\n"
        f"log_MAE={tau_metrics['log_mae']:.4f}, log_corr={tau_metrics['log_corr']:.4f}, "
        f"scale={tau_metrics['scale_ratio']:.2f}",
        fontsize=11
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")

# ── Row 2: Residual plot — log(pred/true) to show time prediction error ──

for col, (name, arrays, eval_data, clr) in enumerate([
    ("v22", v22_arrays, v22_eval, "tab:orange"),
    ("v23", v23_arrays, v23_eval, "tab:green"),
]):
    ax = fig.add_subplot(gs[1, col])
    pred_tau, true_tau_exp, *_ = arrays
    eps = 1e-12
    log_ratio = np.log10(np.clip(pred_tau, eps, None) / np.clip(true_tau_exp, eps, None))

    ax.scatter(true_tau_exp, log_ratio, alpha=0.35, s=12, c=clr, edgecolors="none")
    ax.axhline(0, color="black", linewidth=2, linestyle="--", label="Perfect (pred = KMC)")
    ax.set_xscale("log")
    ax.set_xlabel("Traditional KMC: E[tau]", fontsize=10)
    ax.set_ylabel("log10(Predicted / KMC)", fontsize=10)

    median_ratio = np.median(log_ratio)
    pct_within_2x = np.mean(np.abs(log_ratio) < np.log10(2)) * 100
    pct_within_5x = np.mean(np.abs(log_ratio) < np.log10(5)) * 100
    ax.axhline(np.log10(2), color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(-np.log10(2), color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(median_ratio, color=clr, linewidth=1.5, linestyle="-",
               label=f"Median ratio = {10**median_ratio:.2f}x")

    ax.set_title(
        f"{name} -- Time Prediction Residual vs KMC\n"
        f"Within 2x: {pct_within_2x:.0f}%, Within 5x: {pct_within_5x:.0f}%",
        fontsize=11
    )
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 2)
    ax.grid(True, alpha=0.2)

# ── Row 3: Error distribution comparison (v22 vs v23 side by side) ──

ax_err = fig.add_subplot(gs[2, 0])
eps = 1e-12
v22_log_ratio = np.log10(np.clip(v22_arrays[0], eps, None) / np.clip(v22_arrays[1], eps, None))
v23_log_ratio = np.log10(np.clip(v23_arrays[0], eps, None) / np.clip(v23_arrays[1], eps, None))

bins_err = np.linspace(-2, 2, 60)
ax_err.hist(v22_log_ratio, bins=bins_err, alpha=0.5, color="tab:orange",
            label=f"v22 (std={np.std(v22_log_ratio):.3f})", density=True)
ax_err.hist(v23_log_ratio, bins=bins_err, alpha=0.5, color="tab:green",
            label=f"v23 (std={np.std(v23_log_ratio):.3f})", density=True)
ax_err.axvline(0, color="black", linewidth=2, linestyle="--", label="Perfect (= KMC)")
ax_err.axvline(np.mean(v22_log_ratio), color="tab:orange", linewidth=1.5, linestyle="-",
               label=f"v22 mean bias={np.mean(v22_log_ratio):+.3f}")
ax_err.axvline(np.mean(v23_log_ratio), color="tab:green", linewidth=1.5, linestyle="-",
               label=f"v23 mean bias={np.mean(v23_log_ratio):+.3f}")
ax_err.set_xlabel("log10(Predicted tau / KMC tau)", fontsize=10)
ax_err.set_ylabel("Density", fontsize=10)
ax_err.set_title("Time Prediction Error Distribution\n(centered at 0 = perfect KMC match)", fontsize=11)
ax_err.legend(fontsize=7.5)
ax_err.grid(True, alpha=0.2)

# ── Row 3 right: Boxplot comparison ──

ax_box = fig.add_subplot(gs[2, 1])
bp = ax_box.boxplot([v22_log_ratio, v23_log_ratio],
                     tick_labels=["v22", "v23"],
                     patch_artist=True,
                     widths=0.5,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
bp["boxes"][0].set_facecolor("tab:orange")
bp["boxes"][0].set_alpha(0.4)
bp["boxes"][1].set_facecolor("tab:green")
bp["boxes"][1].set_alpha(0.4)
ax_box.axhline(0, color="black", linewidth=2, linestyle="--", label="= KMC baseline")
ax_box.set_ylabel("log10(Predicted / KMC)", fontsize=10)
ax_box.set_title("Time Prediction Error: v22 vs v23\n(mean = red diamond, closer to 0 = better)", fontsize=11)
ax_box.legend(fontsize=9)
ax_box.grid(True, alpha=0.2, axis="y")

# ── Row 4: Reward scatter (Predicted vs KMC Baseline) ──

for col, (name, arrays, eval_data, clr) in enumerate([
    ("v22", v22_arrays, v22_eval, "tab:orange"),
    ("v23", v23_arrays, v23_eval, "tab:green"),
]):
    ax = fig.add_subplot(gs[3, col])
    _, _, _, pred_reward, true_reward, _, _ = arrays
    rw_metrics = eval_data["reward_sum"]

    ax.scatter(true_reward, pred_reward, alpha=0.4, s=15, c=clr, edgecolors="none")
    lo_r = min(true_reward.min(), pred_reward.min()) - 0.5
    hi_r = max(true_reward.max(), pred_reward.max()) + 0.5
    ax.plot([lo_r, hi_r], [lo_r, hi_r], "--", color="black", alpha=0.6, linewidth=2,
            label="y=x (= KMC baseline)")
    ax.set_xlabel("Traditional KMC: Reward Sum", fontsize=10)
    ax.set_ylabel("World Model: Predicted Reward Sum", fontsize=10)
    ax.set_title(
        f"{name} -- Reward vs KMC Baseline\n"
        f"MAE={rw_metrics['mae']:.4f}, corr={rw_metrics['corr']:.4f}",
        fontsize=11
    )
    ax.legend(fontsize=9, loc="upper left")

# ── Row 5: Energy vs Time (REAL trajectory order, by segment index) ──

ax_real = fig.add_subplot(gs[4, :])

# Segments are consecutive k-step windows from continuous rollouts,
# reset every ~50 segments. Use original index order = real trajectory order.
pred_tau_v22, true_tau_exp_v22, _, _, _, pred_de_v22, true_de_v22 = v22_arrays
pred_tau_v23, true_tau_exp_v23, _, _, _, pred_de_v23, true_de_v23 = v23_arrays

# Detect rollout boundaries: large jump in tau or sign the env was reset
# Heuristic: if tau suddenly drops by >10x from previous, likely a reset
def detect_rollout_boundaries(tau_arr):
    """Find indices where a new rollout starts (env.reset)."""
    boundaries = [0]
    for i in range(1, len(tau_arr)):
        ratio = tau_arr[i] / max(tau_arr[i-1], 1e-15)
        # Also detect if tau pattern breaks -- use a simpler heuristic:
        # every 50 segments is a reset boundary per the collection code
        pass
    # Use the known collection logic: max_segments_per_rollout=50
    for i in range(50, len(tau_arr), 50):
        if i < len(tau_arr):
            boundaries.append(i)
    return boundaries

boundaries = detect_rollout_boundaries(true_tau_exp_v22)

# Plot by original order (real trajectory)
cum_kmc_tau_real = np.cumsum(true_tau_exp_v22)
cum_kmc_de_real = np.cumsum(true_de_v22)
cum_v22_pred_tau_real = np.cumsum(pred_tau_v22)
cum_v22_pred_de_real = np.cumsum(pred_de_v22)
cum_v23_pred_tau_real = np.cumsum(pred_tau_v23)
cum_v23_pred_de_real = np.cumsum(pred_de_v23)

ax_real.plot(cum_kmc_tau_real, -cum_kmc_de_real, color="black", linewidth=2.5, alpha=0.8,
             label="Traditional KMC (baseline)", zorder=5)
ax_real.plot(cum_v22_pred_tau_real, -cum_v22_pred_de_real, color="tab:orange", linewidth=1.5,
             alpha=0.8, label="v22 (pred time + pred energy)")
ax_real.plot(cum_v23_pred_tau_real, -cum_v23_pred_de_real, color="tab:green", linewidth=1.5,
             alpha=0.8, label="v23 (pred time + pred energy)")

# Mark rollout boundaries
for bi, b in enumerate(boundaries[1:]):
    ax_real.axvline(cum_kmc_tau_real[b-1], color="gray", linewidth=0.8, linestyle=":",
                    alpha=0.4, label="Rollout reset" if bi == 0 else None)

ax_real.set_xlabel("Cumulative Physical Time tau (s)  [original trajectory order]", fontsize=11)
ax_real.set_ylabel("Cumulative -Delta_E (eV, energy descent)", fontsize=11)

total_steps = len(true_tau_exp_v22) * 4
total_time = cum_kmc_tau_real[-1]
n_rollouts = len(boundaries)
ax_real.set_title(
    f"Energy vs Time: Real Trajectory Order (original segment sequence)\n"
    f"{len(true_tau_exp_v22)} segments x k=4 = {total_steps} KMC steps, "
    f"~{n_rollouts} rollouts, total time = {total_time:.2f}s  |  "
    f"Dashed vertical = rollout reset",
    fontsize=12
)
ax_real.legend(fontsize=9, ncol=2)
ax_real.grid(True, alpha=0.3)

# ── Row 6: Energy vs Time (sorted by tau, pseudo-trajectory) ──

ax = fig.add_subplot(gs[5, :])

# KMC baseline: sort by true tau, cumulative
order = np.argsort(true_tau_exp_v22)
cum_kmc_tau = np.cumsum(true_tau_exp_v22[order])
cum_kmc_de = np.cumsum(true_de_v22[order])

ax.plot(cum_kmc_tau, -cum_kmc_de, color="black", linewidth=2.5, alpha=0.8,
        label="Traditional KMC (baseline)", zorder=5)

# v22: predicted time axis with same energy ordering
cum_v22_pred_tau = np.cumsum(pred_tau_v22[order])
ax.plot(cum_v22_pred_tau, -cum_kmc_de, color="tab:orange", linewidth=1.8, alpha=0.8,
        linestyle="-", label="v22 predicted time")
# v22: predicted energy on KMC time axis
cum_v22_pred_de = np.cumsum(pred_de_v22[order])
ax.plot(cum_kmc_tau, -cum_v22_pred_de, color="tab:orange", linewidth=1.5, alpha=0.6,
        linestyle="--", label="v22 predicted energy")

# v23: same ordering
cum_v23_pred_tau = np.cumsum(pred_tau_v23[order])
ax.plot(cum_v23_pred_tau, -cum_kmc_de, color="tab:green", linewidth=1.8, alpha=0.8,
        linestyle="-", label="v23 predicted time")
cum_v23_pred_de = np.cumsum(pred_de_v23[order])
ax.plot(cum_kmc_tau, -cum_v23_pred_de, color="tab:green", linewidth=1.5, alpha=0.6,
        linestyle="--", label="v23 predicted energy")

# Mark 90% energy point
total_de = cum_kmc_de[-1]
idx_90 = np.searchsorted(cum_kmc_de, total_de * 0.9)
if idx_90 < len(cum_kmc_tau):
    ax.axvline(cum_kmc_tau[idx_90], color="red", linewidth=1.5, linestyle="--", alpha=0.6)
    pct_time = cum_kmc_tau[idx_90] / cum_kmc_tau[-1] * 100
    ax.annotate(f"90% energy drop\nat {pct_time:.1f}% of total time",
                xy=(cum_kmc_tau[idx_90], -cum_kmc_de[idx_90]),
                xytext=(cum_kmc_tau[-1]*0.4, -total_de*0.5),
                fontsize=10, color="red",
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.6))

ax.set_xlabel("Cumulative Physical Time tau (s)  [sorted by tau, pseudo-trajectory]", fontsize=11)
ax.set_ylabel("Cumulative -Delta_E (eV, energy descent)", fontsize=11)
ax.set_title(
    "Energy vs Time: Sorted by tau (pseudo-trajectory)\n"
    "Solid = model time on KMC energy  |  Dashed = model energy on KMC time  |  "
    "Energy flattens because near-equilibrium segments have large tau but ~0 energy change",
    fontsize=11
)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

out_path = FIG_DIR / "macro_edit_time_alignment.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════
# FIGURE 2: Eval Comparison (v22 vs v23, baseline = KMC)
# ═══════════════════════════════════════════════════════════

fig2 = plt.figure(figsize=(18, 16))
fig2.suptitle(
    "Macro-Edit Dreamer: v22 vs v23 Training & Evaluation\n"
    "Baseline = Traditional KMC  |  2000 train / 400 val segments",
    fontsize=15, fontweight="bold", y=0.98
)

gs2 = fig2.add_gridspec(3, 2, hspace=0.4, wspace=0.3,
                         top=0.92, bottom=0.06, left=0.08, right=0.95)

# ── Panel 1: Training Loss Curve ─────────────────────────

ax1 = fig2.add_subplot(gs2[0, 0])
ax1.plot(range(1, len(v22_loss)+1), v22_loss, color="tab:orange", alpha=0.8, label="v22 (80 epochs)")
ax1.plot(range(1, len(v23_loss)+1), v23_loss, color="tab:green", alpha=0.8, label="v23 (120 epochs)")
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Total Loss", fontsize=10)
ax1.set_title("Training Loss", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Panel 2: VAL tau_log_mae over epochs ──────────────────

ax2 = fig2.add_subplot(gs2[0, 1])
v22_val_epochs = np.arange(1, len(v22_val["tau_log_mae"])+1) * 5
v23_val_epochs = np.arange(1, len(v23_val["tau_log_mae"])+1) * 5
ax2.plot(v22_val_epochs, v22_val["tau_log_mae"], "o-", color="tab:orange", markersize=4, label="v22")
ax2.plot(v23_val_epochs, v23_val["tau_log_mae"], "s-", color="tab:green", markersize=4, label="v23")
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("tau log MAE vs KMC (lower = better)", fontsize=10)
ax2.set_title("VAL: Time Prediction Error vs KMC Baseline", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Panel 3: VAL reward_corr over epochs ──────────────────

ax3 = fig2.add_subplot(gs2[1, 0])
ax3.plot(v22_val_epochs, v22_val["reward_corr"], "o-", color="tab:orange", markersize=4, label="v22")
ax3.plot(v23_val_epochs, v23_val["reward_corr"], "s-", color="tab:green", markersize=4, label="v23")
ax3.set_xlabel("Epoch", fontsize=10)
ax3.set_ylabel("Reward Correlation with KMC (higher = better)", fontsize=10)
ax3.set_title("VAL: Reward Correlation with KMC Baseline", fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Panel 4: VAL tau_log_corr over epochs ─────────────────

ax4 = fig2.add_subplot(gs2[1, 1])
ax4.plot(v22_val_epochs, v22_val["tau_log_corr"], "o-", color="tab:orange", markersize=4, label="v22")
ax4.plot(v23_val_epochs, v23_val["tau_log_corr"], "s-", color="tab:green", markersize=4, label="v23")
ax4.set_xlabel("Epoch", fontsize=10)
ax4.set_ylabel("tau log Correlation with KMC (higher = better)", fontsize=10)
ax4.set_title("VAL: Time Prediction Correlation with KMC", fontsize=12)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# ── Panel 5: Final Metrics Bar Chart (relative to KMC) ──

ax5 = fig2.add_subplot(gs2[2, 0])
metrics_names = ["tau log_MAE\nvs KMC (lower)", "tau log_corr\nvs KMC (higher)", "tau scale\nvs KMC (->1.0)",
                 "reward MAE\nvs KMC (lower)", "reward corr\nvs KMC (higher)"]
v22_finals = [
    v22_eval["tau_expected"]["log_mae"],
    v22_eval["tau_expected"]["log_corr"],
    v22_eval["tau_expected"]["scale_ratio"],
    v22_eval["reward_sum"]["mae"],
    v22_eval["reward_sum"]["corr"],
]
v23_finals = [
    v23_eval["tau_expected"]["log_mae"],
    v23_eval["tau_expected"]["log_corr"],
    v23_eval["tau_expected"]["scale_ratio"],
    v23_eval["reward_sum"]["mae"],
    v23_eval["reward_sum"]["corr"],
]

x = np.arange(len(metrics_names))
w = 0.35
bars1 = ax5.bar(x - w/2, v22_finals, w, label="v22", color="tab:orange", alpha=0.8)
bars2 = ax5.bar(x + w/2, v23_finals, w, label="v23", color="tab:green", alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names, fontsize=8)
ax5.set_title("Final Eval Metrics: v22 vs v23 (relative to KMC)", fontsize=12)
ax5.legend(fontsize=9)
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=8)
ax5.grid(True, alpha=0.3, axis="y")

# ── Panel 6: VAL change_f1 / proj_change_f1 ──────────────

ax6 = fig2.add_subplot(gs2[2, 1])
ax6.plot(v22_val_epochs, v22_val["change_f1"], "o-", color="tab:orange", markersize=3,
         alpha=0.6, label="v22 change_f1")
ax6.plot(v23_val_epochs, v23_val["change_f1"], "s-", color="tab:green", markersize=3,
         alpha=0.6, label="v23 change_f1")
ax6.plot(v22_val_epochs, v22_val["proj_change_f1"], "o--", color="tab:orange", markersize=3,
         alpha=0.8, label="v22 proj_change_f1")
ax6.plot(v23_val_epochs, v23_val["proj_change_f1"], "s--", color="tab:green", markersize=3,
         alpha=0.8, label="v23 proj_change_f1")
ax6.set_xlabel("Epoch", fontsize=10)
ax6.set_ylabel("F1 Score (higher = better)", fontsize=10)
ax6.set_title("VAL: Structure Prediction Quality", fontsize=12)
ax6.legend(fontsize=8, ncol=2)
ax6.grid(True, alpha=0.3)

out_path2 = FIG_DIR / "macro_edit_eval_comparison.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
plt.close(fig2)

# ── Print summary ────────────────────────────────────────

print("\n" + "="*60)
print("Summary: v22 vs v23 Macro-Edit Dreamer (baseline = Traditional KMC)")
print("="*60)
print(f"{'Metric':<25} {'v22':>12} {'v23':>12} {'Better':>8}")
print("-"*60)
for name, k1, k2 in [
    ("tau log_MAE (lower)", "tau_expected", "log_mae"),
    ("tau log_corr (higher)", "tau_expected", "log_corr"),
    ("tau scale_ratio (->1)", "tau_expected", "scale_ratio"),
    ("reward MAE (lower)", "reward_sum", "mae"),
    ("reward corr (higher)", "reward_sum", "corr"),
]:
    v22_v = v22_eval[k1][k2]
    v23_v = v23_eval[k1][k2]
    if "lower" in name:
        better = "v22" if v22_v < v23_v else "v23"
    elif "higher" in name:
        better = "v22" if v22_v > v23_v else "v23"
    else:
        better = "v22" if abs(v22_v - 1) < abs(v23_v - 1) else "v23"
    print(f"{name:<25} {v22_v:>12.4f} {v23_v:>12.4f} {better:>8}")

# Time prediction accuracy summary
print(f"\n--- Time Prediction Accuracy (vs KMC baseline) ---")
eps = 1e-12
for name, arrays in [("v22", v22_arrays), ("v23", v23_arrays)]:
    log_ratio = np.log10(np.clip(arrays[0], eps, None) / np.clip(arrays[1], eps, None))
    pct_2x = np.mean(np.abs(log_ratio) < np.log10(2)) * 100
    pct_5x = np.mean(np.abs(log_ratio) < np.log10(5)) * 100
    median_factor = 10**np.median(log_ratio)
    print(f"  {name}: within 2x of KMC: {pct_2x:.1f}%, within 5x: {pct_5x:.1f}%, "
          f"median pred/KMC = {median_factor:.3f}")

print(f"\nTraditional KMC reward_sum mean: {v22_eval['traditional_energy']['reward_sum_mean']:.4f}")
print(f"v22 predicted reward_sum mean:  {v22_eval['predicted_energy']['reward_sum_mean']:.4f}")
print(f"v23 predicted reward_sum mean:  {v23_eval['predicted_energy']['reward_sum_mean']:.4f}")

print(f"\nv22: 80 epochs x 2000 segments = 160,000 sample updates")
print(f"v23: 120 epochs x 2000 segments = 240,000 sample updates")
