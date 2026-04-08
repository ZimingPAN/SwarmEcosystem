"""
Eval comparison: all methods vs PPO+GNN baseline.

Generates: macro_edit_eval_comparison.png
- PPO+GNN as baseline reference
- Compares PPO, MuZero, Dreamer(v9), macro-edit Dreamer v22/v23
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

# ── Load v9 eval data (PPO, MuZero, Dreamer episode rewards) ──

def load_v9_eval(name: str) -> dict:
    p = ROOT / f"eval_{name}_v9.json"
    with open(p) as f:
        return json.load(f)

ppo_eval = load_v9_eval("ppo")
muzero_eval = load_v9_eval("muzero")
dreamer_eval = load_v9_eval("dreamer")

ppo_means = np.array([r["mean_reward"] for r in ppo_eval["results"]])
muzero_means = np.array([r["mean_reward"] for r in muzero_eval["results"]])
dreamer_means = np.array([r["mean_reward"] for r in dreamer_eval["results"]])

# ── Load macro-edit eval data ──

def load_macro_eval(version_dir: str) -> dict:
    p = ROOT / "dreamer4-main" / "results" / version_dir / "eval_full_samples.json"
    with open(p) as f:
        return json.load(f)

v22_eval = load_macro_eval("dreamer_macro_edit_v22_extreme")
v23_eval = load_macro_eval("dreamer_macro_edit_v23_ultraextreme")

def extract_arrays(eval_data: dict):
    samples = eval_data["all_samples"]
    pred_tau = np.array([s["predicted_tau"] for s in samples])
    true_tau_exp = np.array([s["traditional_kmc_expected_tau"] for s in samples])
    pred_reward = np.array([s["predicted_reward_sum"] for s in samples])
    true_reward = np.array([s["traditional_kmc_reward_sum"] for s in samples])
    pred_de = np.array([s["predicted_delta_e"] for s in samples])
    true_de = np.array([s["traditional_kmc_delta_e"] for s in samples])
    return pred_tau, true_tau_exp, pred_reward, true_reward, pred_de, true_de

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
# FIGURE: Eval Comparison (Baseline = PPO+GNN)
# ═══════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 22))
fig.suptitle(
    "Macro-Edit Dreamer Evaluation\n"
    "Baseline = PPO+GNN  |  All methods on 40x40x40 FCC lattice",
    fontsize=16, fontweight="bold", y=0.99
)

gs = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.3,
                       top=0.94, bottom=0.04, left=0.08, right=0.95)

# ══════════════════════════════════════════
# Row 1: Episode Reward -- all methods vs PPO baseline
# ══════════════════════════════════════════

# ── Panel 1a: Reward over evaluation rounds ──
ax1 = fig.add_subplot(gs[0, 0])
rounds = np.arange(1, 101)

# PPO baseline band
ax1.fill_between(rounds, 0, ppo_means, alpha=0.15, color="red", label=f"PPO+GNN (mean={np.mean(ppo_means):.3f})")
ax1.plot(rounds, ppo_means, color="red", linewidth=1, alpha=0.5)

# Smoothing for readability
def smooth(arr, w=5):
    return np.convolve(arr, np.ones(w)/w, mode="valid")

r_smooth = rounds[:len(smooth(muzero_means))]
ax1.plot(r_smooth, smooth(muzero_means), color="tab:blue", linewidth=1.8, alpha=0.8,
         label=f"MuZero (mean={np.mean(muzero_means):.3f})")
ax1.plot(r_smooth, smooth(dreamer_means), color="tab:purple", linewidth=1.8, alpha=0.8,
         label=f"Dreamer v9 (mean={np.mean(dreamer_means):.3f})")

ax1.axhline(np.mean(ppo_means), color="red", linestyle="--", linewidth=1, alpha=0.6)
ax1.set_xlabel("Evaluation Round", fontsize=10)
ax1.set_ylabel("Mean Episode Reward (5 ep/round)", fontsize=10)
ax1.set_title("Episode Reward: All Methods\n(PPO+GNN baseline in red)", fontsize=12)
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(True, alpha=0.3)

# ── Panel 1b: Reward distribution boxplot ──
ax2 = fig.add_subplot(gs[0, 1])

# Expand all individual episode rewards
ppo_all = np.array([float(r) for res in ppo_eval["results"] for r in res["rewards"]])
muzero_all = np.array([float(r) for res in muzero_eval["results"] for r in res["rewards"]])
dreamer_all = np.array([float(r) for res in dreamer_eval["results"] for r in res["rewards"]])

data_box = [ppo_all, muzero_all, dreamer_all]
bp = ax2.boxplot(data_box,
                  tick_labels=["PPO+GNN\n(baseline)", "MuZero", "Dreamer v9"],
                  patch_artist=True, showmeans=True, widths=0.5,
                  meanprops=dict(marker='D', markerfacecolor='black', markersize=6))
colors_box = ["red", "tab:blue", "tab:purple"]
for patch, c in zip(bp["boxes"], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.35)

# Add mean annotations
for i, (data, c) in enumerate(zip(data_box, colors_box)):
    ax2.text(i+1, np.mean(data) + 0.15, f"mean={np.mean(data):.2f}",
             ha="center", fontsize=9, fontweight="bold", color=c)

ax2.axhline(np.mean(ppo_all), color="red", linewidth=1.5, linestyle="--", alpha=0.4,
            label="PPO mean")
ax2.set_ylabel("Episode Reward", fontsize=10)
ax2.set_title("Reward Distribution (all 500 episodes)\nBaseline = PPO+GNN", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2, axis="y")

# ══════════════════════════════════════════
# Row 2: Improvement ratios vs PPO baseline
# ══════════════════════════════════════════

# ── Panel 2a: Improvement multiplier bar chart ──
ax3 = fig.add_subplot(gs[1, 0])

ppo_mean = max(np.mean(ppo_means), 1e-6)  # avoid div by zero
methods = ["PPO+GNN\n(baseline)", "MuZero\n(v9)", "Dreamer\n(v9)",
           "Traditional\nKMC", "Macro-Edit\nv22 (pred)", "Macro-Edit\nv23 (pred)"]
# For macro-edit, use predicted reward_sum mean (per 4-step segment)
# For v9 models, use mean episode reward (per 200-step episode)
# Scale: v9 rewards are per-episode (200 steps), macro-edit is per-segment (4 steps)
# Both use reward_scale=10.0

values = [
    np.mean(ppo_means),
    np.mean(muzero_means),
    np.mean(dreamer_means),
    v22_eval["traditional_energy"]["reward_sum_mean"],  # KMC teacher segment reward
    v22_eval["predicted_energy"]["reward_sum_mean"],     # v22 predicted segment reward
    v23_eval["predicted_energy"]["reward_sum_mean"],     # v23 predicted segment reward
]
bar_colors = ["red", "tab:blue", "tab:purple", "gray", "tab:orange", "tab:green"]
bar_alphas = [0.4, 0.7, 0.7, 0.5, 0.8, 0.8]

x_pos = np.arange(len(methods))
bars = ax3.bar(x_pos, values, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5)
for i, (bar, val) in enumerate(zip(bars, values)):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.03, f"{val:.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    if i > 0 and ppo_mean > 0.01:
        ratio = val / ppo_mean
        ax3.text(bar.get_x() + bar.get_width()/2, val/2, f"{ratio:.0f}x",
                 ha="center", va="center", fontsize=10, fontweight="bold", color="white")

ax3.set_xticks(x_pos)
ax3.set_xticklabels(methods, fontsize=8)
ax3.set_ylabel("Mean Reward", fontsize=10)
ax3.set_title("Mean Reward: All Methods vs PPO+GNN Baseline\n(v9 = per-episode 200 steps, macro = per-segment 4 steps)", fontsize=11)
ax3.axhline(ppo_mean, color="red", linewidth=2, linestyle="--", alpha=0.5, label=f"PPO baseline = {ppo_mean:.3f}")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2, axis="y")

# ── Panel 2b: Success/positive rate comparison ──
ax4 = fig.add_subplot(gs[1, 1])

ppo_pos_rate = np.mean(ppo_means > 0.01) * 100
muzero_pos_rate = np.mean(muzero_means > 0.01) * 100
dreamer_pos_rate = np.mean(dreamer_means > 0.01) * 100
# For macro-edit: % of segments where predicted reward > 0
v22_pred_pos_rate = np.mean(v22_arrays[2] > 0.01) * 100
v23_pred_pos_rate = np.mean(v23_arrays[2] > 0.01) * 100
kmc_pos_rate = np.mean(v22_arrays[3] > 0.01) * 100

methods_rate = ["PPO+GNN\n(baseline)", "MuZero\n(v9)", "Dreamer\n(v9)",
                "Traditional\nKMC", "Macro-Edit\nv22", "Macro-Edit\nv23"]
rates = [ppo_pos_rate, muzero_pos_rate, dreamer_pos_rate,
         kmc_pos_rate, v22_pred_pos_rate, v23_pred_pos_rate]
bar_colors_rate = ["red", "tab:blue", "tab:purple", "gray", "tab:orange", "tab:green"]

x_pos2 = np.arange(len(methods_rate))
bars2 = ax4.bar(x_pos2, rates, color=bar_colors_rate, alpha=0.7, edgecolor="black", linewidth=0.5)
for bar, rate in zip(bars2, rates):
    ax4.text(bar.get_x() + bar.get_width()/2, rate + 1.5, f"{rate:.0f}%",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

ax4.set_xticks(x_pos2)
ax4.set_xticklabels(methods_rate, fontsize=8)
ax4.set_ylabel("Positive Rate (%)", fontsize=10)
ax4.set_ylim(0, 115)
ax4.set_title("Positive Reward Rate: All Methods vs PPO+GNN\n(% rounds/segments with reward > 0.01)", fontsize=11)
ax4.axhline(ppo_pos_rate, color="red", linewidth=2, linestyle="--", alpha=0.5,
            label=f"PPO baseline = {ppo_pos_rate:.0f}%")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.2, axis="y")

# ══════════════════════════════════════════
# Row 3: Macro-edit training curves
# ══════════════════════════════════════════

# ── Panel 3a: Training Loss ──
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(range(1, len(v22_loss)+1), v22_loss, color="tab:orange", alpha=0.8, label="v22 (80 epochs)")
ax5.plot(range(1, len(v23_loss)+1), v23_loss, color="tab:green", alpha=0.8, label="v23 (120 epochs)")
ax5.set_xlabel("Epoch", fontsize=10)
ax5.set_ylabel("Total Loss", fontsize=10)
ax5.set_title("Macro-Edit Training Loss", fontsize=12)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# ── Panel 3b: VAL tau_log_mae ──
ax6 = fig.add_subplot(gs[2, 1])
v22_val_epochs = np.arange(1, len(v22_val["tau_log_mae"])+1) * 5
v23_val_epochs = np.arange(1, len(v23_val["tau_log_mae"])+1) * 5
ax6.plot(v22_val_epochs, v22_val["tau_log_mae"], "o-", color="tab:orange", markersize=4, label="v22")
ax6.plot(v23_val_epochs, v23_val["tau_log_mae"], "s-", color="tab:green", markersize=4, label="v23")
ax6.set_xlabel("Epoch", fontsize=10)
ax6.set_ylabel("tau log MAE (lower = better)", fontsize=10)
ax6.set_title("VAL: Time Prediction Error\n(PPO+GNN has no time prediction capability)", fontsize=12)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
# Annotate that PPO can't do this
ax6.text(0.5, 0.85, "PPO+GNN: no time prediction",
         transform=ax6.transAxes, ha="center", fontsize=11,
         color="red", fontweight="bold", alpha=0.7,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# ══════════════════════════════════════════
# Row 4: Final metrics & structure prediction
# ══════════════════════════════════════════

# ── Panel 4a: Final Metrics Bar Chart ──
ax7 = fig.add_subplot(gs[3, 0])
metrics_names = ["tau log_MAE\n(lower)", "tau log_corr\n(higher)", "tau scale\n(->1.0)",
                 "reward MAE\n(lower)", "reward corr\n(higher)"]
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
# PPO has no world model metrics, show as 0 or N/A
ppo_finals = [0, 0, 0, 0, 0]

x = np.arange(len(metrics_names))
w = 0.25
ax7.bar(x - w, ppo_finals, w, label="PPO+GNN (N/A)", color="red", alpha=0.25,
        edgecolor="red", linestyle="--", linewidth=1)
bars_v22 = ax7.bar(x, v22_finals, w, label="v22", color="tab:orange", alpha=0.8)
bars_v23 = ax7.bar(x + w, v23_finals, w, label="v23", color="tab:green", alpha=0.8)
ax7.set_xticks(x)
ax7.set_xticklabels(metrics_names, fontsize=8)
ax7.set_title("World Model Metrics: v22 vs v23\n(PPO+GNN has no world model -- shown as 0)", fontsize=11)
ax7.legend(fontsize=9)
for bars in [bars_v22, bars_v23]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax7.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=8)
ax7.grid(True, alpha=0.3, axis="y")

# ── Panel 4b: Structure prediction (change_f1) ──
ax8 = fig.add_subplot(gs[3, 1])
ax8.plot(v22_val_epochs, v22_val["change_f1"], "o-", color="tab:orange", markersize=3,
         alpha=0.6, label="v22 change_f1")
ax8.plot(v23_val_epochs, v23_val["change_f1"], "s-", color="tab:green", markersize=3,
         alpha=0.6, label="v23 change_f1")
ax8.plot(v22_val_epochs, v22_val["proj_change_f1"], "o--", color="tab:orange", markersize=3,
         alpha=0.8, label="v22 proj_change_f1")
ax8.plot(v23_val_epochs, v23_val["proj_change_f1"], "s--", color="tab:green", markersize=3,
         alpha=0.8, label="v23 proj_change_f1")
ax8.set_xlabel("Epoch", fontsize=10)
ax8.set_ylabel("F1 Score (higher = better)", fontsize=10)
ax8.set_title("VAL: Structure Prediction Quality\n(PPO+GNN has no structure prediction)", fontsize=12)
ax8.legend(fontsize=8, ncol=2)
ax8.grid(True, alpha=0.3)
ax8.text(0.5, 0.85, "PPO+GNN: no structure prediction",
         transform=ax8.transAxes, ha="center", fontsize=11,
         color="red", fontweight="bold", alpha=0.7,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

out_path = FIG_DIR / "macro_edit_eval_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close(fig)

# ── Print summary ────────────────────────────────────────

print("\n" + "="*70)
print("Summary: All Methods vs PPO+GNN Baseline")
print("="*70)
print(f"{'Method':<25} {'Mean Reward':>12} {'Std':>8} {'Pos Rate':>10} {'vs PPO':>8}")
print("-"*70)
ppo_m = np.mean(ppo_means)
for name, means in [
    ("PPO+GNN (baseline)", ppo_means),
    ("MuZero (v9)", muzero_means),
    ("Dreamer (v9)", dreamer_means),
]:
    m = np.mean(means)
    s = np.std(means)
    pr = np.mean(means > 0.01) * 100
    ratio = m / ppo_m if ppo_m > 0.01 else float("inf")
    print(f"{name:<25} {m:>12.4f} {s:>8.4f} {pr:>9.0f}% {ratio:>7.1f}x")

print(f"\n{'Method':<25} {'Pred Reward':>12} {'KMC Reward':>12} {'tau log_corr':>12}")
print("-"*70)
for name, ev in [("Macro-Edit v22", v22_eval), ("Macro-Edit v23", v23_eval)]:
    print(f"{name:<25} {ev['predicted_energy']['reward_sum_mean']:>12.4f} "
          f"{ev['traditional_energy']['reward_sum_mean']:>12.4f} "
          f"{ev['tau_expected']['log_corr']:>12.4f}")
print(f"{'PPO+GNN':<25} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
