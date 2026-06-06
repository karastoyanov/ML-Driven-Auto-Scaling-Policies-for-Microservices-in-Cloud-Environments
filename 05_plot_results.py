import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# ── Shared style ───────────────────────────────────────────────────────────────
POLICY_LABELS = {
    "fixed"    : "Fixed Replicas",
    "threshold": "Threshold (HPA)",
    "lstm"     : "LSTM",
    "rf"       : "Random Forest",
}
COLORS = {
    "Fixed Replicas" : "#7f7f7f",
    "Threshold (HPA)": "#d62728",
    "LSTM"           : "#1f77b4",
    "Random Forest"  : "#2ca02c",
}
ORDER = ["Fixed Replicas", "Threshold (HPA)", "LSTM", "Random Forest"]

plt.rcParams.update({
    "font.family"  : "serif",
    "font.size"    : 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi"   : 150,
})

summary_raw = pd.read_csv("results/summary_metrics.csv").set_index("policy")
summary_raw.index = [POLICY_LABELS.get(p, p) for p in summary_raw.index]
summary = summary_raw.reindex(ORDER)

all_metrics = pd.read_csv("results/all_metrics.csv")
all_metrics["policy"] = all_metrics["policy"].map(POLICY_LABELS)
container_ids = pd.read_csv("data/container_ids.csv", header=None)[0].tolist()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Three-metric grouped bar chart (SLA / Over-provisioning / Actions)
# One chart, three metric groups side by side — gives a complete picture at a
# glance and is the most compact way to present the summary table visually.
# ══════════════════════════════════════════════════════════════════════════════
metrics_groups = [
    ("SLA_mean",   "SLA_std",    "SLA Violation (%)"),
    ("Over_mean",  "Over_std",   "Over-Provisioning (%)"),
    ("Acts_mean",  "Acts_std",   "Scaling Actions"),
]
n_groups  = len(metrics_groups)
n_policies= len(ORDER)
bar_w     = 0.18
x         = np.arange(n_groups)

fig, ax = plt.subplots(figsize=(8, 4.5))
for j, label in enumerate(ORDER):
    offsets = x + (j - n_policies / 2 + 0.5) * bar_w
    means   = [summary.loc[label, mc] for mc, _, _ in metrics_groups]
    stds    = [summary.loc[label, sc] for _, sc, _ in metrics_groups]
    ax.bar(offsets, means, bar_w, yerr=stds,
           label=label, color=COLORS[label],
           edgecolor="black", linewidth=0.6, capsize=3)

ax.set_xticks(x)
ax.set_xticklabels([m for _, _, m in metrics_groups], fontsize=10)
ax.set_ylabel("Mean value (± std, n=10 containers)", fontsize=10)
ax.set_title("Auto-Scaling Policy Comparison — All Metrics", fontsize=11)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig("figures/fig1_grouped_bars.pdf", dpi=150)
plt.savefig("figures/fig1_grouped_bars.png", dpi=150)
plt.close()
print("Saved fig1_grouped_bars")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — SLA vs Over-Provisioning scatter (trade-off space)
# Each point = one container × one policy. Shows that no single policy
# dominates — the key qualitative finding of the paper.
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 5))
markers = {"Fixed Replicas": "s", "Threshold (HPA)": "^",
           "LSTM": "o", "Random Forest": "D"}

for pol in ORDER:
    sub = all_metrics[all_metrics["policy"] == pol]
    ax.scatter(sub["Over-Provisioning Rate (%)"],
               sub["SLA Violation Rate (%)"],
               color=COLORS[pol], marker=markers[pol],
               s=55, alpha=0.75, edgecolors="black", linewidths=0.4,
               label=pol)
    # Mark the mean with a larger filled marker
    ax.scatter(sub["Over-Provisioning Rate (%)"].mean(),
               sub["SLA Violation Rate (%)"].mean(),
               color=COLORS[pol], marker=markers[pol],
               s=180, edgecolors="black", linewidths=1.2, zorder=5)

ax.set_xlabel("Over-Provisioning Rate (%)", fontsize=10)
ax.set_ylabel("SLA Violation Rate (%)", fontsize=10)
ax.set_title("SLA Violation vs Over-Provisioning Trade-off\n"
             "(small markers = individual containers, large = mean)", fontsize=10)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(linestyle="--", alpha=0.4)
ax.set_xlim(left=-2)
ax.set_ylim(bottom=-1)
plt.tight_layout()
plt.savefig("figures/fig2_tradeoff_scatter.pdf", dpi=150)
plt.savefig("figures/fig2_tradeoff_scatter.png", dpi=150)
plt.close()
print("Saved fig2_tradeoff_scatter")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Per-container SLA heatmap (rows = containers, cols = policies)
# Shows which policy fails on which workload type — supports the discussion
# that no single policy is universally best.
# ══════════════════════════════════════════════════════════════════════════════
pivot_sla = all_metrics.pivot(index="container", columns="policy",
                               values="SLA Violation Rate (%)")
pivot_sla = pivot_sla.reindex(columns=ORDER)

fig, ax = plt.subplots(figsize=(7, max(3.5, len(pivot_sla) * 0.55)))
im = ax.imshow(pivot_sla.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=50)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, fontsize=9, rotation=15, ha="right")
ax.set_yticks(range(len(pivot_sla.index)))
ax.set_yticklabels(pivot_sla.index, fontsize=8)
for i in range(len(pivot_sla.index)):
    for j in range(len(ORDER)):
        val = pivot_sla.values[i, j]
        txt_color = "white" if val > 35 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8, color=txt_color)
plt.colorbar(im, ax=ax, label="SLA Violation Rate (%)")
ax.set_title("SLA Violation Rate (%) — Per Container and Policy", fontsize=11)
plt.tight_layout()
plt.savefig("figures/fig3_heatmap_sla.pdf", dpi=150)
plt.savefig("figures/fig3_heatmap_sla.png", dpi=150)
plt.close()
print("Saved fig3_heatmap_sla")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Per-container Over-Provisioning heatmap
# Companion to Fig 3 — same structure, other metric.
# ══════════════════════════════════════════════════════════════════════════════
pivot_over = all_metrics.pivot(index="container", columns="policy",
                                values="Over-Provisioning Rate (%)")
pivot_over = pivot_over.reindex(columns=ORDER)

fig, ax = plt.subplots(figsize=(7, max(3.5, len(pivot_over) * 0.55)))
im = ax.imshow(pivot_over.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, fontsize=9, rotation=15, ha="right")
ax.set_yticks(range(len(pivot_over.index)))
ax.set_yticklabels(pivot_over.index, fontsize=8)
for i in range(len(pivot_over.index)):
    for j in range(len(ORDER)):
        val = pivot_over.values[i, j]
        txt_color = "white" if val > 70 else "black"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8, color=txt_color)
plt.colorbar(im, ax=ax, label="Over-Provisioning Rate (%)")
ax.set_title("Over-Provisioning Rate (%) — Per Container and Policy", fontsize=11)
plt.tight_layout()
plt.savefig("figures/fig4_heatmap_over.pdf", dpi=150)
plt.savefig("figures/fig4_heatmap_over.png", dpi=150)
plt.close()
print("Saved fig4_heatmap_over")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Time-series for the most variable container (4 subplots)
# One subplot per policy, shared x-axis. Red shading = SLA violation intervals.
# Shows qualitative behaviour differences between reactive and proactive scaling.
# ══════════════════════════════════════════════════════════════════════════════
# Pick the container with highest replica std (most interesting dynamics)
best_cid = max(container_ids, key=lambda c: (
    pd.read_csv(f"data/containers/{c}_test.csv")["replicas_needed"].std()
))
sim = pd.read_csv(f"results/{best_cid}_simulation.csv")
t   = np.arange(len(sim))

pol_cols = [
    ("fixed_replicas",     "Fixed Replicas",   "#7f7f7f"),
    ("threshold_replicas", "Threshold (HPA)",  "#d62728"),
    ("lstm_replicas",      "LSTM",             "#1f77b4"),
    ("rf_replicas",        "Random Forest",    "#2ca02c"),
]

fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
for ax, (col, label, color) in zip(axes, pol_cols):
    ax.step(t, sim["replicas_needed"], where="post",
            color="black", linewidth=1, linestyle="--", alpha=0.55, label="Demand")
    ax.step(t, sim[col], where="post",
            color=color, linewidth=1.3, label=label)
    viol = sim[col] < sim["replicas_needed"]
    if viol.any():
        ax.fill_between(t, sim[col], sim["replicas_needed"],
                        where=viol, step="post",
                        alpha=0.25, color="red", label="SLA violation")
    sla_pct  = viol.mean() * 100
    over_pct = (sim[col] > sim["replicas_needed"] * 1.2).mean() * 100
    acts     = int((sim[col].diff().fillna(0).abs() > 0).sum())
    ax.set_ylabel("Replicas", fontsize=9)
    ax.legend(fontsize=8, loc="upper right",
              title=f"SLA {sla_pct:.1f}%  Over {over_pct:.1f}%  Acts {acts}",
              title_fontsize=7.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(bottom=0)

axes[-1].set_xlabel("Interval index (1 interval = 60 s)", fontsize=10)
fig.suptitle(f"Replica Count vs Demand — Container {best_cid} "
             f"(highest workload variability)", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f"figures/fig5_timeseries_{best_cid}.pdf", dpi=150, bbox_inches="tight")
plt.savefig(f"figures/fig5_timeseries_{best_cid}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved fig5_timeseries_{best_cid}")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Scaling Actions box plot (distribution across containers)
# Bar charts show the mean but hide the variance. Box plots reveal that
# RF and HPA have very different action distributions across containers.
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4))
data_for_box = [all_metrics[all_metrics["policy"] == p]["Scaling Actions"].values
                for p in ORDER]
bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 4, "alpha": 0.5})
for patch, label in zip(bp["boxes"], ORDER):
    patch.set_facecolor(COLORS[label])
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(ORDER) + 1))
ax.set_xticklabels(ORDER, fontsize=10)
ax.set_ylabel("Scaling Actions (per container test set)", fontsize=10)
ax.set_title("Scaling Action Count Distribution Across Containers", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/fig6_actions_boxplot.pdf", dpi=150)
plt.savefig("figures/fig6_actions_boxplot.png", dpi=150)
plt.close()
print("Saved fig6_actions_boxplot")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — RF feature importance (horizontal bar chart)
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists("results/rf_feature_importance.csv"):
    fi = pd.read_csv("results/rf_feature_importance.csv",
                    index_col=0, header=0).squeeze("columns")
    fi.index = fi.index.astype(str)
    fi = fi.astype(float).sort_values(ascending=True)
    fi = fi.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(fi.index, fi.values,
                   color="#1f77b4", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=10)
    ax.set_title("Random Forest — Feature Importances", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, fi.values):
        if val > 0.01:
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/fig7_rf_feature_importance.pdf", dpi=150)
    plt.savefig("figures/fig7_rf_feature_importance.png", dpi=150)
    plt.close()
    print("Saved fig7_rf_feature_importance")

print("\nAll figures saved to figures/")
print("PDF versions suitable for inclusion in the paper.")
print("PNG versions for preview.")