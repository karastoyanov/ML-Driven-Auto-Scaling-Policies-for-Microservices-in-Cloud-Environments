import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

df_res     = pd.read_csv("results/simulation_detail.csv")
df_metrics = pd.read_csv("results/metrics.csv", index_col=0)

LABELS = {
    "fixed":     "Fixed Replicas",
    "threshold": "Threshold (HPA)",
    "lstm":      "LSTM (proposed)",
    "rf":        "Random Forest (proposed)"
}
COLORS = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8"]

# ── Figure 3: SLA Violation Rate ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
vals   = df_metrics["SLA Violation Rate (%)"]
labels = [LABELS[p] for p in vals.index]
bars   = ax.bar(labels, vals, color=COLORS, edgecolor="white", linewidth=1.2)
ax.set_title("SLA Violation Rate (%)", fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("SLA Violation Rate (%)", fontsize=10)
ax.tick_params(axis="x", rotation=15, labelsize=9)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("results/fig3_sla_violation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig3_sla_violation.png")

# ── Figure 4: Over-Provisioning Rate ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
vals   = df_metrics["Over-Provisioning Rate (%)"]
labels = [LABELS[p] for p in vals.index]
bars   = ax.bar(labels, vals, color=COLORS, edgecolor="white", linewidth=1.2)
ax.set_title("Over-Provisioning Rate (%)", fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("Over-Provisioning Rate (%)", fontsize=10)
ax.tick_params(axis="x", rotation=15, labelsize=9)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("results/fig4_over_provisioning.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig4_over_provisioning.png")

# ── Figure 5: Scaling Actions ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
vals   = df_metrics["Scaling Actions"]
labels = [LABELS[p] for p in vals.index]
bars   = ax.bar(labels, vals, color=COLORS, edgecolor="white", linewidth=1.2)
ax.set_title("Scaling Actions", fontsize=12, fontweight="bold", pad=12)
ax.set_ylabel("Scaling Actions", fontsize=10)
ax.tick_params(axis="x", rotation=15, labelsize=9)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("results/fig5_scaling_actions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig5_scaling_actions.png")

# ── helper: cluster consecutive violation indices ─────────────────────────────
def get_clusters(violations, gap=3):
    if not violations:
        return []
    clusters, cluster = [], [violations[0]]
    for v in violations[1:]:
        if v - cluster[-1] <= gap:
            cluster.append(v)
        else:
            clusters.append(cluster)
            cluster = [v]
    clusters.append(cluster)
    return clusters

# ── Figures 6–9 ───────────────────────────────────────────────────────────────
policies = [
    ("fixed_replicas",     "Fixed Replicas",           "#6baed6", "fig6_ts_fixed.png"),
    ("threshold_replicas", "Threshold (HPA)",           "#fd8d3c", "fig7_ts_threshold.png"),
    ("lstm_replicas",      "LSTM (proposed)",           "#74c476", "fig8_ts_lstm.png"),
    ("rf_replicas",        "Random Forest (proposed)",  "#9e9ac8", "fig9_ts_rf.png"),
]

t = list(range(len(df_res)))

for col, label, color, fname in policies:

    violations = df_res.index[df_res[col] < df_res["replicas_needed"]].tolist()
    clusters   = get_clusters(violations)
    n_clusters = len(clusters)

    if n_clusters == 0:
        # ── Simple figure, no inset row ───────────────────────────────────
        fig, ax_main = plt.subplots(figsize=(11, 4))
        ax_main.plot(t, df_res["replicas_needed"], color="black",
                     lw=1.2, linestyle="--", label="Demand", alpha=0.7)
        ax_main.plot(t, df_res[col], color=color, lw=1.8, label=label, alpha=0.9)
        ax_main.fill_between(t, df_res["replicas_needed"], df_res[col],
                             where=(df_res[col] < df_res["replicas_needed"]),
                             color="red", alpha=0.35, label="SLA Violation")
        ax_main.set_title(f"Replica Count vs. Demand — {label}",
                          fontsize=12, fontweight="bold", pad=12)
        ax_main.set_xlabel("Time (60-second intervals)", fontsize=10)
        ax_main.set_ylabel("Replicas", fontsize=10)
        ax_main.legend(fontsize=9, loc="upper right")
        ax_main.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()

    else:
        # ── Two-row gridspec: main plot top, zoom panels bottom ───────────
        fig = plt.figure(figsize=(11, 7))
        gs  = gridspec.GridSpec(
            2, n_clusters,
            figure=fig,
            height_ratios=[1.6, 1],   # top row taller than bottom
            hspace=0.55,              # vertical gap between rows
            wspace=0.35,              # horizontal gap between zoom panels
            left=0.07, right=0.97,
            top=0.91, bottom=0.09
        )

        # ── Top row: full time-series spanning all columns ────────────────
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(t, df_res["replicas_needed"], color="black",
                     lw=1.2, linestyle="--", label="Demand", alpha=0.7, zorder=2)
        ax_main.plot(t, df_res[col], color=color,
                     lw=1.8, label=label, alpha=0.9, zorder=3)
        ax_main.fill_between(t, df_res["replicas_needed"], df_res[col],
                             where=(df_res[col] < df_res["replicas_needed"]),
                             color="red", alpha=0.35, label="SLA Violation", zorder=1)

        # Light highlight bands on main plot marking violation zones
        for cluster in clusters:
            ax_main.axvspan(cluster[0] - 3, cluster[-1] + 3,
                            color="red", alpha=0.06, zorder=0)

        ax_main.set_title(f"Replica Count vs. Demand — {label}",
                          fontsize=12, fontweight="bold", pad=10)
        ax_main.set_xlabel("Time (60-second intervals)", fontsize=10)
        ax_main.set_ylabel("Replicas", fontsize=10)
        ax_main.legend(fontsize=9, loc="upper right")
        ax_main.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Divider label between rows
        fig.text(0.5, 0.40,
                 "── Violation Detail (zoomed) ──",
                 ha="center", va="center",
                 fontsize=9, color="#cc0000",
                 fontstyle="italic",
                 fontweight="bold")

        # ── Bottom row: one zoom panel per cluster ────────────────────────
        for ci, cluster in enumerate(clusters):
            zoom_lo = max(0, cluster[0] - 6)
            zoom_hi = min(len(df_res) - 1, cluster[-1] + 6)

            ax_z = fig.add_subplot(gs[1, ci])
            ax_z.plot(t, df_res["replicas_needed"], color="black",
                      lw=1.0, linestyle="--", alpha=0.7)
            ax_z.plot(t, df_res[col], color=color, lw=1.5, alpha=0.9)
            ax_z.fill_between(t, df_res["replicas_needed"], df_res[col],
                              where=(df_res[col] < df_res["replicas_needed"]),
                              color="red", alpha=0.5)

            ax_z.set_xlim(zoom_lo, zoom_hi)
            y_slice = pd.concat([
                df_res[col].iloc[zoom_lo:zoom_hi + 1],
                df_res["replicas_needed"].iloc[zoom_lo:zoom_hi + 1]
            ])
            ax_z.set_ylim(y_slice.min() - 0.15, y_slice.max() + 0.2)
            ax_z.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax_z.tick_params(labelsize=8)
            ax_z.set_xlabel("Interval", fontsize=8)
            ax_z.set_ylabel("Replicas", fontsize=8)
            ax_z.set_title(f"Intervals {zoom_lo}–{zoom_hi}",
                           fontsize=9, pad=5, color="#cc0000")
            ax_z.set_facecolor("#fff5f5")
            for spine in ax_z.spines.values():
                spine.set_edgecolor("#cc0000")
                spine.set_linewidth(1.4)

    plt.savefig(f"results/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

print("\nAll figures saved to results/")