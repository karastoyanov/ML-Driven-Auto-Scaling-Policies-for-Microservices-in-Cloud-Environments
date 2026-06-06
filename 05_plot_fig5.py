"""
Fig 5: Focused 3-panel time-series — HPA, LSTM, RF over the 80-interval
window with the highest demand variance. Shows reactive vs proactive
scaling behaviour at a scale readable in a two-column conference paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family"   : "serif",
    "font.size"     : 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
})

container_ids = pd.read_csv("data/container_ids.csv", header=None)[0].tolist()

# Container with highest replica std in test set
best_cid = max(container_ids, key=lambda c: (
    pd.read_csv(f"data/containers/{c}_test.csv")["replicas_needed"].std()
))
sim = pd.read_csv(f"results/{best_cid}_simulation.csv")

# Find 80-interval window with highest demand variance
WINDOW = 80
demand = sim["replicas_needed"].values
best_start = max(
    range(len(demand) - WINDOW),
    key=lambda i: demand[i : i + WINDOW].std()
)
sl = slice(best_start, best_start + WINDOW)
t  = np.arange(WINDOW)
d  = demand[sl]

pol_cols = [
    ("threshold_replicas", "Threshold (HPA)", "#d62728"),
    ("lstm_replicas",      "LSTM",            "#1f77b4"),
    ("rf_replicas",        "Random Forest",   "#2ca02c"),
]

fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)

for ax, (col, label, color) in zip(axes, pol_cols):
    replicas = sim[col].values[sl]
    viol     = replicas < d
    over     = replicas > d * 1.2

    ax.step(t, d, where="post",
            color="black", linewidth=1.1, linestyle="--",
            alpha=0.6, label="Actual demand")
    ax.step(t, replicas, where="post",
            color=color, linewidth=1.5, label=label)

    if viol.any():
        ax.fill_between(t, replicas, d,
                        where=viol, step="post",
                        alpha=0.22, color="red", label="SLA violation")
    if over.any():
        ax.fill_between(t, d, replicas,
                        where=over, step="post",
                        alpha=0.12, color="orange", label="Over-provisioning")

    sla_w  = viol.mean() * 100
    over_w = over.mean() * 100
    acts_w = int((pd.Series(replicas).diff().fillna(0).abs() > 0).sum())

    ax.set_ylabel("Replica count", fontsize=10)
    ax.set_ylim(bottom=0, top=max(d.max(), replicas.max()) + 1)
    ax.legend(loc="upper right",
              title=f"SLA {sla_w:.1f}%  Over {over_w:.1f}%  Actions {acts_w}",
              title_fontsize=7.5, framealpha=0.88)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

axes[-1].set_xlabel(
    f"Interval index (1 interval = 60 s) — "
    f"intervals {best_start}–{best_start + WINDOW - 1} of container {best_cid}",
    fontsize=9
)
fig.suptitle(
    "Reactive vs Proactive Scaling — HPA, LSTM, and Random Forest\n"
    f"(highest-variance 80-interval window, container {best_cid})",
    fontsize=11
)
plt.tight_layout()
plt.savefig("figures/fig5_timeseries_zoom.pdf", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig5_timeseries_zoom.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved fig5_timeseries_zoom  "
      f"(window {best_start}–{best_start + WINDOW - 1}, container {best_cid})")