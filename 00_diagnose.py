"""
00_diagnose.py
Run this BEFORE 01_preprocess.py to understand what the dataset actually contains.
Prints a distribution summary and suggests realistic selection thresholds.
"""

import pandas as pd
import numpy as np

CSV_PATH   = "../vm-right-sizing-ppo-algo/data/container_usage.csv"
NROWS      = 2_000_000
TARGET_CPU = 0.70

print("Loading data...")
df = pd.read_csv(
    CSV_PATH,
    header=None,
    names=["container_id", "machine_id", "timestamp",
           "cpu_util", "mem_util", "cpi", "mpki", "net_in", "net_io", "disk_r", "disk_w"],
    usecols=["container_id", "timestamp", "cpu_util", "mem_util"],
    nrows=NROWS,
)
df["container_id"] = df["container_id"].ffill()
df = df.dropna(subset=["timestamp", "cpu_util", "mem_util"]).reset_index(drop=True)
df["timestamp"] = df["timestamp"].astype("int32")
df["cpu_util"]  = df["cpu_util"].astype("float32")
df["mem_util"]  = df["mem_util"].astype("float32")

total_containers = df["container_id"].nunique()
print(f"Total unique containers in first {NROWS:,} rows: {total_containers}")

# ── Per-container stats (raw rows, no resampling yet) ─────────────────────────
print("\nComputing per-container stats...")
stats = df.groupby("container_id").agg(
    n_rows   =("cpu_util", "count"),
    cpu_mean =("cpu_util", "mean"),
    cpu_std  =("cpu_util", "std"),
    cpu_max  =("cpu_util", "max"),
    mem_mean =("mem_util", "mean"),
).reset_index()

stats["cpu_mean_norm"] = stats["cpu_mean"] / 100.0
stats["cpu_std_norm"]  = stats["cpu_std"]  / 100.0

# n_rows IS the interval count — we are not dividing by 6 here because
# the previous version produced approx_intervals=0 for containers with
# fewer than 6 raw rows, causing the empty-slice IndexError.
# The division by 6 (10s→60s resampling) happens inside process_container;
# here we just want a comparable row count for filtering.
stats["approx_intervals"] = stats["n_rows"]

# Approximate how many distinct replica counts this container would need
stats["rep_max_approx"]   = np.ceil(stats["cpu_max"] / 100.0 / TARGET_CPU).astype(int)
stats["rep_range_approx"] = (stats["rep_max_approx"] - 1).clip(lower=0)

# ── Row-count distribution ────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("CONTAINER COUNT BY RAW ROW COUNT (before resampling):")
for thr in [10, 20, 50, 100, 200, 300, 500, 1000]:
    n = (stats["approx_intervals"] >= thr).sum()
    print(f"  >= {thr:>4} rows: {n:>5} containers  ({100*n/total_containers:.1f}%)")

print(f"\n{'─'*60}")
print("CONTAINER COUNT BY MEAN CPU UTILIZATION (normalized):")
for thr in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]:
    n = (stats["cpu_mean_norm"] >= thr).sum()
    print(f"  cpu_mean >= {thr:.2f}: {n:>5} containers  ({100*n/total_containers:.1f}%)")

print(f"\n{'─'*60}")
print("CONTAINER COUNT BY CPU STD (normalized):")
for thr in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
    n = (stats["cpu_std_norm"] >= thr).sum()
    print(f"  cpu_std  >= {thr:.2f}: {n:>5} containers  ({100*n/total_containers:.1f}%)")

print(f"\n{'─'*60}")
print("CONTAINER COUNT BY APPROXIMATE REPLICA RANGE (max_replicas - 1):")
for thr in [1, 2, 3, 4, 5]:
    n = (stats["rep_range_approx"] >= thr).sum()
    print(f"  rep_range >= {thr}: {n:>5} containers  ({100*n/total_containers:.1f}%)")

# ── Combined filter sweep — use raw row count >= 50 as the interval proxy ─────
print(f"\n{'─'*60}")
print("COMBINED FILTER COUNTS (using raw row count >= 50 as interval proxy):")
for min_mean, min_std, min_rep in [
    (0.15, 0.05, 2),
    (0.10, 0.03, 2),
    (0.05, 0.02, 1),
    (0.03, 0.01, 1),
    (0.01, 0.00, 1),
]:
    mask = (
        (stats["approx_intervals"]   >= 50)
        & (stats["cpu_mean_norm"]    >= min_mean)
        & (stats["cpu_std_norm"]     >= min_std)
        & (stats["rep_range_approx"] >= min_rep)
    )
    n = mask.sum()
    print(f"  mean>={min_mean:.2f}, std>={min_std:.2f}, rep_range>={min_rep}: "
          f"{n:>5} containers")

# ── Top 20 most variable containers (with at least 50 raw rows) ───────────────
ELIGIBLE_MIN_ROWS = 50
eligible_mask = stats["approx_intervals"] >= ELIGIBLE_MIN_ROWS

print(f"\n{'─'*60}")
print(f"TOP 20 CONTAINERS BY CPU STD (>= {ELIGIBLE_MIN_ROWS} raw rows):")
top = (stats[eligible_mask]
       .sort_values("cpu_std_norm", ascending=False)
       .head(20))
print(top[["container_id", "approx_intervals",
           "cpu_mean_norm", "cpu_std_norm", "rep_range_approx"]]
      .to_string(index=False))

# ── Percentile distributions ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"PERCENTILE DISTRIBUTION OF CPU_MEAN (normalized, >= {ELIGIBLE_MIN_ROWS} rows):")
vals_mean = stats[eligible_mask]["cpu_mean_norm"]
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:>2}: {np.percentile(vals_mean, p):.4f}")

print(f"\n{'─'*60}")
print(f"PERCENTILE DISTRIBUTION OF CPU_STD (normalized, >= {ELIGIBLE_MIN_ROWS} rows):")
vals_std = stats[eligible_mask]["cpu_std_norm"]
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:>2}: {np.percentile(vals_std, p):.4f}")

print(f"\n{'─'*60}")
print(f"PERCENTILE DISTRIBUTION OF RAW ROW COUNT:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:>2}: {int(np.percentile(stats['approx_intervals'], p)):>6} rows")
    
# ── Spot-check: actually resample top 5 containers ────────────────────────────
print(f"\n{'─'*60}")
print("ACTUAL RESAMPLED INTERVALS FOR TOP 5 BY CPU_STD:")
top5 = top.head(5)["container_id"].tolist()
for cid in top5:
    grp = df[df["container_id"] == cid].copy()
    grp["bucket"] = (grp["timestamp"] // 6).astype(int)
    agg = grp.groupby("bucket").agg(
        cpu_mean=("cpu_util","mean"),
        cpu_max =("cpu_util","max"),
    ).reset_index(drop=True)
    agg["cpu_mean"] = agg["cpu_mean"] / 100.0
    for tgt in [0.30, 0.50, 0.70]:
        rep_max = int(np.ceil(agg["cpu_max"].max() / tgt))
        rep_min = 1
        print(f"  {cid}: {len(agg)} intervals | "
              f"TARGET={tgt} → replicas {rep_min}–{rep_max}")
    print()