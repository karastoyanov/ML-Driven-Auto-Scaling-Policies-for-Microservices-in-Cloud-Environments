import pandas as pd
import numpy as np
import os

# ── Config ─────────────────────────────────────────────────────────────────────
CSV_PATH      = "../vm-right-sizing-ppo-algo/data/container_usage.csv"
NROWS         = 2_000_000

# TARGET_CPU = 0.30: conservative threshold for latency-sensitive microservices.
# The Alibaba dataset cpu_mean peaks at ~0.56 normalized; at 0.70 every container
# stays at 1 replica. At 0.30 containers with cpu_mean > 0.30 require 2+ replicas.
TARGET_CPU    = 0.30

MIN_ROWS      = 300    # minimum raw 10-second samples before resampling
MIN_CPU_MEAN  = 0.05   # at least 5% mean utilization (normalized)
MIN_REP_RANGE = 1      # replica count must vary: max_replicas - min_replicas >= 1
N_CONTAINERS  = 10

# NOTE: MIN_CPU_STD filter is intentionally removed.
# After resampling, cpu_std per bucket is NaN for single-sample buckets and
# gets zeroed by fillna(0), making the column unreliable as a filter criterion.
# Variation is captured instead by MIN_REP_RANGE on the replicas_needed column.

# ── Load ───────────────────────────────────────────────────────────────────────
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

# ── Resample one container's raw rows to 60-second intervals ───────────────────
def process_container(cdf):
    cdf = cdf.sort_values("timestamp").reset_index(drop=True)
    # Raw timestamps are 10-second units; dividing by 6 gives 60-second buckets.
    cdf["bucket"] = (cdf["timestamp"] // 6).astype(int)
    agg = cdf.groupby("bucket").agg(
        cpu_mean=("cpu_util", "mean"),
        cpu_max =("cpu_util", "max"),
        cpu_std =("cpu_util", "std"),
        mem_mean=("mem_util", "mean"),
    ).reset_index(drop=True)

    # Fill NaN std (single-sample buckets) with 0 AFTER aggregation
    agg["cpu_std"] = agg["cpu_std"].fillna(0)
    agg = agg.fillna(0)

    # Normalize from integer percentage [0, 100] to [0.0, 1.0]
    agg["cpu_mean"] = agg["cpu_mean"] / 100.0
    agg["cpu_max"]  = agg["cpu_max"]  / 100.0
    agg["cpu_std"]  = agg["cpu_std"]  / 100.0
    agg["mem_mean"] = agg["mem_mean"] / 100.0

    # Ground-truth replica count from normalized cpu_mean.
    # Using cpu_mean (not cpu_max) gives a stable interval-level demand signal;
    # cpu_max inflates replica counts because a single peak sample within a
    # 60-second bucket would dominate the result.
    agg["replicas_needed"] = (
        np.ceil(agg["cpu_mean"] / TARGET_CPU)
        .clip(lower=1)
        .astype(int)
    )
    return agg

# ── Scan all containers and apply selection criteria ───────────────────────────
print("Scanning containers...")
candidates = []

for cid, grp in df.groupby("container_id"):
    if len(grp) < MIN_ROWS:
        continue

    agg = process_container(grp)

    rep_range    = int(agg["replicas_needed"].max() - agg["replicas_needed"].min())
    cpu_mean_avg = float(agg["cpu_mean"].mean())
    # Use std of replicas_needed as the variability metric — more reliable than
    # cpu_std after resampling, which is zeroed for single-sample buckets.
    rep_std      = float(agg["replicas_needed"].std())

    if cpu_mean_avg >= MIN_CPU_MEAN and rep_range >= MIN_REP_RANGE:
        candidates.append((cid, agg, rep_std))

print(f"Found {len(candidates)} candidate containers.")

if len(candidates) == 0:
    raise RuntimeError(
        "No containers passed the selection criteria.\n"
        "Run 00_diagnose.py to inspect the data distribution."
    )

# Sort by descending replica std — most variable workloads first
candidates.sort(key=lambda x: -x[2])
selected = candidates[:N_CONTAINERS]
print(f"Selected {len(selected)} containers: {[c for c, _, _ in selected]}")

# ── Add time features and write per-container train / test splits ──────────────
os.makedirs("data/containers", exist_ok=True)
container_ids = []

for cid, agg, _ in selected:
    # Cyclical hour-of-day encoding keeps hour 23 and hour 0 adjacent.
    agg["hour"]     = (agg.index % 1440) // 60
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24).astype("float32")
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24).astype("float32")

    # Chronological 70/30 split — no shuffling, preserves temporal order.
    split = int(len(agg) * 0.70)
    train = agg.iloc[:split]
    test  = agg.iloc[split:]

    safe_cid = cid.replace("/", "_")
    train.to_csv(f"data/containers/{safe_cid}_train.csv", index=False)
    test.to_csv( f"data/containers/{safe_cid}_test.csv",  index=False)
    container_ids.append(safe_cid)

    print(f"  {cid}: {len(agg)} intervals | "
          f"cpu_mean={agg.cpu_mean.mean():.3f} | "
          f"rep_std={agg.replicas_needed.std():.3f} | "
          f"replicas: {agg.replicas_needed.min()}–{agg.replicas_needed.max()}")

# Persist the container ID list for downstream scripts
pd.Series(container_ids).to_csv("data/container_ids.csv", index=False, header=False)

# Concatenate per-container splits into aggregate train/test CSVs.
# Training on the aggregate forces models to generalise across workload types.
all_train = pd.concat(
    [pd.read_csv(f"data/containers/{c}_train.csv") for c in container_ids],
    ignore_index=True,
)
all_test = pd.concat(
    [pd.read_csv(f"data/containers/{c}_test.csv") for c in container_ids],
    ignore_index=True,
)
all_train.to_csv("data/train.csv", index=False)
all_test.to_csv( "data/test.csv",  index=False)

print(f"\nDone.")
print(f"  TARGET_CPU : {TARGET_CPU}")
print(f"  Train      : {len(all_train):,} intervals across {len(container_ids)} containers")
print(f"  Test       : {len(all_test):,} intervals across {len(container_ids)} containers")