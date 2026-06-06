import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = "../vm-right-sizing-ppo-algo/data/container_usage.csv"   # adjust to your filename
NROWS         = 500_000
TARGET_CPU    = 0.25
CONTAINER_ID  = "c_10170" # adjust to your chosen container ID (e.g. "c_10170")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    CSV_PATH,
    header=None,
    names=["container_id","machine_id","timestamp",
           "cpu_util","mem_util","cpi","mpki","net_in","net_io","disk_r","disk_w"],
    usecols=["container_id","timestamp","cpu_util","mem_util"],
    nrows=NROWS,
)
df["container_id"] = df["container_id"].ffill()
df = df.dropna(subset=["timestamp","cpu_util","mem_util"]).reset_index(drop=True)
df["timestamp"] = df["timestamp"].astype("int32")
df["cpu_util"]  = df["cpu_util"].astype("float32")
df["mem_util"]  = df["mem_util"].astype("float32")

# ── Filter to chosen container ────────────────────────────────────────────────
df = df[df["container_id"] == CONTAINER_ID].copy()
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Container '{CONTAINER_ID}' — {len(df):,} rows")

# ── Resample to 60-second intervals ───────────────────────────────────────────
df["bucket"] = (df["timestamp"] // 6).astype(int)

agg = df.groupby("bucket").agg(
    cpu_mean = ("cpu_util", "mean"),
    cpu_max  = ("cpu_util", "max"),
    cpu_std  = ("cpu_util", "std"),
    mem_mean = ("mem_util", "mean"),
).reset_index(drop=True).fillna(0)

# ── Normalize: CPU values are 0–100, convert to 0.0–1.0 ─────────────────────
agg["cpu_mean"] = agg["cpu_mean"] / 100.0
agg["cpu_max"]  = agg["cpu_max"]  / 100.0
agg["cpu_std"]  = agg["cpu_std"]  / 100.0
agg["mem_mean"] = agg["mem_mean"] / 100.0

print(f"Aggregated to {len(agg):,} 60-second intervals")

# ── Time features ─────────────────────────────────────────────────────────────
agg["hour"]     = (agg.index % 1440) // 60
agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24).astype("float32")
agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24).astype("float32")

# ── Ground-truth replica count ─────────────────────────────────────────────────
# cpu_mean is now 0.0–1.0, TARGET_CPU=0.65 means one replica handles 65% load
agg["replicas_needed"] = np.ceil(
    agg["cpu_mean"] / TARGET_CPU
).clip(lower=1).astype(int)

# ── Split ──────────────────────────────────────────────────────────────────────
split = int(len(agg) * 0.70)
train = agg.iloc[:split]
test  = agg.iloc[split:]

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv",   index=False)

print(f"\nDone.")
print(f"  Train : {len(train):,} intervals")
print(f"  Test  : {len(test):,} intervals")
print(f"  CPU   : {agg.cpu_mean.min():.3f} – {agg.cpu_mean.max():.3f}")
print(f"  Replicas: {agg.replicas_needed.min()} – {agg.replicas_needed.max()}")