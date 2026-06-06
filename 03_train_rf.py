import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os

FEATURES = ["cpu_mean", "cpu_max", "cpu_std", "mem_mean", "hour_sin", "hour_cos"]
TARGETS  = ["cpu_mean", "mem_mean"]

# ── Per-container training ─────────────────────────────────────────────────────
# Same reasoning as 02_train_lstm.py: the concatenated multi-container CSV
# contains ~18% boundary jumps that corrupt lag features across containers.
# Per-container training eliminates these artifacts.
# One RF model per container; predictions are aggregated in 04_simulate.py.

container_ids = pd.read_csv("data/container_ids.csv", header=None)[0].tolist()

os.makedirs("models/rf_per_container", exist_ok=True)
os.makedirs("results", exist_ok=True)

ALL_FEATURES = FEATURES + ["cpu_mean_lag1", "cpu_mean_lag2",
                            "mem_mean_lag1", "mem_mean_lag2"]

all_rmse = []

for cid in container_ids:
    print(f"\n── Training RF for {cid} ──────────────────────────────────────")
    train = pd.read_csv(f"data/containers/{cid}_train.csv")
    test  = pd.read_csv(f"data/containers/{cid}_test.csv").reset_index(drop=True)

    # Build lag features within each container — no cross-container contamination
    for col in ["cpu_mean", "mem_mean"]:
        train[f"{col}_lag1"] = train[col].shift(1).fillna(0)
        train[f"{col}_lag2"] = train[col].shift(2).fillna(0)
        test[f"{col}_lag1"]  = test[col].shift(1).fillna(0)
        test[f"{col}_lag2"]  = test[col].shift(2).fillna(0)

    # Shift target to t+1: predict next interval, not the current one
    X_train = train[ALL_FEATURES].iloc[:-1].reset_index(drop=True)
    y_train = train[TARGETS].shift(-1).dropna().reset_index(drop=True)

    X_test  = test[ALL_FEATURES].iloc[:-1].reset_index(drop=True)
    y_test  = test[TARGETS].shift(-1).dropna().reset_index(drop=True)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    preds    = rf.predict(X_test)
    rmse_cpu = float(np.sqrt(mean_squared_error(y_test["cpu_mean"], preds[:, 0])))
    rmse_mem = float(np.sqrt(mean_squared_error(y_test["mem_mean"], preds[:, 1])))

    all_rmse.append({
        "container" : cid,
        "rf_cpu_rmse": round(rmse_cpu, 4),
        "rf_mem_rmse": round(rmse_mem, 4),
    })

    with open(f"models/rf_per_container/{cid}_rf.pkl", "wb") as f:
        pickle.dump(rf, f)

    print(f"  CPU RMSE: {rmse_cpu:.4f} | MEM RMSE: {rmse_mem:.4f}")

# ── Feature importance (aggregate across containers) ───────────────────────────
# Load last trained RF as representative for feature importance reporting
fi = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
print(f"\n── Feature importances (last container, representative) ──────────────")
print(fi.round(3).to_string())
fi.to_csv("results/rf_feature_importance.csv")

# ── Summary ────────────────────────────────────────────────────────────────────
df_rmse = pd.DataFrame(all_rmse)
print(f"\n── RF RMSE summary ───────────────────────────────────────────────────")
print(df_rmse.to_string(index=False))
print(f"\nMean CPU RMSE: {df_rmse['rf_cpu_rmse'].mean():.4f} "
      f"± {df_rmse['rf_cpu_rmse'].std():.4f}")

df_rmse.to_csv("results/rf_rmse_per_container.csv", index=False)
print("\nAll per-container RF models saved to models/rf_per_container/")