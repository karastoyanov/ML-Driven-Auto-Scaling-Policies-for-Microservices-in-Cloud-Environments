import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW        = 10
TARGET_CPU    = 0.30

# HPA cooldown: 3 intervals = 3 minutes
COOLDOWN      = 3

# ML cooldown: minimum intervals between consecutive ML scaling actions.
# Prevents thrashing from prediction fluctuations in high-RMSE containers.
ML_COOLDOWN   = 2

OVERP_MARGIN  = 0.20

# Safety margin applied to ML predictions before replica count conversion.
# Compensates for systematic under-prediction — proactive scaling should
# provision slightly ahead of predicted demand to absorb prediction error.
SAFETY        = 1.15

FEATURES = ["cpu_mean", "cpu_max", "cpu_std", "mem_mean", "hour_sin", "hour_cos"]
ALL_RF   = FEATURES + ["cpu_mean_lag1", "cpu_mean_lag2",
                        "mem_mean_lag1", "mem_mean_lag2"]

container_ids = pd.read_csv("data/container_ids.csv", header=None)[0].tolist()
os.makedirs("results", exist_ok=True)

all_metrics = []

for cid in container_ids:
    print(f"\n── Simulating: {cid} ─────────────────────────────────────────────")

    train = pd.read_csv(f"data/containers/{cid}_train.csv")
    test  = pd.read_csv(f"data/containers/{cid}_test.csv").reset_index(drop=True)

    # ── Load per-container models ────────────────────────────────────────────
    scaler = pickle.load(open(f"models/lstm_per_container/{cid}_scaler.pkl", "rb"))
    lstm   = load_model(f"models/lstm_per_container/{cid}_lstm.keras", compile=False)
    rf     = pickle.load(open(f"models/rf_per_container/{cid}_rf.pkl", "rb"))

    # ── LSTM batch predictions ───────────────────────────────────────────────
    train_scaled = scaler.transform(train[FEATURES])
    test_scaled  = scaler.transform(test[FEATURES])
    seed         = train_scaled[-WINDOW:]
    full_scaled  = np.vstack([seed, test_scaled])
    X_lstm       = np.array([full_scaled[i : i + WINDOW] for i in range(len(test))])

    lstm_preds_scaled = lstm.predict(X_lstm, batch_size=64, verbose=0).flatten()
    dummy = np.zeros((len(lstm_preds_scaled), len(FEATURES)))
    dummy[:, 0] = lstm_preds_scaled
    lstm_preds_cpu = scaler.inverse_transform(dummy)[:, 0].clip(min=0)

    # ── RF batch predictions ─────────────────────────────────────────────────
    rf_df = test[FEATURES].copy()
    for col in ["cpu_mean", "mem_mean"]:
        rf_df[f"{col}_lag1"] = rf_df[col].shift(1).fillna(0)
        rf_df[f"{col}_lag2"] = rf_df[col].shift(2).fillna(0)
    rf_preds     = rf.predict(rf_df[ALL_RF].values)
    rf_preds_cpu = rf_preds[:, 0].clip(min=0)

    # Fixed baseline: median of training demand.
    # The 80th percentile produced fixed_replicas=3 for containers where
    # test demand is 1 in 88-93% of intervals — an unrealistically conservative
    # strategy. The median represents the typical operating point.
    fixed_replicas = max(1, int(np.round(np.percentile(train["replicas_needed"], 50))))

    # ── Simulation loop ───────────────────────────────────────────────────────
    results       = []
    th_replicas   = fixed_replicas
    r_threshold   = fixed_replicas
    r_lstm        = fixed_replicas
    r_rf          = fixed_replicas
    cooldown_ctr  = 0
    lstm_cooldown = 0
    rf_cooldown   = 0

    for i in range(len(test)):
        actual_cpu = float(test.iloc[i]["cpu_mean"])
        demand     = int(test.iloc[i]["replicas_needed"])

        # Policy 1 — Fixed (median training demand)
        r_fixed = fixed_replicas

        # Policy 2 — HPA (Kubernetes-style direct utilization formula)
        # desired = ceil(actual_cpu / target_cpu)
        # Computes required replicas directly from observed utilization,
        # matching the Kubernetes HPA controller:
        #   desiredReplicas = ceil(currentMetricValue / desiredMetricValue)
        # The previous formulation ceil(current * actual/target) compounded
        # the current replica count, causing upward drift under bursty loads.
        if cooldown_ctr == 0:
            desired = max(1, int(np.ceil(actual_cpu / TARGET_CPU)))
            if desired != th_replicas:
                th_replicas  = desired
                cooldown_ctr = COOLDOWN
        else:
            cooldown_ctr = max(0, cooldown_ctr - 1)
        r_threshold = th_replicas

        # Policy 3 — LSTM (proactive, with safety margin and ML cooldown)
        if lstm_cooldown == 0:
            r_lstm_new = max(1, int(np.ceil(lstm_preds_cpu[i] * SAFETY / TARGET_CPU)))
            if r_lstm_new != r_lstm:
                r_lstm        = r_lstm_new
                lstm_cooldown = ML_COOLDOWN
        else:
            lstm_cooldown = max(0, lstm_cooldown - 1)

        # Policy 4 — RF (proactive, with safety margin and ML cooldown)
        if rf_cooldown == 0:
            r_rf_new = max(1, int(np.ceil(rf_preds_cpu[i] * SAFETY / TARGET_CPU)))
            if r_rf_new != r_rf:
                r_rf        = r_rf_new
                rf_cooldown = ML_COOLDOWN
        else:
            rf_cooldown = max(0, rf_cooldown - 1)

        results.append({
            "replicas_needed"   : demand,
            "actual_cpu"        : actual_cpu,
            "fixed_replicas"    : r_fixed,
            "threshold_replicas": r_threshold,
            "lstm_replicas"     : r_lstm,
            "rf_replicas"       : r_rf,
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"results/{cid}_simulation.csv", index=False)

    # ── Per-container metrics ─────────────────────────────────────────────────
    policies = ["fixed", "threshold", "lstm", "rf"]
    cols     = ["fixed_replicas", "threshold_replicas", "lstm_replicas", "rf_replicas"]

    for pol, col in zip(policies, cols):
        sla  = (df_res[col] < df_res["replicas_needed"]).mean() * 100
        over = (df_res[col] > df_res["replicas_needed"] * (1 + OVERP_MARGIN)).mean() * 100
        acts = int((df_res[col].diff().fillna(0).abs() > 0).sum())
        all_metrics.append({
            "container"                 : cid,
            "policy"                    : pol,
            "SLA Violation Rate (%)"    : round(sla,  2),
            "Over-Provisioning Rate (%)": round(over, 2),
            "Scaling Actions"           : acts,
            "n_test_intervals"          : len(test),
        })

    df_ctr = pd.DataFrame([m for m in all_metrics if m["container"] == cid])
    print(df_ctr[["policy", "SLA Violation Rate (%)",
                  "Over-Provisioning Rate (%)", "Scaling Actions"]]
          .to_string(index=False))

# ── Aggregate: mean ± std across containers ───────────────────────────────────
df_all = pd.DataFrame(all_metrics)
df_all.to_csv("results/all_metrics.csv", index=False)

summary = (df_all
           .groupby("policy")
           .agg(
               SLA_mean =("SLA Violation Rate (%)",     "mean"),
               SLA_std  =("SLA Violation Rate (%)",     "std"),
               Over_mean=("Over-Provisioning Rate (%)", "mean"),
               Over_std =("Over-Provisioning Rate (%)", "std"),
               Acts_mean=("Scaling Actions",            "mean"),
               Acts_std =("Scaling Actions",            "std"),
           )
           .round(2))

order   = ["fixed", "threshold", "lstm", "rf"]
summary = summary.reindex([p for p in order if p in summary.index])

print("\n" + "─" * 70)
print("SUMMARY — mean ± std across all containers")
print("─" * 70)
for pol, row in summary.iterrows():
    print(f"\n  {pol.upper()}")
    print(f"    SLA Violation    : {row.SLA_mean:.2f}% ± {row.SLA_std:.2f}%")
    print(f"    Over-Provisioning: {row.Over_mean:.2f}% ± {row.Over_std:.2f}%")
    print(f"    Scaling Actions  : {row.Acts_mean:.1f} ± {row.Acts_std:.1f}")

summary.to_csv("results/summary_metrics.csv")
print("\nResults saved to results/")