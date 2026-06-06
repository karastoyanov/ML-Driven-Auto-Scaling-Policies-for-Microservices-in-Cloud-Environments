import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW        = 10
CPU_THRESHOLD = 0.25
SCALE_UP_TH   = 0.35
SCALE_DOWN_TH = 0.15
COOLDOWN      = 5
OVERP_MARGIN  = 0.20

# ── Load data ─────────────────────────────────────────────────────────────────
test  = pd.read_csv("data/test.csv").reset_index(drop=True)
train = pd.read_csv("data/train.csv")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

lstm = load_model("models/lstm_model.keras", compile=False)

FEATURES = ["cpu_mean","cpu_max","cpu_std","mem_mean","hour_sin","hour_cos"]
ALL_RF   = FEATURES + ["cpu_mean_lag1","cpu_mean_lag2","mem_mean_lag1","mem_mean_lag2"]

# ── Scale full datasets ───────────────────────────────────────────────────────
train_scaled = scaler.transform(train[FEATURES])   # ndarray, no warning
test_scaled  = scaler.transform(test[FEATURES])

# ── Pre-build all LSTM windows ────────────────────────────────────────────────
# Seed buffer = last WINDOW rows of train
seed = train_scaled[-WINDOW:]
full_scaled = np.vstack([seed, test_scaled])       # shape: (WINDOW + len(test), 6)

X_lstm = np.array([
    full_scaled[i : i + WINDOW]
    for i in range(len(test))
])                                                 # shape: (len(test), WINDOW, 6)

print(f"Running LSTM batch prediction on {len(X_lstm)} windows...")
lstm_preds_scaled = lstm.predict(X_lstm, batch_size=64, verbose=1).flatten()

# Inverse-scale cpu_mean only (index 0)
dummy = np.zeros((len(lstm_preds_scaled), len(FEATURES)))
dummy[:, 0] = lstm_preds_scaled
lstm_preds_cpu = scaler.inverse_transform(dummy)[:, 0].clip(min=0)

# ── Pre-build RF features ─────────────────────────────────────────────────────
rf_df = test[FEATURES].copy()
rf_df["cpu_mean_lag1"] = rf_df["cpu_mean"].shift(1).fillna(0)
rf_df["cpu_mean_lag2"] = rf_df["cpu_mean"].shift(2).fillna(0)
rf_df["mem_mean_lag1"] = rf_df["mem_mean"].shift(1).fillna(0)
rf_df["mem_mean_lag2"] = rf_df["mem_mean"].shift(2).fillna(0)

print("Running RF batch prediction...")
rf_preds = rf.predict(rf_df[ALL_RF].values)       # ndarray → no feature name warning
rf_preds_cpu = rf_preds[:, 0].clip(min=0)

# ── Fixed replica count ───────────────────────────────────────────────────────
fixed_replicas = int(np.ceil(np.percentile(train["replicas_needed"], 80)))
print(f"Fixed replica baseline: {fixed_replicas}")

# ── Simulation loop ───────────────────────────────────────────────────────────
results        = []
th_replicas    = fixed_replicas
cooldown_counter = 0

for i in range(len(test)):
    row             = test.iloc[i]
    actual_cpu      = row["cpu_mean"]
    replicas_needed = row["replicas_needed"]

    # Policy 1: Fixed
    r_fixed = fixed_replicas

    # Policy 2: Threshold
    if cooldown_counter == 0:
        if actual_cpu > SCALE_UP_TH:
            th_replicas      = th_replicas + 1
            cooldown_counter = COOLDOWN
        elif actual_cpu < SCALE_DOWN_TH and th_replicas > 1:
            th_replicas      = th_replicas - 1
            cooldown_counter = COOLDOWN
    else:
        cooldown_counter = max(0, cooldown_counter - 1)
    r_threshold = th_replicas

    # Policy 3: LSTM
    r_lstm = max(1, int(np.ceil(lstm_preds_cpu[i] / CPU_THRESHOLD)))

    # Policy 4: RF
    r_rf = max(1, int(np.ceil(rf_preds_cpu[i] / CPU_THRESHOLD)))

    results.append({
        "replicas_needed":    replicas_needed,
        "actual_cpu":         actual_cpu,
        "fixed_replicas":     r_fixed,
        "threshold_replicas": r_threshold,
        "lstm_replicas":      r_lstm,
        "rf_replicas":        r_rf,
    })

# ── Metrics ───────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)

policies = ["fixed",    "threshold",    "lstm",    "rf"]
cols     = ["fixed_replicas","threshold_replicas","lstm_replicas","rf_replicas"]

rows = []
for pol, col in zip(policies, cols):
    sla   = (df_res[col] < df_res["replicas_needed"]).mean() * 100
    over  = (df_res[col] > df_res["replicas_needed"] * (1 + OVERP_MARGIN)).mean() * 100
    acts  = int((df_res[col].diff().abs() > 0).sum())
    rows.append({
        "Policy":                    pol,
        "SLA Violation Rate (%)":    round(sla,  2),
        "Over-Provisioning Rate (%)": round(over, 2),
        "Scaling Actions":           acts,
    })

df_metrics = pd.DataFrame(rows).set_index("Policy")

print("\n── Simulation Results ───────────────────────────────────────")
print(df_metrics.to_string())

df_metrics.to_csv("results/metrics.csv")
df_res.to_csv("results/simulation_detail.csv", index=False)
print("\nSaved to results/")