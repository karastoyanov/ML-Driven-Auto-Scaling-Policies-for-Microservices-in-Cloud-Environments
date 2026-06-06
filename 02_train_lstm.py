import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
import os

WINDOW   = 10
FEATURES = ["cpu_mean", "cpu_max", "cpu_std", "mem_mean", "hour_sin", "hour_cos"]

# ── Per-container training ─────────────────────────────────────────────────────
# Training on the concatenated multi-container CSV introduces ~18% artificial
# discontinuities at container boundaries (jumps up to 0.96 in cpu_mean).
# LSTM treats these as real workload transitions and cannot converge.
# Solution: train one LSTM per container, then aggregate test-set predictions.
# This is also methodologically cleaner — each model learns one workload profile.

container_ids = pd.read_csv("data/container_ids.csv", header=None)[0].tolist()

os.makedirs("models/lstm_per_container", exist_ok=True)
os.makedirs("results", exist_ok=True)

all_rmse   = []
all_test_preds = {}   # cid -> (lstm_preds_cpu, test_df)

def make_windows(data, window):
    X, y = [], []
    for i in range(window, len(data) - 1):
        X.append(data[i - window : i])   # W past steps as input
        y.append(data[i + 1, 0])         # cpu_mean at t+1 as target
    return np.array(X), np.array(y)

def build_model(window, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

for cid in container_ids:
    print(f"\n── Training LSTM for {cid} ────────────────────────────────────")
    train = pd.read_csv(f"data/containers/{cid}_train.csv")
    test  = pd.read_csv(f"data/containers/{cid}_test.csv").reset_index(drop=True)

    # Fit scaler on this container's train set only
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[FEATURES])
    test_scaled  = scaler.transform(test[FEATURES])

    # Save per-container scaler
    with open(f"models/lstm_per_container/{cid}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    X_train, y_train = make_windows(train_scaled, WINDOW)
    X_test,  y_test  = make_windows(test_scaled,  WINDOW)

    if len(X_train) < 20:
        print(f"  Skipping {cid} — not enough windows ({len(X_train)})")
        continue

    model = build_model(WINDOW, len(FEATURES))

    es = EarlyStopping(patience=20, restore_best_weights=True, monitor="val_loss")
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                           min_lr=1e-6, verbose=0)

    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,      # smaller batch — each container has ~700 train rows
        validation_split=0.1,
        callbacks=[es, lr],
        verbose=0,          # suppress per-epoch output; summary printed below
    )

    model.save(f"models/lstm_per_container/{cid}_lstm.keras")

    # Evaluate on test set
    preds_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform: reconstruct full feature matrix, replace col 0, invert
    dummy = np.zeros((len(preds_scaled), len(FEATURES)))
    dummy[:, 0] = preds_scaled
    preds_cpu = scaler.inverse_transform(dummy)[:, 0].clip(min=0)

    rmse = float(np.sqrt(np.mean((preds_scaled - y_test) ** 2)))
    all_rmse.append({"container": cid, "lstm_rmse_scaled": round(rmse, 4)})

    # Store predictions aligned to test rows WINDOW+1 onwards
    # (first WINDOW rows have no prediction window; last row has no t+1 target)
    all_test_preds[cid] = {
        "preds_cpu": preds_cpu,
        "test_df"  : test,
        "window"   : WINDOW,
    }

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val   = min(history.history["val_loss"])
    print(f"  Epochs: {len(history.history['loss'])} | "
          f"Best epoch: {best_epoch} | "
          f"Best val_loss: {best_val:.4f} | "
          f"Test RMSE (scaled): {rmse:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
df_rmse = pd.DataFrame(all_rmse)
print(f"\n── LSTM RMSE summary ─────────────────────────────────────────────────")
print(df_rmse.to_string(index=False))
print(f"\nMean RMSE: {df_rmse['lstm_rmse_scaled'].mean():.4f} "
      f"± {df_rmse['lstm_rmse_scaled'].std():.4f}")

df_rmse.to_csv("results/lstm_rmse_per_container.csv", index=False)

# Save combined predictions for use by 04_simulate.py
import pickle
with open("models/lstm_per_container/all_test_preds.pkl", "wb") as f:
    pickle.dump(all_test_preds, f)

print("\nAll per-container LSTM models saved to models/lstm_per_container/")