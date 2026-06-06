import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

WINDOW = 10   # look-back window in intervals
TARGET = "cpu_mean"
FEATURES = ["cpu_mean","cpu_max","cpu_std","mem_mean","hour_sin","hour_cos"]

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[FEATURES])
test_scaled  = scaler.transform(test[FEATURES])

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ── Build sliding windows ─────────────────────────────────────────────────────
def make_windows(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, 0])         # predict cpu_mean (index 0)
    return np.array(X), np.array(y)

X_train, y_train = make_windows(train_scaled, WINDOW)
X_test,  y_test  = make_windows(test_scaled,  WINDOW)

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, len(FEATURES))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.summary()

es = EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

model.save("models/lstm_model.keras")

# ── Quick RMSE ────────────────────────────────────────────────────────────────
preds = model.predict(X_test).flatten()
rmse  = np.sqrt(np.mean((preds - y_test) ** 2))
print(f"LSTM Test RMSE (scaled): {rmse:.4f}")