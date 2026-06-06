import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

FEATURES = ["cpu_mean","cpu_max","cpu_std","mem_mean","hour_sin","hour_cos"]
TARGETS  = ["cpu_mean","mem_mean"]   # dual output

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

# Lag features: shift by 1 and 2 intervals
for col in ["cpu_mean","mem_mean"]:
    train[f"{col}_lag1"] = train[col].shift(1).fillna(0)
    train[f"{col}_lag2"] = train[col].shift(2).fillna(0)
    test[f"{col}_lag1"]  = test[col].shift(1).fillna(0)
    test[f"{col}_lag2"]  = test[col].shift(2).fillna(0)

ALL_FEATURES = FEATURES + ["cpu_mean_lag1","cpu_mean_lag2",
                            "mem_mean_lag1","mem_mean_lag2"]

X_train = train[ALL_FEATURES]
y_train = train[TARGETS]
X_test  = test[ALL_FEATURES]
y_test  = test[TARGETS]

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
rmse_cpu = np.sqrt(mean_squared_error(y_test["cpu_mean"], preds[:,0]))
rmse_mem = np.sqrt(mean_squared_error(y_test["mem_mean"], preds[:,1]))
print(f"RF CPU RMSE: {rmse_cpu:.4f}")
print(f"RF MEM RMSE: {rmse_mem:.4f}")

with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)