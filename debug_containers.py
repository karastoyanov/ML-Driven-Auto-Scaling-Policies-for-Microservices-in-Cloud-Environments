"""
Quick diagnostic: for each container show the actual demand distribution
and what each model actually predicts, to understand why SLA violations are high.
"""
import pandas as pd
import numpy as np
import pickle
import os

TARGET_CPU = 0.30

container_ids = pd.read_csv("data/container_ids.csv", header=None)[0].tolist()

print(f"{'Container':<12} {'Intervals':>9} {'Demand_mean':>11} {'Demand_std':>10} "
      f"{'Rep_min':>7} {'Rep_max':>7} {'Rep_1_pct':>9} {'Rep_2_pct':>9} {'Rep_3+_pct':>10}")
print("─" * 100)

for cid in container_ids:
    test = pd.read_csv(f"data/containers/{cid}_test.csv")
    sim  = pd.read_csv(f"results/{cid}_simulation.csv")

    demand = sim["replicas_needed"]
    rep1   = (demand == 1).mean() * 100
    rep2   = (demand == 2).mean() * 100
    rep3p  = (demand >= 3).mean() * 100

    print(f"{cid:<12} {len(test):>9} {demand.mean():>11.3f} {demand.std():>10.3f} "
          f"{demand.min():>7} {demand.max():>7} {rep1:>8.1f}% {rep2:>8.1f}% {rep3p:>9.1f}%")

print()
print(f"\n{'Container':<12} {'LSTM_mean':>9} {'RF_mean':>9} {'Fixed':>7} "
      f"{'LSTM_SLA':>9} {'RF_SLA':>9} {'LSTM_Over':>10} {'RF_Over':>9}")
print("─" * 90)

for cid in container_ids:
    sim = pd.read_csv(f"results/{cid}_simulation.csv")
    d   = sim["replicas_needed"]
    fix = sim["fixed_replicas"].iloc[0]

    lstm_sla  = (sim["lstm_replicas"] < d).mean() * 100
    rf_sla    = (sim["rf_replicas"]   < d).mean() * 100
    lstm_over = (sim["lstm_replicas"] > d * 1.2).mean() * 100
    rf_over   = (sim["rf_replicas"]   > d * 1.2).mean() * 100

    print(f"{cid:<12} {sim['lstm_replicas'].mean():>9.2f} {sim['rf_replicas'].mean():>9.2f} "
          f"{fix:>7} {lstm_sla:>8.1f}% {rf_sla:>8.1f}% {lstm_over:>9.1f}% {rf_over:>8.1f}%")

# Show worst container in detail
print("\n\n── WORST CONTAINER DETAIL (highest LSTM SLA) ──────────────────────────")
worst = max(container_ids,
            key=lambda c: (pd.read_csv(f"results/{c}_simulation.csv")["lstm_replicas"]
                           < pd.read_csv(f"results/{c}_simulation.csv")["replicas_needed"])
                          .mean())
sim = pd.read_csv(f"results/{worst}_simulation.csv")
print(f"Container: {worst}")
print(f"Demand distribution:\n{sim['replicas_needed'].value_counts().sort_index()}")
print(f"\nLSTM predictions distribution:\n{sim['lstm_replicas'].value_counts().sort_index()}")
print(f"\nRF predictions distribution:\n{sim['rf_replicas'].value_counts().sort_index()}")
print(f"\nFixed replicas: {sim['fixed_replicas'].iloc[0]}")
print(f"\nSample (first 20 intervals):")
print(sim[["replicas_needed","lstm_replicas","rf_replicas",
           "fixed_replicas","threshold_replicas","actual_cpu"]].head(20).to_string())