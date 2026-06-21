"""
Plots the model's predicted congestion probability against actual
congestion events over a slice of the test period. This is the single
most useful visual for explaining what the model does at a glance.

Usage:
    python plot_predictions.py
Produces:
    assets/prediction_plot.png
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train_lstm import CongestionLSTM, FEATURES, WINDOW, TRAIN_FRACTION, DEVICE

PLOT_SECONDS = 1500  # how much of the test period to show
OUTPUT_PATH = os.path.join("assets", "prediction_plot.png")


def main():
    checkpoint = torch.load("pqpe_lstm.pt", map_location=DEVICE)
    model = CongestionLSTM(n_features=len(FEATURES)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    means = pd.Series(checkpoint["means"])
    stds = pd.Series(checkpoint["stds"])

    df = pd.read_csv("telemetry.csv")
    split = int(len(df) * TRAIN_FRACTION)
    test_df = df.iloc[split:].reset_index(drop=True)

    X = ((test_df[FEATURES] - means) / stds).values.astype(np.float32)
    y_true = test_df["congestion_in_window"].values

    probs = np.full(len(test_df), np.nan)
    with torch.no_grad():
        for end in range(WINDOW - 1, len(test_df)):
            start = end - WINDOW + 1
            x = torch.from_numpy(X[start:end + 1]).unsqueeze(0).to(DEVICE)
            logit = model(x)
            probs[end] = torch.sigmoid(logit).item()

    start_idx = WINDOW
    end_idx = start_idx + PLOT_SECONDS
    t = test_df["timestep"].values[start_idx:end_idx]
    actual = test_df["prb_util"].values[start_idx:end_idx]
    pred_prob = probs[start_idx:end_idx]
    true_label = y_true[start_idx:end_idx]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(t, actual, color="tab:blue", label="PRB utilization (%)")
    ax1.axhline(85, color="tab:red", linestyle="--", linewidth=1, label="congestion threshold")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("PRB utilization (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(t, pred_prob, color="tab:orange", label="predicted congestion probability")
    ax2.set_ylabel("predicted probability", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.title("Predicted congestion probability vs actual network utilization")
    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
