"""
Compares the LSTM against simple non-ML baselines on the same test split.

This answers the obvious question: does the model actually add value, or
would a basic threshold rule do just as well? The baselines here represent
what a purely reactive system effectively does - flag congestion only once
current utilization is already elevated, with no lead time and no use of
the E911/social signals.

Usage:
    python baseline_comparison.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

TRAIN_FRACTION = 0.7
THRESHOLDS_TO_TRY = [55, 60, 65, 70, 75, 80]


def main():
    df = pd.read_csv("telemetry.csv")
    n = len(df)
    split = int(n * TRAIN_FRACTION)
    test_df = df.iloc[split:].reset_index(drop=True)
    y_true = test_df["congestion_in_window"].values

    print("baseline: always predict no congestion")
    zero_pred = np.zeros(len(y_true))
    print(
        f"  precision={precision_score(y_true, zero_pred, zero_division=0):.3f} "
        f"recall={recall_score(y_true, zero_pred, zero_division=0):.3f} "
        f"f1={f1_score(y_true, zero_pred, zero_division=0):.3f}"
    )

    print("\nbaseline: flag congestion if current prb_util already crosses a threshold")
    best_f1 = -1
    best_thresh = None
    for thresh in THRESHOLDS_TO_TRY:
        pred = (test_df["prb_util"].values >= thresh).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        print(f"  threshold={thresh:3d} precision={p:.3f} recall={r:.3f} f1={f:.3f}")
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh

    print(f"\nbest simple-rule f1: {best_f1:.3f} (threshold={best_thresh})")
    print("compare against the LSTM's saved best f1 from train_lstm.py")


if __name__ == "__main__":
    main()
