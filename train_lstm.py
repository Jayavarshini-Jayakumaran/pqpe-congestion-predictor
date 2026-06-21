"""
Trains an LSTM to predict 5G cell congestion before it happens.

Looks at the last 60 seconds of telemetry (PRB utilization, connected
devices, throughput, SINR, E911 call volume, social media velocity) and
predicts whether the cell will hit congestion in the next 60 seconds.

Usage:
    pip install torch scikit-learn pandas numpy
    python generate_data.py
    python train_lstm.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

WINDOW = 60
FEATURES = ["prb_util", "active_ue", "dl_throughput", "sinr", "e911_calls", "social_velocity"]
LABEL_COL = "congestion_in_window"
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
HIDDEN_SIZE = 64
TRAIN_FRACTION = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class WindowedSequenceDataset(Dataset):
    def __init__(self, features, labels, window):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.window = window
        self.valid_indices = np.arange(window - 1, len(features))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end = self.valid_indices[idx]
        start = end - self.window + 1
        x = self.features[start:end + 1]
        y = self.labels[end]
        return torch.from_numpy(x), torch.tensor(y)


def load_and_split(csv_path="telemetry.csv"):
    df = pd.read_csv(csv_path)
    n = len(df)
    split_idx = int(n * TRAIN_FRACTION)

    # split by time, not randomly - we only ever care about predicting forward
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    means = train_df[FEATURES].mean()
    stds = train_df[FEATURES].std().replace(0, 1.0)

    train_X = ((train_df[FEATURES] - means) / stds).values
    test_X = ((test_df[FEATURES] - means) / stds).values
    train_y = train_df[LABEL_COL].values
    test_y = test_df[LABEL_COL].values

    return train_X, train_y, test_X, test_y, means, stds


class CongestionLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = self.dropout(h_n[-1])
        return self.head(last_hidden).squeeze(-1)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs >= threshold).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan"),
    }
    cm = confusion_matrix(all_labels, preds)
    return metrics, cm


def main():
    print(f"device: {DEVICE}")
    train_X, train_y, test_X, test_y, means, stds = load_and_split()

    train_ds = WindowedSequenceDataset(train_X, train_y, WINDOW)
    test_ds = WindowedSequenceDataset(test_X, test_y, WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CongestionLSTM(n_features=len(FEATURES)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pos_rate = train_y.mean()
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6), dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"train windows: {len(train_ds)} | test windows: {len(test_ds)} | positive rate: {pos_rate:.3f}")

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        metrics, cm = evaluate(model, test_loader, criterion)
        print(
            f"epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} | "
            f"test_loss={metrics['loss']:.4f} | precision={metrics['precision']:.3f} | "
            f"recall={metrics['recall']:.3f} | f1={metrics['f1']:.3f} | auc={metrics['auc']:.3f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "means": means.to_dict(),
                    "stds": stds.to_dict(),
                    "features": FEATURES,
                    "window": WINDOW,
                },
                "pqpe_lstm.pt",
            )

    print(f"\nbest test f1: {best_f1:.3f}, saved to pqpe_lstm.pt")
    metrics, cm = evaluate(model, test_loader, criterion)
    print("confusion matrix [[tn, fp], [fn, tp]]:")
    print(cm)


if __name__ == "__main__":
    main()
