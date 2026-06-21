# PQPE - Predictive QoS Preemption Engine

**Predicting 5G network congestion before it happens**, so emergency first-responder traffic can be prioritized proactively instead of reactively.

A machine learning prototype: an LSTM trained on simulated network telemetry to forecast cell congestion 60 seconds ahead of time.

---

## The Problem

During large-scale emergencies - fires, disasters, mass-casualty incidents - civilian phone usage spikes sharply as people call, text, and stream from the area. This overwhelms local cell towers exactly when first responders need bandwidth most, for body-cam video, telemetry, and voice coordination.

Existing solutions (AT&T FirstNet, T-Mobile T-Priority, Verizon FrontLine) are **reactive** - they elevate emergency traffic only *after* congestion has already started. None of them forecast it in advance.

## The Approach

This is framed as a time-series classification problem:

> Given the last 60 seconds of network telemetry, will this cell become congested in the next 60 seconds?

**Key idea:** congestion has leading indicators. Social media activity and emergency call volume in an area tend to rise *before* the network itself saturates - people react to an incident before everyone simultaneously picks up their phone. A model that uses these signals alongside raw network data can predict earlier than one relying on network data alone.

### Why synthetic data

No public dataset combines cell-level RAN telemetry with E911 call volume and social signals at second-level granularity - this data is operator-confidential. So this project generates a synthetic dataset that mimics that structure: normal baseline traffic, with emergency-style events injected on a lead-in → ramp → peak → decay pattern.

This is standard practice for early-stage prototyping where real data isn't accessible. The pipeline is built to retrain on real telemetry without modification once such data exists.

## What the Model Predicts

Given six telemetry signals over a 60-second window - PRB utilization, active devices, downlink throughput, signal quality (SINR), E911 call volume, and a social media activity proxy - the model outputs one number:

**The probability that the cell crosses 85% PRB utilization within the next 60 seconds.**

---

## Architecture

```
generate_data.py  →  telemetry.csv  →  train_lstm.py  →  pqpe_lstm.pt
 (synthetic data)     (raw signals)     (LSTM model)      (trained model)
```

**Data generation** simulates 20,000 seconds of telemetry with 35 injected congestion events, each following four phases: lead-in, ramp, hold, decay. Every timestep is labeled `1` if congestion occurs within the next 60 seconds.

**Model** - a two-layer LSTM processing 60-second sliding windows, with:
- Chronological 70/30 train/test split (no future-data leakage)
- Feature normalization fit on training data only
- Class-weighted loss to handle imbalanced congestion events
- Evaluation via precision, recall, F1, and AUC - not accuracy, which is misleading on imbalanced data

---

## Results

Best epoch (by F1) on the held-out test set:

| Metric | Score |
|---|---|
| Precision | 0.898 |
| Recall | 0.957 |
| F1 | 0.932 |
| AUC | 0.991 |

| | Predicted: no congestion | Predicted: congestion |
|---|---|---|
| **Actual: no congestion** | 3,250 (TN) | 408 (FP) |
| **Actual: congestion** | 52 (FN) | 2,231 (TP) |

Only 52 of 2,283 real congestion events were missed (97.7% recall) - the right tradeoff for an emergency-response system, where a missed event is far costlier than a false alarm.

### Beats a simple baseline

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Always predict no congestion | 0.0 | 0.0 | 0.0 |
| Reactive threshold rule (PRB utilization ≥ 75%) | 0.930 | 0.872 | 0.900 |
| **LSTM (this project)** | **0.898** | **0.957** | **0.932** |

The threshold rule mimics a purely reactive system - flagging congestion only once it's already happening, with no lead time and no external signals. The LSTM's edge comes from higher recall: it catches more events earlier using the leading-indicator signals.

![Prediction plot](assets/prediction_plot.png)

---

## Quick Start

```bash
git clone https://github.com/Jayavarshini-Jayakumaran/pqpe-congestion-predictor.git
cd pqpe-congestion-predictor
pip install -r requirements.txt
```

```bash
python generate_data.py          # regenerate dataset (optional, already included)
python train_lstm.py             # train the model
python baseline_comparison.py    # compare against simple rule-based baselines
python plot_predictions.py       # generate assets/prediction_plot.png
```

---

## Repository Structure

```
.
├── assets/
│   ├── PQPE_Proposal.pdf
│   └── prediction_plot.png
├── generate_data.py
├── train_lstm.py
├── baseline_comparison.py
├── plot_predictions.py
├── telemetry.csv
├── requirements.txt
└── README.md
```

The editable `.docx` proposal source is kept locally in `assets/` but excluded from version control - only the PDF is tracked.

---

## Limitations & Next Steps

This is a prototype, not a production system.

**Current limitations:**
- Trained and evaluated entirely on synthetic data - real-world performance is unverified
- No decision layer yet (probability → throttling action → slice provisioning)
- No live/streaming inference - operates on a static CSV

**Planned:**
- Rule-based decision module mapping prediction confidence to graduated throttling tiers
- Live dashboard with real-time predictions over an animated telemetry feed
- Feature-attribution analysis (SHAP) for operator-facing explainability

---

## Background

Based on a research proposal exploring ML-driven proactive network slicing for emergency communications within an O-RAN architecture, aligned with 3GPP 5G QoS and slicing standards. This repository implements and validates the core predictive component of that proposal.
