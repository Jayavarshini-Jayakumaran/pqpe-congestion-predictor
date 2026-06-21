"""
Generates synthetic telemetry for a single 5G cell site.

The idea: real PRB utilization, throughput, and connected-device counts are
hard to get hold of for a hackathon, so this script fakes a realistic time
series instead. It also injects "emergency events" where social media and
E911 call volume spike a bit before the network itself gets congested -
that lead time is what makes prediction possible instead of just reacting
after the fact.

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd

SEED = 42
N_STEPS = 20000
CONGESTION_THRESHOLD = 85.0
PREDICT_HORIZON = 60
N_EVENTS = 35

rng = np.random.default_rng(SEED)


def simulate_baseline(n_steps):
    t = np.arange(n_steps)

    daily_cycle = 35 + 15 * np.sin(2 * np.pi * t / 3600)
    drift = 5 * np.sin(2 * np.pi * t / (3600 * 6))
    noise = rng.normal(0, 3, n_steps)
    prb_util = np.clip(daily_cycle + drift + noise, 5, 60)

    active_ue = np.clip(prb_util * 4 + rng.normal(0, 15, n_steps), 20, None)
    dl_throughput = np.clip(300 - prb_util * 1.5 + rng.normal(0, 10, n_steps), 10, None)
    sinr = np.clip(20 - prb_util * 0.08 + rng.normal(0, 1.5, n_steps), -5, 30)
    e911_calls = np.clip(rng.poisson(0.5, n_steps), 0, None).astype(float)
    social_velocity = np.clip(10 + rng.normal(0, 3, n_steps), 0, None)

    return dict(
        prb_util=prb_util,
        active_ue=active_ue,
        dl_throughput=dl_throughput,
        sinr=sinr,
        e911_calls=e911_calls,
        social_velocity=social_velocity,
    )


def inject_events(data, n_steps, n_events):
    margin = 600
    starts = rng.choice(np.arange(margin, n_steps - margin), size=n_events, replace=False)
    starts = np.sort(starts)

    for s in starts:
        lead_in = rng.integers(60, 180)
        ramp = rng.integers(30, 90)
        hold = rng.integers(120, 600)
        decay = rng.integers(60, 200)

        # social/E911 signals rise first, before the network shows strain
        li_start, li_end = s - lead_in, s
        li_len = li_end - li_start
        if li_start > 0:
            ramp_signal = np.linspace(0, 1, li_len) ** 1.5
            data["e911_calls"][li_start:li_end] += ramp_signal * rng.uniform(8, 20)
            data["social_velocity"][li_start:li_end] += ramp_signal * rng.uniform(40, 100)

        # network starts to strain
        r_start, r_end = s, min(s + ramp, n_steps)
        r_len = r_end - r_start
        if r_len > 0:
            ramp_curve = np.linspace(0, 1, r_len) ** 1.2
            data["prb_util"][r_start:r_end] += ramp_curve * rng.uniform(35, 50)
            data["active_ue"][r_start:r_end] += ramp_curve * rng.uniform(150, 400)
            data["e911_calls"][r_start:r_end] += ramp_curve * rng.uniform(15, 30)
            data["social_velocity"][r_start:r_end] += ramp_curve * rng.uniform(80, 200)

        # peak congestion
        h_start, h_end = r_end, min(r_end + hold, n_steps)
        h_len = h_end - h_start
        if h_len > 0:
            peak = rng.uniform(35, 50)
            data["prb_util"][h_start:h_end] += peak + rng.normal(0, 3, h_len)
            data["active_ue"][h_start:h_end] += rng.uniform(150, 400) + rng.normal(0, 20, h_len)
            data["e911_calls"][h_start:h_end] += rng.uniform(10, 25, h_len)
            data["social_velocity"][h_start:h_end] += rng.uniform(60, 150, h_len)

        # decay back to normal
        d_start, d_end = h_end, min(h_end + decay, n_steps)
        d_len = d_end - d_start
        if d_len > 0:
            decay_curve = np.linspace(1, 0, d_len) ** 1.5
            data["prb_util"][d_start:d_end] += decay_curve * rng.uniform(20, 35)
            data["social_velocity"][d_start:d_end] += decay_curve * rng.uniform(30, 80)

    data["prb_util"] = np.clip(data["prb_util"], 0, 100)
    data["active_ue"] = np.clip(data["active_ue"], 0, None)
    data["dl_throughput"] = np.clip(300 - data["prb_util"] * 2.2 + rng.normal(0, 8, n_steps), 5, None)
    data["sinr"] = np.clip(20 - data["prb_util"] * 0.12 + rng.normal(0, 1.5, n_steps), -5, 30)

    return data


def label_congestion(prb_util, horizon, threshold):
    n = len(prb_util)
    will_congest = prb_util >= threshold
    label = np.zeros(n, dtype=int)

    future_any = np.zeros(n, dtype=bool)
    count_since_true = horizon + 1
    for i in range(n - 1, -1, -1):
        if will_congest[i]:
            count_since_true = 0
        else:
            count_since_true += 1
        future_any[i] = count_since_true <= horizon

    label[:-1] = future_any[1:]
    return label


def main():
    data = simulate_baseline(N_STEPS)
    data = inject_events(data, N_STEPS, N_EVENTS)
    label = label_congestion(data["prb_util"], PREDICT_HORIZON, CONGESTION_THRESHOLD)

    df = pd.DataFrame(data)
    df["timestep"] = np.arange(N_STEPS)
    df["congestion_now"] = (df["prb_util"] >= CONGESTION_THRESHOLD).astype(int)
    df["congestion_in_window"] = label

    cols = [
        "timestep", "prb_util", "active_ue", "dl_throughput", "sinr",
        "e911_calls", "social_velocity", "congestion_now", "congestion_in_window",
    ]
    df = df[cols]
    df.to_csv("telemetry.csv", index=False)

    print(f"generated {N_STEPS} timesteps -> telemetry.csv")
    print(f"events injected: {N_EVENTS}")
    print(f"positive label rate: {df['congestion_in_window'].mean():.3f}")
    print(f"current congestion rate: {df['congestion_now'].mean():.3f}")


if __name__ == "__main__":
    main()
