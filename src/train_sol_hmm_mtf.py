# How to Execute this file in DOCKER
# docker compose run --rm --entrypoint python3 freqtrade_ui   /freqtrade/user_data/train_sol_hmm_mtf.py


#  After running two output files will get generated so move them to model fodler
# Move the newly compiled 1d and 4h bundles into your models directory
# mv user_data/hmm_bundle_SOL_USDT_1d.joblib user_data/models/hmm_bundle_SOL_USDT_1d.joblib
# mv user_data/hmm_bundle_SOL_USDT_4h.joblib user_data/models/hmm_bundle_SOL_USDT_4h.joblib
#

# divyahari@divyaharihost:~/ft_userdata$ docker compose run --rm --entrypoint python3 freqtrade_ui   /freqtrade/user_data/train_sol_hmm_mtf.py
#
# ================ TRAINING SOL_USDT [1d] ================
# Total Bars: 1707 | Train: 1365 | OOS Test: 342 (From 2025-10-28 00:00:00+00:00)
# Converged: True (82 iterations)
#   State 0 [BEAR]:
#     Expected Duration: 18.5 bars (p_ii=0.9459)
#     Trend ATR (mu +/- sigma):   -1.33 +/- 0.83
#     Impulse ATR (mu +/- sigma): -0.39 +/- 0.96
#     Vol Expansion (NATR log):   +0.01 +/- 0.37
#     Vol Momentum (EMA log):     -0.07 +/- 0.51
#   State 1 [CHOP]:
#     Expected Duration: 10.9 bars (p_ii=0.9083)
#     Trend ATR (mu +/- sigma):   +0.28 +/- 0.67
#     Impulse ATR (mu +/- sigma): +0.07 +/- 0.94
#     Vol Expansion (NATR log):   -0.08 +/- 0.24
#     Vol Momentum (EMA log):     -0.13 +/- 0.34
#   State 2 [BULL]:
#     Expected Duration: 13.7 bars (p_ii=0.9270)
#     Trend ATR (mu +/- sigma):   +2.09 +/- 0.93
#     Impulse ATR (mu +/- sigma): +0.78 +/- 1.23
#     Vol Expansion (NATR log):   -0.16 +/- 0.33
#     Vol Momentum (EMA log):     +0.15 +/- 0.35
#
# Causal Out-of-Sample State Distribution (0=BEAR, 1=CHOP, 2=BULL):
# 0    0.389
# 1    0.357
# 2    0.254
# Name: proportion, dtype: float64
# Exported bundle -> /freqtrade/user_data/hmm_bundle_SOL_USDT_1d.joblib
#
# ================ TRAINING SOL_USDT [4h] ================
# Total Bars: 8055 | Train: 6444 | OOS Test: 1611 (From 2026-01-11 04:00:00+00:00)
# Converged: True (85 iterations)
#   State 0 [BEAR]:
#     Expected Duration: 23.7 bars (p_ii=0.9578)
#     Trend ATR (mu +/- sigma):   -1.72 +/- 1.06
#     Impulse ATR (mu +/- sigma): -0.69 +/- 1.33
#     Vol Expansion (NATR log):   -0.04 +/- 0.36
#     Vol Momentum (EMA log):     -0.14 +/- 0.45
#   State 1 [CHOP]:
#     Expected Duration: 13.0 bars (p_ii=0.9232)
#     Trend ATR (mu +/- sigma):   +0.17 +/- 0.84
#     Impulse ATR (mu +/- sigma): +0.07 +/- 1.35
#     Vol Expansion (NATR log):   -0.13 +/- 0.26
#     Vol Momentum (EMA log):     -0.14 +/- 0.31
#   State 2 [BULL]:
#     Expected Duration: 22.1 bars (p_ii=0.9547)
#     Trend ATR (mu +/- sigma):   +2.17 +/- 1.14
#     Impulse ATR (mu +/- sigma): +0.93 +/- 1.52
#     Vol Expansion (NATR log):   +0.02 +/- 0.35
#     Vol Momentum (EMA log):     +0.12 +/- 0.39
#
# Causal Out-of-Sample State Distribution (0=BEAR, 1=CHOP, 2=BULL):
# 0    0.353
# 1    0.433
# 2    0.214
# Name: proportion, dtype: float64
# Exported bundle -> /freqtrade/user_data/hmm_bundle_SOL_USDT_4h.joblib
# divyahari@divyaharihost:~/ft_userdata$ ^C
# divyahari@divyaharihost:~/ft_userdata$


# -*- coding: utf-8 -*-
import os
import sys
import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

# =====================================================================
# 1. Timeframe-Specific Stationary Configuration (Group C: SOL_USDT)
# =====================================================================
PAIR = "SOL_USDT"
DATA_DIR = "/freqtrade/user_data/data/bybit"
OUTPUT_DIR = "/freqtrade/user_data/models"

TIMEFRAME_CONFIGS = {
    "1d": {
        "atr_len": 14,
        "sma_len": 20,          # 20-day structural trend baseline
        "impulse_len": 3,       # 3-day directional velocity
        "vol_base_len": 90,     # 90-day NATR volatility baseline
        "v_fast_len": 5,        # 5-day volume EMA
        "v_slow_len": 30,       # 30-day volume baseline
        "min_covar": 1e-4,
    },
    "4h": {
        "atr_len": 14,
        "sma_len": 30,          # 30 bars = 5-day trend baseline on 4h
        "impulse_len": 6,       # 6 bars = 24-hour directional velocity
        "vol_base_len": 180,    # 180 bars = 30-day NATR baseline on 4h
        "v_fast_len": 12,       # 12 bars = 2-day volume EMA
        "v_slow_len": 90,       # 90 bars = 15-day volume baseline
        "min_covar": 1e-4,
    }
}

FEATURE_COLS = ["trend_atr", "impulse_atr", "vol_expansion", "vol_momentum"]


# =====================================================================
# 2. Pure Stationary Feature Engineering (Scale-Invariant NATR Fix)
# =====================================================================
def engineer_stationary_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(cfg["atr_len"]).mean()
    sma = df["close"].rolling(cfg["sma_len"]).mean()

    # F1: Trend distance in units of ATR (Dollar units cancel out)
    df["trend_atr"] = (df["close"] - sma) / (atr + 1e-8)

    # F2: Short-term directional velocity in units of ATR (Fixes reversal lag)
    df["impulse_atr"] = (df["close"] - df["close"].shift(cfg["impulse_len"])) / (atr + 1e-8)

    # F3: Volatility Expansion using Normalized Percentage ATR (NATR)
    natr = atr / (df["close"] + 1e-8)
    natr_baseline = natr.shift(1).rolling(cfg["vol_base_len"]).mean()
    df["vol_expansion"] = np.log((natr + 1e-8) / (natr_baseline + 1e-8))

    # F4: Volume Momentum using EMA
    vol_fast = df["volume"].ewm(span=cfg["v_fast_len"], adjust=False).mean()
    vol_slow = df["volume"].shift(1).rolling(cfg["v_slow_len"]).mean()
    df["vol_momentum"] = np.log((vol_fast + 1e-8) / (vol_slow + 1e-8))

    return df


# =====================================================================
# 3. Deterministic K-Means Pre-Seeded HMM Trainer + Canonical Sorting
# =====================================================================
def fit_and_canonicalize_hmm(X: np.ndarray, min_covar: float = 1e-4) -> GaussianHMM:
    n_states = 3
    n_features = X.shape[1]

    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10).fit(X)
    labels = kmeans.labels_
    init_means = kmeans.cluster_centers_

    init_covars = np.zeros((n_states, n_features, n_features))
    for i in range(n_states):
        pts = X[labels == i]
        if len(pts) > n_features:
            cov = np.cov(pts, rowvar=False) + np.eye(n_features) * min_covar
        else:
            cov = np.eye(n_features) * 1e-2
        init_covars[i] = cov

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        min_covar=min_covar,
        n_iter=500,
        tol=1e-4,
        init_params="st",
        random_state=42
    )
    model.means_ = init_means
    model.covars_ = init_covars
    model.fit(X)

    # Canonical State Permutation: 0 = BEAR, 1 = CHOP, 2 = BULL
    composite_trend = model.means_[:, 0] + 0.5 * model.means_[:, 1]
    order = np.argsort(composite_trend)

    model.means_ = model.means_[order]
    model._covars_ = model._covars_[order]
    model.startprob_ = model.startprob_[order]
    model.transmat_ = model.transmat_[order][:, order]

    return model


# =====================================================================
# 4. Causal (Forward-Only) State Inference (Zero Lookahead for Backtests)
# =====================================================================
def causal_predict_proba(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    n_states = model.n_components

    emission_probs = np.zeros((n_samples, n_states))
    for k in range(n_states):
        rv = multivariate_normal(mean=model.means_[k], cov=model.covars_[k], allow_singular=True)
        emission_probs[:, k] = rv.pdf(X) + 1e-300

    filtered_probs = np.zeros((n_samples, n_states))
    alpha = model.startprob_ * emission_probs[0]
    alpha /= np.sum(alpha)
    filtered_probs[0] = alpha

    trans = model.transmat_
    for t in range(1, n_samples):
        alpha = (alpha @ trans) * emission_probs[t]
        norm = np.sum(alpha)
        alpha = alpha / norm if norm > 0 else np.ones(n_states) / n_states
        filtered_probs[t] = alpha

    return filtered_probs


# =====================================================================
# 5. Main Execution Pipeline (80/20 Validation + 100% Live Refit)
# =====================================================================
def train_timeframe(tf: str, cfg: dict):
    data_path = os.path.join(DATA_DIR, f"{PAIR}-{tf}.feather")
    if not os.path.exists(data_path):
        print(f"[SKIP] {data_path} not found.")
        return

    df = pd.read_feather(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = engineer_stationary_features(df, cfg)
    clean_df = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)

    split_idx = int(len(clean_df) * 0.8)
    train_df = clean_df.iloc[:split_idx].copy()
    test_df = clean_df.iloc[split_idx:].copy()
    split_date = str(test_df["date"].iloc[0])

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values
    X_all = clean_df[FEATURE_COLS].values

    print(f"\n================ TRAINING {PAIR} [{tf}] ================")
    print(f"Total Bars: {len(clean_df)} | Train: {len(train_df)} | OOS Test: {len(test_df)} (From {split_date})")

    oos_model = fit_and_canonicalize_hmm(X_train, cfg["min_covar"])
    oos_probs = causal_predict_proba(oos_model, X_test)
    oos_states = np.argmax(oos_probs, axis=1)

    live_model = fit_and_canonicalize_hmm(X_all, cfg["min_covar"])

    means = live_model.means_
    covars = live_model.covars_
    trans_mat = live_model.transmat_

    state_names = {0: "BEAR", 1: "CHOP", 2: "BULL"}
    expected_durations = {}
    bell_curve_stats = {}

    print(f"Converged: {live_model.monitor_.converged} ({live_model.monitor_.iter} iterations)")
    for s in range(3):
        p_ii = trans_mat[s, s]
        exp_bars = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else float("inf")
        stds = np.sqrt(np.diag(covars[s]))

        expected_durations[s] = float(exp_bars)
        bell_curve_stats[s] = {
            "mean": dict(zip(FEATURE_COLS, means[s].tolist())),
            "std": dict(zip(FEATURE_COLS, stds.tolist())),
        }

        print(f"  State {s} [{state_names[s]}]:")
        print(f"    Expected Duration: {exp_bars:.1f} bars (p_ii={p_ii:.4f})")
        print(f"    Trend ATR (mu +/- sigma):   {means[s, 0]:+.2f} +/- {stds[0]:.2f}")
        print(f"    Impulse ATR (mu +/- sigma): {means[s, 1]:+.2f} +/- {stds[1]:.2f}")
        print(f"    Vol Expansion (NATR log):   {means[s, 2]:+.2f} +/- {stds[2]:.2f}")
        print(f"    Vol Momentum (EMA log):     {means[s, 3]:+.2f} +/- {stds[3]:.2f}")

    print("\nCausal Out-of-Sample State Distribution (0=BEAR, 1=CHOP, 2=BULL):")
    print(pd.Series(oos_states).value_counts(normalize=True).sort_index().round(3))

    bundle = {
        "model": live_model,
        "oos_model": oos_model,
        "split_date": split_date,
        "feature_cols": FEATURE_COLS,
        "config": cfg,
        "pair": PAIR,
        "timeframe": tf,
        "regime_labels": {"bear": 0, "chop": 1, "bull": 2},
        "expected_durations": expected_durations,
        "bell_curve_stats": bell_curve_stats,
    }

    out_file = os.path.join(OUTPUT_DIR, f"hmm_bundle_{PAIR}_{tf}.joblib")
    joblib.dump(bundle, out_file)
    print(f"Exported bundle -> {out_file}")


if __name__ == "__main__":
    for tf, cfg in TIMEFRAME_CONFIGS.items():
        train_timeframe(tf, cfg)