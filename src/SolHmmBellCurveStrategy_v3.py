# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import multivariate_normal
from freqtrade.strategy import IStrategy, informative, stoploss_from_open
from freqtrade.persistence import Trade
from freqtrade.exceptions import OperationalException


class SolHmmBellCurveStrategy(IStrategy):
    """
    Multi-Timeframe Gaussian HMM Strategy for SOL/USDT (v3 - Macro Aligned & Stoploss Anchored)
    - Fixes Jan-Apr 2026 1d CHOP bear-rally traps by requiring 1d positive trend/impulse alignment
    - Uses stoploss_from_open() to eliminate custom_stoploss relative-rate math inversions
    - Adds failed-breakout structure exit when 4h Bull probability collapses below 1h EMA50
    """
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    USE_OOS_MODEL_FOR_BACKTEST = True

    minimal_roi = {
        "0": 0.18,
        "720": 0.11,   # After 12 hours: harvest at +11%
        "2160": 0.065, # After 36 hours: harvest at +6.5%
        "4320": 0.03   # After 72 hours: harvest at +3%
    }

    stoploss = -0.052
    trailing_stop = False
    use_custom_stoploss = True

    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count: int = 220

    bundle_1d: dict = None
    bundle_4h: dict = None

    def _load_hmm_bundle(self, filename: str) -> dict:
        user_data_dir = Path(self.config.get("user_data_dir", "/freqtrade/user_data"))
        models_dir = user_data_dir / "models"
        candidate = models_dir / filename

        if not candidate.is_file():
            candidate = Path("/freqtrade/user_data/models") / filename

        if not candidate.is_file():
            raise OperationalException(
                f"HMM model file not found at '{candidate}'. "
                f"Ensure {filename} is in /home/divyahari/ft_userdata/user_data/models/"
            )

        try:
            return joblib.load(candidate)
        except Exception as e:
            raise OperationalException(
                f"Failed to load '{candidate}': {type(e).__name__}: {e}"
            ) from e

    def bot_start(self, **kwargs) -> None:
        self.bundle_1d = self._load_hmm_bundle("hmm_bundle_SOL_USDT_1d.joblib")
        self.bundle_4h = self._load_hmm_bundle("hmm_bundle_SOL_USDT_4h.joblib")

    # -----------------------------------------------------------------
    # Helper: Pure Stationary Features + Causal Forward HMM Filtering
    # -----------------------------------------------------------------
    def _compute_hmm_features_and_states(self, df: DataFrame, bundle: dict) -> DataFrame:
        df = df.copy()
        cfg = bundle["config"]
        feature_cols = bundle["feature_cols"]

        is_backtest = hasattr(self, "dp") and self.dp and self.dp.runmode.value in ("backtest", "hyperopt")
        model = bundle.get("oos_model", bundle["model"]) if (is_backtest and self.USE_OOS_MODEL_FOR_BACKTEST) else bundle["model"]

        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(cfg["atr_len"]).mean()
        sma = df["close"].rolling(cfg["sma_len"]).mean()

        df["trend_atr"] = (df["close"] - sma) / (atr + 1e-8)
        df["impulse_atr"] = (df["close"] - df["close"].shift(cfg["impulse_len"])) / (atr + 1e-8)

        natr = atr / (df["close"] + 1e-8)
        natr_baseline = natr.shift(1).rolling(cfg["vol_base_len"]).mean()
        df["vol_expansion"] = np.log((natr + 1e-8) / (natr_baseline + 1e-8))

        vol_fast = df["volume"].ewm(span=cfg["v_fast_len"], adjust=False).mean()
        vol_slow = df["volume"].shift(1).rolling(cfg["v_slow_len"]).mean()
        df["vol_momentum"] = np.log((vol_fast + 1e-8) / (vol_slow + 1e-8))

        df["hmm_state"] = 1
        df["prob_bear"] = 0.0
        df["prob_chop"] = 1.0
        df["prob_bull"] = 0.0
        df["z_trend_bull"] = 0.0
        df["z_impulse_bull"] = 0.0
        df["regime_age"] = 0

        valid_mask = df[feature_cols].notna().all(axis=1)
        if not valid_mask.any():
            return df

        X = df.loc[valid_mask, feature_cols].values
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

        states = np.argmax(filtered_probs, axis=1)

        ages = np.ones(n_samples, dtype=int)
        for t in range(1, n_samples):
            if states[t] == states[t - 1]:
                ages[t] = ages[t - 1] + 1

        df.loc[valid_mask, "hmm_state"] = states
        df.loc[valid_mask, "prob_bear"] = filtered_probs[:, 0]
        df.loc[valid_mask, "prob_chop"] = filtered_probs[:, 1]
        df.loc[valid_mask, "prob_bull"] = filtered_probs[:, 2]
        df.loc[valid_mask, "regime_age"] = ages

        bull_stats = bundle["bell_curve_stats"][2]
        mu_trend = bull_stats["mean"]["trend_atr"]
        std_trend = bull_stats["std"]["trend_atr"]
        mu_imp = bull_stats["mean"]["impulse_atr"]
        std_imp = bull_stats["std"]["impulse_atr"]

        df.loc[valid_mask, "z_trend_bull"] = (df.loc[valid_mask, "trend_atr"] - mu_trend) / (std_trend + 1e-8)
        df.loc[valid_mask, "z_impulse_bull"] = (df.loc[valid_mask, "impulse_atr"] - mu_imp) / (std_imp + 1e-8)

        return df

    # -----------------------------------------------------------------
    # Informative Timeframes (1d Macro Filter & 4h Primary Engine)
    # -----------------------------------------------------------------
    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.bundle_1d is None:
            self.bot_start()
        return self._compute_hmm_features_and_states(dataframe, self.bundle_1d)

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.bundle_4h is None:
            self.bot_start()
        return self._compute_hmm_features_and_states(dataframe, self.bundle_4h)

    # -----------------------------------------------------------------
    # Base Timeframe (1h Execution Indicators)
    # -----------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema_50"] = dataframe["close"].ewm(span=50, adjust=False).mean()

        delta = dataframe["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        dataframe["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        return dataframe

    # -----------------------------------------------------------------
    # Entry Logic (1d Macro Structure Alignment + 4h Pullback Sweet Spot)
    # -----------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.bundle_4h is None:
            self.bot_start()

        max_bull_age_4h = int(self.bundle_4h["expected_durations"][2] * 0.80)

        # Macro 1d Alignment: Either 1d is in BULL (State 2), OR 1d is in CHOP (State 1)
        # with positive daily trend/impulse (blocks Jan-Apr bear market dead-cat bounces)
        macro_1d_ok = (
            (dataframe["hmm_state_1d"] == 2) |
            (
                (dataframe["hmm_state_1d"] == 1) &
                (dataframe["prob_bear_1d"] < 0.30) &
                (dataframe["trend_atr_1d"] >= -0.10) &
                (dataframe["impulse_atr_1d"] >= 0.0)
            )
        )

        long_conditions = (
            macro_1d_ok &

            # Primary 4h Engine: High-conviction BULL State (State 2)
            (dataframe["hmm_state_4h"] == 2) &
            (dataframe["prob_bull_4h"] >= 0.72) &

            # Market Clock: 4h Bull regime is within active lifespan (<= 18 bars)
            (dataframe["regime_age_4h"] <= max_bull_age_4h) &

            # Bell Curve Pullback Sweet Spot (-1.20 <= Z <= +0.25)
            (dataframe["z_trend_bull_4h"] >= -1.20) &
            (dataframe["z_trend_bull_4h"] <= 0.25) &

            # Short-term 4h Velocity Safeguard
            (dataframe["z_impulse_bull_4h"] >= -0.90) &
            (dataframe["z_impulse_bull_4h"] <= 0.85) &

            # 1h Micro Trigger: Confirmed uptrend structure (Close > EMA20 > EMA50)
            (dataframe["close"] > dataframe["ema_20"]) &
            (dataframe["ema_20"] > dataframe["ema_50"]) &
            (dataframe["rsi_14"] >= 45) &
            (dataframe["rsi_14"] <= 67) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[long_conditions, ["enter_long", "enter_tag"]] = (1, "hmm_4h_pullback_sweetspot")
        return dataframe

    # -----------------------------------------------------------------
    # Anchored Custom Stoploss Using stoploss_from_open()
    # -----------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        """
        Uses stoploss_from_open() so profit locks are anchored to open_rate
        and never choke out 4h swing pullbacks prematurely.
        """
        if current_profit >= 0.11:
            return stoploss_from_open(0.080, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        if current_profit >= 0.08:
            return stoploss_from_open(0.050, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        if current_profit >= 0.055:
            return stoploss_from_open(0.022, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        if current_profit >= 0.038:
            return stoploss_from_open(0.006, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        return self.stoploss

    # -----------------------------------------------------------------
    # Exit Logic (Regime Breakdown or Bell Curve Climax)
    # -----------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = (
            # 1. 4h flips to BEAR
            ((dataframe["hmm_state_4h"] == 0) & (dataframe["prob_bear_4h"] > 0.50)) |
            # 2. Failed 4h BULL breakout: 4h drops back to CHOP with negative impulse & 1h breaks EMA50
            (
                (dataframe["hmm_state_4h"] != 2) &
                (dataframe["prob_bull_4h"] < 0.25) &
                (dataframe["z_impulse_bull_4h"] < -1.10) &
                (dataframe["close"] < dataframe["ema_50"])
            ) |
            # 3. Bell Curve Climax Exhaustion (> +1.6 sigma) + 1h breaks below EMA20
            ((dataframe["z_trend_bull_4h"] > 1.60) & (dataframe["close"] < dataframe["ema_20"]))
        )

        dataframe.loc[exit_long_conditions, ["exit_long", "exit_tag"]] = (1, "hmm_regime_flip_or_climax")
        return dataframe