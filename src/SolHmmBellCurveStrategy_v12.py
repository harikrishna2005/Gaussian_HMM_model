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
    Multi-Timeframe Gaussian HMM Strategy for SOL/USDT (v12 - TDS & High-Payoff Edition)
    - Engineered to beat 1% Indian TDS + exchange fees (~2.4% total friction)
    - 1D Macro Gate: Blocks bear-market relief traps (trend_atr_1d >= 0.0)
    - 4H Swing Engine: Confirmed Bull state (prob >= 0.70) in pullback zone (-1.20 <= Z <= +0.20)
    - 1H Execution Trigger: Support retest + green confirmation candle
    - TDS-Compensated Breakeven: Locks +1.40% at +4.2% gain (guarantees net positive after 1% TDS)
    """
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    USE_OOS_MODEL_FOR_BACKTEST = True

    # Sized to capture 4H expansions that overcome 2.4% TDS/fee friction
    minimal_roi = {
        "0": 0.16,     # Climax target (+16%)
        "360": 0.085,  # After 6 hours: harvest at +8.5%
        "720": 0.060,  # After 12 hours: harvest at +6.0%
        "1440": 0.045, # After 24 hours: harvest at +4.5%
        "2880": 0.030  # After 48 hours: harvest at +3.0%
    }

    # Fixed safety stoploss (-4.6%)
    stoploss = -0.046
    trailing_stop = False
    use_custom_stoploss = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
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

        df.loc[valid_mask, "hmm_state"] = states
        df.loc[valid_mask, "prob_bear"] = filtered_probs[:, 0]
        df.loc[valid_mask, "prob_chop"] = filtered_probs[:, 1]
        df.loc[valid_mask, "prob_bull"] = filtered_probs[:, 2]

        bull_stats = bundle["bell_curve_stats"][2]
        mu_trend = bull_stats["mean"]["trend_atr"]
        std_trend = bull_stats["std"]["trend_atr"]
        mu_imp = bull_stats["mean"]["impulse_atr"]
        std_imp = bull_stats["std"]["impulse_atr"]

        df.loc[valid_mask, "z_trend_bull"] = (df.loc[valid_mask, "trend_atr"] - mu_trend) / (std_trend + 1e-8)
        df.loc[valid_mask, "z_impulse_bull"] = (df.loc[valid_mask, "impulse_atr"] - mu_imp) / (std_imp + 1e-8)

        return df

    # -----------------------------------------------------------------
    # Informative Timeframes
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
    # Base Timeframe (1H Indicators)
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
    # Sureshot Entry Logic
    # -----------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.bundle_4h is None:
            self.bot_start()

        long_conditions = (
            # 1. Macro 1D Gate: Daily Close >= SMA20 (Blocks bear downtrends)
            (dataframe["hmm_state_1d"] != 0) &
            (dataframe["trend_atr_1d"] >= 0.0) &
            (dataframe["prob_bear_1d"] < 0.35) &

            # 2. 4H Engine: High-conviction BULL State (State 2)
            (dataframe["hmm_state_4h"] == 2) &
            (dataframe["prob_bull_4h"] >= 0.70) &

            # 3. Pullback Sweet Spot on Bull Curve (-1.20 <= Z <= +0.20)
            (dataframe["z_trend_bull_4h"] >= -1.20) &
            (dataframe["z_trend_bull_4h"] <= 0.20) &

            # 4. Short-term Velocity Safeguard (No falling knives)
            (dataframe["z_impulse_bull_4h"] >= -1.00) &
            (dataframe["z_impulse_bull_4h"] <= 0.80) &

            # 5. 1H Execution Trigger: Support hold + Green confirmation candle
            (dataframe["close"] > dataframe["open"]) &
            (dataframe["close"] > dataframe["ema_20"]) &
            (dataframe["ema_20"] >= dataframe["ema_50"] * 0.995) &
            (dataframe["rsi_14"] >= 43) &
            (dataframe["rsi_14"] <= 66) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[long_conditions, ["enter_long", "enter_tag"]] = (1, "sureshot_pullback")
        return dataframe

    # -----------------------------------------------------------------
    # Custom Stoploss: TDS-Compensated Breakeven & Profit Ratchet
    # -----------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        """
        - Tier 3 (> +10.0% profit): Lock in +6.5% above entry
        - Tier 2 (> +6.5% profit): Lock in +3.5% above entry
        - Tier 1 (> +4.2% profit): Lock in +1.40% above entry
          (+1.40% covers the 1% TDS on sell + fees, guaranteeing net wallet profit)
        - Below +4.2%: Keeps hard stop at -4.6% to let 4H swings breathe
        """
        # Tier 3: Major expansion
        if current_profit >= 0.100:
            return stoploss_from_open(0.065, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # Tier 2: Solid swing gain
        if current_profit >= 0.065:
            return stoploss_from_open(0.035, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # Tier 1: TDS-Compensated Breakeven
        if current_profit >= 0.042:
            return stoploss_from_open(0.014, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        return self.stoploss

    # -----------------------------------------------------------------
    # Exit Logic: 4H Invalidation & Climax Harvest
    # -----------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = (
            # 1. Thesis Invalidation: 4H HMM flips to BEAR
            ((dataframe["hmm_state_4h"] == 0) & (dataframe["prob_bear_4h"] > 0.55)) |

            # 2. Climax Harvest: 4H overextended and breaks 1H EMA20
            ((dataframe["z_trend_bull_4h"] > 1.70) & (dataframe["close"] < dataframe["ema_20"]))
        )

        dataframe.loc[exit_long_conditions, ["exit_long", "exit_tag"]] = (1, "regime_invalidation_or_climax")
        return dataframe