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
    Multi-Timeframe Gaussian HMM Strategy for SOL/USDT (v22 - Structural Support Ratchet)
    1. Restores v18's exact 2023-2024 winning entry engine (+1,165 USDT ROI winners).
    2. Replaces arbitrary tight percentage stops with a 2-Layer Structural Support Stop:
       - Layer A (Chart Support): Exits if price breaks below BOTH 4H SMA20 support and 15m EMA200 support.
       - Layer B (One-Way Support Shelf Ratchet): Once profit reaches +4.8%, ratchets stop to +1.60%
         (leaving 3.2% breathing room above support so bull pullbacks never wick out early).
    3. Blocks 2022 bear-market crashes by requiring the 1D 50-SMA support slope not to be in freefall.
    """
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = False

    USE_OOS_MODEL_FOR_BACKTEST = True

    # Proven ROI ladder from v18 (+38.06% in 2023-24, 6.18% DD in 2026)
    minimal_roi = {
        "0": 0.16,     # Immediate climax target (+16%)
        "240": 0.09,   # After 4 hours: harvest at +9.0%
        "720": 0.06,   # After 12 hours: harvest at +6.0%
        "1440": 0.04,  # After 24 hours: harvest at +4.0%
        "2880": 0.025  # After 48 hours: harvest at +2.5%
    }

    # Wide catastrophe stoploss (-7.5%) so normal bull-market pullbacks never wick out prematurely;
    # true structural breakdowns are cut much earlier (-2.5% to -3.8%) by the 4H SMA20 / 15m EMA200 support exit.
    stoploss = -0.075
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
        df["sma_20"] = sma
        df["sma_50"] = df["close"].rolling(50).mean()
        # Structural slope of 50-period SMA (identifies if macro support is rising or collapsing)
        df["sma_50_rising"] = (df["sma_50"] >= df["sma_50"].shift(3) * 0.995).astype(int)

        df["trend_atr"] = (df["close"] - sma) / (atr + 1e-8)
        df["impulse_atr"] = (df["close"] - df["close"].shift(cfg["impulse_len"])) / (atr + 1e-8)

        natr = atr / (df["close"] + 1e-8)
        natr_baseline = natr.shift(1).rolling(cfg["vol_base_len"]).mean()
        df["vol_expansion"] = np.log((natr + 1e-8) / (natr_baseline + 1e-8))

        vol_fast = df["volume"].ewm(span=cfg["v_fast_len"], adjust=False).mean()
        vol_slow = df["volume"].shift(1).rolling(cfg["v_slow_len"]).mean()
        df["vol_momentum"] = np.log((vol_fast + 1e-8) / (vol_slow + 1e-8))

        # Default uninitialized warmup rows to BEAR (State 0)
        df["hmm_state"] = 0
        df["prob_bear"] = 1.0
        df["prob_chop"] = 0.0
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
    # Informative Timeframes (1D Macro & 4H Swing Engine)
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
    # Base Timeframe (15m Fast Execution & Support Indicators)
    # -----------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_20"] = dataframe["close"].ewm(span=20, adjust=False).mean()
        dataframe["ema_50"] = dataframe["close"].ewm(span=50, adjust=False).mean()
        dataframe["ema_200"] = dataframe["close"].ewm(span=200, adjust=False).mean()

        # 24-hour (96 bars of 15m) major structural support floor
        dataframe["support_floor_24h"] = dataframe["low"].rolling(96).min()

        delta = dataframe["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        dataframe["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        return dataframe

    # -----------------------------------------------------------------
    # 15m Entry Logic (v18 Winning Core + 1D Rising Support Filter)
    # -----------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.bundle_4h is None:
            self.bot_start()

        long_conditions = (
            # 1. Macro 1D Uptrend Gate: Restores v18's exact thresholds (+38% in 2023-24)
            #    plus sma_50_rising_1d to block the May-Aug 2022 bear market crash
            (dataframe["hmm_state_1d"] != 0) &
            (dataframe["prob_bear_1d"] < 0.25) &
            (dataframe["trend_atr_1d"] >= 0.10) &
            (dataframe["sma_20_1d"] > dataframe["sma_50_1d"]) &
            (dataframe["sma_50_rising_1d"] == 1) &

            # 2. 4H HMM Bull Regime Confirmation
            (dataframe["hmm_state_4h"] == 2) &
            (dataframe["prob_bull_4h"] >= 0.72) &
            (dataframe["z_trend_bull_4h"] >= -1.00) &
            (dataframe["z_trend_bull_4h"] <= 0.40) &
            (dataframe["z_impulse_bull_4h"] >= -0.60) &

            # 3. 15m Support Bounce Trigger: Bouncing off 15m EMA20 above EMA50 & EMA200 support
            (dataframe["close"] > dataframe["ema_20"]) &
            (dataframe["ema_20"] > dataframe["ema_50"]) &
            (dataframe["ema_50"] > dataframe["ema_200"]) &
            (dataframe["low"].rolling(4).min() <= dataframe["ema_20"] * 1.003) &
            (dataframe["close"] > dataframe["open"]) &
            (dataframe["rsi_14"] >= 48) &
            (dataframe["rsi_14"] <= 64) &
            (dataframe["volume"] > 0)
        )

        dataframe.loc[long_conditions, ["enter_long", "enter_tag"]] = (1, "15m_hmm_regime_rider")
        return dataframe

    # -----------------------------------------------------------------
    # Custom Stoploss: One-Way Support Shelf Ratchet (Never Moves Down!)
    # -----------------------------------------------------------------
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        """
        One-Way Support Shelf Ratchet:
        - Freqtrade guarantees stop_loss NEVER moves down once raised.
        - We wait until price clears +4.8% (establishing a new higher support shelf) before raising
          the stop to +1.60% (leaving 3.2% breathing room so normal 15m retests never wick us out).
        """
        # Shelf 3: Major breakout (> +10.0% profit) -> Lock in +7.0% support floor
        if current_profit >= 0.100:
            return stoploss_from_open(0.070, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # Shelf 2: Mid-swing expansion (> +7.0% profit) -> Lock in +4.0% support floor
        if current_profit >= 0.070:
            return stoploss_from_open(0.040, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # Shelf 1: First impulse complete (> +4.8% profit) -> Lock in +1.60% (Covers 1% TDS + fees,
        #          with 3.20% buffer above stop so 2023-2024 bull pullbacks ride to full ROI)
        if current_profit >= 0.048:
            return stoploss_from_open(0.016, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # Below +4.8%: Keep anchored at -7.5% from open price (never trails up prematurely on noise)
        return stoploss_from_open(self.stoploss, current_profit, is_short=trade.is_short, leverage=trade.leverage)

    # -----------------------------------------------------------------
    # 15m Exit Logic: Structural Support Breakdown & Climax Harvest
    # -----------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit 1: 4H Bull Regime genuinely breaks down AND price loses 15m EMA200 support
        regime_breakdown = (
            (dataframe["hmm_state_4h"] != 2) &
            (dataframe["prob_bull_4h"] < 0.35) &
            (dataframe["close"] < dataframe["ema_200"])
        )

        # Exit 2: Major Structural Support Breakdown!
        # Price breaks below BOTH the 4H 20-SMA support AND 15m EMA200 support with weak RSI (< 36).
        # Cuts failed trades at ~ -3.2% when support genuinely breaks, preventing -7.5% stop_loss hits!
        support_level_broken = (
            (dataframe["close"] < dataframe["sma_20_4h"] * 0.985) &
            (dataframe["close"] < dataframe["ema_200"] * 0.990) &
            (dataframe["rsi_14"] < 36)
        )

        # Exit 3: 4H flips directly to high-conviction BEAR
        bear_flip = (
            (dataframe["hmm_state_4h"] == 0) &
            (dataframe["prob_bear_4h"] > 0.55)
        )

        # Exit 4: Overextended 4H Climax Harvest
        climax_harvest = (
            (dataframe["z_trend_bull_4h"] > 1.60) &
            (dataframe["rsi_14"] < 45) &
            (dataframe["close"] < dataframe["ema_50"])
        )

        dataframe.loc[regime_breakdown | support_level_broken | bear_flip, ["exit_long", "exit_tag"]] = (1, "structural_support_broken")
        dataframe.loc[climax_harvest, ["exit_long", "exit_tag"]] = (1, "climax_profit_harvest")
        return dataframe