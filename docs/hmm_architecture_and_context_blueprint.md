# Institutional Quant Blueprint: SOL/USDT Multi-Timeframe Gaussian HMM Architecture & Context Guide

> **Purpose of this Document:**  
> Pass this file as context to any future AI coding assistant or quantitative session. It documents the exact mathematical equations, solved bugs, verified `1d` and `4h` model diagnostics, strict anti-patterns ("what NOT to do"), and the active Freqtrade multi-timeframe execution architecture for `SOL_USDT`.

---

## 1. Core Architectural Philosophy (Preserved Foundations)

This system builds upon the **Zero-Scaler Stationary Hidden Markov Model (HMM)** framework[cite: 3]:

1. **Zero-Scaler Stationary Feature Engineering:**  
   Never use `StandardScaler` or `MinMaxScaler` on raw price or volume[cite: 3]. Crypto assets like `SOL_USDT` move across orders of magnitude, causing global scalers to suffer from severe train-test distribution shift[cite: 3]. All input features must be natively dimensionless and scale-invariant (ratios, ATR-normalized distances, and log transformations)[cite: 2, 3].
2. **Deterministic K-Means Spatial Pre-Seeding:**  
   Standard Baum-Welch Expectation-Maximization (EM) with random initialization gets trapped in poor local minima. We run a deterministic `KMeans(n_clusters=3, random_state=42, n_init=10)` pass on the training matrix to pre-seed `model.means_` and sample cluster covariance matrices into `model.covars_`, locking `init_params="st"` so EM only initializes start and transition probabilities[cite: 2, 3].
3. **Full Covariance (`covariance_type="full"`):**  
   For Group C Ecosystem Altcoins (`SOL`, `AVAX`, `NEAR`), directional trend excursions, volatility expansion, and volume velocity are tightly coupled[cite: 3]. Retaining `covariance_type="full"` with `min_covar=1e-4` captures cross-feature correlations without matrix singularity[cite: 2, 3].
4. **Two Mathematical Pillars for Execution:**
   * **The Market Clock (Expected Duration):** Derived from the diagonal persistence probability ($p_{ii}$) of the transition matrix[cite: 3]:
     $$D = \frac{1}{1 - p_{ii}}$$
     Used to block late-cycle entries when a regime's age exceeds $80\%$ of its expected lifespan ($0.80 \times D$)[cite: 3].
   * **The Variance Map (Gaussian Bell Curves):** Each hidden state $k$ emits a multivariate Gaussian distribution $\mathcal{N}(\mu_k, \Sigma_k)$[cite: 3]. Feature standard deviations ($\sigma_k = \sqrt{\text{diag}(\Sigma_k)}$) define real-time Z-score bands for pullback entries and climax exits.

---

## 2. Forensic Audit: Legacy `1d` Model Bugs That Were Solved

When auditing the legacy 3-feature `1d` script (`train_sol_hmm_stationary.py`)[cite: 2] and its binary bundle (`hmm_bundle_SOL_USDT_1d.joblib`)[cite: 1], five structural defects were identified and permanently fixed in `train_sol_hmm_mtf.py`:

### Bug 1: The Nominal Dollar ATR Trap in `vol_expansion` (SOLVED)
* **Legacy Behavior:** Computed `atr14 = tr.rolling(14).mean()` in nominal USDT dollars and calculated `vol_expansion = np.log(atr14 / atr_baseline_90)`[cite: 2, 3].
* **Why It Failed:** In the legacy `.joblib` binary, `State 2 (BEAR)` had `trend_atr = -1.38` but reported a negative `vol_expansion = -0.31` (lower volatility than `CHOP` at `-0.11` and `BULL` at `+0.27`)[cite: 1, 3]. When SOL dropped 50% in dollar price over a 90-day bear market, a 6% daily candle at \$90 produced half the dollar ATR of a 6% candle at \$180 ninety days earlier. Nominal dollar decay tricked the HMM into classifying bear crashes as volatility compression[cite: 1, 2].
* **The Fix:** Normalize ATR by Close price (**Normalized ATR / `NATR`**) *before* computing the rolling baseline log-ratio:
  $$\text{NATR}_t = \frac{\text{ATR}_t}{\text{Close}_t}, \quad \text{vol\_expansion}_t = \ln\left(\frac{\text{NATR}_t}{\text{SMA}(\text{NATR}, N)_{t-1}}\right)$$

### Bug 2: Exporting the 80% Train Slice as the Live Production Model (SOLVED)
* **Legacy Behavior:** Split data 80/20 (`X_train`, `X_test`), called `model.fit(X_train)`, and dumped that exact `model` to `.joblib`[cite: 2]. The exported production model was blind to the most recent 20% (~1 year) of market history[cite: 2].
* **The Fix:** The upgraded trainer fits `oos_model` on the first 80% of bars (saved in the bundle for strict out-of-sample backtesting after `split_date`) and fits `model` on **100% of bars** for Live/Dry-Run deployment.

### Bug 3: Regime Reversal Lag & Boxcar Volume Echo (SOLVED)
* **Legacy Behavior:** Used only 3 features (`trend_atr`, `vol_expansion`, `vol_momentum`)[cite: 1, 2, 3]. Because `trend_atr = (close - sma20) / atr14` measures static position relative to a 20-bar mean[cite: 2, 3], a violent reversal from $+3.0\text{ ATR}$ down to $+1.0\text{ ATR}$ still looked "Bullish" to the HMM. Furthermore, `vol_fast = volume.rolling(5).mean()`[cite: 2, 3] created a 5-bar boxcar echo after single-bar volume spikes.
* **The Fix:** Added a 4th feature, **ATR-Normalized Impulse (`impulse_atr`)**, to measure short-term directional velocity, and replaced the `vol_fast` SMA with an **Exponential Moving Average (`EMA`)**:
  $$\text{impulse\_atr}_t = \frac{\text{Close}_t - \text{Close}_{t - L_{\text{imp}}}}{\text{ATR}_{14, t}}$$

### Bug 4: Random State Label-Switching Across Retrains (SOLVED)
* **Legacy Behavior:** K-Means assigned arbitrary state indices (`bull=0, chop=1, bear=2` in the legacy `1d` bundle)[cite: 1, 2].
* **The Fix:** Immediately after EM convergence, the trainer permutes `model.means_`, `model._covars_`, `model.startprob_`, and `model.transmat_` using `np.argsort(means[:, 0] + 0.5 * means[:, 1])`. Across all timeframes and retrains, state indices are permanently locked to:
  * **`State 0` = `BEAR`**
  * **`State 1` = `CHOP`**
  * **`State 2` = `BULL`**

### Bug 5: Viterbi Lookahead Bias in Backtesting (SOLVED)
* **Legacy Behavior:** Calling `model.predict(X)`[cite: 1, 2] or `model.predict_proba(X)` on a full historical dataframe uses the forward-backward / Viterbi algorithm across the entire array, leaking future candle information into past state assignments.
* **The Fix:** Implemented `causal_predict_proba()`, a strictly forward-only recursive Bayesian filter:
  $$\alpha_t = \frac{(\alpha_{t-1} A) \odot B(x_t)}{\sum (\alpha_{t-1} A) \odot B(x_t)}$$
  where $A$ is `transmat_` and $B(x_t)$ is the multivariate Gaussian PDF value at bar $t$.

---

## 3. Verified Production Model Diagnostics (`SOL_USDT`)

The following diagnostics were generated by `train_sol_hmm_mtf.py` using the 4-feature stationary pipeline (`trend_atr`, `impulse_atr`, `vol_expansion`, `vol_momentum`).

### A. Feature Lookback Configuration Matrix

| Parameter | `1d` Timeframe Config (Macro Filter) | `4h` Timeframe Config (Primary Engine) | Physical Time Horizon (`4h`) |
| :--- | :--- | :--- | :--- |
| **`atr_len`** | `14` bars | `14` bars | 2.33 days |
| **`sma_len`** (`trend_atr`) | `20` bars[cite: 2, 3] | `30` bars | 5.0 days |
| **`impulse_len`** (`impulse_atr`) | `3` bars | `6` bars | 24 hours (1 day) |
| **`vol_base_len`** (`vol_expansion`) | `90` bars[cite: 2, 3] | `180` bars | 30.0 days |
| **`v_fast_len`** (`vol_momentum` EMA) | `5` bars[cite: 2, 3] | `12` bars | 48 hours (2 days) |
| **`v_slow_len`** (`vol_momentum` SMA) | `30` bars[cite: 2, 3] | `90` bars | 15.0 days |
| **`min_covar`** | `1e-4`[cite: 2, 3] | `1e-4` | Regularization floor |

---

### B. `1d` Model Diagnostic Table (`hmm_bundle_SOL_USDT_1d.joblib`)
* **Dataset Coverage:** `1,707` total bars | `1,365` Train bars | `342` Out-of-Sample Test bars (OOS Start: `2025-10-28 00:00:00+00:00`)
* **Convergence:** `True` (`82` EM iterations)

| State | Regime | Persistence ($p_{ii}$) | Expected Duration ($D$) | 80% Clock Cutoff ($0.8D$) | `trend_atr` ($\mu \pm \sigma$) | `impulse_atr` ($\mu \pm \sigma$) | `vol_expansion` ($\mu \pm \sigma$) | `vol_momentum` ($\mu \pm \sigma$) | Causal OOS Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`0`** | **`BEAR`** | `0.9459` | **18.5 days** | 14.8 days | `-1.33 +/- 0.83` | `-0.39 +/- 0.96` | **`+0.01 +/- 0.37`** | `-0.07 +/- 0.51` | **38.9%** |
| **`1`** | **`CHOP`** | `0.9083` | **10.9 days** | 8.7 days | `+0.28 +/- 0.67` | `+0.07 +/- 0.94` | `-0.08 +/- 0.24` | `-0.13 +/- 0.34` | **35.7%** |
| **`2`** | **`BULL`** | `0.9270` | **13.7 days** | **11.0 days** | **`+2.09 +/- 0.93`** | **`+0.78 +/- 1.23`** | `-0.16 +/- 0.33` | **`+0.15 +/- 0.35`** | **25.4%** |

*Diagnostic Verification (`1d`):*
1. **Bear Volatility Restored:** Unlike the legacy `1d` model where Bear `vol_expansion` was `-0.31`[cite: 1], the `NATR` fix correctly reveals that `1d BEAR` has the **highest volatility expansion (`+0.01 +/- 0.37`)** and widest volume dispersion (`+/- 0.51`).
2. **Steady Daily Bull Grinder:** On the `1d` macro horizon, `State 2 [BULL]` represents steady structural uptrends (`trend_atr = +2.09`, `vol_expansion = -0.16`, strong positive volume inflows `+0.15`) lasting an average of **13.7 days**.

---

### C. `4h` Model Diagnostic Table (`hmm_bundle_SOL_USDT_4h.joblib`)
* **Dataset Coverage:** `8,055` total bars | `6,444` Train bars | `1,611` Out-of-Sample Test bars (OOS Start: `2026-01-11 04:00:00+00:00`)
* **Convergence:** `True` (`85` EM iterations)

| State | Regime | Persistence ($p_{ii}$) | Expected Duration ($D$) | 80% Clock Cutoff ($0.8D$) | `trend_atr` ($\mu \pm \sigma$) | `impulse_atr` ($\mu \pm \sigma$) | `vol_expansion` ($\mu \pm \sigma$) | `vol_momentum` ($\mu \pm \sigma$) | Causal OOS Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`0`** | **`BEAR`** | `0.9578` | **23.7 bars** *(94.8h / ~4.0d)* | 19 bars | `-1.72 +/- 1.06` | `-0.69 +/- 1.33` | `-0.04 +/- 0.36` | `-0.14 +/- 0.45` | **35.3%** |
| **`1`** | **`CHOP`** | `0.9232` | **13.0 bars** *(52.0h / ~2.2d)* | 10 bars | `+0.17 +/- 0.84` | `+0.07 +/- 1.35` | **`-0.13 +/- 0.26`** | `-0.14 +/- 0.31` | **43.3%** |
| **`2`** | **`BULL`** | `0.9547` | **22.1 bars** *(88.4h / ~3.7d)* | **18 bars** *(72h)* | **`+2.17 +/- 1.14`** | **`+0.93 +/- 1.52`** | **`+0.02 +/- 0.35`** | **`+0.12 +/- 0.39`** | **21.4%** |

*Diagnostic Verification (`4h`):*
1. **Clean Noise Isolation:** In the blind 2026 out-of-sample test (`1,611` bars), `SOL_USDT` spent **43.3%** of the time in `CHOP` (`vol_expansion = -0.13 +/- 0.26`, tightest variance) and **35.3%** in `BEAR`. Only **21.4%** of `4h` candles qualify as genuine `State 2 [BULL]` regimes.
2. **True Breakout Signature on `4h`:** `State 2 [BULL]` on `4h` is the *only* state with both positive volatility expansion (`+0.02`) and positive volume momentum (`+0.12`), cleanly separating real institutional breakouts from low-volume fakeouts.

---

## 4. Multi-Timeframe Strategy Calibration (`SolHmmBellCurveStrategy_v12.py`)

The Freqtrade strategy operates on a **3-Layer Hierarchy** (`1d` Macro Filter $\rightarrow$ `4h` Primary HMM Engine $\rightarrow$ `1h` Base Execution):

### 1. Bell Curve Z-Score Equations (`4h` Bull State)
Using the learned `4h` `State 2 [BULL]` Gaussian parameters ($\mu_{\text{trend}} = +2.17, \sigma_{\text{trend}} = 1.14$ and $\mu_{\text{imp}} = +0.93, \sigma_{\text{imp}} = 1.52$):
$$Z_{\text{trend}, 4h} = \frac{\text{trend\_atr}_{4h} - 2.17}{1.14}, \quad Z_{\text{impulse}, 4h} = \frac{\text{impulse\_atr}_{4h} - 0.93}{1.52}$$

* **Entry Sweet Spot ($-1.20 \le Z_{\text{trend}, 4h} \le +1.20$):**  
  Translates to `trend_atr_4h` between **`+0.80` and `+3.54` ATRs** above the 30-bar SMA. This ensures we enter on healthy pullbacks or mid-trend continuations inside the Bull bell curve rather than buying exhausted blow-off wicks.
* **Climax Exit Trigger ($Z_{\text{trend}, 4h} > +2.20$):**  
  Translates to `trend_atr_4h > +4.68 ATRs` (beyond the 98.6th percentile of the Bull bell curve). When price reaches this extreme tail and breaks below the `1h` `EMA(20)`, the strategy harvests profits immediately.

### 2. Market Clock Exhaustion Filter
* **`4h` Bull Expected Duration ($D_{4h}$):** `22.1` bars (`88.4` hours)[cite: 3].
* **80% Exhaustion Rule:** $\lfloor 22.1 \times 0.80 \rfloor = \mathbf{18\text{ bars}}$ (`72` hours)[cite: 3].
* Any new long entry is blocked if `regime_age_4h > 18`.

---

## 5. Strict Guardrails: What NOT To Do in Future Iterations

When modifying or extending this framework in future sessions, **NEVER** do any of the following:

1. **DO NOT feed raw OHLC prices, raw dollar ATR, or raw volume into `GaussianHMM`.** Every feature must remain dimensionless and stationary across \$10 SOL and \$500 SOL[cite: 2, 3].
2. **DO NOT introduce `StandardScaler`, `RobustScaler`, or `MinMaxScaler`.** Our features naturally inhabit approximately $[-3, +3]$[cite: 3]. External scalers reintroduce lookahead leakage and train-test drift[cite: 3].
3. **DO NOT call `model.predict(X)` or `model.predict_proba(X)` inside Freqtrade's `populate_indicators()`.** In backtesting, Freqtrade passes the full historical slice at once. Standard `hmmlearn` inference runs the backward/Viterbi pass from the end of the array, creating severe lookahead bias. Always use `causal_predict_proba()`.
4. **DO NOT hardcode state numbers without canonical sorting.** If you modify the training script, preserve the `np.argsort(composite_trend)` permutation block so `0=BEAR, 1=CHOP, 2=BULL` never scrambles.
5. **DO NOT use non-ASCII characters (like `±`, `μ`, `σ`) inside Python `print()` strings.** To prevent `SyntaxError: Non-UTF-8 code starting with '\xb1'` across Windows editors and Linux Docker containers, always include `# -*- coding: utf-8 -*-` at line 1 and use pure ASCII (`+/-`, `mu`, `sigma`).

---

## 6. Future Optimization Roadmap (What To Build Next)

The following enhancements have been designed conceptually[cite: 3] but are **not yet wired** into `SolHmmBellCurveStrategy_v12.py`:

1. **Dynamic Bell Curve Stop-Loss (`custom_stoploss`):**  
   Currently, the strategy uses a static `-0.06` (6%) stoploss and fixed trailing offset. Per Section 1.B of the Quant Framework[cite: 3], wire `custom_stoploss()` to scale dynamically with the active `4h` state's `vol_expansion` and `NATR` (widening stops when `vol_expansion_4h > 0` during high-velocity Bull runs and tightening stops when `vol_expansion_4h < 0` in low-volatility compression)[cite: 3].
2. **Position Sizing by Regime Age & Probability (`custom_stake_amount`):**  
   Scale position size proportionally to `prob_bull_4h` (e.g., 1.0x base stake at `0.70` probability, 1.5x stake at `>0.90` probability when `regime_age_4h <= 6` bars) and scale down to 0.5x stake when `regime_age_4h` approaches the 50%–80% Market Clock window[cite: 3].
3. **Bi-Directional Futures Execution (`can_short = True`):**  
   Activate short entries on Bybit USDT perpetual futures when `hmm_state_1d == 0` (`BEAR`), `hmm_state_4h == 0` (`BEAR`), `prob_bear_4h >= 0.75`, and `regime_age_4h <= 19` bars ($0.80 \times 23.7$), targeting `-1.0` to `+1.0` Z-score bounces on the `State 0 [BEAR]` bell curve (`trend_atr_4h` around `-1.72 +/- 1.06`).
4. **Multi-Pair Portfolio Expansion (Group A & Group B):**  
   Extend `train_sol_hmm_mtf.py` to loop over Group A (`BTC_USDT`, `ETH_USDT`) and Group B (`WIF_USDT`, `PEPE_USDT`, lifting `vol_base_len` to `120–150` days and `min_covar` to `5e-4` per the Dynamic Asset Group Playbook)[cite: 3].