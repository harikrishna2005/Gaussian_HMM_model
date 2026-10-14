# Gaussian HMM Quantitative Trading Blueprint & Context Guide
**Target Asset Class:** Cryptocurrency (Primary Benchmark: `SOL_USDT` — Group C Ecosystem Altcoins)  
**Execution Engine:** Freqtrade (Python, `hmmlearn`, `scipy.stats`, `pandas`, `joblib`)  
**Purpose of This Document:** This file serves as the canonical engineering context for AI assistants and quantitative developers. It documents **what has already been tested**, **which mathematical bugs were discovered and fixed**, **what must never be reintroduced (Anti-Patterns)**, **the current active training architecture**, and **what remains to be implemented or improvised**.

---

## 1. Core Architectural Philosophy (Why We Built It This Way)

### A. The Zero-Scaler Stationary Paradigm
Traditional machine learning pipelines apply `StandardScaler` or `MinMaxScaler` to input features before fitting a Gaussian Hidden Markov Model (`GaussianHMM`). In cryptocurrency markets, **global scalers fail catastrophically** because nominal asset prices (`SOL_USDT` moving from $\$10$ to $\$250+$) and nominal trading volumes shift across orders of magnitude between market cycles. A scaler fitted on 2022–2024 data produces distorted Z-scores in 2025–2026.

* **Our Solution:** Every input feature fed into the HMM is **self-scaling and strictly stationary** (constructed via volatility-normalized ratios or rolling logarithmic ratios).
* **Result:** All features naturally oscillate within a bounded decimal range (approximately $[-4.0, +4.0]$) with preserved fat tails, requiring **zero external feature scalers** and guaranteeing zero train-test distribution shift.

### B. Deterministic K-Means Spatial Pre-Seeding
The Baum-Welch Expectation-Maximization (EM) algorithm used by `hmmlearn` is highly sensitive to initial parameters and easily gets trapped in suboptimal local maxima when initialized randomly.
* **Our Solution:** We run a deterministic `KMeans(n_clusters=3, random_state=42, n_init=10)` pass on the training matrix to extract spatial cluster centers (`means_`) and sample cluster covariance matrices (`covars_`), adding a regularization floor (`min_covar = 1e-4`) to the diagonal.
* We then lock `init_params="st"` in `GaussianHMM` so Baum-Welch only auto-initializes the start probabilities (`s`) and transition matrix (`t`) while refining our pre-seeded Gaussian bell curves.

### C. Full Covariance Matrix (`covariance_type="full"`)
For high-beta ecosystem altcoins (`SOL`, `AVAX`, `NEAR`), directional trend expansion, candle velocity, and volume acceleration are tightly coupled. Using `covariance_type="diag"` or `"spherical"` ignores the cross-correlation between volume surges and volatility expansion. We strictly retain `"full"` covariance matrices.

---

## 2. Forensic Audit of the Original `1d` Model: Problems Identified & Solved

When auditing the initial 3-feature `1d` model (`hmm_bundle_SOL_USDT_1d.joblib`), the model converged in 85 iterations with the following state means (`[trend_atr, vol_expansion, vol_momentum]`):
* **State 0 (`bull`):** $[+1.73, +0.27, +0.07]$
* **State 1 (`chop`):** $[-0.003, -0.11, -0.07]$
* **State 2 (`bear`):** $[-1.38, -0.31, -0.12]$

While functional, forensic inspection revealed **six critical structural flaws** that have now been solved in the upgraded pipeline:

### Problem 1: The Nominal Dollar ATR Bug in `vol_expansion` (SOLVED)
* **The Bug:** In the original `1d` script, `atr14` was calculated in raw USDT dollars, and `vol_expansion` was computed as:
  $$\text{vol\_expansion}_{\text{old}} = \ln\left(\frac{\text{ATR}_{14, t}}{\text{SMA}(\text{ATR}_{14}, 90)_t}\right)$$
  Because `ATR` was in nominal dollars, when `SOL_USDT` dropped $50\%$ in price during a 60-day Bear market, a $6\%$ daily candle at $\$90$ ($\text{ATR} = \$5.40$) looked $50\%$ smaller than a $6\%$ daily candle 60 days earlier at $\$180$ ($\text{ATR} = \$10.80$). Consequently, the HMM learned that Bear markets have **negative volatility expansion ($-0.31$)**—falsely classifying violent bear sell-offs as low-volatility compression!
* **The Fix:** Convert `ATR` into **Normalized Percentage ATR ($\text{NATR}$)** before taking the 90-bar log ratio:
  $$\text{NATR}_{14, t} = \frac{\text{ATR}_{14, t}}{\text{Close}_t}, \quad \text{vol\_expansion}_{\text{new}} = \ln\left(\frac{\text{NATR}_{14, t}}{\text{SMA}(\text{NATR}_{14}, L_{\text{vol}})_{t-1}}\right)$$

### Problem 2: The 80/20 Train-Split Production Blindness (SOLVED)
* **The Bug:** The original script split the dataset chronologically ($80\%$ `X_train`, $20\%$ `X_test`), fitted the HMM on `X_train`, and exported that exact `X_train` model to `.joblib`. This meant the live trading model had never seen the most recent $20\%$ ($\sim 1+$ year) of `SOL_USDT` market structure.
* **The Fix:** The training script now performs a **Dual-Model Export**:
  1. `oos_model`: Trained strictly on the first $80\%$ of data (saved with `split_date`) for honest, zero-leakage Out-of-Sample backtesting.
  2. `model`: Refitted on $100\%$ of available historical data for live/dry-run deployment.

### Problem 3: Reversal Lag at Blow-Off Tops & Capitulation Bottoms (SOLVED)
* **The Bug:** `trend_atr = (close - sma20) / atr14` only measures **positional distance** from a 20-period moving average. If `SOL_USDT` rallies to $+3.5\text{ ATR}$ above `SMA(20)` and then crashes $-15\%$ in one candle, price is still $+1.5\text{ ATR}$ above `SMA(20)`. Relying solely on `trend_atr` keeps the HMM trapped in the `BULL` state during the initial cascade of a crash.
* **The Fix:** Added a 4th stationary feature—**ATR-Normalized Impulse (`impulse_atr`)**—which measures short-term directional velocity over $L_{\text{imp}}$ bars in units of `ATR`:
  $$\text{impulse\_atr}_t = \frac{\text{Close}_t - \text{Close}_{t - L_{\text{imp}}}}{\text{ATR}_{14, t} + \epsilon}$$

### Problem 4: Boxcar SMA Echoes in Volume Momentum (SOLVED)
* **The Bug:** Using a 5-bar Simple Moving Average (`SMA(5)`) for fast volume meant a single 1-day liquidation spike kept `vol_momentum` artificially elevated for 5 consecutive days, and then caused an artificial cliff drop on day 6 when the spike exited the rolling window.
* **The Fix:** Replaced the fast volume `SMA` with an **Exponential Moving Average (`EMA`)** so volume spikes decay smoothly and exponentially.

### Problem 5: Random HMM Label Switching Across Retrains (SOLVED)
* **The Bug:** `hmmlearn` assigns integer state labels (`0, 1, 2`) arbitrarily depending on cluster initialization. Having to look up a dictionary (`regime_labels['bull']`) makes strategy code fragile and hard to compare across timeframes.
* **The Fix:** Immediately after EM convergence, we compute a composite directional score ($\mu_{\text{trend\_atr}} + 0.5 \cdot \mu_{\text{impulse\_atr}}$) and **canonically permute** `model.means_`, `model._covars_`, `model.startprob_`, and `model.transmat_` so that across **every pair and every timeframe**:
  * **`State 0` = `BEAR`** (Lowest directional mean)
  * **`State 1` = `CHOP`** (Neutral/Consolidation mean)
  * **`State 2` = `BULL`** (Highest directional mean)

### Problem 6: Viterbi Lookahead Bias in Freqtrade Backtesting (SOLVED)
* **The Bug:** Calling `model.predict(X)` or `model.predict_proba(X)` on an entire historical DataFrame inside Freqtrade's `populate_indicators()` uses the forward-backward / Viterbi algorithm, which passes information from **future candles ($t+1 \dots T$)** backward to smooth state classifications at time $t$. This creates severe lookahead bias in backtesting.
* **The Fix:** Implemented `causal_predict_proba(model, X)`, a strictly **forward-only recursive Bayesian filter** that computes $P(S_t = k \mid X_{1:t})$ using only past and current observations:
  $$\alpha_t(j) \propto \left(\sum_{i=0}^{2} \alpha_{t-1}(i) A_{ij}\right) \cdot \mathcal{N}\left(x_t \mid \mu_j, \Sigma_j\right)$$

---

## 3. Strict Anti-Patterns: What NOT To Consider in Future Iterations

When extending or improvising this framework, **NEVER** do any of the following:
1. **DO NOT feed raw OHLC prices, raw Volume, or un-normalized ATR/MACD values into the HMM.** Any feature bound to nominal token price will destroy stationarity.
2. **DO NOT introduce `StandardScaler`, `RobustScaler`, or `MinMaxScaler`.** Keep all features naturally stationary through volatility normalization and log ratios.
3. **DO NOT trade directly off a `1d`-only HMM without a lower-timeframe regime engine.** On `SOL_USDT`, `1d` has too few historical bars ($\sim 1,800$ candles) for high-confidence 4D covariance estimation and suffers from a 24-hour execution lag.
4. **DO NOT use `model.predict()` or `model.predict_proba()` on full backtest DataFrames in Freqtrade.** Always use the forward-only `causal_predict_proba()` function to guarantee zero lookahead bias.
5. **DO NOT enter trades purely because `state == 2` (`BULL`).** Always combine state conviction ($P(\text{State}=2) > 0.70$), **Bell Curve Z-score position**, and **Market Clock regime age**.

---

## 4. Current Active Architecture: Multi-Timeframe (`1d` + `4h` + `1h`)

### A. Multi-Timeframe Hierarchy for Freqtrade
To solve both the `1d` sample-size limitation and execution lag on `SOL_USDT`, the system uses a 3-layer hierarchy:

| Architectural Layer | Timeframe | Sample Size (`SOL_USDT`) | Primary Role | Feature Lookback Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **1. Macro Filter** | **`1d`** (`@informative('1d')`) | $\sim 1,800\text{ bars}$ | **Macro Tail-Risk & Exhaustion Filter:** Blocks long setups if `1d` is in `BEAR` (`State 0`) or if `1d BULL` age exceeds $80\%$ of its Expected Duration ($D_{1\text{d}}$). | `atr=14`, `sma=20` ($20\text{d}$), `impulse=3` ($3\text{d}$), `vol_base=90` ($90\text{d}$), `v_fast=5` ($5\text{d}$), `v_slow=30` ($30\text{d}$) |
| **2. Primary Regime Engine** | **`4h`** (`@informative('4h')`) | $\sim 11,000+\text{ bars}$ | **Active Regime & Bell Curve Engine:** Provides $6\times$ more training data for rock-solid Gaussian bell curves and catches trend transitions $6\times$ faster than `1d`. | `atr=14`, `sma=30` ($5\text{d}$), `impulse=6` ($24\text{h}$), `vol_base=180` ($30\text{d}$), `v_fast=12` ($2\text{d}$), `v_slow=90` ($15\text{d}$) |
| **3. Execution Base** | **`1h`** (or **`15m`**) | $\sim 44,000+\text{ bars}$ | **Precision Entry & Dynamic Risk:** Executes pullback entries when `4h` is `BULL` and manages trailing stops scaled by the active state's variance. | Uses `4h` Bell Curve Z-scores + `1h` local microstructure triggers. |

### B. Mathematical Specification of the 4 Stationary Features
For any timeframe with lookback configuration $\{L_{\text{atr}}, L_{\text{sma}}, L_{\text{imp}}, L_{\text{vol}}, L_{\text{vfast}}, L_{\text{vslow}}\}$:

1. **Trend ATR Distance (`trend_atr`):**
   $$\text{trend\_atr}_t = \frac{\text{Close}_t - \text{SMA}(\text{Close}, L_{\text{sma}})_t}{\text{ATR}(L_{\text{atr}})_t + 10^{-8}}$$
2. **ATR-Normalized Impulse (`impulse_atr`):**
   $$\text{impulse\_atr}_t = \frac{\text{Close}_t - \text{Close}_{t - L_{\text{imp}}}}{\text{ATR}(L_{\text{atr}})_t + 10^{-8}}$$
3. **NATR Volatility Expansion (`vol_expansion`):**
   $$\text{NATR}_t = \frac{\text{ATR}(L_{\text{atr}})_t}{\text{Close}_t + 10^{-8}}, \quad \text{vol\_expansion}_t = \ln\left(\frac{\text{NATR}_t + 10^{-8}}{\text{SMA}(\text{NATR}, L_{\text{vol}})_{t-1} + 10^{-8}}\right)$$
4. **EMA Volume Momentum (`vol_momentum`):**
   $$\text{vol\_momentum}_t = \ln\left(\frac{\text{EMA}(\text{Volume}, L_{\text{vfast}})_t + 10^{-8}}{\text{SMA}(\text{Volume}, L_{\text{vslow}})_{t-1} + 10^{-8}}\right)$$

### C. Bell Curve ($\mu_k, \sigma_k$) & Market Clock ($D_k$) Extraction
Each exported `.joblib` bundle contains precomputed statistical metadata for each canonical state $k \in \{0, 1, 2\}$:
* **Market Clock (Expected Regime Duration in Bars):**
  $$D_k = \frac{1}{1 - p_{kk}}$$
  where $p_{kk}$ is the diagonal persistence probability in `model.transmat_[k, k]`.
* **Bell Curve Marginal Parameters:**
  For each feature $f$ in state $k$, the bundle stores $\mu_{k, f} = \text{means\_}[k, f]$ and $\sigma_{k, f} = \sqrt{\text{covars\_}[k, f, f]}$.
* **Real-Time Bell Curve Z-Score:**
  At any candle $t$ where active state is $k$, the position of feature $f$ on that regime's Gaussian bell curve is:
  $$Z_{t, f}^{(k)} = \frac{x_{t, f} - \mu_{k, f}}{\sigma_{k, f}}$$

---

## 5. Conceptualized / Unfitted Components (Roadmap for Next Implementation Steps)

The following components have been mathematically designed and prepared in the `.joblib` bundle metadata, but **have not yet been wired or stress-tested inside a live Freqtrade `IStrategy` file**:

### 1. Freqtrade `IStrategy` Multi-Timeframe Wiring (Next Immediate Step)
* **Goal:** Create the Freqtrade strategy file (`SolHmmRegimeStrategy.py`) with `timeframe = '1h'`, loading `hmm_bundle_SOL_USDT_1d.joblib` inside `@informative('1d')` and `hmm_bundle_SOL_USDT_4h.joblib` inside `@informative('4h')`.
* **Requirement:** Must include a switch (`USE_OOS_MODEL = True/False`) so backtests after `split_date` automatically use `bundle["oos_model"]` while dry/live runs use `bundle["model"]`.

### 2. Bell Curve Z-Score Pullback Entry Logic (Unfitted / To Be Tuned via Hyperopt)
* **Concept:** Instead of buying blindly when `4h` enters `BULL` (`State 2`), use the `BULL` state's Gaussian bell curve (`bell_curve_stats[2]`) to avoid buying overextended blow-off tops ($Z_{\text{trend}} > +2.0$) and instead buy healthy pullbacks within the regime:
  * **Macro Gate (`1d`):** `hmm_state_1d != 0` (Not in Macro Bear) AND `hmm_prob_bear_1d < 0.40`.
  * **Primary Regime (`4h`):** `hmm_state_4h == 2` (`BULL`) AND `hmm_prob_bull_4h >= 0.70`.
  * **Bell Curve Value Zone (`4h`):** $-1.5 \le Z_{\text{trend\_atr}, 4\text{h}}^{(2)} \le +0.8$ and $Z_{\text{impulse\_atr}, 4\text{h}}^{(2)} > -1.0$ (price is in the lower-to-middle region of the Bull bell curve, not breaking down).
  * **Execution Trigger (`1h`):** Local `1h` momentum turn (e.g., `1h` close reclaiming `1h EMA(9)` or `1h impulse_atr` turning positive).

### 3. Market Clock Exhaustion & Patience Filters (To Be Tested)
* **Exhaustion Filter:** Track `regime_age_4h` (consecutive `4h` bars spent in `State 2`). When $\text{regime\_age\_4h} > 0.80 \cdot D_2$, block new long entries or tighten trailing stops.
* **Bear Patience Boundary:** When `4h` transitions into `BEAR` (`State 0`), block mean-reversion long entries until at least $0.50 \cdot D_0$ bars have elapsed.

### 4. Variance-Adaptive Dynamic Stop-Loss (`custom_stoploss`)
* **Concept:** Use the active `4h` state's `vol_expansion` mean and standard deviation ($\mu_{k, \text{vol}}, \sigma_{k, \text{vol}}$) inside Freqtrade's `custom_stoploss()` callback.
* When the market is in a wide-variance `BULL` expansion, allow a wider ATR stop ($2.5 \times \text{ATR}_{4\text{h}}$) so normal regime volatility doesn't prematurely stop out the position; when variance compresses toward `CHOP` (`State 1`), tighten the stop dynamically ($1.2 \times \text{ATR}_{4\text{h}}$).

### 5. Walk-Forward Retraining Automation
* **Concept:** Currently, `train_sol_hmm_mtf.py` is run offline to generate `.joblib` bundles. A future upgrade can wrap `fit_and_canonicalize_hmm()` into a weekly cron job or a custom `FreqAI` `IFreqaiModel` class so the HMM updates its covariance matrices on a rolling window without manual intervention.