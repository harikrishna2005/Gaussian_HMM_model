# How to run
# python visualize_4h_engine.py


# What to see
# audit_4h_engine_swings.html


#  ALGORITHM - TRADING STRATEGY INFORMATION
"""

Decoding the Color Mapping Disconnect

In your original 1D model, the background states match what your brain naturally expects from macro market cycles: broad, multi-week blocks of color where price drifts steadily.
On the 4-Hour Swing Engine, your model's parameters reveal a completely different microstructure behavioral profile:
• 🟢 State 2 (BULL - Shaded Green): The Impulsive Expansion Phase
	• The Math: Your 4H training log shows an Impulse ATR score of +0.93 and a positive Volume Momentum of +0.12.
	• What it looks like on the chart: These are not long, slow uptrends. These represent high-velocity, short-term momentum bursts where price aggressively jumps upward over 3 to 4 days, backed by a surge in buying volume.
• 🟡 State 1 (CHOP - Shaded Yellow): The Mean Reversion / Mean Equilibrium Phase
	• The Math: The Trend ATR score sits at +0.17, while Volume Momentum collapses to -0.14.
	• What it looks like on the chart: This is the most common intraday state. Price tightly winds or mean-reverts around the 5-day baseline, volume dries up completely, and indicators lose directional momentum.
• 🔴 State 0 (BEAR - Shaded Red): The Liquidation / Pullback Phase
	• The Math: The model clocked a deeply negative Trend ATR score of -1.72 and an aggressive Impulse ATR drop of -0.69.
	• What it looks like on the chart: These represent sharp, swift intraday sell-offs, stop-hunts, or deeper multi-day market corrections.

How to Use the Oscillating Orange Z-Score Line

To clear up any remaining confusion, bypass the background colors on the 4H chart completely and look straight at the Orange Z-Score Oscillator line in your bottom subplot.
This line is the definitive guide for your strategy execution layer:
                  [ 4H BULL STATE BELL CURVE DISTRIBUTION ]

                                 Sweet Spot
                                 Entry Zone
                              ┌──────────────┐
  Blow-Off Top  ──────────────┼───────┐      │   (+0.20 Z-Score)
                              │       │      │
                              │       ▼      │
  Center of Bell Curve ───────┼───────●──────┼─── ( 0.00 Z-Score)
                              │              │
                              │       ▲      │
  Oversold Value Dip ─────────┼───────┘      │   (-1.20 Z-Score)
                              └──────────────┘
1. When the Orange Line is ABOVE +0.20: Price is over-extended in a minor blow-off phase (too far to the right side of the Gaussian bell curve). The strategy blocks long entries here to keep you from chasing the top.
2. When the Orange Line dips BETWEEN -1.20 and +0.20: This is the pullback sweet spot. The 4H environment is structurally bullish, but price has temporarily dipped into the lower half of the expected distribution. This is where your 1-hour RSI trigger is allowed to execute entries.
3. When the Orange Line drops BELOW -1.20: Price has dropped too far, indicating that the 4-hour intraday markup structure is failing. The strategy freezes entries because the move is no longer a standard pullback.
Now that the behavioral meaning of the 4H swing colors is clarified, you are fully prepared to test the system. Please upload the text or screenshots of your backtest results summary table once your Multi-Timeframe run completes!

"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATHER_PATH = os.path.join(BASE_DIR, "data", "bybit", "SOL_USDT-4h.feather")
JOBLIB_PATH = os.path.join(BASE_DIR, "models", "hmm_bundle_SOL_USDT_4h.joblib")
OUTPUT_HTML = os.path.join(BASE_DIR, "audit_4h_engine_swings.html")

if not os.path.exists(FEATHER_PATH) or not os.path.exists(JOBLIB_PATH):
    print("❌ Critical Error: Verify your data/bybit and models paths match your script folder.")
    sys.exit(1)

print("📦 Loading 4H Engine Bundle...")
bundle = joblib.load(JOBLIB_PATH)
model = bundle["model"]
cfg = bundle["config"]
stats = bundle["bell_curve_stats"]
regimes = bundle["regime_labels"]

df = pd.read_feather(FEATHER_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("⚙️ Processing intraday features...")
prev_close = df['close'].shift(1)
tr = pd.concat([df['high'] - df['low'], (df['high'] - prev_close).abs(), (df['low'] - prev_close).abs()], axis=1).max(axis=1)
atr = tr.rolling(cfg["atr_len"]).mean()
sma = df['close'].rolling(cfg["sma_len"]).mean()

df['trend_atr'] = (df['close'] - sma) / (atr + 1e-8)
df['impulse_atr'] = (df['close'] - df['close'].shift(cfg["impulse_len"])) / (atr + 1e-8)
natr = atr / (df['close'] + 1e-8)
df['vol_expansion'] = np.log((natr + 1e-8) / (natr.shift(1).rolling(cfg["vol_base_len"]).mean() + 1e-8))
df['vol_momentum'] = np.log((df['volume'].ewm(span=cfg["v_fast_len"], adjust=False).mean() + 1e-8) / (df['volume'].shift(1).rolling(cfg["v_slow_len"]).mean() + 1e-8))

clean_df = df.dropna(subset=bundle["feature_cols"]).copy().reset_index(drop=True)
clean_df['state'] = model.predict(clean_df[bundle["feature_cols"]].values)

# Target state 2 (BULL) explicitly using our dictionary index configuration maps
bull_idx = regimes["bull"]
mu_trend = stats[bull_idx]["mean"]["trend_atr"]
sigma_trend = stats[bull_idx]["std"]["trend_atr"]
clean_df['z_score'] = (clean_df['trend_atr'] - mu_trend) / (sigma_trend + 1e-8)

color_map = {
    regimes["bear"]: {"name": "Bear Swing", "color": "rgba(239, 83, 80, 0.06)"},
    regimes["chop"]: {"name": "Chop Range", "color": "rgba(255, 235, 59, 0.04)"},
    regimes["bull"]: {"name": "Bull Markup", "color": "rgba(38, 166, 154, 0.06)"}
}

print("🎨 Structuring multi-subplot graphics engine...")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)

# Subplot 1: Candlesticks
fig.add_trace(go.Candlestick(
    x=clean_df['date'], open=clean_df['open'], high=clean_df['high'], low=clean_df['low'], close=clean_df['close'],
    name="SOL 4H"
), row=1, col=1)

print("⚡ Vectorizing background shape arrays to bypass loops...")
shapes = []
clean_df['change'] = clean_df['state'].ne(clean_df['state'].shift()).cumsum()

for _, group in clean_df.groupby('change'):
    state = int(group['state'].iloc[0]) # FIXED: Explicit row-zero indexing
    shapes.append(dict(
        type="rect",
        xref="x", yref="paper",
        x0=group['date'].iloc[0], x1=group['date'].iloc[-1],
        y0=0, y1=1,
        fillcolor=color_map[state]["color"],
        opacity=1,
        layer="below",
        line_width=0
    ))

# Subplot 2: Dynamic Bell Curve Z-score Trailing Oscillations
fig.add_trace(go.Scatter(
    x=clean_df['date'], y=clean_df['z_score'], name="Bull Curve Z-Score", line=dict(color='#ff9900', width=1.5)
), row=2, col=1)

# Add clear upper and lower boundary thresholds for the pullback "Sweet Spot" entry zone (-1.2 to +0.2)
fig.add_hline(y=0.2, line_dash="dash", line_color="rgba(255,255,255,0.2)", row=2, col=1)
fig.add_hline(y=-1.2, line_dash="dash", line_color="rgba(0,255,0,0.4)", row=2, col=1)

# Inject shapes all at once
fig.update_layout(shapes=shapes)

fig.update_layout(
    title="<b>SOL/USDT 4-Hour Swing Engine & Bell Curve Tracker</b>",
    template="plotly_dark",
    xaxis_rangeslider_visible=False
)

fig.write_html(OUTPUT_HTML)
print(f"✨ Success! 4H Engine audit file safely generated at: {OUTPUT_HTML}")
