#  How to run
# python visualize_1d_macro.py


# Output
# src\audit_1d_macro_regimes.html

# What to verify
# In audit_1d_macro_regimes.html: Verify that the background bands track large, multi-week blocks of color. Look closely at the major historical bull cycles—the green zones should lock in early and persist cleanly without toggling back and forth rapidly.

import os
import sys
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Local Relative Path Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATHER_PATH = os.path.join(BASE_DIR, "data", "bybit", "SOL_USDT-1d.feather")
JOBLIB_PATH = os.path.join(BASE_DIR, "models", "hmm_bundle_SOL_USDT_1d.joblib")
OUTPUT_HTML = os.path.join(BASE_DIR, "audit_1d_macro_regimes.html")

if not os.path.exists(FEATHER_PATH) or not os.path.exists(JOBLIB_PATH):
    print("❌ Critical Error: Verify your data/bybit and models paths match your script folder.")
    sys.exit(1)

print("📦 Loading 1D Canonical Bundle...")
bundle = joblib.load(JOBLIB_PATH)
model = bundle["model"]
cfg = bundle["config"]

df = pd.read_feather(FEATHER_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("⚙️ Building stationary features...")
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

# Canonical Reordering enforced: 0=BEAR, 1=CHOP, 2=BULL
color_map = {
    0: {"name": "🐻 BEAR REGIME", "color": "rgba(255, 0, 0, 0.08)"},
    1: {"name": "⚡ CHOP / RANGE", "color": "rgba(255, 255, 0, 0.06)"},
    2: {"name": "🐂 BULL MARKUP", "color": "rgba(0, 255, 0, 0.08)"}
}

print("🎨 Rendering chart canvas...")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=clean_df['date'], open=clean_df['open'], high=clean_df['high'], low=clean_df['low'], close=clean_df['close'],
    name="SOL Daily"
))

clean_df['change'] = clean_df['state'].ne(clean_df['state'].shift()).cumsum()
for _, group in clean_df.groupby('change'):
    state = int(group['state'].iloc[0])
    fig.add_vrect(
        x0=group['date'].iloc[0], x1=group['date'].iloc[-1],
        fillcolor=color_map[state]["color"], opacity=1, layer="below", line_width=0,
        annotation_text=color_map[state]["name"] if len(group) > 8 else ""
    )

fig.update_layout(title="<b>SOL/USDT 1-Day Macro HMM Audit</b>", template="plotly_dark", xaxis_rangeslider_visible=False)
fig.write_html(OUTPUT_HTML)
print(f"✨ Macro audit file safely generated at: {OUTPUT_HTML}")
