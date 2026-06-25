"""
LONG/SHORT TRADING SIGNAL GENERATOR
=====================================
A reusable script that generates buy/sell/hold signals (1/-1/0) for ANY
price dataset using a combination of technical indicators:
  - SMA (20/50/200) trend filters
  - Bollinger Bands (mean reversion)
  - RSI (overbought/oversold)
  - 20-day Breakout (momentum)
  - Z-score (statistical extremes)
  - Stop-loss / Take-profit risk management

HOW TO USE:
1. Update the CONFIG section below (file path, column names, date format)
2. Run the script
3. Output: signals.csv with columns [Date, Signal] where:
     1  = go/stay LONG
    -1  = go/stay SHORT
     0  = stay FLAT (no position)

REQUIREMENTS: pandas, numpy
    pip install pandas numpy
"""

import pandas as pd
import numpy as np

# =========================================================================
# CONFIG — EDIT THESE FOR YOUR DATASET
# =========================================================================
INPUT_FILE = "Nifty 50 Historical Data.csv"          # path to your CSV file
OUTPUT_FILE = "signals.csv"           # output file name

DATE_COLUMN = "Date"                  # name of the date column in your CSV
PRICE_COLUMN = "Price"                # name of the closing-price column
DATE_FORMAT = "%d-%m-%Y"              # format of dates in your CSV (e.g. "%Y-%m-%d" for 2024-01-15)

# Strategy parameters (tune these as needed)
SMA_SHORT = 20
SMA_MED = 50
SMA_LONG = 200          # trend filter window
BB_PERIOD = 20           # Bollinger Band lookback
BB_STD_MULT = 2.0        # Bollinger Band std-dev multiplier
RSI_PERIOD = 14
BREAKOUT_LOOKBACK = 20   # breakout high/low lookback window
STOP_LOSS_PCT = 0.10     # 10% stop-loss
TAKE_PROFIT_PCT = 0.20   # 20% take-profit
ZSCORE_EXTREME = 2.0     # Z-score threshold for mean-reversion exit
RSI_LONG_FILTER = 70     # don't go long if RSI above this (avoid overbought entries)
RSI_SHORT_FILTER = 30    # don't go short if RSI below this (avoid oversold entries)

# =========================================================================
# LOAD DATA
# =========================================================================
df = pd.read_csv(INPUT_FILE)

# Clean numeric columns that may contain commas (e.g. "24,812.50" -> 24812.50)
if not pd.api.types.is_numeric_dtype(df[PRICE_COLUMN]):
    df[PRICE_COLUMN] = df[PRICE_COLUMN].astype(str).str.replace(',', '').astype(float)

df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT)
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

close = df[PRICE_COLUMN]

# =========================================================================
# INDICATOR FUNCTIONS
# =========================================================================

def calculate_sma(series, period):
    """Simple Moving Average"""
    return series.rolling(period).mean()


def calculate_bollinger_bands(series, period=20, std_multiplier=2.0):
    """Returns (upper_band, middle_band, lower_band)"""
    middle_band = calculate_sma(series, period)
    std_dev = series.rolling(period).std(ddof=1)
    upper_band = middle_band + (std_multiplier * std_dev)
    lower_band = middle_band - (std_multiplier * std_dev)
    return upper_band, middle_band, lower_band


def calculate_rsi(series, period=14):
    """Relative Strength Index using Wilder's smoothing approximation (simple rolling mean version)"""
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_zscore(series, window=20):
    """Z-score: how many std-devs the price is from its rolling mean"""
    moving_avg = calculate_sma(series, window)
    moving_std = series.rolling(window).std(ddof=1)
    zscore = (series - moving_avg) / moving_std
    return zscore


def calculate_breakout_levels(series, lookback=20):
    """Rolling high/low for breakout detection"""
    recent_high = series.rolling(lookback).max()
    recent_low = series.rolling(lookback).min()
    return recent_high, recent_low


def calculate_momentum(series, lookback=5):
    """Simple N-day momentum (price change over lookback period)"""
    return series - series.shift(lookback)


# =========================================================================
# COMPUTE ALL INDICATORS
# =========================================================================
df['SMA_short'] = calculate_sma(close, SMA_SHORT)
df['SMA_med'] = calculate_sma(close, SMA_MED)
df['SMA_long'] = calculate_sma(close, SMA_LONG)

df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = calculate_bollinger_bands(
    close, BB_PERIOD, BB_STD_MULT
)

df['RSI'] = calculate_rsi(close, RSI_PERIOD)
df['Z_Score'] = calculate_zscore(close, BB_PERIOD)
df['Recent_High'], df['Recent_Low'] = calculate_breakout_levels(close, BREAKOUT_LOOKBACK)
df['Momentum'] = calculate_momentum(close, lookback=5)

# =========================================================================
# SIGNAL GENERATION LOGIC
# =========================================================================
# LONG Entry:  (Breakout above recent high OR price <= lower BB w/ RSI < filter)
#              AND price > SMA_long (uptrend context)
# SHORT Entry: (Breakout below recent low OR price >= upper BB w/ RSI > filter)
#              AND price < SMA_long (downtrend context)
#
# LONG Exit:   stop-loss hit, OR take-profit hit, OR SMA_short < SMA_med (trend break),
#              OR Z-score > +extreme (overbought)
# SHORT Exit:  stop-loss hit, OR take-profit hit, OR SMA_short > SMA_med (trend break),
#              OR Z-score < -extreme (oversold)

warmup_period = max(SMA_LONG, BB_PERIOD, RSI_PERIOD, BREAKOUT_LOOKBACK)

position = 0       # 0 = flat, 1 = long, -1 = short
entry_price = 0
signals = []

for i in range(len(df)):
    if i < warmup_period or pd.isna(df['SMA_long'].iloc[i]):
        signals.append(0)
        continue

    price = close.iloc[i]
    sma_short = df['SMA_short'].iloc[i]
    sma_med = df['SMA_med'].iloc[i]
    sma_long = df['SMA_long'].iloc[i]
    bb_upper = df['BB_Upper'].iloc[i]
    bb_lower = df['BB_Lower'].iloc[i]
    rsi = df['RSI'].iloc[i]
    z_score = df['Z_Score'].iloc[i]
    # use previous day's breakout levels to avoid lookahead bias
    recent_high = df['Recent_High'].iloc[i - 1]
    recent_low = df['Recent_Low'].iloc[i - 1]

    if position == 0:
        long_breakout = price > recent_high
        long_bb_touch = price <= bb_lower and rsi < RSI_LONG_FILTER
        long_uptrend = price > sma_long

        short_breakout = price < recent_low
        short_bb_touch = price >= bb_upper and rsi > RSI_SHORT_FILTER
        short_downtrend = price < sma_long

        if long_uptrend and (long_breakout or long_bb_touch):
            position = 1
            entry_price = price
        elif short_downtrend and (short_breakout or short_bb_touch):
            position = -1
            entry_price = price

    elif position == 1:  # currently LONG
        stop_loss_price = entry_price * (1 - STOP_LOSS_PCT)
        take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)

        hit_stop = price <= stop_loss_price
        hit_target = price >= take_profit_price
        trend_break = sma_short < sma_med
        overbought = z_score > ZSCORE_EXTREME

        if hit_stop or hit_target or trend_break or overbought:
            position = 0

    elif position == -1:  # currently SHORT
        stop_loss_price = entry_price * (1 + STOP_LOSS_PCT)
        take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)

        hit_stop = price >= stop_loss_price
        hit_target = price <= take_profit_price
        trend_break = sma_short > sma_med
        oversold = z_score < -ZSCORE_EXTREME

        if hit_stop or hit_target or trend_break or oversold:
            position = 0

    signals.append(position)

df['Signal'] = signals

# =========================================================================
# SAVE OUTPUT
# =========================================================================
out = df[[DATE_COLUMN, 'Signal']].copy()
out[DATE_COLUMN] = out[DATE_COLUMN].dt.strftime(DATE_FORMAT)
out.to_csv(OUTPUT_FILE, index=False)

print(f"Signals saved to: {OUTPUT_FILE}")
print(f"\nSignal distribution:\n{out['Signal'].value_counts().sort_index()}")

# =========================================================================
# OPTIONAL: QUICK BACKTEST SUMMARY
# =========================================================================
df['Returns'] = close.pct_change()
df['StratReturns'] = df['Returns'] * df['Signal'].shift(1)

total_return = (1 + df['StratReturns'].fillna(0)).prod() - 1
bh_return = (1 + df['Returns'].fillna(0)).prod() - 1

strat_equity = (1 + df['StratReturns'].fillna(0)).cumprod()
bh_equity = (1 + df['Returns'].fillna(0)).cumprod()
strat_dd = (strat_equity / strat_equity.cummax() - 1).min()
bh_dd = (bh_equity / bh_equity.cummax() - 1).min()

print(f"\n=== BACKTEST SUMMARY ===")
print(f"Strategy total return:   {total_return*100:.2f}%")
print(f"Buy & Hold total return: {bh_return*100:.2f}%")
print(f"Strategy max drawdown:   {strat_dd*100:.2f}%")
print(f"Buy & Hold max drawdown: {bh_dd*100:.2f}%")
