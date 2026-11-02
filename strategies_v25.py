
"""Research-only, close-confirmed signals for four long-only strategies.

Signals at candle t are eligible for execution no earlier than candle t+1 open.
No live orders, exchange integration, or portfolio accounting.
"""
import numpy as np
import pandas as pd

STRATEGIES = (
    "EMA crossover",
    "EMA + RSI",
    "Volume breakout",
    "EMA + ATR trailing exit",
)


def _validate(df):
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected timestamp-indexed OHLCV DataFrame")
    if df.index.tz is None or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError("Index must be timezone-aware, unique and ascending")
    columns = ("open", "high", "low", "close", "volume")
    if not set(columns).issubset(df.columns):
        raise ValueError("Missing OHLCV columns")
    out = df.loc[:, columns].astype(float).copy()
    if len(out) < 220 or not np.isfinite(out.to_numpy()).all():
        raise ValueError("Need >=220 finite OHLCV candles")
    if (out[["open", "high", "low", "close"]] <= 0).any().any() or (out.volume < 0).any():
        raise ValueError("Invalid prices or volume")
    if (out.high < out[["open", "low", "close"]].max(axis=1)).any() or (out.low > out[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Inconsistent OHLC")
    return out


def _rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    denominator = avg_up + avg_down
    rsi = 100 * avg_up / denominator.replace(0, np.nan)
    rsi = rsi.mask((avg_down == 0) & (avg_up > 0), 100)
    rsi = rsi.mask((avg_up == 0) & (avg_down > 0), 0)
    return rsi.fillna(50)


def _atr(df, period=14):
    previous = df.close.shift(1)
    true_range = pd.concat([
        df.high - df.low,
        (df.high - previous).abs(),
        (df.low - previous).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def generate_signals(frame, strategy, warmup=200):
    """Return OHLCV plus boolean entry/exit signals (flat-to-long state machine).

    All decisions use information available at the close of the signal candle.
    Trailing stop is evaluated on the CLOSE, not as an intrabar stop fill.
    """
    df = _validate(frame)
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if not isinstance(warmup, int) or warmup < 200 or warmup >= len(df)-1:
        raise ValueError("Warmup must be >=200 and leave >=2 candles")

    close = df.close
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi = _rsi(close)
    atr = _atr(df)
    prev_high = df.high.shift(1).rolling(20, min_periods=20).max()
    prev_volume = df.volume.shift(1).rolling(20, min_periods=20).mean()
    bullish = ema20 > ema50
    crossover = bullish & ~bullish.shift(1, fill_value=False)
    crossunder = ~bullish & bullish.shift(1, fill_value=False)

    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    position = False
    highest_close = np.nan
    for i in range(warmup, len(df)):
        if strategy == "EMA crossover":
            enter = bool(crossover.iloc[i])
            leave = bool(crossunder.iloc[i])
        elif strategy == "EMA + RSI":
            enter = bool(
                bullish.iloc[i] and close.iloc[i] > ema200.iloc[i]
                and 50 < rsi.iloc[i] < 70
                and (rsi.iloc[i-1] <= 50 or crossover.iloc[i])
            )
            leave = bool(rsi.iloc[i] < 45 or not bullish.iloc[i])
        elif strategy == "Volume breakout":
            enter = bool(
                close.iloc[i] > prev_high.iloc[i]
                and df.volume.iloc[i] > 1.5 * prev_volume.iloc[i]
                and close.iloc[i] > ema200.iloc[i]
            )
            leave = bool(close.iloc[i] < ema20.iloc[i])
        else:
            enter = bool(crossover.iloc[i] and close.iloc[i] > ema200.iloc[i])
            if position:
                highest_close = max(highest_close, float(close.iloc[i]))
            stop = highest_close - 3 * atr.iloc[i] if position else np.nan
            leave = bool(position and (
                close.iloc[i] < stop or not bullish.iloc[i]
            ))

        if position:
            if leave:
                exits.iloc[i] = True
                position = False
                highest_close = np.nan
        elif enter:
            entries.iloc[i] = True
            position = True
            highest_close = float(close.iloc[i])

    result = df.copy()
    result["entry"] = entries
    result["exit"] = exits
    return result
