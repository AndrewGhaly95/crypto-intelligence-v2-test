
"""Trading signal generation for Crypto Intelligence V2."""

import numpy as np
import pandas as pd


def prepare_candles(df):
    """Validate and prepare historical OHLCV candles."""

    required = ["open", "high", "low", "close", "volume"]

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Candles must be a pandas DataFrame.")

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing candle columns: {missing}")

    data = df.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Candles must have a DatetimeIndex.")

    if data.index.tz is None:
        raise ValueError("Candle timestamps must be timezone-aware.")

    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError("Duplicate candle timestamps detected.")

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=required)

    if data.empty:
        raise ValueError("No valid candles available.")

    if (
        (data[["open", "high", "low", "close"]] <= 0).any().any()
        or (data["volume"] < 0).any()
    ):
        raise ValueError("Invalid price or volume values.")

    if (
        (data["high"] < data[["open", "close", "low"]].max(axis=1)).any()
        or (data["low"] > data[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise ValueError("Inconsistent OHLC candle values.")

    return data


def signals(df, strategy="Scalping"):
    """Return entry and exit signals using closed candles."""

    data = prepare_candles(df)

    close = data["close"]
    volume = data["volume"]

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    if strategy == "Scalping":
        typical_price = (
            data["high"] + data["low"] + close
        ) / 3.0

        rolling_pv = (
            typical_price * volume
        ).rolling(20, min_periods=20).sum()

        rolling_volume = volume.rolling(
            20, min_periods=20
        ).sum()

        vwap = rolling_pv / rolling_volume.replace(0, np.nan)

        average_volume = volume.shift(1).rolling(
            20, min_periods=20
        ).mean()

        previous_high = data["high"].shift(1).rolling(
            10, min_periods=10
        ).max()

        entry = (
            (close > vwap)
            & (ema9 > ema21)
            & (volume > average_volume * 1.2)
            & (close > previous_high)
        )

        exit_signal = (
            (close < vwap)
            | (ema9 < ema21)
        )

    elif strategy == "Intraday":
        uptrend = ema20 > ema50

        pullback_recovery = (
            (close.shift(1) <= ema20.shift(1))
            & (close > ema20)
        )

        bullish_crossover = (
            (ema20 > ema50)
            & (ema20.shift(1) <= ema50.shift(1))
        )

        entry = uptrend & (
            pullback_recovery | bullish_crossover
        )

        exit_signal = (
            (close < ema20)
            | (ema20 < ema50)
        )

    else:
        raise ValueError(
            "Strategy must be 'Scalping' or 'Intraday'."
        )

    result = data.copy()

    result["entry"] = entry.fillna(False).astype(bool)
    result["exit"] = exit_signal.fillna(False).astype(bool)

    return result
