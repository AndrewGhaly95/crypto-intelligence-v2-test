
"""Historical market candle retrieval for Crypto Intelligence V2."""

import pandas as pd
import requests


BINANCE_URL = "https://api.binance.com/api/v3/klines"

SUPPORTED_COINS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BNB": "BNBUSDT",
}

SUPPORTED_INTERVALS = {
    "1m",
    "5m",
    "15m",
    "1h",
    "4h",
}


def get_candles(symbol, interval="1h", limit=500):
    """
    Retrieve historical OHLCV candles from Binance.

    Returns a pandas DataFrame with UTC timestamps.
    The latest candle is excluded because it may
    still be forming.
    """

    symbol = symbol.upper().strip()

    if symbol not in SUPPORTED_COINS:
        raise ValueError(
            f"Unsupported cryptocurrency: {symbol}"
        )

    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported candle interval: {interval}"
        )

    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError(
            "Candle limit must be between 1 and 1000."
        )

    params = {
        "symbol": SUPPORTED_COINS[symbol],
        "interval": interval,
        "limit": min(limit + 1, 1000),
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=15,
    )

    response.raise_for_status()
    candles = response.json()

    if not isinstance(candles, list) or not candles:
        raise ValueError(
            "No valid market candle data returned."
        )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    data = pd.DataFrame(candles, columns=columns)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    data["timestamp"] = pd.to_datetime(
        data["open_time"],
        unit="ms",
        utc=True,
    )

    data["close_timestamp"] = pd.to_datetime(
        data["close_time"],
        unit="ms",
        utc=True,
    )

    now = pd.Timestamp.now(tz="UTC")

    # Only retain candles whose closing time has passed.
    data = data[
        data["close_timestamp"] < now
    ].copy()

    data = data.set_index("timestamp")

    data = data[
        numeric_columns
    ].sort_index()

    if data.empty:
        raise ValueError(
            "No completed candles were available."
        )

    return data.tail(limit)
