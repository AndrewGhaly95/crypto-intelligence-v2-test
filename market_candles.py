
"""Historical candle retrieval for Crypto Intelligence V2."""

import pandas as pd
import requests


KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"

SUPPORTED_COINS = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "XRP": "XRPUSD",
    "BNB": "BNBUSD",
}

SUPPORTED_INTERVALS = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}


def get_candles(symbol, interval="1h", limit=500):
    """Retrieve completed OHLCV candles from Kraken."""

    symbol = symbol.upper().strip()

    if symbol not in SUPPORTED_COINS:
        raise ValueError(f"Unsupported coin: {symbol}")

    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")

    if not isinstance(limit, int) or not 1 <= limit <= 719:
        raise ValueError("Limit must be between 1 and 719.")

    response = requests.get(
        KRAKEN_URL,
        params={
            "pair": SUPPORTED_COINS[symbol],
            "interval": SUPPORTED_INTERVALS[interval],
        },
        timeout=20,
    )

    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise ValueError(
            f"Kraken API error: {payload['error']}"
        )

    results = payload.get("result", {})

    pair_keys = [
        key for key in results
        if key != "last"
    ]

    if not pair_keys:
        raise ValueError("No market data returned.")

    candles = results[pair_keys[0]]

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "count",
    ]

    data = pd.DataFrame(candles, columns=columns)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        unit="s",
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    # Kraken includes the current unfinished candle.
    # Remove it to prevent look-ahead bias.
    data = data.iloc[:-1].copy()

    if data.empty:
        raise ValueError("No completed candles available.")

    data = data.set_index("timestamp")

    return data[
        ["open", "high", "low", "close", "volume"]
    ].sort_index().tail(limit)

