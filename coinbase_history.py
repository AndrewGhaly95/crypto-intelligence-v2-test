
"""Paginated Coinbase Exchange OHLCV history.

Research only. No orders or account access.
Prices are quoted in USD, not AUD.
"""

import time
import pandas as pd
import requests


BASE_URL = "https://api.exchange.coinbase.com"

INTERVALS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}

COLUMNS = ["open", "high", "low", "close", "volume"]


def get_coinbase_history(
    symbol="BTC",
    interval="1h",
    days=90,
    max_requests=150,
):
    """Download completed historical candles in pages."""

    symbol = symbol.upper().strip()

    if not symbol.isalpha():
        raise ValueError("Invalid cryptocurrency symbol")

    if interval not in INTERVALS:
        raise ValueError("Unsupported interval")

    if not isinstance(days, int) or not 1 <= days <= 180:
        raise ValueError("Days must be between 1 and 180")

    seconds = INTERVALS[interval]

    # Use the most recent completed candle boundary.
    now = pd.Timestamp.now(tz="UTC")
    end = pd.Timestamp(
        (int(now.timestamp()) // seconds) * seconds,
        unit="s",
        tz="UTC",
    )

    start = end - pd.Timedelta(days=days)

    # Keep pages below the 300-candle API limit.
    page_size = 250
    page_span = pd.Timedelta(seconds=page_size * seconds)

    estimated_requests = (
        (end - start) / page_span
    )

    if estimated_requests > max_requests:
        raise ValueError(
            "Requested history exceeds the request budget. "
            "Use fewer days or a longer interval."
        )

    url = (
        f"{BASE_URL}/products/"
        f"{symbol}-USD/candles"
    )

    session = requests.Session()
    session.headers.update({
        "User-Agent": "CryptoIntelligenceResearch/2.1",
        "Accept": "application/json",
    })

    frames = []
    cursor = start

    try:
        while cursor < end:
            page_end = min(cursor + page_span, end)

            params = {
                "start": cursor.isoformat(),
                "end": page_end.isoformat(),
                "granularity": seconds,
            }

            for attempt in range(4):
                response = session.get(
                    url,
                    params=params,
                    timeout=20,
                )

                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt == 3:
                        response.raise_for_status()

                    time.sleep(2 ** attempt)
                    continue

                response.raise_for_status()
                break

            payload = response.json()

            if not isinstance(payload, list):
                raise ValueError(
                    "Unexpected Coinbase API response"
                )

            if payload:
                # Coinbase order:
                # time, low, high, open, close, volume
                page = pd.DataFrame(
                    payload,
                    columns=[
                        "timestamp",
                        "low",
                        "high",
                        "open",
                        "close",
                        "volume",
                    ],
                )

                page["timestamp"] = pd.to_datetime(
                    page["timestamp"],
                    unit="s",
                    utc=True,
                )

                frames.append(page)

            cursor = page_end

            # Avoid aggressive requests.
            time.sleep(0.15)

    finally:
        session.close()

    if not frames:
        raise ValueError(
            "No historical candles returned. "
            "Check the Coinbase trading pair."
        )

    data = pd.concat(frames, ignore_index=True)

    for column in COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    data = data.set_index("timestamp")

    data = data[
        ~data.index.duplicated(keep="last")
    ].sort_index()

    data = data[
        (data.index >= start)
        & (data.index < end)
    ]

    data = data[COLUMNS]

    if data.empty:
        raise ValueError("No completed candles available")

    if data.isna().any().any():
        raise ValueError("Missing OHLCV values")

    if (
        (data[["open", "high", "low", "close"]] <= 0)
        .any()
        .any()
        or (data["volume"] < 0).any()
    ):
        raise ValueError("Invalid market values")

    if (
        (
            data["high"]
            < data[["open", "close", "low"]].max(axis=1)
        ).any()
        or (
            data["low"]
            > data[["open", "close", "high"]].min(axis=1)
        ).any()
    ):
        raise ValueError("Invalid OHLC relationships")

    expected = pd.date_range(
        start=start,
        end=end - pd.Timedelta(seconds=seconds),
        freq=pd.Timedelta(seconds=seconds),
    )

    missing = expected.difference(data.index)

    data.attrs["source"] = "Coinbase Exchange"
    data.attrs["quote_currency"] = "USD"
    data.attrs["requested_start"] = start.isoformat()
    data.attrs["requested_end"] = end.isoformat()
    data.attrs["missing_candles"] = len(missing)
    data.attrs["expected_candles"] = len(expected)

    return data
