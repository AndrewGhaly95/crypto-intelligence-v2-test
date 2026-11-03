

"""Coinbase Exchange historical OHLCV with gap recovery.

Research only. No orders or account access.
Prices are quoted in USD, not AUD.

Missing candles are retried but never fabricated.
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


def _fetch_page(session, url, start, end, seconds):
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "granularity": seconds,
    }

    for attempt in range(4):
        try:
            response = session.get(
                url,
                params=params,
                timeout=20,
            )

            if response.status_code in (
                429, 500, 502, 503, 504
            ):
                response.raise_for_status()

            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, list):
                raise ValueError(
                    "Unexpected Coinbase API response"
                )

            if not payload:
                return None

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

            return page

        except (
            requests.RequestException,
            ValueError,
        ):
            if attempt == 3:
                raise

            time.sleep(min(2 ** attempt, 8))

    return None


def _normalise(frames, start, end):
    data = pd.concat(
        frames,
        ignore_index=True,
    )

    for column in COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    data = (
        data.set_index("timestamp")
        .sort_index()
    )

    data = data[
        ~data.index.duplicated(keep="last")
    ]

    data = data[
        (data.index >= start)
        & (data.index < end)
    ]

    data = data[COLUMNS]

    if data.empty:
        raise ValueError(
            "No completed historical candles available"
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Historical timestamps are not ascending"
        )

    if data.isna().any().any():
        raise ValueError(
            "Missing OHLCV values"
        )

    prices = data[
        ["open", "high", "low", "close"]
    ]

    if (
        (prices <= 0).any().any()
        or (data["volume"] < 0).any()
    ):
        raise ValueError(
            "Invalid price or volume values"
        )

    if (
        (
            data["high"]
            < data[
                ["open", "low", "close"]
            ].max(axis=1)
        ).any()
        or (
            data["low"]
            > data[
                ["open", "high", "close"]
            ].min(axis=1)
        ).any()
    ):
        raise ValueError(
            "Invalid OHLC relationships"
        )

    return data


def get_coinbase_history(
    symbol="BTC",
    interval="1h",
    days=90,
    max_requests=150,
    repair_gaps=True,
    max_repair_requests=30,
):
    """Download completed candles and retry missing intervals."""

    symbol = symbol.upper().strip()

    if not symbol.isalpha():
        raise ValueError(
            "Invalid cryptocurrency symbol"
        )

    if interval not in INTERVALS:
        raise ValueError(
            "Unsupported interval"
        )

    if (
        not isinstance(days, int)
        or not 1 <= days <= 180
    ):
        raise ValueError(
            "Days must be between 1 and 180"
        )

    if (
        not isinstance(max_requests, int)
        or max_requests < 1
        or not isinstance(max_repair_requests, int)
        or max_repair_requests < 0
    ):
        raise ValueError(
            "Invalid API request limits"
        )

    seconds = INTERVALS[interval]

    now = pd.Timestamp.now(tz="UTC")

    end = pd.Timestamp(
        (int(now.timestamp()) // seconds) * seconds,
        unit="s",
        tz="UTC",
    )

    start = end - pd.Timedelta(days=days)

    page_size = 250

    page_span = pd.Timedelta(
        seconds=page_size * seconds
    )

    estimated_requests = int(
        (
            (end - start) / page_span
        ).__ceil__()
    )

    if estimated_requests > max_requests:
        raise ValueError(
            "Requested history exceeds the "
            "API request budget"
        )

    url = (
        f"{BASE_URL}/products/"
        f"{symbol}-USD/candles"
    )

    expected = pd.date_range(
        start=start,
        end=end - pd.Timedelta(seconds=seconds),
        freq=pd.Timedelta(seconds=seconds),
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent":
            "CryptoIntelligenceResearch/2.7",
        "Accept": "application/json",
    })

    frames = []
    cursor = start
    pages_requested = 0
    repair_requests = 0

    try:
        # Initial paginated retrieval.
        while cursor < end:
            page_end = min(
                cursor + page_span,
                end,
            )

            page = _fetch_page(
                session,
                url,
                cursor,
                page_end,
                seconds,
            )

            pages_requested += 1

            if page is not None:
                frames.append(page)

            cursor = page_end
            time.sleep(0.15)

        if not frames:
            raise ValueError(
                "Coinbase returned no historical candles"
            )

        data = _normalise(
            frames,
            start,
            end,
        )

        initial_missing = expected.difference(
            data.index
        )

        # Retry missing candle intervals.
        if repair_gaps and len(initial_missing):
            for timestamp in initial_missing[
                :max_repair_requests
            ]:
                repair_end = (
                    timestamp
                    + pd.Timedelta(seconds=seconds)
                )

                page = _fetch_page(
                    session,
                    url,
                    timestamp,
                    repair_end,
                    seconds,
                )

                repair_requests += 1

                if page is not None:
                    frames.append(page)

                time.sleep(0.15)

            data = _normalise(
                frames,
                start,
                end,
            )

    finally:
        session.close()

    missing = expected.difference(
        data.index
    )

    data.attrs["source"] = (
        "Coinbase Exchange"
    )

    data.attrs["quote_currency"] = "USD"

    data.attrs["requested_start"] = (
        start.isoformat()
    )

    data.attrs["requested_end"] = (
        end.isoformat()
    )

    data.attrs["expected_candles"] = (
        len(expected)
    )

    data.attrs["missing_candles"] = (
        len(missing)
    )

    data.attrs["initial_missing_candles"] = (
        len(initial_missing)
    )

    data.attrs["recovered_candles"] = (
        len(initial_missing) - len(missing)
    )

    data.attrs["pages_requested"] = (
        pages_requested
    )

    data.attrs["repair_requests"] = (
        repair_requests
    )

    data.attrs["missing_timestamps"] = [
        timestamp.isoformat()
        for timestamp in missing
    ]

    return data

