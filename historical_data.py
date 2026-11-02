
"""Historical candle storage for Crypto Intelligence V2.1.

Research only. No trading or order execution.
"""

from pathlib import Path
import pandas as pd

from market_candles import get_candles


DATA_DIR = Path("historical_cache")

INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
}

COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_history(data):
    """Validate a historical OHLCV dataset."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("History must be a DataFrame.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("History requires a DatetimeIndex.")

    if data.index.tz is None:
        raise ValueError("History timestamps must be timezone-aware.")

    if any(col not in data.columns for col in COLUMNS):
        raise ValueError("Missing OHLCV columns.")

    result = data[COLUMNS].copy()

    for col in COLUMNS:
        result[col] = pd.to_numeric(
            result[col],
            errors="raise",
        )

    if result.isna().any().any():
        raise ValueError("History contains missing values.")

    if (
        (result[["open", "high", "low", "close"]] <= 0)
        .any()
        .any()
    ):
        raise ValueError("History contains invalid prices.")

    if (result["volume"] < 0).any():
        raise ValueError("History contains negative volume.")

    if (
        (
            result["high"]
            < result[["open", "low", "close"]].max(axis=1)
        ).any()
        or (
            result["low"]
            > result[["open", "high", "close"]].min(axis=1)
        ).any()
    ):
        raise ValueError("Inconsistent OHLC prices.")

    result.index = result.index.tz_convert("UTC")

    result = result.sort_index()

    if result.index.has_duplicates:
        raise ValueError("Duplicate candle timestamps.")

    return result


def cache_path(symbol, interval):
    """Return the local cache filename."""

    symbol = symbol.upper()

    if not symbol.isalpha():
        raise ValueError("Invalid cryptocurrency symbol.")

    if interval not in INTERVAL_MINUTES:
        raise ValueError("Unsupported interval.")

    return DATA_DIR / f"{symbol}_{interval}.csv"


def load_history(symbol, interval):
    """Load previously collected candles."""

    path = cache_path(symbol, interval)

    if not path.exists():
        return pd.DataFrame(
            columns=COLUMNS,
            index=pd.DatetimeIndex([], tz="UTC"),
        )

    data = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    return validate_history(data)


def save_history(symbol, interval, data):
    """Save validated historical candles."""

    clean = validate_history(data)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_csv(
        cache_path(symbol, interval)
    )


def update_history(symbol, interval, limit=700):
    """Merge recent completed candles with cached history."""

    existing = load_history(symbol, interval)

    recent = get_candles(
        symbol,
        interval,
        limit,
    )

    recent = validate_history(recent)

    combined = pd.concat(
        [existing, recent]
    )

    combined = combined[
        ~combined.index.duplicated(keep="last")
    ].sort_index()

    combined = validate_history(combined)

    save_history(
        symbol,
        interval,
        combined,
    )

    return combined


def history_quality(data, interval):
    """Summarise coverage and missing candle intervals."""

    clean = validate_history(data)

    if clean.empty:
        return {
            "candles": 0,
            "missing_intervals": 0,
            "coverage_pct": 0.0,
        }

    minutes = INTERVAL_MINUTES[interval]

    expected = pd.date_range(
        start=clean.index.min(),
        end=clean.index.max(),
        freq=f"{minutes}min",
        tz="UTC",
    )

    missing = expected.difference(clean.index)

    return {
        "candles": len(clean),
        "start": clean.index.min(),
        "end": clean.index.max(),
        "missing_intervals": len(missing),
        "coverage_pct": round(
            100 * len(clean) / len(expected),
            2,
        ),
    }
