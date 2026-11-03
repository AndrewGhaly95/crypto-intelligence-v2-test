
"""V2.7 expanding-history, rolling out-of-sample evaluation.

Research only. No live orders or exchange integration.
"""
import numpy as np
import pandas as pd

from backtest_v27 import backtest_v27, CostsV27
from strategy_v27 import STRATEGIES


def run_rolling_v27(
    frame,
    train_days=45,
    test_days=15,
    allocation=0.5,
    initial_cash=500.0,
    costs=None,
    warmup=250,
):
    """
    Evaluate fixed strategies in non-overlapping future windows.

    Earlier history is used for indicator calculation.
    Each test window begins flat with fresh simulated cash.

    These are expanding-context holdout evaluations.
    No parameter optimisation is performed.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Expected OHLCV DataFrame")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")

    if (
        frame.index.tz is None
        or frame.index.has_duplicates
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Timestamps must be timezone-aware, "
            "unique and ascending"
        )

    if not isinstance(train_days, int) or train_days <= 0:
        raise ValueError("Invalid training duration")

    if not isinstance(test_days, int) or test_days <= 0:
        raise ValueError("Invalid testing duration")

    if not isinstance(warmup, int) or warmup < 200:
        raise ValueError("Warmup must be at least 200 candles")

    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("Invalid initial capital")

    if not np.isfinite(allocation) or not 0 < allocation <= 1:
        raise ValueError("Invalid allocation")

    costs = CostsV27() if costs is None else costs

    # V2.7 currently expects continuous hourly candles.
    expected = pd.date_range(
        start=frame.index[0],
        end=frame.index[-1],
        freq="h",
    )

    if not frame.index.equals(expected):
        raise ValueError(
            "Missing or irregular hourly candles. "
            "Evaluation cancelled."
        )

    train_candles = train_days * 24
    test_candles = test_days * 24

    if train_candles < warmup + 2:
        raise ValueError(
            "Training history is shorter than indicator warmup"
        )

    if len(frame) < train_candles + test_candles:
        raise ValueError(
            "Insufficient history for one complete testing window"
        )

    rows = []
    fold = 0

    for start in range(
        train_candles,
        len(frame) - test_candles + 1,
        test_candles,
    ):
        fold += 1
        end = start + test_candles

        # Historical data up to the end of this fold.
        # No subsequent candles are supplied.
        history = frame.iloc[:end].copy()

        evaluation_start = frame.index[start]
        evaluation_end = (
            frame.index[end]
            if end < len(frame)
            else None
        )

        for strategy in STRATEGIES:
            summary, trades, equity = backtest_v27(
                history,
                strategy,
                initial_cash=initial_cash,
                allocation=allocation,
                warmup=warmup,
                costs=costs,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_end,
            )

            excess = (
                summary["Return %"]
                - summary["Buy and Hold Return %"]
            )

            rows.append({
                "Fold": fold,
                "Strategy": strategy,
                "Testing Start": summary["Evaluation Start"],
                "Testing End": summary["Evaluation End"],
                "Initial USD": summary["Initial USD"],
                "Final USD": summary["Final USD"],
                "Return %": summary["Return %"],
                "Buy and Hold Return %":
                    summary["Buy and Hold Return %"],
                "Excess Return %": round(excess, 3),
                "Maximum Drawdown %":
                    summary["Maximum Drawdown %"],
                "Trades": summary["Trades"],
                "Win Rate %": summary["Win Rate %"],
                "Profit Factor": summary["Profit Factor"],
                "Fees USD": summary["Fees USD"],
            })

    if not rows:
        raise ValueError(
            "No complete testing windows were available"
        )

    return pd.DataFrame(rows)
