
"""Chronological research-only EMA strategy evaluation.

Uses backtest_v2.py.
No optimisation or real trading.
All portfolio figures are USD.
"""

from dataclasses import dataclass
import pandas as pd

from backtest_v2 import run_backtest, TradingCosts


@dataclass(frozen=True)
class WalkForwardConfig:
    train_fraction: float = 0.65
    warmup_candles: int = 200
    initial_cash: float = 500.0
    allocation: float = 0.5


def run_walk_forward(
    candles,
    config=None,
    costs=None,
):
    """Evaluate a fixed strategy in chronological windows.

    Training and testing each use a separate simulated
    account. Earlier candles provide indicator warm-up.
    """

    config = config or WalkForwardConfig()
    costs = costs or TradingCosts()

    if not 0.3 <= config.train_fraction <= 0.8:
        raise ValueError(
            "train_fraction must be between 0.3 and 0.8"
        )

    if config.warmup_candles < 100:
        raise ValueError(
            "warmup_candles must be at least 100"
        )

    if (
        not isinstance(candles, pd.DataFrame)
        or len(candles) < 2 * config.warmup_candles + 20
    ):
        raise ValueError(
            "Not enough candles for both evaluation windows"
        )

    if (
        not isinstance(candles.index, pd.DatetimeIndex)
        or candles.index.tz is None
    ):
        raise ValueError(
            "Candle timestamps must be timezone-aware"
        )

    if (
        not candles.index.is_monotonic_increasing
        or candles.index.has_duplicates
    ):
        raise ValueError(
            "Candle timestamps must be sorted and unique"
        )

    split = int(
        len(candles) * config.train_fraction
    )

    if (
        split <= config.warmup_candles + 2
        or len(candles) - split < 20
    ):
        raise ValueError(
            "Training or testing period is too short"
        )

    train_frame = candles.iloc[:split].copy()

    # The testing window includes earlier candles
    # for indicator calculation, not for trading.
    test_frame = candles.iloc[
        split - config.warmup_candles:
    ].copy()

    test_start = candles.index[split]

    train_summary, train_trades, train_equity = (
        run_backtest(
            train_frame,
            initial_cash=config.initial_cash,
            allocation=config.allocation,
            warmup=config.warmup_candles,
            costs=costs,
        )
    )

    test_summary, test_trades, test_equity = (
        run_backtest(
            test_frame,
            initial_cash=config.initial_cash,
            allocation=config.allocation,
            warmup=config.warmup_candles,
            costs=costs,
            evaluation_start=test_start,
        )
    )

    comparison = pd.DataFrame([
        {
            "Period": "Training",
            **train_summary,
        },
        {
            "Period": "Testing",
            **test_summary,
        },
    ])

    return {
        "comparison": comparison,
        "train_trades": train_trades,
        "test_trades": test_trades,
        "train_equity": train_equity,
        "test_equity": test_equity,
        "split_time": test_start,
    }
