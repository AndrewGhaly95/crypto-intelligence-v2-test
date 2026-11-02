
import unittest
import numpy as np
import pandas as pd
from backtest_v2 import run_backtest, TradingCosts


def candles(prices):
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": p,
            "high": p * 1.001,
            "low": p * 0.999,
            "close": p,
            "volume": np.ones(len(p)),
        },
        index=pd.date_range(
            "2026-01-01",
            periods=len(p),
            freq="h",
            tz="UTC",
        ),
    )


class EngineTests(unittest.TestCase):

    def setUp(self):
        self.free = TradingCosts(0, 0, 0)

    def test_flat_market_no_trades(self):
        summary, trades, equity = run_backtest(
            candles(np.full(300, 100)),
            warmup=100,
            costs=self.free,
        )
        self.assertEqual(summary["Trades"], 0)
        self.assertEqual(summary["Final USD"], 500)
        self.assertTrue(trades.empty)
        self.assertTrue(
            (equity["Equity USD"] == 500).all()
        )

    def test_rejects_duplicate_timestamps(self):
        data = candles(np.full(300, 100))
        idx = list(data.index)
        idx[1] = idx[0]
        data.index = pd.DatetimeIndex(idx)

        with self.assertRaises(ValueError):
            run_backtest(data, warmup=100)

    def test_rejects_bad_ohlc(self):
        data = candles(np.full(300, 100))
        data.iloc[
            0,
            data.columns.get_loc("high")
        ] = 50

        with self.assertRaises(ValueError):
            run_backtest(data, warmup=100)

    def test_no_trades_in_warmup(self):
        prices = np.r_[
            np.linspace(100, 200, 100),
            np.full(200, 200),
        ]
        data = candles(prices)

        _, trades, _ = run_backtest(
            data,
            warmup=100,
            costs=self.free,
        )

        if not trades.empty:
            self.assertTrue(
                (
                    trades["Entry Time"]
                    >= data.index[100]
                ).all()
            )

    def test_costs_cannot_improve_same_trades(self):
        prices = np.r_[
            np.full(120, 100),
            np.linspace(100, 130, 50),
            np.linspace(130, 90, 50),
            np.linspace(90, 120, 50),
        ]
        data = candles(prices)

        free, free_trades, _ = run_backtest(
            data,
            warmup=100,
            costs=self.free,
        )

        costly, cost_trades, _ = run_backtest(
            data,
            warmup=100,
            costs=TradingCosts(
                0.001,
                0.0005,
                0.001,
            ),
        )

        self.assertEqual(
            len(free_trades),
            len(cost_trades),
        )

        self.assertLessEqual(
            costly["Final USD"],
            free["Final USD"] + 1e-8,
        )

    def test_holdout_start_not_before_requested(self):
        prices = np.r_[
            np.full(100, 100),
            np.linspace(100, 140, 100),
            np.linspace(140, 90, 100),
        ]
        data = candles(prices)
        start = data.index[200]

        _, trades, equity = run_backtest(
            data,
            warmup=100,
            costs=self.free,
            evaluation_start=start,
        )

        self.assertEqual(
            equity.index[0],
            start,
        )

        if not trades.empty:
            self.assertTrue(
                (
                    trades["Entry Time"] >= start
                ).all()
            )


def run_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        EngineTests
    )

    result = unittest.TestResult()
    suite.run(result)

    return {
        "passed": (
            result.testsRun
            - len(result.failures)
            - len(result.errors)
        ),
        "total": result.testsRun,
        "failures": [
            str(test) + "\n" + message
            for test, message in result.failures
        ],
        "errors": [
            str(test) + "\n" + message
            for test, message in result.errors
        ],
    }
