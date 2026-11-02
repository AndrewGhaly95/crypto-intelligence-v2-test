
import unittest
import numpy as np
import pandas as pd

from backtest_v3 import run_backtest_v3, TradingCostsV3


def sample(prices, opening=None):
    close = np.asarray(prices, dtype=float)
    op = (
        close.copy()
        if opening is None
        else np.asarray(opening, dtype=float)
    )

    return pd.DataFrame(
        {
            "open": op,
            "high": np.maximum(op, close) * 1.01,
            "low": np.minimum(op, close) * 0.99,
            "close": close,
            "volume": 100.0,
        },
        index=pd.date_range(
            "2026-01-01",
            periods=len(close),
            freq="h",
            tz="UTC",
        ),
    )


def fixture():
    # Flat warm-up, bullish crossover,
    # followed by bearish crossover.
    prices = np.r_[
        np.full(110, 100.0),
        np.full(30, 120.0),
        np.full(40, 80.0),
    ]
    return sample(prices)


class V3ExecutionTests(unittest.TestCase):

    def setUp(self):
        self.zero = TradingCostsV3(0, 0, 0)

    def test_next_open_and_no_lookahead(self):
        data = fixture()

        _, trades, _ = run_backtest_v3(
            data,
            warmup=100,
            costs=self.zero,
        )

        self.assertGreaterEqual(len(trades), 1)

        # Signal at candle 110 close.
        # Entry at candle 111 open.
        self.assertEqual(
            trades.iloc[0]["Entry Time"],
            data.index[111],
        )

        self.assertEqual(
            trades.iloc[0]["Entry Price USD"],
            data.open.iloc[111],
        )

    def test_position_size_and_cash_budget(self):
        data = fixture()

        _, trades, _ = run_backtest_v3(
            data,
            warmup=100,
            allocation=0.5,
            costs=self.zero,
        )

        first = trades.iloc[0]

        self.assertAlmostEqual(
            first["Cost Basis USD"],
            250,
            places=8,
        )

        self.assertAlmostEqual(
            first["Quantity"],
            250 / 120,
            places=8,
        )

    def test_fee_once_on_each_side(self):
        data = fixture()
        fee = 0.01

        _, trades, _ = run_backtest_v3(
            data,
            warmup=100,
            allocation=0.5,
            costs=TradingCostsV3(fee, 0, 0),
        )

        first = trades.iloc[0]

        expected_qty = 250 / (120 * (1 + fee))
        expected_proceeds = (
            expected_qty * 80 * (1 - fee)
        )

        self.assertAlmostEqual(
            first["Quantity"],
            expected_qty,
            places=8,
        )

        self.assertAlmostEqual(
            first["Net Proceeds USD"],
            expected_proceeds,
            places=8,
        )

        self.assertAlmostEqual(
            first["Net PnL USD"],
            expected_proceeds - 250,
            places=8,
        )

    def test_ledger_reconciles(self):
        data = fixture()

        summary, trades, equity = run_backtest_v3(
            data,
            warmup=100,
            costs=TradingCostsV3(
                0.001,
                0.0005,
                0.001,
            ),
        )

        self.assertAlmostEqual(
            equity["Equity USD"].iloc[-1],
            summary["Final USD"],
            places=2,
        )

        self.assertAlmostEqual(
            500 + trades["Net PnL USD"].sum(),
            equity["Equity USD"].iloc[-1],
            places=7,
        )

    def test_end_liquidation(self):
        data = sample(
            np.r_[
                np.full(110, 100.0),
                np.full(40, 120.0),
            ]
        )

        _, trades, equity = run_backtest_v3(
            data,
            warmup=100,
            costs=self.zero,
        )

        self.assertEqual(len(trades), 1)

        self.assertEqual(
            trades.iloc[0]["Exit Reason"],
            "End of test (last close)",
        )

        self.assertEqual(
            trades.iloc[0]["Exit Time"],
            data.index[-1],
        )

        self.assertAlmostEqual(
            equity["Equity USD"].iloc[-1],
            500,
            places=8,
        )

    def test_drawdown_independent_calculation(self):
        data = fixture()

        summary, _, equity = run_backtest_v3(
            data,
            warmup=100,
            costs=self.zero,
        )

        values = equity["Equity USD"].to_numpy()

        peak = 500.0
        worst = 0.0

        for value in values:
            peak = max(peak, float(value))
            worst = min(
                worst,
                100 * (float(value) / peak - 1),
            )

        self.assertAlmostEqual(
            summary["Maximum Drawdown %"],
            round(worst, 3),
            places=3,
        )


def run_v3_checks():
    suite = (
        unittest.defaultTestLoader.loadTestsFromTestCase(
            V3ExecutionTests
        )
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
            f"{test}: {message}"
            for test, message in result.failures
        ],
        "errors": [
            f"{test}: {message}"
            for test, message in result.errors
        ],
    }
