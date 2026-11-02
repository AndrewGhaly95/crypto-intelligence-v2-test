
"""Deterministic execution/accounting checks for V2.5. No live orders."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from backtest_multi_v25 import backtest_strategy_v25, CostsV25
from strategies_v25 import generate_signals, STRATEGIES


def candles():
    n = 230
    close = np.full(n, 100.0)
    close[203:208] = [110, 115, 120, 110, 90]
    close[208:] = 90
    opening = close.copy()
    opening[202] = 105
    opening[207] = 95
    return pd.DataFrame({
        'open': opening,
        'high': np.maximum(opening, close) + 1,
        'low': np.minimum(opening, close) - 1,
        'close': close,
        'volume': np.full(n, 100.0),
    }, index=pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC'))


def controlled_signals(df, strategy, warmup=200):
    out = df.copy()
    out['entry'] = False
    out['exit'] = False
    out.loc[out.index[201], 'entry'] = True
    out.loc[out.index[206], 'exit'] = True
    return out


class MultiV25Tests(unittest.TestCase):
    def setUp(self):
        self.df = candles()
        self.zero = CostsV25(0, 0, 0)

    def run_controlled(self, **kwargs):
        with patch('backtest_multi_v25.generate_signals',
                   side_effect=controlled_signals):
            return backtest_strategy_v25(
                self.df, STRATEGIES[0], warmup=200, **kwargs
            )

    def test_all_four_signal_generators(self):
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                result = generate_signals(self.df, strategy)
                self.assertEqual(len(result), len(self.df))
                self.assertFalse(result['entry'].iloc[:200].any())
                self.assertFalse(result['exit'].iloc[:200].any())
                self.assertFalse(
                    (result['entry'] & result['exit']).any()
                )

    def test_next_candle_open(self):
        _, trades, _ = self.run_controlled(costs=self.zero)
        self.assertEqual(len(trades), 1)
        self.assertEqual(
            trades.iloc[0]['Entry Time'], self.df.index[202]
        )
        self.assertEqual(
            trades.iloc[0]['Exit Time'], self.df.index[207]
        )
        self.assertAlmostEqual(
            trades.iloc[0]['Entry Price USD'], 105
        )
        self.assertAlmostEqual(
            trades.iloc[0]['Exit Price USD'], 95
        )

    def test_allocation_and_fees(self):
        fee = 0.01
        summary, trades, _ = self.run_controlled(
            costs=CostsV25(fee, 0, 0)
        )
        row = trades.iloc[0]
        qty = 250 / (105 * (1 + fee))
        proceeds = qty * 95 * (1 - fee)
        self.assertAlmostEqual(
            row['Quantity'], qty, places=9
        )
        self.assertAlmostEqual(
            row['Cost Basis USD'], 250, places=9
        )
        self.assertAlmostEqual(
            row['Net Proceeds USD'], proceeds, places=9
        )
        self.assertAlmostEqual(
            summary['Fees USD'],
            round(qty * 105 * fee + qty * 95 * fee, 4)
        )

    def test_costs_reduce_returns(self):
        no_cost, _, _ = self.run_controlled(costs=self.zero)
        with_cost, _, _ = self.run_controlled(
            costs=CostsV25(0.001, 0.0005, 0.001)
        )
        self.assertLess(
            with_cost['Final USD'], no_cost['Final USD']
        )

    def test_reconciliation_and_drawdown(self):
        summary, trades, equity = self.run_controlled(
            costs=self.zero
        )
        self.assertAlmostEqual(
            500 + trades['Net PnL USD'].sum(),
            equity['Equity USD'].iloc[-1],
            places=8
        )
        peak, worst = 500.0, 0.0
        for value in equity['Equity USD']:
            peak = max(peak, float(value))
            worst = min(
                worst, 100 * (float(value) / peak - 1)
            )
        self.assertAlmostEqual(
            summary['Maximum Drawdown %'],
            round(worst, 3)
        )

    def test_end_of_test_liquidation(self):
        def entry_only(df, strategy, warmup=200):
            out = controlled_signals(
                df, strategy, warmup
            )
            out['exit'] = False
            return out

        with patch('backtest_multi_v25.generate_signals',
                   side_effect=entry_only):
            summary, trades, equity = backtest_strategy_v25(
                self.df, STRATEGIES[0],
                warmup=200, costs=self.zero
            )

        self.assertEqual(len(trades), 1)
        self.assertEqual(
            trades.iloc[0]['Exit Reason'],
            'End of test'
        )
        self.assertEqual(
            trades.iloc[0]['Exit Time'],
            self.df.index[-1]
        )
        self.assertAlmostEqual(
            summary['Final USD'],
            equity['Equity USD'].iloc[-1],
            places=2
        )


def run_v25_checks():
    suite = (
        unittest.defaultTestLoader.loadTestsFromTestCase(
            MultiV25Tests
        )
    )
    result = unittest.TestResult()
    suite.run(result)
    return {
        'passed': (
            result.testsRun
            - len(result.failures)
            - len(result.errors)
        ),
        'total': result.testsRun,
        'failures': [
            f'{case}: {detail}'
            for case, detail in result.failures
        ],
        'errors': [
            f'{case}: {detail}'
            for case, detail in result.errors
        ],
    }


if __name__ == '__main__':
    unittest.main()
