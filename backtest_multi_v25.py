
"""Long-only research backtester for strategies_v25. No live orders."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from strategies_v25 import generate_signals, STRATEGIES

@dataclass(frozen=True)
class CostsV25:
    fee_per_side: float = 0.001
    slippage_per_side: float = 0.0005
    spread_round_trip: float = 0.001


def backtest_strategy_v25(frame, strategy, initial_cash=500.0, allocation=0.5,
                          warmup=200, costs=None, evaluation_start=None):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown strategy')
    costs = CostsV25() if costs is None else costs
    if not np.isfinite(initial_cash) or initial_cash <= 0 or not np.isfinite(allocation) or not 0 < allocation <= 1:
        raise ValueError('Invalid capital or allocation')
    rates = (costs.fee_per_side, costs.slippage_per_side, costs.spread_round_trip)
    if any(not np.isfinite(v) or v < 0 or v >= 1 for v in rates):
        raise ValueError('Invalid trading costs')
    buy_multiplier = 1 + costs.slippage_per_side + costs.spread_round_trip / 2
    sell_multiplier = 1 - costs.slippage_per_side - costs.spread_round_trip / 2
    if sell_multiplier <= 0:
        raise ValueError('Invalid sell multiplier')
    data = generate_signals(frame, strategy, warmup=warmup)
    first = warmup
    if evaluation_start is not None:
        start = pd.Timestamp(evaluation_start)
        if start.tzinfo is None:
            raise ValueError('Evaluation start must be timezone-aware')
        first = max(first, int(data.index.searchsorted(start)))
    if first >= len(data) - 1:
        raise ValueError('Insufficient evaluation candles')

    cash = float(initial_cash)
    quantity = 0.0
    basis = 0.0
    entry_time = None
    entry_price = None
    total_fees = 0.0
    impact = 0.0
    trades = []
    equity_points = []

    def liquidate(raw_price, timestamp, reason):
        nonlocal cash, quantity, basis, entry_time, entry_price, total_fees, impact
        price = float(raw_price) * sell_multiplier
        gross = quantity * price
        fee = gross * costs.fee_per_side
        proceeds = gross - fee
        cash += proceeds
        total_fees += fee
        impact += quantity * (float(raw_price) - price)
        trades.append({
            'Strategy': strategy, 'Entry Time': entry_time, 'Exit Time': timestamp,
            'Entry Price USD': entry_price, 'Exit Price USD': price,
            'Quantity': quantity, 'Cost Basis USD': basis,
            'Net Proceeds USD': proceeds, 'Net PnL USD': proceeds - basis,
            'Return %': 100 * (proceeds / basis - 1), 'Exit Reason': reason,
        })
        quantity, basis, entry_time, entry_price = 0.0, 0.0, None, None

    for i in range(first, len(data)):
        if i > first:
            previous = data.iloc[i - 1]
            if quantity > 0 and bool(previous['exit']):
                liquidate(data.open.iloc[i], data.index[i], 'Signal exit')
            elif quantity == 0 and bool(previous['entry']):
                raw = float(data.open.iloc[i])
                price = raw * buy_multiplier
                budget = cash * allocation
                quantity = budget / (price * (1 + costs.fee_per_side))
                notional = quantity * price
                fee = notional * costs.fee_per_side
                basis = notional + fee
                cash -= basis
                total_fees += fee
                impact += quantity * (price - raw)
                entry_time, entry_price = data.index[i], price
        net_liquidation = quantity * float(data.close.iloc[i]) * sell_multiplier * (1 - costs.fee_per_side)
        equity_points.append((data.index[i], cash + net_liquidation))

    if quantity > 0:
        liquidate(data.close.iloc[-1], data.index[-1], 'End of test')
        equity_points[-1] = (data.index[-1], cash)

    equity = pd.DataFrame(equity_points, columns=['Timestamp', 'Equity USD']).set_index('Timestamp')
    trade_log = pd.DataFrame(trades)
    realised = float(trade_log['Net PnL USD'].sum()) if len(trade_log) else 0.0
    if not np.isclose(cash, initial_cash + realised, atol=1e-7):
        raise AssertionError('Trade ledger does not reconcile')
    peaks = np.maximum(float(initial_cash), equity['Equity USD'].cummax())
    drawdown = float((100 * (equity['Equity USD'] / peaks - 1)).min())
    wins = trade_log.loc[trade_log['Net PnL USD'] > 0, 'Net PnL USD'] if len(trade_log) else pd.Series(dtype=float)
    losses = trade_log.loc[trade_log['Net PnL USD'] < 0, 'Net PnL USD'] if len(trade_log) else pd.Series(dtype=float)
    profit_factor = (float(wins.sum() / -losses.sum()) if losses.sum() < 0 else (float('inf') if wins.sum() > 0 else None))

    # Cost-adjusted buy-and-hold with the same allocation and evaluation window.
    buy_price = float(data.open.iloc[first]) * buy_multiplier
    budget = initial_cash * allocation
    hold_qty = budget / (buy_price * (1 + costs.fee_per_side))
    hold_final = initial_cash - budget + hold_qty * float(data.close.iloc[-1]) * sell_multiplier * (1 - costs.fee_per_side)
    summary = {
        'Strategy': strategy,
        'Initial USD': round(initial_cash, 2),
        'Final USD': round(cash, 2),
        'Return %': round(100 * (cash / initial_cash - 1), 3),
        'Buy and Hold Return %': round(100 * (hold_final / initial_cash - 1), 3),
        'Maximum Drawdown %': round(drawdown, 3),
        'Trades': len(trade_log),
        'Win Rate %': round(100 * len(wins) / len(trade_log), 2) if len(trade_log) else None,
        'Profit Factor': profit_factor,
        'Fees USD': round(total_fees, 4),
        'Spread and Slippage USD (approx)': round(impact, 4),
        'Evaluation Start': str(data.index[first]),
        'Evaluation End': str(data.index[-1]),
    }
    return summary, trade_log, equity
