
"""Research-only EMA20/50 backtester; no live orders. USD quote accounting."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradingCostsV3:
    fee_per_side: float = 0.001
    slippage_per_side: float = 0.0005
    spread_round_trip: float = 0.001


def run_backtest_v3(frame, initial_cash=500.0, allocation=0.5,
                    warmup=200, costs=None, evaluation_start=None):
    costs = TradingCostsV3() if costs is None else costs
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('Provide a DataFrame indexed by timestamps')
    if frame.index.tz is None or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError('Timestamps must be unique, ordered and timezone-aware')
    required = ['open', 'high', 'low', 'close', 'volume']
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing OHLCV columns')
    df = frame[required].astype(float).copy()
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Missing or nonfinite values')
    if (df[['open','high','low','close']] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError('Invalid OHLCV values')
    if (df.high < df[['open','low','close']].max(axis=1)).any() or (df.low > df[['open','high','close']].min(axis=1)).any():
        raise ValueError('Inconsistent OHLC values')
    if not isinstance(warmup, int) or warmup < 100 or len(df) < warmup + 3:
        raise ValueError('Need at least 100 warmup candles and three evaluation candles')
    if not 0 < initial_cash or not 0 < allocation <= 1:
        raise ValueError('Invalid cash or allocation')
    rates = [costs.fee_per_side, costs.slippage_per_side, costs.spread_round_trip]
    if any(not np.isfinite(x) or x < 0 or x >= 1 for x in rates):
        raise ValueError('Invalid cost rate')
    buy_mult = 1 + costs.slippage_per_side + costs.spread_round_trip / 2
    sell_mult = 1 - costs.slippage_per_side - costs.spread_round_trip / 2
    if sell_mult <= 0:
        raise ValueError('Sell execution multiplier must be positive')

    ema20 = df.close.ewm(span=20, adjust=False).mean()
    ema50 = df.close.ewm(span=50, adjust=False).mean()
    bullish = ema20 > ema50
    entries = bullish & ~bullish.shift(1, fill_value=False)
    exits = ~bullish & bullish.shift(1, fill_value=False)
    first = warmup
    if evaluation_start is not None:
        stamp = pd.Timestamp(evaluation_start)
        if stamp.tzinfo is None:
            raise ValueError('evaluation_start requires a timezone')
        first = max(first, int(df.index.searchsorted(stamp)))
    if first >= len(df) - 1:
        raise ValueError('Evaluation period too short')

    cash, qty = float(initial_cash), 0.0
    fees, impact = 0.0, 0.0
    entry_basis, entry_price, entry_time = 0.0, None, None
    records, marks = [], []

    def buy(i):
        nonlocal cash, qty, fees, impact, entry_basis, entry_price, entry_time
        raw = float(df.open.iloc[i])
        price = raw * buy_mult
        budget = cash * allocation
        acquired = budget / (price * (1 + costs.fee_per_side))
        notional = acquired * price
        fee = notional * costs.fee_per_side
        cash -= notional + fee
        qty = acquired
        fees += fee
        impact += acquired * (price - raw)
        entry_basis = notional + fee
        entry_price, entry_time = price, df.index[i]

    def sell(raw, stamp, reason):
        nonlocal cash, qty, fees, impact, entry_basis, entry_price, entry_time
        price = raw * sell_mult
        gross = qty * price
        fee = gross * costs.fee_per_side
        proceeds = gross - fee
        fees += fee
        impact += qty * (raw - price)
        cash += proceeds
        records.append({
            'Entry Time': entry_time, 'Exit Time': stamp,
            'Entry Price USD': entry_price, 'Exit Price USD': price,
            'Quantity': qty, 'Cost Basis USD': entry_basis,
            'Net Proceeds USD': proceeds, 'Net PnL USD': proceeds - entry_basis,
            'Return %': 100 * (proceeds / entry_basis - 1),
            'Exit Reason': reason,
        })
        qty, entry_basis, entry_price, entry_time = 0.0, 0.0, None, None

    for i in range(first, len(df)):
        # Only signals at the previous completed close may trade at this open.
        if i > first:
            if qty > 0 and bool(exits.iloc[i - 1]):
                sell(float(df.open.iloc[i]), df.index[i], 'EMA bearish crossover')
            elif qty == 0 and bool(entries.iloc[i - 1]):
                buy(i)
        close = float(df.close.iloc[i])
        # Mark open positions at net hypothetical liquidation proceeds.
        liquidatable = qty * close * sell_mult * (1 - costs.fee_per_side)
        marks.append((df.index[i], cash + liquidatable))

    if qty > 0:
        sell(float(df.close.iloc[-1]), df.index[-1], 'End of test (last close)')
        marks[-1] = (df.index[-1], cash)

    equity = pd.DataFrame(marks, columns=['timestamp','Equity USD']).set_index('timestamp')
    peak = np.maximum(float(initial_cash), equity['Equity USD'].cummax())
    max_drawdown = float((100 * (equity['Equity USD'] / peak - 1)).min())
    trades = pd.DataFrame(records)
    pnl = trades['Net PnL USD'] if len(trades) else pd.Series(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    factor = (float(wins.sum() / -losses.sum()) if losses.sum() < 0
              else (float('inf') if wins.sum() > 0 else None))

    # Identical evaluation window, allocation and execution costs.
    first_price = float(df.open.iloc[first]) * buy_mult
    hold_budget = initial_cash * allocation
    hold_qty = hold_budget / (first_price * (1 + costs.fee_per_side))
    hold_cash = initial_cash - hold_budget
    hold_final = hold_cash + hold_qty * float(df.close.iloc[-1]) * sell_mult * (1 - costs.fee_per_side)
    realised_pnl = float(pnl.sum()) if len(trades) else 0.0
    if not np.isclose(cash, initial_cash + realised_pnl, atol=1e-7):
        raise AssertionError('Trade ledger does not reconcile with final cash')

    summary = {
        'Initial USD': round(initial_cash, 2),
        'Final USD': round(cash, 2),
        'Return %': round(100 * (cash / initial_cash - 1), 3),
        'Buy and Hold Final USD': round(hold_final, 2),
        'Buy and Hold Return %': round(100 * (hold_final / initial_cash - 1), 3),
        'Maximum Drawdown %': round(max_drawdown, 3),
        'Trades': len(trades),
        'Win Rate %': round(100 * len(wins) / len(trades), 2) if len(trades) else None,
        'Profit Factor': factor,
        'Fees USD': round(fees, 4),
        'Spread and Slippage USD (approx)': round(impact, 4),
        'Realised PnL USD': round(realised_pnl, 4),
        'Evaluation Start': str(df.index[first]),
        'Evaluation End': str(df.index[-1]),
        'Warmup Candles': first,
    }
    return summary, trades, equity
