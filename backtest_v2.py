
"""Research-only long-only backtester with closed-bar signals and next-open fills."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradingCosts:
    fee_per_side: float = 0.001
    slippage_per_side: float = 0.0005
    spread_round_trip: float = 0.001


def _check(frame, warmup, costs, allocation):
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError('Supply an OHLCV DataFrame with a DatetimeIndex')
    if frame.index.tz is None or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError('Index must be timezone-aware, increasing and unique')
    required = ['open', 'high', 'low', 'close', 'volume']
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing OHLCV columns')
    df = frame[required].astype(float).copy()
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Missing or nonfinite values')
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any() or (df.volume < 0).any():
        raise ValueError('Invalid prices or volume')
    if (df.high < df[['open', 'low', 'close']].max(axis=1)).any() or (df.low > df[['open', 'high', 'close']].min(axis=1)).any():
        raise ValueError('Inconsistent OHLC')
    if len(df) < warmup + 3:
        raise ValueError(f'Need at least {warmup + 3} candles')
    if not 0 < allocation <= 1:
        raise ValueError('allocation must be in (0,1]')
    for x in (costs.fee_per_side, costs.slippage_per_side, costs.spread_round_trip):
        if x < 0 or x >= 1:
            raise ValueError('Invalid cost rate')
    return df


def run_backtest(frame, initial_cash=500.0, allocation=0.5,
                 warmup=200, costs=None, evaluation_start=None):
    """EMA20/EMA50 crossover, next-open fills, no intrabar stop/target.

    The first warmup candles are indicators-only. Optionally set evaluation_start
    (UTC timestamp) to align comparisons. Signals from close i fill at open i+1.
    Buy-and-hold invests the same allocation at evaluation start and holds.
    """
    costs = costs or TradingCosts()
    if initial_cash <= 0 or warmup < 100:
        raise ValueError('initial_cash must be positive and warmup >= 100')
    df = _check(frame, warmup, costs, allocation)
    ema20 = df.close.ewm(span=20, adjust=False).mean()
    ema50 = df.close.ewm(span=50, adjust=False).mean()
    bullish = ema20 > ema50
    entry_signal = bullish & ~bullish.shift(1, fill_value=False)
    exit_signal = ~bullish & bullish.shift(1, fill_value=False)
    first = warmup
    if evaluation_start is not None:
        when = pd.Timestamp(evaluation_start)
        if when.tzinfo is None:
            raise ValueError('evaluation_start must include timezone')
        first = max(first, int(df.index.searchsorted(when)))
    if first >= len(df) - 1:
        raise ValueError('Not enough evaluation candles after warmup')

    half_spread = costs.spread_round_trip / 2
    buy_multiplier = 1 + half_spread + costs.slippage_per_side
    sell_multiplier = 1 - half_spread - costs.slippage_per_side
    cash = float(initial_cash)
    qty = 0.0
    entry_cost = 0.0
    entry_time = None
    entry_price = None
    trades = []
    curve = []
    fees_paid = 0.0
    slippage_spread_paid = 0.0

    def sell(i, reason):
        nonlocal cash, qty, entry_cost, entry_time, entry_price
        nonlocal fees_paid, slippage_spread_paid
        raw = float(df.open.iloc[i])
        execution = raw * sell_multiplier
        gross = qty * execution
        fee = gross * costs.fee_per_side
        proceeds = gross - fee
        fees_paid += fee
        slippage_spread_paid += qty * (raw - execution)
        cash += proceeds
        trades.append({
            'Entry Time': entry_time,
            'Exit Time': df.index[i],
            'Entry Price USD': entry_price,
            'Exit Price USD': execution,
            'Net PnL USD': proceeds - entry_cost,
            'Return %': 100 * (proceeds / entry_cost - 1),
            'Exit Reason': reason,
        })
        qty = 0.0
        entry_cost = 0.0
        entry_time = None
        entry_price = None

    # Process candle opens: signal is based strictly on previous close.
    for i in range(first, len(df)):
        if i > first:
            if qty > 0 and bool(exit_signal.iloc[i - 1]):
                sell(i, 'EMA bearish crossover')
            elif qty == 0 and bool(entry_signal.iloc[i - 1]):
                budget = cash * allocation
                raw = float(df.open.iloc[i])
                execution = raw * buy_multiplier
                qty = budget / (execution * (1 + costs.fee_per_side))
                notional = qty * execution
                fee = notional * costs.fee_per_side
                cash -= notional + fee
                fees_paid += fee
                slippage_spread_paid += qty * (execution - raw)
                entry_cost = notional + fee
                entry_time = df.index[i]
                entry_price = execution
        curve.append((df.index[i], cash + qty * float(df.close.iloc[i])))

    # End-of-test liquidation at last close, not an imaginary next candle.
    if qty > 0:
        i = len(df) - 1
        raw = float(df.close.iloc[i])
        execution = raw * sell_multiplier
        gross = qty * execution
        fee = gross * costs.fee_per_side
        proceeds = gross - fee
        fees_paid += fee
        slippage_spread_paid += qty * (raw - execution)
        cash += proceeds
        trades.append({
            'Entry Time': entry_time,
            'Exit Time': df.index[i],
            'Entry Price USD': entry_price,
            'Exit Price USD': execution,
            'Net PnL USD': proceeds - entry_cost,
            'Return %': 100 * (proceeds / entry_cost - 1),
            'Exit Reason': 'End of test (last close)',
        })
        curve[-1] = (df.index[i], cash)

    equity = pd.DataFrame(curve, columns=['timestamp', 'Equity USD']).set_index('timestamp')
    running_peak = np.maximum(initial_cash, equity['Equity USD'].cummax())
    drawdown = 100 * (equity['Equity USD'] / running_peak - 1)
    trade_df = pd.DataFrame(trades)
    wins = trade_df.loc[trade_df['Net PnL USD'] > 0, 'Net PnL USD'] if len(trade_df) else pd.Series(dtype=float)
    losses = trade_df.loc[trade_df['Net PnL USD'] < 0, 'Net PnL USD'] if len(trade_df) else pd.Series(dtype=float)
    profit_factor = (float(wins.sum() / -losses.sum()) if -losses.sum() > 0 else (float('inf') if wins.sum() > 0 else None))

    # Matched-window buy-and-hold using the same capital allocation and costs.
    first_open = float(df.open.iloc[first])
    hold_budget = initial_cash * allocation
    hold_qty = hold_budget / (first_open * buy_multiplier * (1 + costs.fee_per_side))
    last_close = float(df.close.iloc[-1])
    hold_final = (initial_cash - hold_budget + hold_qty * last_close * sell_multiplier * (1 - costs.fee_per_side))
    summary = {
        'Initial USD': round(initial_cash, 2),
        'Final USD': round(cash, 2),
        'Return %': round(100 * (cash / initial_cash - 1), 3),
        'Buy and Hold Final USD': round(hold_final, 2),
        'Buy and Hold Return %': round(100 * (hold_final / initial_cash - 1), 3),
        'Maximum Drawdown %': round(float(drawdown.min()), 3),
        'Trades': len(trade_df),
        'Win Rate %': round(100 * len(wins) / len(trade_df), 2) if len(trade_df) else None,
        'Profit Factor': profit_factor,
        'Fees USD': round(fees_paid, 4),
        'Spread and Slippage USD (approx)': round(slippage_spread_paid, 4),
        'Evaluation Start': str(df.index[first]),
        'Evaluation End': str(df.index[-1]),
        'Warmup Candles': first,
    }
    return summary, trade_df, equity
