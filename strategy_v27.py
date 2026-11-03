
"""Causal indicators and stateless long-only trading decisions."""
import numpy as np
import pandas as pd
from strategies_v25 import _validate, _rsi, _atr, STRATEGIES


def indicators(frame):
    df = _validate(frame)
    close = df['close']

    df['ema20'] = close.ewm(
        span=20, adjust=False
    ).mean()

    df['ema50'] = close.ewm(
        span=50, adjust=False
    ).mean()

    df['ema200'] = close.ewm(
        span=200, adjust=False
    ).mean()

    df['rsi'] = _rsi(close)
    df['atr'] = _atr(df)

    df['previous_high20'] = (
        df['high']
        .shift(1)
        .rolling(20, min_periods=20)
        .max()
    )

    df['previous_volume20'] = (
        df['volume']
        .shift(1)
        .rolling(20, min_periods=20)
        .mean()
    )

    return df


def decide(df, i, strategy, holding, highest_close):
    """
    Decision at close of candle i.

    Caller executes at the opening of candle i+1.
    Actual position state is supplied by the backtester.
    """
    if strategy not in STRATEGIES:
        raise ValueError('Unknown strategy')

    row = df.iloc[i]
    previous = df.iloc[i - 1]

    bullish = row.ema20 > row.ema50
    previous_bullish = previous.ema20 > previous.ema50

    cross_up = bullish and not previous_bullish
    cross_down = not bullish and previous_bullish

    if strategy == 'EMA crossover':
        enter = cross_up
        leave = cross_down

    elif strategy == 'EMA + RSI':
        enter = (
            bullish
            and row.close > row.ema200
            and 50 < row.rsi < 70
            and (previous.rsi <= 50 or cross_up)
        )

        leave = (
            row.rsi < 45
            or not bullish
        )

    elif strategy == 'Volume breakout':
        enter = (
            row.close > row.previous_high20
            and row.volume > 1.5 * row.previous_volume20
            and row.close > row.ema200
        )

        leave = row.close < row.ema20

    else:
        enter = (
            cross_up
            and row.close > row.ema200
        )

        leave = holding and (
            row.close < highest_close - 3 * row.atr
            or not bullish
        )

    return (
        bool(leave)
        if holding
        else bool(enter)
    )
