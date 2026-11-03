
"""Research-only long-only backtest with execution-owned state."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from strategy_v27 import indicators, decide, STRATEGIES


@dataclass(frozen=True)
class CostsV27:
    fee_per_side: float = 0.001
    slippage_per_side: float = 0.0005
    spread_round_trip: float = 0.001


def backtest_v27(
    frame,
    strategy,
    initial_cash=500.0,
    allocation=0.5,
    warmup=250,
    costs=None,
    evaluation_start=None,
    evaluation_end=None,
):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")

    costs = CostsV27() if costs is None else costs

    if (
        not np.isfinite(initial_cash)
        or initial_cash <= 0
        or not np.isfinite(allocation)
        or not 0 < allocation <= 1
    ):
        raise ValueError("Invalid capital or allocation")

    rates = (
        costs.fee_per_side,
        costs.slippage_per_side,
        costs.spread_round_trip,
    )

    if any(
        not np.isfinite(x) or x < 0 or x >= 1
        for x in rates
    ):
        raise ValueError("Invalid costs")

    buy_mult = (
        1
        + costs.slippage_per_side
        + costs.spread_round_trip / 2
    )

    sell_mult = (
        1
        - costs.slippage_per_side
        - costs.spread_round_trip / 2
    )

    if sell_mult <= 0 or warmup < 200:
        raise ValueError(
            "Invalid execution assumptions or warmup"
        )

    df = indicators(frame)
    first = warmup

    if evaluation_start is not None:
        t = pd.Timestamp(evaluation_start)

        if t.tzinfo is None:
            raise ValueError(
                "evaluation_start requires timezone"
            )

        first = max(
            first,
            int(df.index.searchsorted(t)),
        )

    stop = len(df)

    if evaluation_end is not None:
        t = pd.Timestamp(evaluation_end)

        if t.tzinfo is None:
            raise ValueError(
                "evaluation_end requires timezone"
            )

        stop = min(
            stop,
            int(df.index.searchsorted(t)),
        )

    if first >= stop - 1:
        raise ValueError(
            "Need at least two evaluation candles after warmup"
        )

    cash = float(initial_cash)
    qty = 0.0
    basis = 0.0

    entry_time = None
    entry_price = None
    highest_close = np.nan

    total_fees = 0.0
    impact = 0.0

    trades = []
    curve = []
    pending = None

    def sell(raw, ts, reason):
        nonlocal cash, qty, basis
        nonlocal entry_time, entry_price
        nonlocal highest_close, total_fees, impact

        px = float(raw) * sell_mult
        gross = qty * px
        fee = gross * costs.fee_per_side
        proceeds = gross - fee

        cash += proceeds
        total_fees += fee
        impact += qty * (float(raw) - px)

        trades.append({
            "Strategy": strategy,
            "Entry Time": entry_time,
            "Exit Time": ts,
            "Entry Price USD": entry_price,
            "Exit Price USD": px,
            "Quantity": qty,
            "Cost Basis USD": basis,
            "Net Proceeds USD": proceeds,
            "Net PnL USD": proceeds - basis,
            "Return %": 100 * (proceeds / basis - 1),
            "Exit Reason": reason,
        })

        qty = 0.0
        basis = 0.0
        entry_time = None
        entry_price = None
        highest_close = np.nan

    for i in range(first, stop):
        row = df.iloc[i]
        ts = df.index[i]

        # Execute the prior candle's confirmed decision.
        if pending == "sell" and qty > 0:
            sell(
                row.open,
                ts,
                "Signal exit",
            )

        elif pending == "buy" and qty == 0:
            px = float(row.open) * buy_mult
            budget = cash * allocation

            qty = budget / (
                px * (1 + costs.fee_per_side)
            )

            fee = qty * px * costs.fee_per_side
            basis = qty * px + fee

            cash -= basis
            total_fees += fee

            impact += qty * (
                px - float(row.open)
            )

            entry_time = ts
            entry_price = px
            highest_close = float(row.close)

        pending = None

        if qty > 0:
            highest_close = max(
                highest_close,
                float(row.close),
            )

        # Decide only after the candle has closed.
        # Execution occurs at the next candle's open.
        if i < stop - 1:
            action = decide(
                df,
                i,
                strategy,
                qty > 0,
                highest_close,
            )

            if action:
                pending = (
                    "sell" if qty > 0 else "buy"
                )

        liquidation_value = (
            qty
            * float(row.close)
            * sell_mult
            * (1 - costs.fee_per_side)
        )

        curve.append((
            ts,
            cash + liquidation_value,
        ))

    # Close remaining position for fold accounting.
    if qty > 0:
        sell(
            df.iloc[stop - 1].close,
            df.index[stop - 1],
            "End of evaluation",
        )

        curve[-1] = (
            df.index[stop - 1],
            cash,
        )

    equity = pd.DataFrame(
        curve,
        columns=["Timestamp", "Equity USD"],
    ).set_index("Timestamp")

    log = pd.DataFrame(trades)

    realised = (
        float(log["Net PnL USD"].sum())
        if len(log)
        else 0.0
    )

    if not np.isclose(
        cash,
        initial_cash + realised,
        atol=1e-7,
    ):
        raise AssertionError(
            "Ledger does not reconcile"
        )

    peaks = np.maximum(
        float(initial_cash),
        equity["Equity USD"].cummax(),
    )

    drawdown = float(
        (
            100
            * (equity["Equity USD"] / peaks - 1)
        ).min()
    )

    if len(log):
        wins = log.loc[
            log["Net PnL USD"] > 0,
            "Net PnL USD",
        ]

        losses = log.loc[
            log["Net PnL USD"] < 0,
            "Net PnL USD",
        ]
    else:
        wins = pd.Series(dtype=float)
        losses = pd.Series(dtype=float)

    if losses.sum() < 0:
        profit_factor = float(
            wins.sum() / -losses.sum()
        )
    elif wins.sum() > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None

    # Cost-adjusted buy-and-hold benchmark.
    budget = initial_cash * allocation

    buy_px = (
        float(df.iloc[first].open)
        * buy_mult
    )

    hold_qty = budget / (
        buy_px * (1 + costs.fee_per_side)
    )

    hold_final = (
        initial_cash
        - budget
        + hold_qty
        * float(df.iloc[stop - 1].close)
        * sell_mult
        * (1 - costs.fee_per_side)
    )

    summary = {
        "Strategy": strategy,
        "Initial USD": round(initial_cash, 2),
        "Final USD": round(cash, 2),
        "Return %": round(
            100 * (cash / initial_cash - 1),
            3,
        ),
        "Buy and Hold Return %": round(
            100 * (hold_final / initial_cash - 1),
            3,
        ),
        "Maximum Drawdown %": round(
            drawdown,
            3,
        ),
        "Trades": len(log),
        "Win Rate %": (
            round(
                100 * len(wins) / len(log),
                2,
            )
            if len(log)
            else None
        ),
        "Profit Factor": profit_factor,
        "Fees USD": round(
            total_fees,
            4,
        ),
        "Spread and Slippage USD (approx)": round(
            impact,
            4,
        ),
        "Evaluation Start": str(
            df.index[first]
        ),
        "Evaluation End": str(
            df.index[stop - 1]
        ),
    }

    return summary, log, equity
