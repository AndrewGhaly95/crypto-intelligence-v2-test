
"""Conservative long-only OHLCV backtester. No real orders."""

from dataclasses import dataclass
import math
import pandas as pd
from strategies import signals


@dataclass
class Costs:
    fee_per_side: float = 0.001
    slippage_per_side: float = 0.0005
    spread_round_trip: float = 0.001

    def __post_init__(self):
        for key in (
            "fee_per_side",
            "slippage_per_side",
            "spread_round_trip",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"Invalid {key}")

        if (
            self.spread_round_trip / 2
            + self.slippage_per_side >= 1
        ):
            raise ValueError("Execution costs are excessive")


def backtest(
    frame,
    strategy,
    initial_cash=500.0,
    allocation=0.5,
    costs=None,
    stop_pct=0.012,
    target_pct=0.024,
):
    costs = costs if costs is not None else Costs()

    if not isinstance(costs, Costs):
        raise TypeError("costs must be Costs")

    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    if not math.isfinite(allocation) or not 0 < allocation <= 1:
        raise ValueError("Invalid allocation")

    if not math.isfinite(stop_pct) or not 0 < stop_pct < 1:
        raise ValueError("Invalid stop percentage")

    if not math.isfinite(target_pct) or not 0 < target_pct < 1:
        raise ValueError("Invalid target percentage")

    data = signals(frame, strategy)

    if len(data) < 2:
        raise ValueError("At least two candles required")

    buy_factor = (
        1
        + costs.spread_round_trip / 2
        + costs.slippage_per_side
    )

    sell_factor = (
        1
        - costs.spread_round_trip / 2
        - costs.slippage_per_side
    )

    cash = float(initial_cash)
    qty = 0.0

    entry_price = None
    entry_cost = None
    entry_time = None

    trades = []
    equity = []

    def sell(time, raw_price, reason):
        nonlocal cash, qty
        nonlocal entry_price, entry_cost, entry_time

        px = raw_price * sell_factor

        proceeds = (
            qty * px * (1 - costs.fee_per_side)
        )

        pnl = proceeds - entry_cost
        cash += proceeds

        trades.append({
            "Strategy": strategy,
            "Entry Time": entry_time,
            "Exit Time": time,
            "Entry Price": entry_price,
            "Exit Price": px,
            "Quantity": qty,
            "PnL AUD": pnl,
            "Return %": 100 * pnl / entry_cost,
            "Exit Reason": reason,
        })

        qty = 0.0
        entry_price = None
        entry_cost = None
        entry_time = None

    for i, (time, bar) in enumerate(data.iterrows()):
        op = float(bar["open"])
        hi = float(bar["high"])
        lo = float(bar["low"])
        cl = float(bar["close"])

        # Signals are acted upon at the next candle open.
        if i > 0:
            prev = data.iloc[i - 1]

            if qty > 0 and bool(prev["exit"]):
                sell(time, op, "Signal exit")

            elif qty == 0 and bool(prev["entry"]):
                budget = cash * allocation
                px = op * buy_factor

                new_qty = budget / (
                    px * (1 + costs.fee_per_side)
                )

                if new_qty > 0:
                    qty = new_qty
                    entry_price = px
                    entry_cost = (
                        qty
                        * px
                        * (1 + costs.fee_per_side)
                    )
                    entry_time = time
                    cash -= entry_cost

        if qty > 0:
            stop_bid = entry_price * (
                1 - stop_pct
            )

            target_bid = entry_price * (
                1 + target_pct
            )

            open_bid = op * sell_factor
            low_bid = lo * sell_factor
            high_bid = hi * sell_factor

            # Opening gaps take precedence.
            if open_bid <= stop_bid:
                sell(time, op, "Stop gap/open")

            elif open_bid >= target_bid:
                sell(time, op, "Target gap/open")

            # If both intrabar levels are reached,
            # assume the stop is triggered first.
            elif low_bid <= stop_bid:
                sell(
                    time,
                    stop_bid / sell_factor,
                    "Stop loss",
                )

            elif high_bid >= target_bid:
                sell(
                    time,
                    target_bid / sell_factor,
                    "Take profit",
                )

        marked_equity = (
            cash
            + qty
            * cl
            * sell_factor
            * (1 - costs.fee_per_side)
        )

        equity.append({
            "Timestamp": time,
            "Equity AUD": marked_equity,
        })

    if qty > 0:
        sell(
            data.index[-1],
            float(data.iloc[-1]["close"]),
            "End of test",
        )

        equity[-1]["Equity AUD"] = cash

    trade_columns = [
        "Strategy",
        "Entry Time",
        "Exit Time",
        "Entry Price",
        "Exit Price",
        "Quantity",
        "PnL AUD",
        "Return %",
        "Exit Reason",
    ]

    trade_df = pd.DataFrame(
        trades,
        columns=trade_columns,
    )

    curve = pd.DataFrame(equity).set_index(
        "Timestamp"
    )

    peak = curve["Equity AUD"].cummax().clip(
        lower=initial_cash
    )

    max_dd = float(
        (
            curve["Equity AUD"] / peak - 1
        ).min() * 100
    )

    summary = {
        "Strategy": strategy,
        "Initial Capital AUD": initial_cash,
        "Final Equity AUD": round(cash, 2),
        "Net Profit AUD": round(
            cash - initial_cash, 2
        ),
        "Total Return %": round(
            100 * (cash / initial_cash - 1), 2
        ),
        "Maximum Drawdown %": round(max_dd, 2),
        "Trades": len(trade_df),
        "Win Rate %": (
            round(
                100 * (
                    trade_df["PnL AUD"] > 0
                ).mean(),
                2,
            )
            if len(trade_df)
            else 0.0
        ),
    }

    return summary, trade_df, curve

