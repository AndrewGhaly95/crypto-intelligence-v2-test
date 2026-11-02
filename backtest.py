
"""Historical backtesting engine for Crypto Intelligence V2.

Research and simulation only. No real orders are placed.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies import signals


@dataclass
class Costs:
    fee_per_side: float = 0.001
    slippage_per_side: float = 0.0005
    spread_round_trip: float = 0.001

    def __post_init__(self):
        for name in (
            "fee_per_side",
            "slippage_per_side",
            "spread_round_trip",
        ):
            value = getattr(self, name)

            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )


def backtest(
    frame,
    strategy,
    initial_cash=500.0,
    allocation=0.5,
    costs=None,
    stop_pct=0.012,
    target_pct=0.024,
):
    """
    Run a long-only historical trading simulation.

    Signals are calculated on completed candles.
    Entries and signal-based exits occur at the
    following candle's opening price.

    Returns:
        summary: dictionary of performance statistics
        trades: DataFrame of completed trades
        curve: DataFrame of simulated account equity
    """

    if costs is None:
        costs = Costs()

    if not isinstance(costs, Costs):
        raise TypeError("costs must be a Costs instance.")

    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive.")

    if not np.isfinite(allocation) or not 0 < allocation <= 1:
        raise ValueError("allocation must be between 0 and 1.")

    if not 0 < stop_pct < 1:
        raise ValueError("Invalid stop percentage.")

    if not 0 < target_pct < 1:
        raise ValueError("Invalid target percentage.")

    data = signals(frame, strategy)

    if len(data) < 2:
        raise ValueError(
            "At least two candles are required."
        )

    cash = float(initial_cash)
    quantity = 0.0

    entry_price = None
    entry_time = None
    entry_cost = 0.0

    trade_records = []
    equity_records = []

    # Half of the assumed round-trip spread per side.
    half_spread = costs.spread_round_trip / 2.0

    buy_multiplier = (
        1.0 + half_spread + costs.slippage_per_side
    )

    sell_multiplier = (
        1.0 - half_spread - costs.slippage_per_side
    )

    def close_position(timestamp, raw_price, reason):
        nonlocal cash
        nonlocal quantity
        nonlocal entry_price
        nonlocal entry_time
        nonlocal entry_cost

        execution_price = raw_price * sell_multiplier

        gross_proceeds = quantity * execution_price
        exit_fee = gross_proceeds * costs.fee_per_side
        net_proceeds = gross_proceeds - exit_fee

        cash += net_proceeds

        pnl = net_proceeds - entry_cost

        trade_records.append(
            {
                "Strategy": strategy,
                "Entry Time": entry_time,
                "Exit Time": timestamp,
                "Entry Price": entry_price,
                "Exit Price": execution_price,
                "Quantity": quantity,
                "PnL AUD": pnl,
                "Return %": (
                    pnl / entry_cost * 100
                    if entry_cost > 0
                    else 0.0
                ),
                "Exit Reason": reason,
            }
        )

        quantity = 0.0
        entry_price = None
        entry_time = None
        entry_cost = 0.0

    for i in range(len(data)):
        timestamp = data.index[i]
        candle = data.iloc[i]

        raw_open = float(candle["open"])
        raw_high = float(candle["high"])
        raw_low = float(candle["low"])
        raw_close = float(candle["close"])

        # Execute signals from the previous closed candle.
        if i > 0:
            previous = data.iloc[i - 1]

            if quantity > 0 and bool(previous["exit"]):
                close_position(
                    timestamp,
                    raw_open,
                    "Signal exit",
                )

            elif quantity == 0 and bool(previous["entry"]):
                budget = cash * allocation

                execution_price = (
                    raw_open * buy_multiplier
                )

                # Budget includes the entry trading fee.
                quantity = budget / (
                    execution_price
                    * (1.0 + costs.fee_per_side)
                )

                entry_cost = (
                    quantity
                    * execution_price
                    * (1.0 + costs.fee_per_side)
                )

                cash -= entry_cost

                entry_price = execution_price
                entry_time = timestamp

        # Intracandle stop and target checks.
        if quantity > 0:
            stop_price = entry_price * (
                1.0 - stop_pct
            )

            target_price = entry_price * (
                1.0 + target_pct
            )

            # Conservative assumption when both levels
            # are touched in the same candle: stop first.
            if raw_low <= stop_price:
                close_position(
                    timestamp,
                    min(raw_open, stop_price),
                    "Stop loss",
                )

            elif raw_high >= target_price:
                close_position(
                    timestamp,
                    max(raw_open, target_price),
                    "Take profit",
                )

        equity = (
            cash + quantity * raw_close * sell_multiplier
            * (1.0 - costs.fee_per_side)
        )

        equity_records.append(
            {
                "Timestamp": timestamp,
                "Equity AUD": equity,
            }
        )

    # Close remaining positions at the final candle close.
    if quantity > 0:
        close_position(
            data.index[-1],
            float(data.iloc[-1]["close"]),
            "End of backtest",
        )

        equity_records[-1]["Equity AUD"] = cash

    trades = pd.DataFrame(
        trade_records,
        columns=[
            "Strategy",
            "Entry Time",
            "Exit Time",
            "Entry Price",
            "Exit Price",
            "Quantity",
            "PnL AUD",
            "Return %",
            "Exit Reason",
        ],
    )

    curve = pd.DataFrame(equity_records).set_index(
        "Timestamp"
    )

    final_equity = float(cash)

    total_return = (
        final_equity / initial_cash - 1.0
    ) * 100.0

    running_peak = curve["Equity AUD"].cummax()

    drawdown = (
        curve["Equity AUD"] / running_peak - 1.0
    ) * 100.0

    max_drawdown = float(drawdown.min())

    total_trades = len(trades)

    winning_trades = (
        int((trades["PnL AUD"] > 0).sum())
        if total_trades
        else 0
    )

    win_rate = (
        winning_trades / total_trades * 100.0
        if total_trades
        else 0.0
    )

    summary = {
        "Strategy": strategy,
        "Initial Capital AUD": initial_cash,
        "Final Equity AUD": round(final_equity, 2),
        "Net Profit AUD": round(
            final_equity - initial_cash, 2
        ),
        "Total Return %": round(total_return, 2),
        "Maximum Drawdown %": round(
            max_drawdown, 2
        ),
        "Trades": total_trades,
        "Win Rate %": round(win_rate, 2),
    }

    return summary, trades, curve
