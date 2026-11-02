
"""Crypto Intelligence V2: matched-period strategy comparison."""

import pandas as pd
import streamlit as st

from historical_data import update_history, history_quality

from coinbase_history import get_coinbase_history

from backtest_v2 import run_backtest, TradingCosts




from market_candles import get_candles
from backtest import backtest, Costs

st.set_page_config(
    page_title="Crypto Intelligence V2",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Crypto Intelligence V2")
st.caption("Matched-period scalping and intraday research")

st.warning(
    "Simulation only. No live orders. Historical results "
    "do not establish future profitability."
)

coin = st.selectbox(
    "Cryptocurrency",
    ["BTC", "ETH", "SOL", "XRP", "BNB"],
)

count = st.slider(
    "Historical 5-minute candles",
    100, 700, 500, 100,
)

st.subheader("Transaction costs")

fee = st.number_input(
    "Fee per side (%)",
    0.0, 5.0, 0.10, 0.05,
)

spread = st.number_input(
    "Round-trip spread (%)",
    0.0, 5.0, 0.10, 0.05,
)

slippage = st.number_input(
    "Slippage per side (%)",
    0.0, 5.0, 0.05, 0.05,
)

if st.button("Run matched-period comparison", type="primary"):
    try:
        costs = Costs(
            fee_per_side=fee / 100,
            spread_round_trip=spread / 100,
            slippage_per_side=slippage / 100,
        )

        five_min = get_candles(coin, "5m", count)
        hourly = get_candles(coin, "1h", 100)

        # The hourly candles must be fully contained
        # within the available five-minute history.
        start = five_min.index.min()
        end = five_min.index.max() + pd.Timedelta(minutes=5)

        hourly = hourly[
            (hourly.index >= start)
            & (
                hourly.index + pd.Timedelta(hours=1)
                <= end
            )
        ].copy()

        if len(hourly) < 2:
            st.error(
                "Insufficient overlapping hourly candles. "
                "Choose more five-minute candles."
            )
            st.stop()

        st.caption(
            f"5-minute period: {start} to {end} UTC"
        )

        rows = []

        for strategy, candles in [
            ("Scalping", five_min),
            ("Intraday", hourly),
        ]:
            summary, trades, curve = backtest(
                candles,
                strategy,
                initial_cash=500.0,
                costs=costs,
            )

            rows.append(summary)

            with st.expander(f"{strategy} details"):
                st.line_chart(curve["Equity AUD"])
                st.dataframe(
                    trades,
                    use_container_width=True,
                )

        st.subheader("Strategy comparison")

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "Both strategies now use overlapping calendar "
            "history. The sample is short and results "
            "remain exploratory."
        )

    except Exception as exc:
        st.error("Backtest failed")
        st.caption(str(exc))
        
st.divider()
st.header("Historical Data Diagnostics")

st.caption(
    "Inspect available historical candles and identify "
    "missing intervals before running longer backtests."
)

diagnostic_interval = st.selectbox(
    "Diagnostic candle interval",
    ["5m", "15m", "1h", "4h"],
)

if st.button("Update and inspect history"):
    try:
        with st.spinner("Retrieving historical candles..."):
            history = update_history(
                coin,
                diagnostic_interval,
                limit=700,
            )

            quality = history_quality(
                history,
                diagnostic_interval,
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Candles",
            quality["candles"],
        )

        col2.metric(
            "Coverage",
            f"{quality['coverage_pct']:.1f}%",
        )

        col3.metric(
            "Missing intervals",
            quality["missing_intervals"],
        )

        if quality["candles"] > 0:
            st.write(
                "First candle:",
                quality["start"],
            )

            st.write(
                "Last candle:",
                quality["end"],
            )

            st.line_chart(
                history["close"]
            )

            if quality["candles"] < 200:
                st.warning(
                    "Limited historical observations. "
                    "Strategy indicators may not have "
                    "adequate warm-up history."
                )

            st.info(
                "Historical data is cached locally. "
                "Streamlit Community Cloud may erase "
                "this cache during app restarts."
            )

    except Exception as exc:
        st.error("Historical data update failed")
        st.caption(str(exc))
        
st.divider()

st.header("Extended Historical Data")

st.caption(
    "Retrieve longer historical cryptocurrency datasets "
    "from Coinbase Exchange. Prices are in USD."
)

history_coin = st.selectbox(
    "Historical cryptocurrency",
    ["BTC", "ETH"],
    key="extended_history_coin",
)

history_interval = st.selectbox(
    "Historical interval",
    ["1h", "6h", "1d", "5m", "15m"],
    key="extended_history_interval",
)

history_days = st.selectbox(
    "Historical period (days)",
    [30, 90, 180],
    index=1,
    key="extended_history_days",
)

if st.button(
    "Retrieve extended history",
    type="primary",
    key="retrieve_extended_history",
):
    try:
        with st.spinner(
            "Downloading and validating historical candles..."
        ):
            historical = get_coinbase_history(
                symbol=history_coin,
                interval=history_interval,
                days=history_days,
            )

        expected = historical.attrs.get(
            "expected_candles", 0
        )

        missing = historical.attrs.get(
            "missing_candles", 0
        )

        actual = len(historical)

        coverage = (
            100 * actual / expected
            if expected
            else 0
        )

        st.success(
            "Historical data retrieval completed."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Candles retrieved",
            f"{actual:,}",
        )

        col2.metric(
            "Coverage",
            f"{coverage:.2f}%",
        )

        col3.metric(
            "Missing candles",
            f"{missing:,}",
        )

        st.write(
            "First candle:",
            historical.index.min(),
        )

        st.write(
            "Last candle:",
            historical.index.max(),
        )

        st.write(
            "Source: Coinbase Exchange"
        )

        st.write(
            "Quote currency: USD"
        )

        st.line_chart(
            historical["close"]
        )

        if missing:
            st.warning(
                "Missing candle intervals detected. "
                "These must be reviewed before backtesting."
            )

        if actual < 200:
            st.warning(
                "Limited historical observations. "
                "Indicator warm-up may be insufficient."
            )

        st.download_button(
            "Download historical CSV",
            data=historical.to_csv().encode("utf-8"),
            file_name=(
                f"{history_coin}_"
                f"{history_interval}_"
                f"{history_days}d.csv"
            ),
            mime="text/csv",
        )

    except Exception as exc:
        st.error(
            "Extended historical retrieval failed."
        )

        st.code(str(exc))
        
st.divider()
st.header("V2.1 Validated Backtesting")

st.caption(
    "Research-only EMA20/EMA50 crossover backtest "
    "using Coinbase USD market data. "
    "No real trades are placed."
)

bt_coin = st.selectbox(
    "Backtest cryptocurrency",
    ["BTC", "ETH"],
    key="v21_coin",
)

bt_days = st.selectbox(
    "Backtest historical period",
    [30, 90, 180],
    index=1,
    key="v21_days",
)

bt_allocation = st.slider(
    "Capital allocation (%)",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
    key="v21_allocation",
)

if st.button(
    "Run V2.1 backtest",
    type="primary",
    key="run_v21",
):
    try:
        with st.spinner(
            "Retrieving history and calculating backtest..."
        ):
            candles = get_coinbase_history(
                symbol=bt_coin,
                interval="1h",
                days=bt_days,
            )

            missing = candles.attrs.get(
                "missing_candles", 0
            )

            if missing:
                raise ValueError(
                    f"Dataset has {missing} missing candles. "
                    "Backtest cancelled."
                )

            costs = TradingCosts(
                fee_per_side=0.001,
                slippage_per_side=0.0005,
                spread_round_trip=0.001,
            )

            summary, trades, equity = run_backtest(
                candles,
                initial_cash=500.0,
                allocation=bt_allocation / 100,
                warmup=200,
                costs=costs,
            )

        st.success("Backtest completed.")

        st.subheader("Strategy Performance")

        c1, c2 = st.columns(2)

        c1.metric(
            "Final Equity (USD)",
            f"${summary['Final USD']:,.2f}",
        )

        c2.metric(
            "Strategy Return",
            f"{summary['Return %']:+.2f}%",
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "Buy & Hold Return",
            f"{summary['Buy and Hold Return %']:+.2f}%",
        )

        c4.metric(
            "Maximum Drawdown",
            f"{summary['Maximum Drawdown %']:.2f}%",
        )

        st.subheader("Trade Statistics")

        st.write("Trades:", summary["Trades"])

        st.write(
            "Win rate:",
            summary["Win Rate %"],
        )

        st.write(
            "Profit factor:",
            summary["Profit Factor"],
        )

        st.write(
            "Trading fees (USD):",
            summary["Fees USD"],
        )

        st.write(
            "Spread and slippage (USD):",
            summary["Spread and Slippage USD (approx)"],
        )

        st.write(
            "Evaluation start:",
            summary["Evaluation Start"],
        )

        st.write(
            "Evaluation end:",
            summary["Evaluation End"],
        )

        st.subheader("Equity Curve")

        st.line_chart(equity["Equity USD"])

        st.subheader("Trade Log")

        if trades.empty:
            st.info(
                "No trades occurred during this period."
            )
        else:
            st.dataframe(
                trades,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        st.error("Backtest failed.")
        st.code(str(exc))




