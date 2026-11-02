
"""Crypto Intelligence V2: matched-period strategy comparison."""

import pandas as pd
import streamlit as st

from historical_data import update_history, history_quality


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


