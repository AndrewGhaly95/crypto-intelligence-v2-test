
"""Standalone Crypto Intelligence V2 research dashboard."""

import pandas as pd
import streamlit as st

from market_candles import get_candles
from backtest import backtest, Costs


st.set_page_config(
    page_title="Crypto Intelligence V2",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Crypto Intelligence V2")
st.caption("Experimental scalping and intraday backtesting")

st.warning(
    "Research only. No real orders. Historical simulations "
    "are not validated trading performance."
)

coin = st.selectbox(
    "Cryptocurrency",
    ["BTC", "ETH", "SOL", "XRP", "BNB"],
)

count = st.slider(
    "Historical candles",
    min_value=100,
    max_value=1000,
    value=500,
    step=100,
)

st.subheader("Transaction costs")

fee = st.number_input(
    "Fee per side (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.1,
    step=0.05,
)

spread = st.number_input(
    "Round-trip spread (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.1,
    step=0.05,
)

slippage = st.number_input(
    "Slippage per side (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.05,
    step=0.05,
)

if st.button("Run strategy comparison", type="primary"):
    try:
        costs = Costs(
            fee_per_side=fee / 100,
            spread_round_trip=spread / 100,
            slippage_per_side=slippage / 100,
        )

        rows = []

        for strategy, interval in [
            ("Scalping", "5m"),
            ("Intraday", "1h"),
        ]:
            candles = get_candles(coin, interval, count)

            summary, trades, curve = backtest(
                candles,
                strategy,
                initial_cash=500.0,
                costs=costs,
            )

            rows.append(summary)

            with st.expander(f"{strategy} details"):
                st.line_chart(curve["Equity AUD"])
                st.dataframe(trades)

        st.subheader("Strategy comparison")

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Strategies use different historical windows. "
            "Results are not directly comparable until "
            "matched-period testing is implemented."
        )

    except Exception as exc:
        st.error("Backtest failed")
        st.caption(str(exc))
