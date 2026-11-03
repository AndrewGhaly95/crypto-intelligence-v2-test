
"""Crypto Intelligence V2.7 research dashboard.

Research and simulation only. No live trading.
Historical prices and simulated balances are in USD.
"""

import pandas as pd
import streamlit as st

from historical_data import update_history, history_quality
from coinbase_history import get_coinbase_history
from market_candles import get_candles
from backtest import backtest, Costs
from backtest_v2 import run_backtest, TradingCosts
from backtest_v3 import run_backtest_v3, TradingCostsV3
from walk_forward import run_walk_forward, WalkForwardConfig
from test_engine import run_checks
from test_engine_v3 import run_v3_checks
from test_multi_v25 import run_v25_checks
from backtest_multi_v25 import backtest_strategy_v25, CostsV25
from strategies_v25 import STRATEGIES
from walk_forward_v27 import run_rolling_v27


st.set_page_config(
    page_title="Crypto Intelligence V2.7",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Crypto Intelligence V2.7")

st.warning(
    "Simulation only. No live orders. Historical results "
    "do not establish future profitability. Coinbase USD "
    "backtests are not executable CoinSpot AUD returns."
)


# ==========================================================
# SHARED HELPERS
# ==========================================================

def history_for(symbol, days):
    data = get_coinbase_history(
        symbol=symbol,
        interval="1h",
        days=days,
    )

    if (
        not isinstance(data, pd.DataFrame)
        or data.empty
        or not isinstance(data.index, pd.DatetimeIndex)
    ):
        raise ValueError("Invalid or empty historical dataset")

    if (
        data.index.tz is None
        or data.index.has_duplicates
        or not data.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Historical timestamps must be timezone-aware, "
            "unique and ascending."
        )

    expected = pd.date_range(
        start=data.index[0],
        end=data.index[-1],
        freq="h",
    )

    missing_times = expected.difference(data.index)
    off_grid = data.index.difference(expected)

    reported_missing = int(
        data.attrs.get("missing_candles", 0) or 0
    )

    stats = {
        "Candles retrieved": len(data),
        "Provider missing": reported_missing,
        "Missing within span": len(missing_times),
        "Off-grid timestamps": len(off_grid),
        "First candle": str(data.index[0]),
        "Last candle": str(data.index[-1]),
        "Coverage %": round(
            100 * len(data) / len(expected), 3
        ),
    }

    return data, stats, missing_times


def require_contiguous(stats):
    if (
        stats["Provider missing"]
        or stats["Missing within span"]
        or stats["Off-grid timestamps"]
    ):
        raise ValueError(
            "Incomplete hourly history: "
            f"{stats['Provider missing']} provider-reported "
            "missing candles, "
            f"{stats['Missing within span']} gaps within "
            "the returned span, and "
            f"{stats['Off-grid timestamps']} off-grid "
            "timestamps. Evaluation cancelled."
        )


def display_history_diagnostics(stats, missing_times):
    st.subheader("Historical Data Diagnostics")

    st.dataframe(
        pd.DataFrame([stats]),
        hide_index=True,
        use_container_width=True,
    )

    if len(missing_times):
        st.subheader("Missing Candle Timestamps")

        st.dataframe(
            pd.DataFrame({
                "Missing UTC Timestamp": [
                    timestamp.isoformat()
                    for timestamp in missing_times
                ]
            }),
            hide_index=True,
            use_container_width=True,
        )

        st.warning(
            f"{len(missing_times)} missing hourly candle(s) "
            "detected. No synthetic prices will be inserted."
        )
    else:
        st.success(
            "No missing hourly timestamps within "
            "the returned data span."
        )


def show_backtest(summary, trades, equity):
    st.dataframe(
        pd.DataFrame([summary]),
        hide_index=True,
        use_container_width=True,
    )

    st.line_chart(equity["Equity USD"])

    with st.expander("Trade log"):
        st.dataframe(
            trades,
            hide_index=True,
            use_container_width=True,
        )


# ==========================================================
# MATCHED-PERIOD SCALPING AND INTRADAY RESEARCH
# ==========================================================

st.divider()
st.header("Matched-Period Scalping and Intraday Research")

coin = st.selectbox(
    "Cryptocurrency",
    ["BTC", "ETH", "SOL", "XRP", "BNB"],
    key="legacy_coin",
)

count = st.slider(
    "Historical 5-minute candles",
    100, 700, 500, 100,
)

fee = st.number_input(
    "Fee per side (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.10,
    step=0.05,
)

spread = st.number_input(
    "Round-trip spread (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.10,
    step=0.05,
)

slippage = st.number_input(
    "Slippage per side (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.05,
    step=0.05,
)

if st.button(
    "Run matched-period comparison",
    key="legacy_run",
):
    try:
        costs = Costs(
            fee_per_side=fee / 100,
            spread_round_trip=spread / 100,
            slippage_per_side=slippage / 100,
        )

        five_min = get_candles(coin, "5m", count)
        hourly = get_candles(coin, "1h", 100)

        start = five_min.index.min()
        end = (
            five_min.index.max()
            + pd.Timedelta(minutes=5)
        )

        hourly = hourly[
            (hourly.index >= start)
            & (
                hourly.index + pd.Timedelta(hours=1)
                <= end
            )
        ].copy()

        if len(hourly) < 2:
            raise ValueError(
                "Insufficient overlapping hourly candles."
            )

        rows = []

        for strategy, candles in (
            ("Scalping", five_min),
            ("Intraday", hourly),
        ):
            summary, trades, equity = backtest(
                candles,
                strategy,
                initial_cash=500.0,
                costs=costs,
            )

            rows.append(summary)

            with st.expander(f"{strategy} details"):
                st.line_chart(equity["Equity AUD"])
                st.dataframe(
                    trades,
                    hide_index=True,
                    use_container_width=True,
                )

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

    except Exception as exc:
        st.error("Matched-period comparison failed.")
        st.code(str(exc))


# ==========================================================
# HISTORICAL DATA DIAGNOSTICS
# ==========================================================

st.divider()
st.header("Historical Data Diagnostics")

diagnostic_interval = st.selectbox(
    "Diagnostic candle interval",
    ["5m", "15m", "1h", "4h"],
)

if st.button(
    "Update and inspect history",
    key="diagnostic_run",
):
    try:
        history = update_history(
            coin,
            diagnostic_interval,
            limit=700,
        )

        quality = history_quality(
            history,
            diagnostic_interval,
        )

        st.dataframe(
            pd.DataFrame([quality]),
            hide_index=True,
            use_container_width=True,
        )

        st.line_chart(history["close"])

        st.caption(
            "Locally cached history may be cleared "
            "when Streamlit restarts."
        )

    except Exception as exc:
        st.error("History diagnostics failed.")
        st.code(str(exc))


# ==========================================================
# EXTENDED HISTORICAL DATA
# ==========================================================

st.divider()
st.header("Extended Historical Data")

history_coin = st.selectbox(
    "Historical cryptocurrency",
    ["BTC", "ETH"],
    key="extended_coin",
)

history_interval = st.selectbox(
    "Historical interval",
    ["1h", "6h", "1d", "5m", "15m"],
    key="extended_interval",
)

history_days = st.selectbox(
    "Historical period (days)",
    [30, 90, 180],
    index=1,
    key="extended_days",
)

if st.button(
    "Retrieve extended history",
    key="extended_run",
):
    try:
        data = get_coinbase_history(
            symbol=history_coin,
            interval=history_interval,
            days=history_days,
        )

        st.write("Candles retrieved:", len(data))

        st.write(
            "Missing candles:",
            data.attrs.get("missing_candles", "Unknown"),
        )

        st.write(
            "Recovered candles:",
            data.attrs.get("recovered_candles", "Unknown"),
        )

        st.write(
            "First candle:", data.index.min()
        )

        st.write(
            "Last candle:", data.index.max()
        )

        st.line_chart(data["close"])

        st.download_button(
            "Download historical CSV",
            data=data.to_csv().encode("utf-8"),
            file_name=(
                f"{history_coin}_"
                f"{history_interval}_"
                f"{history_days}d.csv"
            ),
            mime="text/csv",
        )

    except Exception as exc:
        st.error("Extended historical retrieval failed.")
        st.code(str(exc))


# ==========================================================
# V2.1 VALIDATED BACKTESTING
# ==========================================================

st.divider()
st.header("V2.1 Validated Backtesting")

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
    "Backtest allocation (%)",
    10, 100, 50, 10,
    key="v21_allocation",
)

if st.button(
    "Run V2.1 backtest",
    key="v21_run",
):
    try:
        data, stats, missing_times = history_for(
            bt_coin, bt_days
        )

        require_contiguous(stats)

        summary, trades, equity = run_backtest(
            data,
            initial_cash=500.0,
            allocation=bt_allocation / 100,
            warmup=200,
            costs=TradingCosts(),
        )

        show_backtest(summary, trades, equity)

    except Exception as exc:
        st.error("V2.1 backtest failed.")
        st.code(str(exc))


# ==========================================================
# V2.2 CHRONOLOGICAL EVALUATION
# ==========================================================

st.divider()
st.header("V2.2 Chronological Holdout Evaluation")

wf_coin = st.selectbox(
    "Holdout cryptocurrency",
    ["BTC", "ETH"],
    key="wf_coin",
)

wf_days = st.selectbox(
    "Holdout historical days",
    [90, 180],
    key="wf_days",
)

wf_train_pct = st.slider(
    "Training proportion (%)",
    50, 75, 65, 5,
    key="wf_train_pct",
)

if st.button(
    "Run chronological evaluation",
    key="wf_run",
):
    try:
        data, stats, missing_times = history_for(
            wf_coin, wf_days
        )

        require_contiguous(stats)

        results = run_walk_forward(
            data,
            config=WalkForwardConfig(
                train_fraction=wf_train_pct / 100,
                warmup_candles=200,
                initial_cash=500.0,
                allocation=0.5,
            ),
            costs=TradingCosts(),
        )

        st.write(
            "Testing begins:",
            results["split_time"],
        )

        st.dataframe(
            results["comparison"],
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Training Equity")
        st.line_chart(
            results["train_equity"]["Equity USD"]
        )

        st.subheader("Testing Equity")
        st.line_chart(
            results["test_equity"]["Equity USD"]
        )

        with st.expander("Training trade log"):
            st.dataframe(
                results["train_trades"],
                hide_index=True,
            )

        with st.expander("Testing trade log"):
            st.dataframe(
                results["test_trades"],
                hide_index=True,
            )

        st.info(
            "This is a single chronological holdout, "
            "not rolling walk-forward validation."
        )

    except Exception as exc:
        st.error("Chronological evaluation failed.")
        st.code(str(exc))


# ==========================================================
# V2.3 AND V2.5 ENGINE VALIDATION
# ==========================================================

st.divider()
st.header("V2.3 and V2.5 Engine Validation")

validation_runners = (
    ("V2 engine", "v2_checks", run_checks),
    ("V3 engine", "v3_checks", run_v3_checks),
    (
        "Multi-strategy engine",
        "v25_checks",
        run_v25_checks,
    ),
)

for label, key, runner in validation_runners:
    if st.button(
        f"Run {label} checks",
        key=key,
    ):
        try:
            result = runner()

            st.write(
                f"{label}: "
                f"{result['passed']} / "
                f"{result['total']} passed"
            )

            if result["passed"] == result["total"]:
                st.success("All available checks passed.")
            else:
                st.error("Some checks failed.")

                for message in (
                    result.get("failures", [])
                    + result.get("errors", [])
                ):
                    st.code(str(message))

        except Exception as exc:
            st.error("Validation runner failed.")
            st.code(str(exc))


# ==========================================================
# V2.4 ENGINE COMPARISON
# ==========================================================

st.divider()
st.header("V2.4 Backtesting Engine Comparison")

compare_coin = st.selectbox(
    "Comparison cryptocurrency",
    ["BTC", "ETH"],
    key="v24_coin",
)

compare_days = st.selectbox(
    "Comparison historical days",
    [90, 180],
    key="v24_days",
)

compare_allocation = st.slider(
    "Comparison allocation (%)",
    10, 100, 50, 10,
    key="v24_allocation",
)

if st.button(
    "Run V2 versus V3 comparison",
    key="v24_run",
):
    try:
        data, stats, missing_times = history_for(
            compare_coin, compare_days
        )

        require_contiguous(stats)

        rates = {
            "fee_per_side": 0.001,
            "slippage_per_side": 0.0005,
            "spread_round_trip": 0.001,
        }

        v2_summary, v2_trades, v2_equity = (
            run_backtest(
                data,
                initial_cash=500.0,
                allocation=compare_allocation / 100,
                warmup=200,
                costs=TradingCosts(**rates),
            )
        )

        v3_summary, v3_trades, v3_equity = (
            run_backtest_v3(
                data,
                initial_cash=500.0,
                allocation=compare_allocation / 100,
                warmup=200,
                costs=TradingCostsV3(**rates),
            )
        )

        comparison = pd.DataFrame([
            {"Engine": "V2", **v2_summary},
            {"Engine": "V3", **v3_summary},
        ])

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )

        st.line_chart(v3_equity["Equity USD"])

        with st.expander("V2 trade log"):
            st.dataframe(v2_trades)

        with st.expander("V3 trade log"):
            st.dataframe(v3_trades)

    except Exception as exc:
        st.error("Engine comparison failed.")
        st.code(str(exc))


# ==========================================================
# V2.6 MULTI-ASSET STRATEGY RESEARCH
# ==========================================================

st.divider()
st.header("V2.6 Multi-Asset Strategy Research")

v26_days = st.selectbox(
    "Multi-asset history (days)",
    [90, 180],
    key="v26_days",
)

v26_train_pct = st.slider(
    "Training proportion (%)",
    50, 80, 65, 5,
    key="v26_train_pct",
)

v26_allocation = st.slider(
    "Multi-asset allocation (%)",
    10, 100, 50, 10,
    key="v26_allocation",
)

v26_coins = st.multiselect(
    "Cryptocurrencies",
    ["BTC", "ETH", "SOL"],
    default=["BTC", "ETH", "SOL"],
    key="v26_coins",
)

if st.button(
    "Run multi-asset research",
    key="v26_run",
):
    results = []
    failures = []
    details = {}

    for asset in v26_coins:
        try:
            data, stats, missing_times = history_for(
                asset, v26_days
            )

            require_contiguous(stats)

            split = int(
                len(data) * v26_train_pct / 100
            )

            if split < 220 or len(data) - split < 2:
                raise ValueError(
                    "Insufficient training or testing candles."
                )

            for strategy in STRATEGIES:
                try:
                    train, train_trades, train_equity = (
                        backtest_strategy_v25(
                            data.iloc[:split],
                            strategy,
                            initial_cash=500.0,
                            allocation=v26_allocation / 100,
                            costs=CostsV25(),
                        )
                    )

                    test, test_trades, test_equity = (
                        backtest_strategy_v25(
                            data,
                            strategy,
                            initial_cash=500.0,
                            allocation=v26_allocation / 100,
                            costs=CostsV25(),
                            evaluation_start=data.index[split],
                        )
                    )

                    results.append({
                        "Asset": asset,
                        "Strategy": strategy,
                        "Training Return %":
                            train["Return %"],
                        "Testing Return %":
                            test["Return %"],
                        "Testing Buy & Hold %":
                            test["Buy and Hold Return %"],
                        "Testing Excess Return %":
                            round(
                                test["Return %"]
                                - test["Buy and Hold Return %"],
                                3,
                            ),
                        "Testing Drawdown %":
                            test["Maximum Drawdown %"],
                        "Testing Trades":
                            test["Trades"],
                        "Testing Win Rate %":
                            test["Win Rate %"],
                        "Testing Final USD":
                            test["Final USD"],
                    })

                    details[(asset, strategy)] = {
                        "train_trades": train_trades,
                        "train_equity": train_equity,
                        "test_trades": test_trades,
                        "test_equity": test_equity,
                    }

                except Exception as exc:
                    failures.append(
                        f"{asset} / {strategy}: {exc}"
                    )

        except Exception as exc:
            failures.append(f"{asset}: {exc}")

    if failures:
        with st.expander(
            f"{len(failures)} evaluation errors"
        ):
            for failure in failures:
                st.write(failure)

    if results:
        table = pd.DataFrame(results)

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Testing Return Comparison")

        st.bar_chart(
            table.pivot(
                index="Strategy",
                columns="Asset",
                values="Testing Return %",
            ),
            stack=False,
        )

        st.subheader("Testing Excess Return")

        st.bar_chart(
            table.pivot(
                index="Strategy",
                columns="Asset",
                values="Testing Excess Return %",
            ),
            stack=False,
        )

        asset = st.selectbox(
            "Inspect asset",
            sorted(table["Asset"].unique()),
            key="v26_detail_asset",
        )

        available = table.loc[
            table["Asset"] == asset,
            "Strategy",
        ].tolist()

        strategy = st.selectbox(
            "Inspect strategy",
            available,
            key="v26_detail_strategy",
        )

        detail = details[(asset, strategy)]

        st.subheader("Training Equity")
        st.line_chart(
            detail["train_equity"]["Equity USD"]
        )

        st.subheader("Testing Equity")
        st.line_chart(
            detail["test_equity"]["Equity USD"]
        )

        with st.expander("Training trade log"):
            st.dataframe(
                detail["train_trades"],
                hide_index=True,
            )

        with st.expander("Testing trade log"):
            st.dataframe(
                detail["test_trades"],
                hide_index=True,
            )

    else:
        st.info(
            "No successful evaluations. "
            "Review the errors above."
        )


# ==========================================================
# V2.7 ROLLING WALK-FORWARD RESEARCH
# ==========================================================

st.divider()
st.header("V2.7 Rolling Walk-Forward Research")

st.caption(
    "Fixed long-only strategies across non-overlapping "
    "testing windows. Earlier history is retained for "
    "indicators. Each fold starts with fresh simulated USD."
)

v27_coin = st.selectbox(
    "Research cryptocurrency",
    ["BTC", "ETH", "SOL"],
    key="v27_coin",
)

v27_days = st.selectbox(
    "Historical data (days)",
    [90, 180],
    index=1,
    key="v27_days",
)

v27_train = st.selectbox(
    "Initial training history (days)",
    [30, 45, 60],
    index=1,
    key="v27_train",
)

v27_test = st.selectbox(
    "Testing window (days)",
    [7, 15, 30],
    index=1,
    key="v27_test",
)

v27_allocation = st.slider(
    "Capital allocation (%)",
    10, 100, 50, 10,
    key="v27_allocation",
)

if st.button(
    "Run V2.7 rolling evaluation",
    type="primary",
    key="v27_run",
):
    try:
        with st.spinner(
            "Retrieving historical data and "
            "checking hourly candle continuity..."
        ):
            data, stats, missing_times = history_for(
                v27_coin,
                v27_days,
            )

        # Always display diagnostics before validation.
        display_history_diagnostics(
            stats,
            missing_times,
        )

        # Missing candles are never silently filled.
        require_contiguous(stats)

        with st.spinner(
            "Running rolling walk-forward evaluations..."
        ):
            results = run_rolling_v27(
                data,
                train_days=v27_train,
                test_days=v27_test,
                allocation=v27_allocation / 100,
                initial_cash=500.0,
            )

        st.success(
            f"Completed {results['Fold'].nunique()} "
            "testing windows across "
            f"{results['Strategy'].nunique()} strategies."
        )

        st.subheader("Individual Testing Windows")

        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Strategy Performance Summary")

        summary = results.groupby("Strategy").agg(
            Folds=("Fold", "count"),
            Mean_Return=("Return %", "mean"),
            Median_Return=("Return %", "median"),
            Mean_Excess=("Excess Return %", "mean"),
            Worst_Return=("Return %", "min"),
            Best_Return=("Return %", "max"),
            Total_Trades=("Trades", "sum"),
        ).round(3)

        st.dataframe(
            summary,
            use_container_width=True,
        )

        st.subheader("Return by Testing Window")

        st.line_chart(
            results.pivot(
                index="Fold",
                columns="Strategy",
                values="Return %",
            )
        )

        st.info(
            "Each fold begins flat with fresh US$500. "
            "Returns are not compounded across folds. "
            "Historical results are not evidence of "
            "future profitability."
        )

    except Exception as exc:
        st.error(
            "V2.7 evaluation could not complete."
        )
        st.code(str(exc))








