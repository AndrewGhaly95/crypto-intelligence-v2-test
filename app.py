

"""Crypto Intelligence V2.7: research-only Streamlit dashboard."""
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
    "Simulation only. No live orders. Historical results do not "
    "establish future profitability. USD backtests do not "
    "represent executable CoinSpot AUD returns."
)


def history_for(symbol, days):
    data = get_coinbase_history(
        symbol=symbol,
        interval="1h",
        days=days,
    )
    if not isinstance(data.index, pd.DatetimeIndex) or data.empty:
        raise ValueError(
            "Historical dataset is empty or has an invalid index"
        )
    if (
        data.index.tz is None
        or data.index.has_duplicates
        or not data.index.is_monotonic_increasing
    ):
        raise ValueError(
            "Historical timestamps must be timezone-aware, "
            "unique and ascending"
        )

    expected = pd.date_range(
        data.index[0],
        data.index[-1],
        freq="h",
    )
    missing = len(expected.difference(data.index))
    extra = len(data.index.difference(expected))
    reported = int(data.attrs.get("missing_candles", 0) or 0)

    return data, {
        "retrieved": len(data),
        "reported_missing": reported,
        "missing_in_span": missing,
        "irregular": extra,
        "first": data.index[0],
        "last": data.index[-1],
    }


def require_contiguous(stats):
    if (
        stats["reported_missing"]
        or stats["missing_in_span"]
        or stats["irregular"]
    ):
        raise ValueError(
            "Incomplete hourly history: provider reports "
            f"{stats['reported_missing']} missing; "
            f"{stats['missing_in_span']} gaps within returned span; "
            f"{stats['irregular']} off-grid timestamps. "
            "Backtest cancelled."
        )


def show_results(summary, trades, equity):
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

st.header("Matched-period scalping and intraday research")

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

if st.button("Run matched-period comparison"):
    try:
        costs = Costs(
            fee_per_side=fee / 100,
            spread_round_trip=spread / 100,
            slippage_per_side=slippage / 100,
        )

        five = get_candles(coin, "5m", count)
        hourly = get_candles(coin, "1h", 100)

        start = five.index.min()
        end = five.index.max() + pd.Timedelta(minutes=5)

        hourly = hourly[
            (hourly.index >= start)
            & (
                hourly.index + pd.Timedelta(hours=1)
                <= end
            )
        ]

        if len(hourly) < 2:
            raise ValueError(
                "Insufficient overlapping hourly candles"
            )

        rows = []

        for strategy, candles in (
            ("Scalping", five),
            ("Intraday", hourly),
        ):
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

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            f"Matched-period comparison failed: {exc}"
        )


# ==========================================================
# HISTORICAL DATA DIAGNOSTICS
# ==========================================================

st.divider()
st.header("Historical Data Diagnostics")

interval = st.selectbox(
    "Diagnostic candle interval",
    ["5m", "15m", "1h", "4h"],
)

if st.button("Update and inspect history"):
    try:
        data = update_history(
            coin,
            interval,
            limit=700,
        )

        quality = history_quality(
            data,
            interval,
        )

        st.dataframe(
            pd.DataFrame([quality]),
            hide_index=True,
        )

        st.line_chart(data["close"])

        st.caption(
            "Local history cache may be cleared "
            "on Streamlit restarts."
        )

    except Exception as exc:
        st.error(
            f"History diagnostics failed: {exc}"
        )


# ==========================================================
# EXTENDED HISTORICAL DATA
# ==========================================================

st.divider()
st.header("Extended Historical Data")

h_coin = st.selectbox(
    "Historical cryptocurrency",
    ["BTC", "ETH"],
    key="history_coin",
)

h_interval = st.selectbox(
    "Historical interval",
    ["1h", "6h", "1d", "5m", "15m"],
)

h_days = st.selectbox(
    "Historical period (days)",
    [30, 90, 180],
    index=1,
    key="history_days",
)

if st.button("Retrieve extended history"):
    try:
        data = get_coinbase_history(
            symbol=h_coin,
            interval=h_interval,
            days=h_days,
        )

        st.write(
            "Candles:", len(data),
            "Missing reported:",
            data.attrs.get("missing_candles", "unknown"),
        )

        st.write(
            "First:", data.index.min(),
            "Last:", data.index.max(),
        )

        st.line_chart(data["close"])

        st.download_button(
            "Download historical CSV",
            data.to_csv().encode("utf-8"),
            file_name=f"{h_coin}_{h_interval}_{h_days}d.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(
            f"Historical retrieval failed: {exc}"
        )


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

bt_alloc = st.slider(
    "Backtest allocation (%)",
    10, 100, 50, 10,
    key="v21_alloc",
)

if st.button("Run V2.1 backtest"):
    try:
        data, stats = history_for(bt_coin, bt_days)
        require_contiguous(stats)

        summary, trades, equity = run_backtest(
            data,
            initial_cash=500.0,
            allocation=bt_alloc / 100,
            warmup=200,
            costs=TradingCosts(),
        )

        show_results(summary, trades, equity)

    except Exception as exc:
        st.error(f"V2.1 failed: {exc}")


# ==========================================================
# V2.2 CHRONOLOGICAL HOLDOUT
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

wf_pct = st.slider(
    "Training proportion (%)",
    50, 75, 65, 5,
    key="wf_pct",
)

if st.button("Run chronological evaluation"):
    try:
        data, stats = history_for(wf_coin, wf_days)
        require_contiguous(stats)

        result = run_walk_forward(
            data,
            config=WalkForwardConfig(
                train_fraction=wf_pct / 100,
                warmup_candles=200,
                initial_cash=500.0,
                allocation=0.5,
            ),
            costs=TradingCosts(),
        )

        st.write("Split time:", result["split_time"])

        st.dataframe(
            result["comparison"],
            hide_index=True,
            use_container_width=True,
        )

        st.line_chart(
            result["train_equity"]["Equity USD"]
        )

        st.line_chart(
            result["test_equity"]["Equity USD"]
        )

        with st.expander("Training trades"):
            st.dataframe(result["train_trades"])

        with st.expander("Testing trades"):
            st.dataframe(result["test_trades"])

        st.caption(
            "One chronological holdout, "
            "not rolling validation."
        )

    except Exception as exc:
        st.error(f"V2.2 failed: {exc}")


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
    if st.button(f"Run {label} checks", key=key):
        try:
            result = runner()

            st.write(
                f"{label}: "
                f"{result['passed']} / "
                f"{result['total']} passed"
            )

            if result["passed"] != result["total"]:
                st.error("Some checks failed")

                for item in (
                    result.get("failures", [])
                    + result.get("errors", [])
                ):
                    st.code(str(item))
            else:
                st.success("All available checks passed")

        except Exception as exc:
            st.error(
                f"Validation failed: {exc}"
            )


# ==========================================================
# V2.4 ENGINE COMPARISON
# ==========================================================

st.divider()
st.header("V2.4 Backtesting Engine Comparison")

c_coin = st.selectbox(
    "Comparison cryptocurrency",
    ["BTC", "ETH"],
    key="compare_coin",
)

c_days = st.selectbox(
    "Comparison historical days",
    [90, 180],
    key="compare_days",
)

c_alloc = st.slider(
    "Comparison allocation (%)",
    10, 100, 50, 10,
    key="compare_alloc",
)

if st.button("Run V2 versus V3 comparison"):
    try:
        data, stats = history_for(c_coin, c_days)
        require_contiguous(stats)

        rates = dict(
            fee_per_side=0.001,
            slippage_per_side=0.0005,
            spread_round_trip=0.001,
        )

        s2, t2, e2 = run_backtest(
            data,
            initial_cash=500.0,
            allocation=c_alloc / 100,
            warmup=200,
            costs=TradingCosts(**rates),
        )

        s3, t3, e3 = run_backtest_v3(
            data,
            initial_cash=500.0,
            allocation=c_alloc / 100,
            warmup=200,
            costs=TradingCostsV3(**rates),
        )

        comparison = pd.DataFrame([
            dict(Engine="V2", **s2),
            dict(Engine="V3", **s3),
        ])

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )

        st.line_chart(e3["Equity USD"])

        with st.expander("V2 trades"):
            st.dataframe(t2)

        with st.expander("V3 trades"):
            st.dataframe(t3)

    except Exception as exc:
        st.error(
            f"Engine comparison failed: {exc}"
        )


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

v26_pct = st.slider(
    "Training proportion (%)",
    50, 80, 65, 5,
    key="v26_pct",
)

v26_alloc = st.slider(
    "Multi-asset allocation (%)",
    10, 100, 50, 10,
    key="v26_alloc",
)

v26_coins = st.multiselect(
    "Cryptocurrencies",
    ["BTC", "ETH", "SOL"],
    default=["BTC", "ETH", "SOL"],
    key="v26_coins",
)

if st.button("Run multi-asset research"):
    results = []
    failures = []
    details = {}

    for asset in v26_coins:
        try:
            data, stats = history_for(
                asset,
                v26_days,
            )

            require_contiguous(stats)

            split = int(
                len(data) * v26_pct / 100
            )

            if split < 220 or len(data) - split < 2:
                raise ValueError(
                    "Insufficient training or testing candles"
                )

            for strategy in STRATEGIES:
                try:
                    train, tt, te = backtest_strategy_v25(
                        data.iloc[:split],
                        strategy,
                        allocation=v26_alloc / 100,
                        costs=CostsV25(),
                    )

                    test, vt, ve = backtest_strategy_v25(
                        data,
                        strategy,
                        allocation=v26_alloc / 100,
                        costs=CostsV25(),
                        evaluation_start=data.index[split],
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

                    details[(asset, strategy)] = (
                        tt, te, vt, ve
                    )

                except Exception as exc:
                    failures.append(
                        f"{asset}/{strategy}: {exc}"
                    )

        except Exception as exc:
            failures.append(
                f"{asset}: {exc}"
            )

    if failures:
        with st.expander(
            f"{len(failures)} evaluation errors"
        ):
            for error in failures:
                st.write(error)

    if results:
        table = pd.DataFrame(results)

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            table.pivot(
                index="Strategy",
                columns="Asset",
                values="Testing Return %",
            ),
            stack=False,
        )

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

        strategy = st.selectbox(
            "Inspect strategy",
            table.loc[
                table["Asset"] == asset,
                "Strategy",
            ].tolist(),
            key="v26_detail_strategy",
        )

        tt, te, vt, ve = details[(asset, strategy)]

        st.line_chart(te["Equity USD"])
        st.line_chart(ve["Equity USD"])

        with st.expander("Training trade log"):
            st.dataframe(tt)

        with st.expander("Testing trade log"):
            st.dataframe(vt)

    else:
        st.info(
            "No successful evaluations. "
            "Review errors above."
        )


# ==========================================================
# V2.7 ROLLING WALK-FORWARD RESEARCH
# ==========================================================

st.divider()
st.header("V2.7 Rolling Walk-Forward Research")

st.caption(
    "Fixed strategies, independent non-overlapping "
    "test windows, expanding historical indicator context; "
    "simulated USD only."
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

v27_alloc = st.slider(
    "Capital allocation (%)",
    10, 100, 50, 10,
    key="v27_alloc",
)

if st.button(
    "Run V2.7 rolling evaluation",
    type="primary",
    key="v27_run",
):
    try:
        with st.spinner(
            "Retrieving data and checking candle continuity..."
        ):
            data, stats = history_for(
                v27_coin,
                v27_days,
            )

        st.subheader("Historical Data Diagnostics")

        st.dataframe(
            pd.DataFrame([stats]),
            hide_index=True,
            use_container_width=True,
        )

        require_contiguous(stats)

        with st.spinner(
            "Evaluating rolling test windows..."
        ):
            results = run_rolling_v27(
                data,
                train_days=v27_train,
                test_days=v27_test,
                allocation=v27_alloc / 100,
                initial_cash=500.0,
            )

        st.success(
            f"Completed {results['Fold'].nunique()} "
            "testing windows across "
            f"{results['Strategy'].nunique()} strategies"
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
            "Each fold starts flat with fresh US$500. "
            "Returns are not compounded. "
            "No parameter fitting is performed. "
            "Results remain unvalidated for live execution."
        )

    except Exception as exc:
        st.error(
            "V2.7 evaluation could not complete"
        )
        st.code(str(exc))







