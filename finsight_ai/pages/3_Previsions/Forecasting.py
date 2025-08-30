"""📈 Prévisions (Forecasting) - simple et cohérente"""

import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import importlib.util

# Add parent directory to path for imports
# Dynamic load local modules to avoid collisions with other pages
_this_dir = Path(__file__).parent

_spec_q = importlib.util.spec_from_file_location("previsions_queries", str(_this_dir / "queries.py"))
_q = importlib.util.module_from_spec(_spec_q)
assert _spec_q and _spec_q.loader
_spec_q.loader.exec_module(_q)

_spec_v = importlib.util.spec_from_file_location("previsions_viz", str(_this_dir / "visualization.py"))
_viz = importlib.util.module_from_spec(_spec_v)
assert _spec_v and _spec_v.loader
_spec_v.loader.exec_module(_viz)

SCHEMA_NAME = "aml"
CACHE_TTL = 300

BANK_MAPPING = {
    0: "BNP Paribas",
    1: "Société Générale",
    2: "Crédit Agricole",
    3: "Banque Populaire",
    4: "Caisse d'Épargne",
    5: "Crédit Mutuel",
    6: "La Banque Postale",
    7: "HSBC France",
    8: "LCL (Crédit Lyonnais)",
    9: "Boursorama Banque",
}


def main() -> None:
    st.subheader("Forecasting")

    with st.expander("How does it work?", expanded=False):
        st.markdown(
            """
            ## 🎯 Goal
            Forecast daily transaction patterns for the near future (7/14/30 days).
            We can predict either total amounts or transaction volumes, 
            optionally filtered by bank.

            ## 📈 What we forecast
            - **Amount**: sum of transaction amounts per day  
            - **Volume**: count of transactions per day
            - **Bank filter**: focus on a specific bank if needed

            ## 🔍 How the model works
            We use **SARIMA** (Seasonal ARIMA), which is designed for time series
            with weekly patterns. It looks at:
            - Trends (going up/down over time)
            - Seasonality (weekly patterns - e.g., weekends vs weekdays)
            - Recent fluctuations to adjust predictions

            ## 📊 What you get
            **Graph**: Historical data + future predictions with confidence bands
            **Table**: Day-by-day forecasts with upper/lower bounds
            **Metrics**: How accurate the model is on recent data

            ## 🧪 Validation process
            We test the model on recent data (last 7-30 days) to measure:
            - **MAE**: Average prediction error in absolute terms
            - **MAPE**: Average prediction error as percentage  
            - **Coverage**: How often real values fall within confidence bands

            ## ⚠️ Keep in mind
            - Forecasting works best with stable patterns
            - Unusual events (holidays, crises) can affect accuracy
            - Shorter horizons (7 days) are typically more reliable than longer ones
            """
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        period_choice = st.selectbox("Period", [
            "Last 90 days", "Last 180 days", "Last 365 days", "Last 540 days", "Last 720 days"
        ], index=0)
        import re as _re
        _m = _re.search(r"(\d+)", period_choice)
        history_days = int(_m.group(1)) if _m else 90
    with c2:
        series_choice = st.selectbox("Series", ["Amount", "Volume"], index=0)
    with c3:
        bank_options = ["All"] + list(BANK_MAPPING.values())
        bank_label = st.selectbox("Bank (optional)", bank_options, index=0)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("**Model**")
        st.markdown("SARIMA")
    with c5:
        horizon = st.selectbox("Horizon", [7, 14, 30], index=2)
    with c6:
        test_len = st.selectbox("Test window (days)", [7, 14, 30], index=0)

    model_choice = "SARIMA"

    df = _q.load_daily_series(history_days, series_choice, bank_label)
    if df.empty:
        st.info("No data available.")
        return

    y_col = "value"

    df = df.sort_values("day")
    test_len = min(int(test_len), max(1, len(df) // 5))  # borne
    df_train = df.iloc[:-test_len].copy()
    df_test = df.iloc[-test_len:].copy()

    try:
        fc_valid, fc_future = fit_and_forecast(df_train, df_test, y_col, model_choice, int(horizon))
    except ImportError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Model error: {e}")
        return

    merged = df_test[["day", "date", y_col]].merge(fc_valid[["date", "yhat", "yhat_lower", "yhat_upper"]], on="date", how="inner")
    if merged.empty:
        n = min(len(df_test), len(fc_valid))
        merged = pd.DataFrame({
            "day": df_test["day"].values[-n:],
            "date": df_test["date"].values[-n:],
            y_col: df_test[y_col].values[-n:],
            "yhat": fc_valid["yhat"].values[-n:],
            "yhat_lower": fc_valid.get("yhat_lower", pd.Series([np.nan] * n)).values,
            "yhat_upper": fc_valid.get("yhat_upper", pd.Series([np.nan] * n)).values,
        })

    mae, mape = compute_metrics(merged[y_col].values, merged["yhat"].values)
    coverage = compute_coverage(merged[y_col].values, merged.get("yhat_lower"), merged.get("yhat_upper"))

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("MAE (test)", f"{mae:,.2f}")
    with k2:
        st.metric("MAPE (test)", f"{mape:.2%}")
    with k3:
        st.metric("Coverage", f"{coverage:.0%}")


    last_day = int(df["day"].max())
    test_start = int(df_test["day"].min())
    horizon_start = last_day + 1

    fc_future = fc_future.copy()
    fc_future["day"] = np.arange(horizon_start, horizon_start + len(fc_future))

    fig = _viz.create_forecast_figure(
        df=df.rename(columns={"value": y_col}),
        merged_valid=merged,
        fc_future=fc_future,
        y_label=series_choice,
        test_start=test_start,
        horizon_start=horizon_start,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    _viz.render_chart_help()
    
    st.subheader("Forecasts")
    fut = fc_future[["day", "yhat"]].copy()
    if "yhat_lower" in fc_future.columns:
        fut["yhat_lower"] = fc_future["yhat_lower"]
        fut["yhat_upper"] = fc_future["yhat_upper"]
    else:
        fut["yhat_lower"] = np.nan
        fut["yhat_upper"] = np.nan

    table = fut
    st.dataframe(table, use_container_width=True, hide_index=True)

    _viz.render_csv_download(table, "forecast.csv")


@st.cache_data(ttl=CACHE_TTL)
def load_daily_series(history_days: int, series_choice: str, bank_label: str) -> pd.DataFrame:
    """Delegates to queries.load_daily_series (kept here for backward-compat imports)."""
    return _q.load_daily_series(history_days, series_choice, bank_label)


def fit_and_forecast(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str,
    model_choice: str,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_series = df_train.set_index("date")[y_col].astype(float)

    if model_choice == "Prophet":
        try:
            from prophet import Prophet  # type: ignore
        except Exception:
            raise ImportError("Prophet n'est pas installé. Exécutez: pip install prophet")

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.02,
            interval_width=0.8,
        )
        m.fit(pd.DataFrame({"ds": train_series.index, "y": train_series.values}))

        valid_dates = df_test["date"].values
        fc_valid_raw = m.predict(pd.DataFrame({"ds": valid_dates}))
        fc_valid = pd.DataFrame({
            "date": pd.to_datetime(fc_valid_raw["ds"]).dt.normalize(),
            "yhat": fc_valid_raw["yhat"].values,
            "yhat_lower": fc_valid_raw.get("yhat_lower"),
            "yhat_upper": fc_valid_raw.get("yhat_upper"),
        })

        last_date = df_train["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=int(horizon), freq="D")
        fc_future_raw = m.predict(pd.DataFrame({"ds": future_dates}))
        fc_future = pd.DataFrame({
            "date": future_dates,
            "yhat": fc_future_raw["yhat"].values,
            "yhat_lower": fc_future_raw.get("yhat_lower"),
            "yhat_upper": fc_future_raw.get("yhat_upper"),
        })
        return fc_valid, fc_future

    if model_choice == "SARIMA":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
        except Exception:
            raise ImportError("statsmodels n'est pas installé. Exécutez: pip install statsmodels")

        model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)

        steps_valid = len(df_test)
        fc_valid_res = result.get_forecast(steps=steps_valid)
        ci_v = fc_valid_res.conf_int(alpha=0.1)
        fc_valid = pd.DataFrame({
            "date": df_test["date"].values,
            "yhat": fc_valid_res.predicted_mean.values,
            "yhat_lower": ci_v.iloc[:, 0].values,
            "yhat_upper": ci_v.iloc[:, 1].values,
        })

        from statsmodels.tsa.statespace.sarimax import SARIMAX as SARIMAX_ALL  # reuse
        model_all = SARIMAX_ALL(pd.concat([train_series, df_test.set_index("date")[y_col].astype(float)]), order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
        res_all = model_all.fit(disp=False)
        fc_fut_res = res_all.get_forecast(steps=int(horizon))
        ci_f = fc_fut_res.conf_int(alpha=0.1)
        future_dates = pd.date_range(df_test["date"].max() + pd.Timedelta(days=1), periods=int(horizon), freq="D")
        fc_future = pd.DataFrame({
            "date": future_dates,
            "yhat": fc_fut_res.predicted_mean.values,
            "yhat_lower": ci_f.iloc[:, 0].values,
            "yhat_upper": ci_f.iloc[:, 1].values,
        })
        return fc_valid, fc_future

    # Default: ETS (Holt-Winters) avec trend=None pour stabilité
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    except Exception:
        raise ImportError("statsmodels n'est pas installé. Exécutez: pip install statsmodels")

    ets = ExponentialSmoothing(train_series, trend=None, seasonal="add", seasonal_periods=7)
    ets_fit = ets.fit(optimized=True)

    steps_valid = len(df_test)
    yhat_v = ets_fit.forecast(steps_valid)
    resid_std = float(np.std(ets_fit.resid)) if hasattr(ets_fit, "resid") else 0.0
    fc_valid = pd.DataFrame({
        "date": df_test["date"].values,
        "yhat": yhat_v.values,
        "yhat_lower": yhat_v.values - 1.96 * resid_std,
        "yhat_upper": yhat_v.values + 1.96 * resid_std,
    })

    yhat_f = ets_fit.forecast(int(horizon))
    last_date = df_test["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=int(horizon), freq="D")
    fc_future = pd.DataFrame({
        "date": future_dates,
        "yhat": yhat_f.values,
        "yhat_lower": yhat_f.values - 1.96 * resid_std,
        "yhat_upper": yhat_f.values + 1.96 * resid_std,
    })
    return fc_valid, fc_future


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.maximum(1e-6, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)))
    return mae, mape


def compute_coverage(y_true: np.ndarray, lower: Optional[pd.Series], upper: Optional[pd.Series]) -> float:
    if lower is None or upper is None:
        return 0.0
    try:
        l = np.array(lower).astype(float)
        u = np.array(upper).astype(float)
        y = y_true.astype(float)
        inside = (y >= l) & (y <= u)
        return float(np.mean(inside)) if len(inside) else 0.0
    except Exception:
        return 0.0


def compare_models(df: pd.DataFrame, y_col: str, test_len: int, folds: int = 3) -> pd.DataFrame:
    models = ["ETS (Holt-Winters)", "SARIMA"]
    # Ajouter Prophet si dispo
    prophet_ok = True
    try:
        import importlib
        importlib.import_module("prophet")
    except Exception:
        prophet_ok = False
    if prophet_ok:
        models.append("Prophet")

    records = []
    n = len(df)
    for m in models:
        maes, mapes, covs = [], [], []
        for k in range(folds):
            split = n - (folds - k) * test_len
            if split <= max(14, test_len * 2):
                continue
            train = df.iloc[:split].copy()
            test = df.iloc[split: split + test_len].copy()
            try:
                fc_v, _ = fit_and_forecast(train, test, y_col, m, horizon=test_len)
            except Exception:
                continue
            join = test[["date", y_col]].merge(fc_v[["date", "yhat", "yhat_lower", "yhat_upper"]], on="date", how="inner")
            if join.empty:
                continue
            mae, mape = compute_metrics(join[y_col].values, join["yhat"].values)
            cov = compute_coverage(join[y_col].values, join.get("yhat_lower"), join.get("yhat_upper"))
            maes.append(mae); mapes.append(mape); covs.append(cov)
        if maes:
            records.append({"modèle": m, "MAE": np.mean(maes), "MAPE": np.mean(mapes), "Coverage": np.mean(covs)})

    return pd.DataFrame(records).sort_values(["MAPE", "MAE"]).reset_index(drop=True)


if __name__ == "__main__":
    main() 