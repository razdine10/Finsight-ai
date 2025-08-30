"""💸 Fraud Loss Forecast - Prévisions des pertes dues à la fraude"""

import os
import sys
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import importlib.util

# Add parent directory to path for imports
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
    st.subheader("Fraud Loss Forecast")

    with st.expander("How does it work?", expanded=False):
        st.markdown(
            """
            ## 🎯 Goal
            Forecast daily fraud losses to anticipate financial impact and help
            with budgeting and risk management decisions.

            ## 💸 What we forecast
            - **Fraud losses**: daily sum of amounts where transactions are marked as suspicious
            - **Optional bank filter**: focus on specific institutions
            - **Time horizon**: 7, 14, or 30 days ahead

            ## 🔍 How the model works
            We use **SARIMA** (same as regular forecasting) because fraud patterns
            often show weekly seasonality:
            - Weekdays vs weekends have different fraud patterns
            - The model learns from historical fraud amounts
            - It accounts for trends and seasonal variations

            ## 📊 What you get
            **Graph**: Historical fraud losses + future predictions with confidence bands
            **Table**: Day-by-day loss forecasts with uncertainty ranges
            **Validation metrics**: How well the model performed on recent data

            ## ⚠️ Keep in mind
            - Fraud patterns can be irregular and harder to predict than normal transactions
            - External factors (new fraud schemes, security updates) can break patterns
            - Use forecasts as guidance, not exact predictions
            """
        )

    # Filtres
    c1, c2, c3 = st.columns(3)
    with c1:
        history_days = st.selectbox("Period (days)", [90, 180, 365, 540, 720], index=0)
    with c2:
        series_choice = st.selectbox("Series", ["Amount", "Volume"], index=0)
    with c3:
        bank_options = ["All"] + list(BANK_MAPPING.values())
        bank_label = st.selectbox("Bank (optional)", bank_options, index=0)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("**Modèle**")
        st.markdown("SARIMA")
    with c5:
        horizon = st.selectbox("Horizon", [7, 14, 30], index=2)
    with c6:
        test_len = st.selectbox("Test window (days)", [7, 14, 30], index=0)

    # Données
    df = _q.load_daily_fraud_series(history_days, series_choice, bank_label)
    if df.empty:
        st.info("No data available.")
        return

    y_col = "value"

    # Split train/test
    df = df.sort_values("day")
    test_len = min(int(test_len), max(1, len(df) // 5))
    df_train = df.iloc[:-test_len].copy()
    df_test = df.iloc[-test_len:].copy()

    try:
        fc_valid, fc_future = fit_and_forecast_sarima(df_train, df_test, y_col, int(horizon))
    except Exception as e:
        st.error(f"Model error: {e}")
        return

    # KPIs
    merged = df_test[["day", "date", y_col]].merge(
        fc_valid[["date", "yhat", "yhat_lower", "yhat_upper"]], on="date", how="inner"
    )
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

    # Graphique
    last_day = int(df["day"].max())
    test_start = int(df_test["day"].min())
    horizon_start = last_day + 1

    fc_future = fc_future.copy()
    fc_future["day"] = np.arange(horizon_start, horizon_start + len(fc_future))

    y_label = "Fraud losses (amount)" if series_choice == "Amount" else "Suspicious transactions"
    fig = _viz.create_forecast_figure(
        df=df.rename(columns={"value": y_col}),
        merged_valid=merged,
        fc_future=fc_future,
        y_label=y_label,
        test_start=test_start,
        horizon_start=horizon_start,
    )
    st.plotly_chart(fig, use_container_width=True)

    _viz.render_chart_help()

    # Tableau horizon
    st.subheader("Forecasts (horizon)")
    fut = fc_future[["day", "yhat"]].copy()
    if "yhat_lower" in fc_future.columns:
        fut["yhat_lower"] = fc_future["yhat_lower"]
        fut["yhat_upper"] = fc_future["yhat_upper"]
    else:
        fut["yhat_lower"] = np.nan
        fut["yhat_upper"] = np.nan

    st.dataframe(fut, use_container_width=True, hide_index=True)
    _viz.render_csv_download(fut, "fraud_loss_forecast.csv")


@st.cache_data(ttl=CACHE_TTL)
def load_daily_fraud_series(history_days: int, series_choice: str, bank_label: str) -> pd.DataFrame:
    """Delegates to queries.load_daily_fraud_series (no SQL here)."""
    return _q.load_daily_fraud_series(history_days, series_choice, bank_label)


def fit_and_forecast_sarima(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str,
    horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

    series = df_train.set_index("date")[y_col].astype(float)
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    steps_valid = len(df_test)
    fc_v = res.get_forecast(steps=steps_valid)
    ci_v = fc_v.conf_int(alpha=0.1)
    fc_valid = pd.DataFrame({
        "date": df_test["date"].values,
        "yhat": fc_v.predicted_mean.values,
        "yhat_lower": ci_v.iloc[:, 0].values,
        "yhat_upper": ci_v.iloc[:, 1].values,
    })

    series_all = pd.concat([series, df_test.set_index("date")[y_col].astype(float)])
    model_all = SARIMAX(series_all, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
    res_all = model_all.fit(disp=False)
    fc_f = res_all.get_forecast(steps=int(horizon))
    ci_f = fc_f.conf_int(alpha=0.1)
    future_dates = pd.date_range(df_test["date"].max() + pd.Timedelta(days=1), periods=int(horizon), freq="D")
    fc_future = pd.DataFrame({
        "date": future_dates,
        "yhat": fc_f.predicted_mean.values,
        "yhat_lower": ci_f.iloc[:, 0].values,
        "yhat_upper": ci_f.iloc[:, 1].values,
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


if __name__ == "__main__":
    main() 