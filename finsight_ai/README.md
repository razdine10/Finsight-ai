# Finsight AI

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/razdine10/Finsight-ai/main/finsight_ai/app.py)

AI-first AML and finance analytics built with Streamlit. Explore data, detect fraud and anomalies, forecast trends, and assess risk with clean, fast UX.

## Features

- Global dashboard KPIs and time-series views
- Analysis hub with: Exploration, Fraud detection (RandomForest), and Anomalies (IsolationForest)
- Forecasts: Amount/Volume forecasting and Fraud Loss forecasting (SARIMA)
- Risks: simple Risk & CFaR with clear interpretation
- Consistent Plotly hovers (unified hover, templated tooltips)
- Clean architecture per page (queries / visualization / style)

## Quick start

```bash
pip install -r requirements.txt
streamlit run finsight_ai/app.py
```

## Project layout

```
finsight_ai/
├── app.py                     # Main entry, top nav, dynamic page loading
├── assets/
│   └── img/                   # Shared images (logo, bank logos, csv_logo.png)
├── pages/
│   ├── 1_Dashboard_Global/
│   │   ├── Dashboard.py       # Orchestrator
│   │   ├── queries.py         # SQL only
│   │   ├── visualization.py   # Plotly + UI only (bank logos restored)
│   │   ├── constants.py       # Static config
│   │   └── style.css          # Local CSS
│   ├── 2_Analyse/
│   │   ├── Analysis.py        # Orchestrator (Exploration / Fraud / Anomalies)
│   │   ├── queries.py         # SQL only
│   │   ├── visualization.py   # Plotly + UI only
│   │   └── style.css          # Local CSS
│   ├── 3_Previsions/
│   │   ├── Forecasting.py     # Orchestrator
│   │   ├── FraudLoss.py       # Orchestrator
│   │   ├── queries.py         # SQL only
│   │   ├── visualization.py   # Plotly + UI only
│   │   └── style.css          # Local CSS
│   ├── 4_Risques/
│   │   ├── RiskCFaR.py        # Orchestrator
│   │   ├── queries.py         # SQL only
│   │   ├── visualization.py   # Plotly + UI only
│   │   └── style.css          # Local CSS
│   └── 5_Client/
│       ├── ProfilClient.py    # Orchestrator
│       ├── GrapheTransactions.py
│       ├── queries.py         # SQL only
│       ├── visualization.py   # Plotly + UI only
│       └── style.css          # Local CSS
└── README.md
```

## Architecture conventions

Each page directory follows strict separation of concerns:
- `queries.py`: SQL queries and data access only
- `visualization.py`: Plotly charts and Streamlit UI only
- `style.css`: Local CSS scoped to the page
- The main page file (e.g., `Dashboard.py`, `Analysis.py`, `Forecasting.py`, `FraudLoss.py`, `RiskCFaR.py`, `ProfilClient.py`) orchestrates by importing from `queries` and `visualization`.

Dynamic imports are used inside orchestrator files to avoid relative-import issues with Streamlit’s multi-page loader (`importlib.util.spec_from_file_location`).

## UX consistency

- Period selectors use the "Last X days" pattern everywhere
- Radio groups have clear labels (e.g., "Choose your analysis:") with a simple separator underneath
- Plotly hover configuration follows official docs (unified hover + templated tooltips)

## Run tips

- Default landing is the "Global" page. Clicking the top logo toggles the Home intro
- If port 8501 is busy, run with `--server.port 8502`

---
Made with Streamlit, Plotly, Pandas, Statsmodels, and scikit-learn.
