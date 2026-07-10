---
title: Layoff Risk Monitor Demo
emoji: ⚠️
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
pinned: false
---

# Corporate Layoff Risk Monitor — Portfolio Demo

A static, illustrative demo of a multi-modal corporate distress prediction
pipeline that estimates the probability of a company executing a mass layoff
within a 90-day forward window.

**This Space shows a bundled snapshot, not a live system** — the full
project runs on Postgres, Airflow, and MLflow, which aren't available on this
tier. It fuses:

- **Ground truth labels** — state WARN Act notifications (`warn-scraper`)
- **Market data** — daily OHLCV + volatility via `yfinance`
- **Fundamentals** — quarterly balance sheet ratios (Alpha Vantage / FMP)
- **Text sentiment** — news + social mention velocity (NewsAPI, RSS/Reddit)
- **Macro headwinds** — sector labor market data from BLS

into a daily feature matrix served through an XGBoost classifier, with
MLflow experiment tracking and an Airflow-orchestrated drift/precision
monitoring loop.

Full source and architecture: see the model card and GitHub repo linked in
the app.
