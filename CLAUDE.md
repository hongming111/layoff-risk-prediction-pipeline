# CLAUDE.md - Corporate Distress & Layoff Prediction Pipeline

## 1. Project Overview & Objective
This end-to-end data platform predicts the probability of a company executing a mass layoff event within a 90-day forward window. To achieve this, it constructs an **Alternative Data Pipeline** that fuses standard numerical market/fundamental data with unstructured social, textual, and macroeconomic indicators. 

### Core Use Case
Predicting corporate downsizing *before* it happens by treating mass layoffs as a multi-modal binary classification/survival analysis task ($Y \in \{0, 1\}$).

---

## 2. Multi-Modal Data Sources (Free APIs)
The pipeline ingests and merges five distinct data domains:
1. **The Ground Truth Labels ($Y$):** Official state WARN Act notifications scraped dynamically via the open-source `warn-scraper` Python package.
2. **Market & Price Metrics ($X_{market}$):** Daily historical OHLCV data and rolling volatility trends via the free `yfinance` API.
3. **Fundamental Metrics ($X_{fundamentals}$):** Quarterly balance sheets and financial ratios (Debt-to-Equity, Current Ratio, Profit Margins) from Alpha Vantage or Financial Modeling Prep (FMP) free tiers.
4. **Textual Sentiment & Noise ($X_{text}$):** Corporate news headlines and public rumor mill velocities captured using NewsAPI and Reddit JSON feeds.
5. **Macro Headwinds ($X_{macro}$):** Monthly sector-specific labor market contraction metrics via the US Bureau of Labor Statistics (BLS) API.

---

## 3. Architecture & Heavy ETL Challenges

[ Daily WARN Scraping ]      [ NewsAPI/Reddit ]    [ yfinance/AlphaVantage ]
│                           │                         │
(Incremental)                 (Hourly)                  (Daily)
▼                           ▼                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Orchestration Layer (Apache Airflow)                  │
└────────────────────────────────────────────────────────────────────────┘
│                           │                         │
▼                           ▼                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│             ETL Processing Layer (PySpark / Pandas + DuckDB)           │
│  • Entity Resolution (Fuzzy matching Company Strings -> Stock Tickers)  │
│  • Temporal Feature Alignment (Resampling Quarterly/Daily to Feature)  │
│  • Stateful Rolling Windows (7d Sentiment Moving Avg, 14d Volatility)  │
└────────────────────────────────────────────────────────────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Storage & Feature Store Layer                        │
│       • Postgres (Structured Aggregates & Relational Metadata)        │
│       • Local Parquet Feature Files (Versioned Data Matrices)          │
└────────────────────────────────────────────────────────────────────────┘


### Key Engineering Hurdle: Entity Resolution & Temporal Alignment
* **Entity Resolution:** WARN entries report unstructured legal names (e.g., "Google LLC, Mountain View"), while market datasets use stock tickers (`GOOG`). The pipeline implements cleaning, corporate suffix stripping, and fuzzy string matching to map elements to a universal `Company_ID`.
* **Temporal Realignment:** Financial ratios update quarterly, stock prices daily, news hourly, and BLS data monthly. The pipeline resamples and forward-fills these irregular intervals into a unified daily/weekly analytical feature matrix.

---

## 4. MLOps Framework & Considerations
To secure maximum rubric marks for MLOps validation, tracking, and closed-loop feedback, the system architecture treats models as living assets:

### A. Experimentation & Version Control
* **Tooling:** **MLflow** container running alongside the database.
* **Tracking:** Every training run explicitly logs model hyper-parameters, feature importance matrices, and validation curves (ROC-AUC, Precision-Recall).
* **Artifacts:** Processed data matrices are timestamped and hash-versioned alongside the generated model binaries (`.json` or `.bin` model states).

### B. Daily Evaluation & Model Drift DAG
* **Automated Schedule:** Every morning at 09:00 AM, an Airflow DAG initiates a validation loop.
* **Data Drift Detection:** Calculates Kolmogorov-Smirnov test scores on incoming numerical features (e.g., sudden spikes in average stock volatility across the tech sector) to flag incoming data distribution changes.
* **The "Prediction vs. Actual" Loop:** Checks the newly published daily WARN data against predictions generated 60–90 days prior. It dynamically updates historical system precision metrics and logs accuracy degradation directly to the dashboard.

### C. Continuous Deployment Guardrails
* If model performance falls below an established threshold (e.g., Precision drops below 70%), an automated flag prevents auto-deployment of retrained weights, falling back to a safe baseline model while triggering an Airflow alert log.

---

## 5. System Components & Project Structure
Organize the codebase into the following containerized microservices:

├── .github/workflows/       # CI/CD pipelines
├── airflow/                 # Ingestion & Evaluation DAGs
│   ├── dags/
│   │   ├── Ingestion_Pipeline_DAG.py
│   │   └── MLOps_Evaluation_DAG.py
│   └── config/
├── core_etl/                # Transformation & Alignment Engine
│   ├── entity_resolver.py   # Text-matching mapping logic
│   └── feature_engineer.py  # Forward-filling & temporal windowing
├── ml_engine/               # Training, Inference & Drift Scripts
│   ├── train.py             # XGBoost / Random Forest Pipeline
│   └── evaluate_drift.py    # Prediction vs Actual loop + Drift metrics
├── dashboard/               # Frontend interface
│   └── app.py               # Streamlit application script
├── docker-compose.yml       # Orchestrates Postgres, Airflow, MLflow, Dashboard
├── requirements.txt         # Package dependencies (yfinance, warn-scraper, mlflow, etc.)
└── CLAUDE.md                # This instruction file

---

## 6. Technical Depth & Robustness Features
* **Rate-Limit Resilience:** Ingestion scripts must use `tenacity` retry decorators with exponential backoff to handle free tier API limitations safely.
* **Data Validation Gates:** Before running inference or model training, the incoming data matrices are evaluated using `Pydantic` or `Great Expectations` to ensure crucial identifier fields or ratio rows are not empty or null.
* **Real-Time Threshold Alerts:** The dashboard monitors the 90-day probability output stream. If an individual company's distress score breaches a critical threshold ($> 0.85$), an active incident flag triggers visually on the front-end interface.