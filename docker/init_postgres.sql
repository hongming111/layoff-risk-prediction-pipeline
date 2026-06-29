-- Run once at first container start (docker-entrypoint-initdb.d)
-- Creates the Airflow metadata DB alongside the main application DB.

-- Airflow needs its own database; the main DB is created by POSTGRES_DB env var
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO retrench;

-- Switch to the application database and create core tables
\c retrenchment_db;

CREATE TABLE IF NOT EXISTS warn_notices (
    id                  SERIAL PRIMARY KEY,
    company             TEXT NOT NULL,
    state               VARCHAR(2),
    event_date          DATE,
    employees_affected  INTEGER,
    ticker              VARCHAR(10),
    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (company, state, event_date)
);

CREATE TABLE IF NOT EXISTS predictions_log (
    id               SERIAL PRIMARY KEY,
    ticker           VARCHAR(10)   NOT NULL,
    prediction_date  DATE          NOT NULL,
    target_date      DATE          NOT NULL,
    score            FLOAT         NOT NULL,
    model_version    TEXT,
    created_at       TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (ticker, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_warn_ticker_date     ON warn_notices   (ticker, event_date);
CREATE INDEX IF NOT EXISTS idx_preds_ticker_date    ON predictions_log (ticker, prediction_date);
CREATE INDEX IF NOT EXISTS idx_preds_score          ON predictions_log (score DESC);
