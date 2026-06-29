{{ config(materialized='table') }}

/*
  Final feature matrix for layoff risk prediction.

  Adds the 90-day forward label: label=1 if a confirmed WARN Act event
  falls within the next 90 calendar days for that ticker, else 0.

  Written to data/features/feature_matrix.parquet by the ETL Airflow task
  after dbt run completes (via a DuckDB export step in _run_etl).
*/
SELECT
    a.ticker,
    a.date,
    -- Market features
    a.close,
    a.vol_7d,
    a.vol_14d,
    a.vol_21d,
    -- Fundamental features (ASOF forward-filled from quarterly)
    a.debt_to_equity,
    a.current_ratio,
    a.profit_margin,
    -- Sentiment features
    a.sentiment_score,
    a.sentiment_score_ma7d,
    a.mention_velocity,
    -- Macro features (ASOF forward-filled from monthly)
    a.unemployment_rate_total,
    a.layoff_rate_total,
    a.layoff_rate_tech,
    -- Label: 1 if a WARN event occurs within the next 90 calendar days
    COALESCE(MAX(CASE
        WHEN l.event_date BETWEEN a.date AND a.date + INTERVAL '90 days'
        THEN 1 ELSE 0
    END), 0) AS label
FROM {{ ref('int_daily_aligned') }} a
LEFT JOIN {{ ref('stg_warn_labels') }} l
    ON a.ticker = l.ticker
GROUP BY
    a.ticker, a.date,
    a.close, a.vol_7d, a.vol_14d, a.vol_21d,
    a.debt_to_equity, a.current_ratio, a.profit_margin,
    a.sentiment_score, a.sentiment_score_ma7d, a.mention_velocity,
    a.unemployment_rate_total, a.layoff_rate_total, a.layoff_rate_tech
ORDER BY a.ticker, a.date
