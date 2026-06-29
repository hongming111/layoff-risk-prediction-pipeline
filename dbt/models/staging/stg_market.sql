{{ config(materialized='view') }}

SELECT
    ticker,
    date::DATE  AS date,
    close,
    vol_7d,
    vol_14d,
    vol_21d
FROM read_parquet('/opt/airflow/data/processed/market.parquet')
WHERE ticker IS NOT NULL
  AND date   IS NOT NULL
