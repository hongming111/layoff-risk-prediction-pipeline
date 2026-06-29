{{ config(materialized='view') }}

SELECT
    ticker,
    event_date::DATE    AS event_date,
    employees_affected
FROM read_parquet('/opt/airflow/data/processed/warn_labels.parquet')
WHERE ticker     IS NOT NULL
  AND event_date IS NOT NULL
