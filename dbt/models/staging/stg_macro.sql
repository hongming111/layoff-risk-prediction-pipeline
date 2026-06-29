{{ config(materialized='view') }}

SELECT
    date::DATE                                       AS date,
    unemployment_rate_total,
    layoff_rate_total,
    COALESCE(layoff_rate_tech, layoff_rate_total)    AS layoff_rate_tech
FROM read_parquet('/opt/airflow/data/processed/macro.parquet')
WHERE date IS NOT NULL
  AND unemployment_rate_total IS NOT NULL
  AND layoff_rate_total IS NOT NULL
