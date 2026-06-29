{{ config(materialized='view') }}

SELECT
    ticker,
    date::DATE      AS date,
    debt_to_equity,
    current_ratio,
    profit_margin
FROM read_parquet('/opt/airflow/data/processed/fundamentals.parquet')
WHERE ticker IS NOT NULL
