{{ config(materialized='view') }}

SELECT
    ticker,
    date::DATE                              AS date,
    COALESCE(sentiment_score,      0.0)     AS sentiment_score,
    COALESCE(sentiment_score_ma7d, 0.0)     AS sentiment_score_ma7d,
    COALESCE(mention_velocity,     0.0)     AS mention_velocity
FROM read_parquet('/opt/airflow/data/processed/sentiment.parquet')
WHERE ticker IS NOT NULL
  AND date   IS NOT NULL
