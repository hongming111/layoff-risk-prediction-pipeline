{{ config(materialized='view') }}

/*
  Temporally aligns all five data sources to a daily market spine.

  ASOF JOIN forward-fills sparse series to daily resolution:
  - stg_fundamentals (quarterly): per-ticker, carries the most recent
    reporting-date row whose date <= market date forward indefinitely.
  - stg_macro (monthly): carries the most recent BLS monthly observation
    whose date <= market date forward to every trading day.
  - stg_sentiment (daily): exact-date match; days with no news coverage
    receive 0.0 via the COALESCE in stg_sentiment.

  This replaces the previous exact-date LEFT JOIN which silently left
  quarterly fundamentals NULL on every non-reporting day.
*/
WITH market AS (
    SELECT * FROM {{ ref('stg_market') }}
),
fundamentals AS (
    SELECT * FROM {{ ref('stg_fundamentals') }}
),
macro AS (
    SELECT * FROM {{ ref('stg_macro') }}
),
sentiment AS (
    SELECT * FROM {{ ref('stg_sentiment') }}
),
market_with_fundamentals AS (
    SELECT
        m.ticker,
        m.date,
        m.close,
        m.vol_7d,
        m.vol_14d,
        m.vol_21d,
        f.debt_to_equity,
        f.current_ratio,
        f.profit_margin
    FROM market m
    ASOF LEFT JOIN fundamentals f
        ON m.ticker = f.ticker
       AND m.date   >= f.date
),
aligned AS (
    SELECT
        mf.*,
        mac.unemployment_rate_total,
        mac.layoff_rate_total,
        mac.layoff_rate_tech
    FROM market_with_fundamentals mf
    ASOF LEFT JOIN macro mac
        ON mf.date >= mac.date
)
SELECT
    a.*,
    COALESCE(s.sentiment_score,      0.0) AS sentiment_score,
    COALESCE(s.sentiment_score_ma7d, 0.0) AS sentiment_score_ma7d,
    COALESCE(s.mention_velocity,     0.0) AS mention_velocity
FROM aligned a
LEFT JOIN sentiment s
    ON a.ticker = s.ticker
   AND a.date   = s.date
