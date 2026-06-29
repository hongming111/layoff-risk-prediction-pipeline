-- Migration 001: Add unique constraints for idempotent upserts
-- Run this once against an existing retrenchment_db if the container was
-- already initialised before this constraint was added to init_postgres.sql.
--
-- Apply with:
--   docker exec -it <postgres-container> psql -U retrench -d retrenchment_db \
--       -f /docker-entrypoint-initdb.d/migrations/001_add_unique_constraints.sql
--
-- Or interactively:
--   docker exec -it <postgres-container> psql -U retrench -d retrenchment_db

-- Remove duplicate predictions first (keep the row with the highest score).
DELETE FROM predictions_log p1
USING predictions_log p2
WHERE p1.id < p2.id
  AND p1.ticker          = p2.ticker
  AND p1.prediction_date = p2.prediction_date;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'uq_predictions_ticker_date'
  ) THEN
    ALTER TABLE predictions_log
      ADD CONSTRAINT uq_predictions_ticker_date UNIQUE (ticker, prediction_date);
  END IF;
END $$;

-- Remove duplicate warn notices first (keep the row with the highest id).
DELETE FROM warn_notices w1
USING warn_notices w2
WHERE w1.id < w2.id
  AND w1.company    = w2.company
  AND w1.state      = w2.state
  AND w1.event_date = w2.event_date;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'uq_warn_company_state_date'
  ) THEN
    ALTER TABLE warn_notices
      ADD CONSTRAINT uq_warn_company_state_date UNIQUE (company, state, event_date);
  END IF;
END $$;
