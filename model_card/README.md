---
tags:
  - xgboost
  - tabular-classification
  - finance
  - layoff-prediction
  - alternative-data
pipeline_tag: tabular-classification
---

# Layoff Risk Prediction Model (XGBoost)

Binary classifier estimating the probability that a public company will
execute a mass layoff (WARN Act notice) within a 90-day forward window,
using fused market, fundamental, sentiment, and macro features.

- **Model type:** XGBoost (`XGBClassifier`), `n_estimators=300`, `max_depth=6`,
  `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`
- **Task:** Binary classification, $Y \in \{0, 1\}$ — layoff notice within 90 days
- **Framework:** scikit-learn API / XGBoost, tracked via MLflow
- **Source:** [github.com/hongming111/layoff-risk-prediction-pipeline](https://github.com/hongming111/layoff-risk-prediction-pipeline)

## Training data

13 features fused from five data domains, resampled/forward-filled onto a
shared daily grid:

| Domain | Features |
|---|---|
| Market | `close`, `vol_7d`, `vol_14d`, `vol_21d` |
| Fundamentals | `debt_to_equity`, `current_ratio`, `profit_margin` |
| Sentiment | `sentiment_score`, `sentiment_score_ma7d`, `mention_velocity` |
| Macro (BLS) | `unemployment_rate_total`, `layoff_rate_total`, `layoff_rate_tech` |

Ground truth labels come from state WARN Act notices (`warn-scraper`),
entity-resolved to stock tickers via fuzzy company-name matching.

## Evaluation

5-fold stratified cross-validation on the training snapshot (497 positive
labels):

| Metric | Value |
|---|---|
| ROC-AUC | 0.999 |
| Precision | 0.978 |
| Recall | 0.938 |
| Average Precision | 0.993 |
| Accuracy | 0.992 |

## Known limitations

- **Suspiciously high CV AUC.** These numbers likely reflect the small
  current training set and ticker universe rather than a fully validated
  production signal — `close` (raw price level) is included as a feature,
  which can make separability easier than it would be in a true forward-looking
  deployment. Before trusting this for anything beyond a portfolio demo, it
  needs out-of-time (not just stratified k-fold) validation and a leakage
  audit.
- **Small, non-representative ticker universe.** Currently evaluated on a
  handful of large, well-known public companies — not a broad or
  industry-representative sample.
- **Not financial, HR, or legal advice.** This is a research/portfolio
  project. Scores should not be used to make real decisions about any
  company or its employees.

## Intended use

Educational/portfolio demonstration of a multi-modal MLOps pipeline
(entity resolution, temporal feature alignment, drift monitoring, and
closed-loop prediction-vs-actual evaluation). See the linked GitHub repo for
the full architecture, Airflow DAGs, and evaluation loop.
