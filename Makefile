.PHONY: up down build logs lint test train backfill

up:           ## Start all services
	docker compose up -d

down:         ## Stop and remove containers
	docker compose down

build:        ## Rebuild images after code changes
	docker compose build

logs:         ## Tail all container logs
	docker compose logs -f

lint:         ## Run ruff linter
	ruff check .

test:         ## Run test suite
	pytest tests/ --cov=. --cov-report=term-missing -q

train:        ## Trigger a training run locally (outside Docker)
	python -m ml_engine.train

backfill:     ## Re-run ingestion DAG from a given date (DATE=YYYY-MM-DD)
	docker compose exec airflow-scheduler airflow dags backfill ingestion_pipeline --start-date $(DATE)

airflow-ui:   ## Open Airflow webserver in default browser
	open http://localhost:8080 || start http://localhost:8080

dashboard:    ## Open Streamlit dashboard in default browser
	open http://localhost:8501 || start http://localhost:8501

mlflow-ui:    ## Open MLflow UI in default browser
	open http://localhost:5000 || start http://localhost:5000
