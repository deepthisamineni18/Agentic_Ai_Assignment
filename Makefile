.PHONY: run build up down test verify clean logs

# Builds the image, starts Redis + app, runs the pipeline against topics.json,
# and leaves results in ./output. Single command per requirements.
run: build
	mkdir -p output
	docker compose up --abort-on-container-exit --exit-code-from app
	$(MAKE) verify
	$(MAKE) down

build:
	docker compose build

run-rag: build
	docker compose run --rm app --pipeline rag

# Fires the daily RAG ingestion job once immediately and exits (CI/demo).
run-rag-ingest-once:
	python -m research_pipeline.rag.scheduler --run-once

# Starts the long-running CRON-style scheduler (default: daily at 00:00 UTC).
# Override with e.g. `make run-rag-scheduler CRON="30 6 * * *"`.
CRON ?= 0 0 * * *
run-rag-scheduler:
	python -m research_pipeline.rag.scheduler --cron "$(CRON)"

run-active-learning: build
	docker compose run --rm app --pipeline active-learning

run-customer-intelligence: build
	docker compose run --rm app --pipeline customer-intelligence

up:
	docker compose up -d redis

down:
	docker compose down -v

logs:
	docker compose logs -f

test:
	pip install -r requirements.txt --break-system-packages -q || pip install -r requirements.txt -q
	pytest tests/ -v

verify:
	python verify.py

clean:
	docker compose down -v --remove-orphans
	rm -rf output/*.json
