.PHONY: setup run sweep score dashboard test clean

RESULTS ?= database/runs/results_top10k.json

setup:
	bash setup.sh

run:
	python main.py $(ARGS)

sweep:
	bash scripts/run_monthly_and_top10k.sh

score:
	python tests/eval/eval_score.py $(RESULTS)

dashboard:
	python -m http.server 8080 --bind 127.0.0.1

test:
	pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache
