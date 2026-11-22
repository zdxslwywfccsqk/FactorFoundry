PYTHON ?= python3
export PYTHONPATH := src

.PHONY: download build test run report diagnose agentic agentic-holdout agentic-mini score brand all clean

download:
	$(PYTHON) -m alpha_foundry.pipeline --config config.json --stage download

build:
	$(PYTHON) -m alpha_foundry.pipeline --config config.json --stage build

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m alpha_foundry.pipeline --config config.json --stage all

report:
	$(PYTHON) -m alpha_foundry.pipeline --config config.json --stage report

agentic:
	$(PYTHON) -m alpha_foundry.agentic.cli run --config config.json --mode deterministic --factor-limit 12 --max-audits 4

agentic-holdout:
	$(PYTHON) -m alpha_foundry.agentic.cli run --config config.json --mode deterministic --factor-limit 12 --max-audits 4 --release-holdout

agentic-mini:
	$(PYTHON) -m alpha_foundry.agentic.cli run --config config.json --mode mini-agent --factor-limit 8 --max-audits 4

score:
	@test -n "$(RUN)" || (echo "Usage: make score RUN=<run_id>" && exit 2)
	$(PYTHON) -m alpha_foundry.agentic.cli score --config config.json $(RUN)

brand:
	$(PYTHON) scripts/render_brand_assets.py

diagnose:
	$(PYTHON) scripts/detailed_diagnostics.py

all: run

clean:
	rm -rf output/figures experiments/*.csv experiments/*.sqlite experiments/*.jsonl
