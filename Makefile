# Empathic Memory Bench — reproducibility commands
#
# Quick start:
#   1. cp .env.example .env  && fill in at least COHERE_API_KEY + one judge provider
#   2. make install           (creates venv, installs deps)
#   3. make bench-v3          (≈2-3 minutes with Qwen judge, ≈8-10 min with 7-judge)
#   4. make judge-agreement   (Krippendorff α on the 8-judge snapshot)
#
# External benchmarks:
#   make locomo     (LoCoMo, ACL 2024, requires data — see locomo target)
#   make lme-s      (LongMemEval_S, ICLR 2025)
#
# Conventions:
#   - Uses .venv/bin/python3 if present, else python3.
#   - Sources .env so scripts see COHERE_API_KEY etc. (common.py reads env fallbacks).

.DEFAULT_GOAL := help
SHELL         := /bin/bash
PY            ?= $(shell [ -x .venv/bin/python3 ] && echo .venv/bin/python3 || echo python3)

# Export variables from .env to all sub-commands (if .env exists)
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install deps
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install cohere openai numpy

bench-v3: ## Run empathic-bench v3 (default: Qwen judge, fast single-judge)
	$(PY) bench-empathic-memory-v3.py \
		--systems cosine,bm25,hybrid,pulse_v3 \
		--judges qwen \
		--parallel-judges 3

bench-v3-8judge: ## Run empathic-bench v3 with the full 8-judge pool (expensive: ~280 judge calls)
	$(PY) bench-empathic-memory-v3.py \
		--systems cosine,bm25,hybrid,pulse_v3 \
		--judges kimi,kimi-preview,glm,glm-51,qwen,deepseek,openai \
		--parallel-judges 4
	@echo ""
	@echo "8th judge (Claude Opus) is run in-chat via Claude Code — see scripts/opus_verdicts_*.py"
	@echo "Then merge with: python external-evals/scripts/merge_opus_into_snapshot.py"

judge-agreement: ## Compute Krippendorff α on the published 8-judge snapshot
	$(PY) external-evals/scripts/compute_judge_agreement.py \
		--raw external-evals/results/bench-v3-20260424-1714.json \
		--opus external-evals/results/opus_verdicts_2026-04-24_v2.json \
		--out external-evals/snapshots/2026-04-24-bench-v3-pulse-v3-8judge/agreement.md

locomo: ## Run LoCoMo benchmark (requires LoCoMo data, see external-evals/locomo/README)
	@if [ ! -f "$${LOCOMO_DATA:-$$HOME/dev/ai/locomo-data/locomo/data/locomo10.json}" ]; then \
		echo "LoCoMo data not found. Clone github.com/snap-research/locomo to $$HOME/dev/ai/locomo-data"; \
		exit 1; \
	fi
	$(PY) external-evals/scripts/run_pulse_locomo.py \
		--data "$${LOCOMO_DATA:-$$HOME/dev/ai/locomo-data/locomo/data/locomo10.json}" \
		--out external-evals/locomo/run.jsonl \
		--mode cosine --provider qwen
	$(PY) external-evals/scripts/evaluate_locomo.py \
		--hyps external-evals/locomo/run.jsonl \
		--out external-evals/locomo/run-scored.json

lme-s: ## Run LongMemEval_S benchmark (requires LME data)
	@if [ ! -f "$${LME_DATA:-$$HOME/dev/ai/longmemeval_data/longmemeval_s.json}" ]; then \
		echo "LongMemEval data not found. Clone github.com/xiaowu0162/LongMemEval"; \
		exit 1; \
	fi
	$(PY) external-evals/scripts/run_pulse_lme.py \
		--data "$${LME_DATA:-$$HOME/dev/ai/longmemeval_data/longmemeval_s.json}" \
		--out external-evals/longmemeval/run.jsonl \
		--mode cosine --provider kimi

paper: ## Paper has moved — see ../pulse-paper (private repo: github.com/nikshilov/pulse-paper)
	@echo "Paper has moved to its own private repo: github.com/nikshilov/pulse-paper"
	@echo ""
	@echo "If you have it cloned alongside this repo:"
	@echo "    cd ../pulse-paper && tectonic paper.tex"
	@echo ""
	@echo "Otherwise: gh repo clone nikshilov/pulse-paper ../pulse-paper"
	@exit 1

clean: ## Remove caches and temp files
	rm -rf __pycache__ .pytest_cache */__pycache__ */*/__pycache__

.PHONY: help install bench-v3 bench-v3-8judge judge-agreement locomo lme-s paper clean
