# Project tasks. Every target reports a real exit code -- never pipe these
# through `tail` or `head` in a `&&` chain, which masks the status.

PY ?= $(HOME)/anaconda3/envs/capstone/python.exe
export HF_HOME ?= C:\hf

.PHONY: help test lint check reproduce tables install-hooks

help:
	@echo "make test           run the test suite"
	@echo "make lint           run ruff"
	@echo "make check          lint + test (what the pre-commit hook runs)"
	@echo "make reproduce      run all cells on 100 dev pairs and rebuild tables"
	@echo "make tables         rebuild tables from existing raw results, no models"
	@echo "make install-hooks  install the pre-commit hook that blocks failing commits"

test:
	"$(PY)" -m pytest -q

lint:
	"$(PY)" -m ruff check src/ tests/ scripts/ reproduce.py

check: lint test

reproduce:
	"$(PY)" reproduce.py

tables:
	"$(PY)" reproduce.py --tables-only

install-hooks:
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "installed .git/hooks/pre-commit"
