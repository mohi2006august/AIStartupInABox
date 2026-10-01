PY ?= python
export PYTHONPATH := src

.PHONY: dev evals economics bench ledger test

dev:
	$(PY) -m sib serve

evals:
	$(PY) -m sib evals

economics:
	$(PY) -m sib economics

bench:
	$(PY) -m sib bench --change latest

ledger:
	$(PY) -m sib ledger

test:
	$(PY) -m pytest -q
