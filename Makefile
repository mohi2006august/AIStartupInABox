PY ?= python

dev:
	$(PY) -m sib serve

evals:
	$(PY) -m sib evals

economics:
	$(PY) -m sib economics

test:
	$(PY) -m pytest -q
