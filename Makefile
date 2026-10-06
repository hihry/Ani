.PHONY: install dev lint test format run serve ui

install:
	pip install -e .

dev:
	pip install -e .[dev,ui]

lint:
	ruff check .
	mypy .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

run:
	m2a

serve:
	uvicorn m2a.api:app --reload

ui:
	streamlit run m2a/ui.py
