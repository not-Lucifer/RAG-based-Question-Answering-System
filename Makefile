# Run inside an activated virtualenv. Windows without make: use ./make.ps1 <target>.
PY ?= python

.PHONY: install samples ingest api ui test eval reset lint format

install:
	$(PY) -m pip install -r requirements.txt

samples:
	$(PY) scripts/make_samples.py

ingest:
	$(PY) scripts/ingest_folder.py --path data/samples

api:
	$(PY) -m uvicorn app.main:app --reload --port 8000

ui:
	$(PY) -m streamlit run frontend/streamlit_app.py

test:
	$(PY) -m pytest -q

eval:
	$(PY) scripts/run_eval.py

reset:
	$(PY) scripts/reset_index.py

lint:
	ruff check . && black --check .

format:
	ruff check --fix . && black .
