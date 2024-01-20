setup-directories:
	mkdir -p data/raw
	mkdir -p data/processed
	mkdir -p notebooks/exploratory
	mkdir -p notebooks/report
	mkdir -p src

setup-venv:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt


make-dataset:
	python src/data/make_dataset.py

.PHONY: setup-venv