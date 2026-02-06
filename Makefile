.PHONY: all data pipeline test

all: pipeline

pipeline: data
	python -m src.pipeline

data:
	python -m src.prepare_data --artifacts-zip yenepoya_predictor_artifacts.zip --output-dir data

test:
	python -m unittest discover -s tests -p "test_*.py"
