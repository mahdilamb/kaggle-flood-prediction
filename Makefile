.PHONY: help requirements venv install install-all qc test ruff dataset mypy
default: help

PYTHON_VERSION=$(shell python3 -c "import re; print(re.findall(r'^\[project\]$$.*?^requires-python\s*?=.*?(3\.\d+).*?(?:^\[|\Z)', open('pyproject.toml', 'r').read(), flags=re.MULTILINE|re.DOTALL)[0])")
PACKAGE_DIR=flood_prediction
SRC_FILES=${PACKAGE_DIR} tests

requirements: # Compile the pinned requirements if they've changed.
	@[ -f "requirements.in.md5" ] && md5sum --status -c requirements.in.md5 ||\
	( md5sum requirements.in > requirements.in.md5 && pip-compile requirements.in -o requirements.txt )

venv: # Create the virtual environment. 
	@[ ! -d .venv ] && python${PYTHON_VERSION} -m venv .venv || echo "Virtual environment already exists at './.venv'."

install: # Install minimum required packages.
	@make venv && make requirements && . .venv/bin/activate && pip install -e .${extras}

install-all: # Install all packages
	@make install extras='[all]'

ruff: # Run ruff
	@ruff check ${SRC_FILES} --fix

mypy: # Run mypy
	@mypy ${SRC_FILES}

test: # Run pytest
	@pytest --cov=${PACKAGE_DIR} tests --cov-report term-missing

qc:  # Format and test
	@make ruff; make mypy; make test

dataset: # Download the kaggle dataset
	@[ -f flood_prediction/data/train.csv ] || (make install-all && . .venv/bin/activate && kaggle competitions download -c playground-series-s4e5 && unzip -o playground-series-s4e5.zip -d flood_prediction/data && rm -rf playground-series-s4e5.zip && echo 'Sucessfully downloaded data to "./data"')

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m\n\t$$(echo $$l | cut -f 2- -d'#')\n"; done
