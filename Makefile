.PHONY: help requirements install install-all qc test ruff dataset mypy prune-branches
default: help

PYTHON_VERSION=$(shell python3 -c "import re; print(re.findall(r'^\[project\]$$.*?^requires-python\s*?=.*?(3\.\d+).*?(?:^\[|\Z)', open('pyproject.toml', 'r').read(), flags=re.MULTILINE|re.DOTALL)[0])")
PACKAGE_DIR=flood_prediction
SRC_FILES=${PACKAGE_DIR} tests

requirements: # Compile the pinned requirements if they've changed.
	@[ -f "requirements.in.md5" ] && md5sum --status -c requirements.in.md5 ||\
	( md5sum requirements.in > requirements.in.md5 && (python3 -c 'import piptools' || pip install pip-tools ) && pip-compile requirements.in -o requirements.txt )

install: # Install minimum required packages.
	@make requirements && pip install -e .${extras}

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
	@[ -f flood_prediction/data/train.csv ] || (make install-all && kaggle competitions download -c playground-series-s4e5 && unzip -o playground-series-s4e5.zip -d flood_prediction/data && rm -rf playground-series-s4e5.zip && echo 'Sucessfully downloaded data to "./data"')

prune-branches: # Remove all branches except one
	@git branch | grep -v "${except}" | xargs git branch -D

prune-branches: except=main

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m\n\t$$(echo $$l | cut -f 2- -d'#')\n"; done
