#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${MODEL_PYTHON:-.venv/bin/python}
"$PYTHON" inspect_model.py
