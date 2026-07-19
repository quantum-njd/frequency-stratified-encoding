#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

"$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "Python environment ready: $PROJECT_DIR/.venv"
echo "Run all experiments with:"
echo "  $PROJECT_DIR/.venv/bin/python $PROJECT_DIR/experiments/run_all.py"
