#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Python environment not found. Run ./setup_python_env.sh first." >&2
    exit 1
fi

if "$PYTHON" -c "import matplotlib, numpy, PIL" 2>/dev/null; then
    exec "$PYTHON" "$PROJECT_DIR/experiments/run_all.py"
fi

echo "Python dependencies are incomplete. Run ./setup_python_env.sh first." >&2
exit 1
