#!/usr/bin/env bash
set -euo pipefail

PY=python
command -v python >/dev/null 2>&1 || PY=python3

$PY -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pre-commit install || true

echo "✅ Bootstrapped (.venv created, dev deps installed)"
