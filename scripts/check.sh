#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

ruff check .
ruff format --check .
pytest -q

echo "✅ Checks passed"
