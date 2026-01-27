.\.venv\Scripts\Activate.ps1

ruff check .
ruff format --check .
pytest -q

Write-Host "✅ Checks passed"
