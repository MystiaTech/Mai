python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pre-commit install

Write-Host "✅ Bootstrapped (.venv created, dev deps installed)"
