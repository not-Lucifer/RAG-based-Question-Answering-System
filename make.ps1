# PowerShell equivalent of the Makefile for Windows machines without `make`.
# Usage: .\make.ps1 install | samples | ingest | api | ui | test | eval | reset | lint | format
param([Parameter(Mandatory = $true)][string]$Target)

$ErrorActionPreference = "Stop"
$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

switch ($Target) {
    "install" { & $py -m pip install -r requirements.txt }
    "samples" { & $py scripts/make_samples.py }
    "ingest"  { & $py scripts/ingest_folder.py --path data/samples }
    "api"     { & $py -m uvicorn app.main:app --reload --port 8000 }
    "ui"      { & $py -m streamlit run frontend/streamlit_app.py }
    "test"    { & $py -m pytest -q }
    "eval"    { & $py scripts/run_eval.py }
    "reset"   { & $py scripts/reset_index.py }
    "lint"    { & $py -m ruff check .; if ($?) { & $py -m black --check . } }
    "format"  { & $py -m ruff check --fix .; & $py -m black . }
    default   { Write-Error "Unknown target '$Target'" }
}
exit $LASTEXITCODE
