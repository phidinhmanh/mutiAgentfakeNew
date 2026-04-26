# scripts/quick_check.ps1
# A lightweight check script for developers before committing code

Write-Host "Running Quick Check..." -ForegroundColor Cyan

Write-Host "`n1. Running Ruff Linter & Formatter..." -ForegroundColor Yellow
uv run ruff check src tests scripts
$lintExit = $LASTEXITCODE

Write-Host "`n2. Running Unit Tests (Fast)..." -ForegroundColor Yellow
uv run pytest tests/unit -m "not slow"
$testExit = $LASTEXITCODE

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
if ($lintExit -eq 0 -and $testExit -eq 0) {
    Write-Host "✅ All checks passed! Ready to commit." -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ Some checks failed. Please fix them before committing." -ForegroundColor Red
    exit 1
}
