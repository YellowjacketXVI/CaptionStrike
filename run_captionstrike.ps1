# CaptionStrike PowerShell Launcher
# Run this script after activating your conda environment

param(
    [string]$Root = "D:\Datasets",
    [string]$ModelsDir = ".\models",
    [int]$Port = 7860,
    [switch]$Debug,
    [switch]$Check
)

Write-Host "🎯 CaptionStrike - Local Dataset Builder" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Check if conda environment is activated
if ($env:CONDA_DEFAULT_ENV -ne "CaptionStrike") {
    Write-Host "⚠️  WARNING: CaptionStrike conda environment not detected!" -ForegroundColor Yellow
    Write-Host "Please run: conda activate CaptionStrike" -ForegroundColor Yellow
    Write-Host ""
}

# Show current environment info
Write-Host "📍 Current Environment:" -ForegroundColor Green
Write-Host "   Conda Env: $($env:CONDA_DEFAULT_ENV)" -ForegroundColor White
Write-Host "   Python: $(python --version 2>$null)" -ForegroundColor White
Write-Host "   Working Dir: $(Get-Location)" -ForegroundColor White
Write-Host ""

# Check if showing checklist
if ($Check) {
    Write-Host "📋 Running acceptance checklist..." -ForegroundColor Green
    python app.py --check
    exit
}

# Validate paths
if (-not (Test-Path $Root)) {
    Write-Host "📁 Creating root directory: $Root" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $Root -Force | Out-Null
}

if (-not (Test-Path $ModelsDir)) {
    Write-Host "📁 Creating models directory: $ModelsDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
}

Write-Host "🚀 Starting CaptionStrike..." -ForegroundColor Green
Write-Host "   Root Directory: $Root" -ForegroundColor White
Write-Host "   Models Directory: $ModelsDir" -ForegroundColor White
Write-Host "   Port: $Port" -ForegroundColor White
Write-Host ""

# Build command arguments
$args = @("app.py", "--root", $Root, "--models_dir", $ModelsDir, "--port", $Port)

if ($Debug) {
    $args += "--debug"
    Write-Host "🔍 Debug mode enabled" -ForegroundColor Yellow
}

Write-Host "💻 Command: python $($args -join ' ')" -ForegroundColor Cyan
Write-Host ""
Write-Host "🌐 Web interface will open at: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "🛑 Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

# Execute the application
try {
    & python @args
}
catch {
    Write-Host "❌ Error starting CaptionStrike: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "   1. Make sure conda environment is activated: conda activate CaptionStrike" -ForegroundColor White
    Write-Host "   2. Install dependencies: pip install -r requirements.txt" -ForegroundColor White
    Write-Host "   3. Check Python version: python --version" -ForegroundColor White
    Write-Host "   4. Run with debug: .\run_captionstrike.ps1 -Debug" -ForegroundColor White
    exit 1
}
