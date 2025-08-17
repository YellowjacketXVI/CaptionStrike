# Quick Environment Test Script
# Run this to verify your setup before launching CaptionStrike

Write-Host "🧪 CaptionStrike Environment Test" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Check conda environment
Write-Host "📍 Environment Check:" -ForegroundColor Green
Write-Host "   Conda Environment: $($env:CONDA_DEFAULT_ENV)" -ForegroundColor White

if ($env:CONDA_DEFAULT_ENV -ne "CaptionStrike") {
    Write-Host "   ❌ CaptionStrike environment not active!" -ForegroundColor Red
    Write-Host "   💡 Run: conda activate CaptionStrike" -ForegroundColor Yellow
} else {
    Write-Host "   ✅ CaptionStrike environment active" -ForegroundColor Green
}

# Check Python
Write-Host ""
Write-Host "🐍 Python Check:" -ForegroundColor Green
try {
    $pythonVersion = python --version 2>$null
    Write-Host "   Version: $pythonVersion" -ForegroundColor White
    Write-Host "   ✅ Python available" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python not found!" -ForegroundColor Red
}

# Check key dependencies
Write-Host ""
Write-Host "📦 Dependencies Check:" -ForegroundColor Green

$dependencies = @(
    "torch",
    "transformers", 
    "gradio",
    "PIL",
    "cv2",
    "numpy"
)

foreach ($dep in $dependencies) {
    try {
        $result = python -c "import $dep; print('OK')" 2>$null
        if ($result -eq "OK") {
            Write-Host "   ✅ $dep" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $dep" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ❌ $dep" -ForegroundColor Red
    }
}

# Check CUDA
Write-Host ""
Write-Host "🚀 CUDA Check:" -ForegroundColor Green
try {
    $cudaResult = python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" 2>$null
    Write-Host "   $cudaResult" -ForegroundColor White
} catch {
    Write-Host "   ❌ Could not check CUDA" -ForegroundColor Red
}

# Check directories
Write-Host ""
Write-Host "📁 Directory Check:" -ForegroundColor Green

$dirs = @("src", "tests", "models")
foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        Write-Host "   ✅ $dir/" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $dir/" -ForegroundColor Red
    }
}

# Check key files
$files = @("app.py", "requirements.txt", "environment.yml")
foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. If any ❌ above, install missing components" -ForegroundColor White
Write-Host "   2. Run: .\run_captionstrike.ps1" -ForegroundColor White
Write-Host "   3. Or: python app.py --root 'D:\Datasets' --models_dir '.\models'" -ForegroundColor White
Write-Host ""
