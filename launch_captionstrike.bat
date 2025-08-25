@echo off
REM CaptionStrike Launcher - Windows Batch File
REM This script activates the conda environment and launches CaptionStrike

echo.
echo ========================================
echo 🎯 CaptionStrike - Local Dataset Builder
echo ========================================
echo.

conda activate Captionstrike

REM Set default parameters (can be overridden by command line arguments)
set ROOT_DIR=D:\Datasets
set MODELS_DIR=.\models
set PORT=7860
set VERBOSE=false
set PREFETCH_QWEN=false

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--root" (
    set ROOT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--models_dir" (
    set MODELS_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--verbose" (
    set VERBOSE=true
    shift
    goto :parse_args
)
if /i "%~1"=="--prefetch-qwen" (
    set PREFETCH_QWEN=true
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    echo Usage: launch_captionstrike.bat [options]
    echo.
    echo Options:
    echo   --root DIR          Root directory for datasets (default: D:\Datasets)
    echo   --models_dir DIR    Directory for model files (default: .\models)
    echo   --port PORT         Port for web interface (default: 7860)
    echo   --verbose           Enable verbose logging
    echo   --prefetch-qwen     Download Qwen model and exit
    echo   --help              Show this help message
    echo.
    echo Examples:
    echo   launch_captionstrike.bat
    echo   launch_captionstrike.bat --root "E:\MyDatasets" --port 8080
    echo   launch_captionstrike.bat --prefetch-qwen
    echo.
    pause
    exit /b 0
)
shift
goto :parse_args

:done_parsing

echo 📍 Configuration:
echo    Root Directory: %ROOT_DIR%
echo    Models Directory: %MODELS_DIR%
echo    Port: %PORT%
echo    Verbose: %VERBOSE%
echo.

REM Activate conda environment
echo 🔄 Activating conda environment 'CaptionStrike'...
call conda activate CaptionStrike
if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to activate conda environment 'CaptionStrike'
    echo.
    echo 🔧 Please create the environment first:
    echo    conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

REM Verify Python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python not found in activated environment
    echo.
    pause
    exit /b 1
)

echo ✅ Environment activated successfully
echo    Python:
python --version
echo    Conda Environment: %CONDA_DEFAULT_ENV%
echo.

REM Create directories if they don't exist
if not exist "%ROOT_DIR%" (
    echo 📁 Creating root directory: %ROOT_DIR%
    mkdir "%ROOT_DIR%" 2>nul
)

if not exist "%MODELS_DIR%" (
    echo 📁 Creating models directory: %MODELS_DIR%
    mkdir "%MODELS_DIR%" 2>nul
)
REM Force all model and HF caches to stay inside the project models directory
set HF_HOME=%MODELS_DIR%
set TRANSFORMERS_CACHE=%MODELS_DIR%
set HUGGINGFACE_HUB_CACHE=%MODELS_DIR%
set TORCH_HOME=%MODELS_DIR%

echo 🧩 Caches configured:
echo    HF_HOME=%HF_HOME%
echo    TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%
echo    HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%
echo    TORCH_HOME=%TORCH_HOME%
echo.

REM Build command arguments
set ARGS=--root "%ROOT_DIR%" --models_dir "%MODELS_DIR%" --port %PORT%

if "%VERBOSE%"=="true" (
    set ARGS=%ARGS% --verbose
)

if "%PREFETCH_QWEN%"=="true" (
    set ARGS=%ARGS% --prefetch-qwen
    echo 🤖 Downloading Qwen model files...
    echo    This may take several minutes depending on your internet connection
    echo.
)

echo 🚀 Starting CaptionStrike...
echo    Command: python app.py %ARGS%
echo.

if "%PREFETCH_QWEN%"=="false" (
    echo 🌐 Web interface will be available at: http://localhost:%PORT%
    echo 🛑 Press Ctrl+C to stop the server
    echo.
)

REM Launch the application
python app.py %ARGS%

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo ❌ CaptionStrike exited with error code %errorlevel%
    echo.
    echo 🔧 Troubleshooting:
    echo    1. Check that all dependencies are installed
    echo    2. Verify directory permissions
    echo    3. Run with --verbose for detailed error information
    echo    4. Check the log file: captionstrike.log
    echo.
) else (
    echo.
    echo ✅ CaptionStrike exited successfully
    echo.
)

pause
