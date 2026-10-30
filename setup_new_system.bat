@echo off
title RTV - Setup on New System
cd /d "%~dp0"

echo ======================================================================
echo           Real-Time Virtual Try-On (RTV) Environment Setup
echo ======================================================================
echo This script sets up a clean Python virtual environment for RTV on this PC.
echo.

:: 1. Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH!
    echo Please install Python 3.10 (64-bit) from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do echo [OK] Found Python: %%i

:: 2. Create virtual environment if not already present
if not exist "venv" (
    echo.
    echo [1/5] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Virtual environment 'venv' already exists.
)

:: 3. Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

:: 4. Upgrade pip and wheel
echo [3/5] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

:: 5. Install PyTorch with CUDA 12.1 (falls back to CPU if no NVIDIA GPU)
echo.
echo Installing PyTorch (CUDA 12.1 acceleration)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: 6. Install dependencies from RTV/requirements.txt
echo.
echo [4/5] Installing RTV requirements...
pip install -r RTV\requirements.txt

:: 7. Install simple_romp
if exist "ROMP_repo\simple_romp" (
    echo Installing ROMP mesh recovery module...
    cd ROMP_repo\simple_romp
    pip install -e .
    cd ..\..
)

:: 8. Check and download garment models if missing (e.g. fresh git clone)
if not exist "RTV\rtv_ckpts\coat_seq_vmssdp2ta_576\latest_net_G.pth" (
    echo.
    echo ======================================================================
    echo [5/5] Pretrained garment models are not present.
    echo Downloading core models is recommended to run the try-on demo.
    echo ======================================================================
    set /p dl="Download core outfits now (~2.1 GB)? [Y/n]: "
    if /i not "%dl%"=="n" (
        python download_models.py --mode recommended
    )
) else (
    echo [5/5] Pretrained garment models found.
)

echo.
echo ======================================================================
echo               RTV SETUP COMPLETED SUCCESSFULLY!
echo ======================================================================
echo You can now run RTV on this machine:
echo   - Double-click 'run_demo.bat'      : Full-body / Upper-body desktop UI
echo   - Double-click 'run_web_ui.bat'    : Gradio Web Studio (Browser UI)
echo ======================================================================
pause
