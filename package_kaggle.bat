@echo off
title Package RTV for Kaggle Cloud
cd /d "%~dp0RTV"
call "%~dp0venv\Scripts\activate.bat"

echo ========================================================
echo        RTV Kaggle & Cloud GPU Packager
echo ========================================================
echo This tool builds a clean, lightweight zip archive of RTV
echo without Windows venv or .git bloat, so Kaggle uploads and
echo extracts in seconds without timing out.
echo.
echo Select packaging mode:
echo   [1] Recommended (Code + Assets + Winter Coat + Lab Coat + Hanfu) ~1.4 GB [DEFAULT]
echo   [2] All Garments (Code + Assets + All 10 Garment Models) ~7 GB
echo   [3] Code Only (Code + Assets + SMPL, no heavy models) ~85 MB
echo ========================================================
set /p choice="Enter choice [1, 2, or 3, default is 1]: "

if "%choice%"=="2" (
    python package_for_kaggle.py --mode all
) else if "%choice%"=="3" (
    python package_for_kaggle.py --mode code_only
) else (
    python package_for_kaggle.py --mode recommended
)

echo.
pause

