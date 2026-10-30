@echo off
title Package RTV for Another System
cd /d "%~dp0"

echo ======================================================================
echo           Real-Time Virtual Try-On (RTV) - Portable Packager
echo ======================================================================
echo This tool builds a clean, lightweight zip package of RTV ready to run
echo on another system without 16+ GB of machine-specific junk.
echo.
echo Select packaging mode:
echo   [1] Recommended (Code + Assets + Winter Coat + Hanfu + Lab Coat + RCB) ~2.1 GB [DEFAULT]
echo   [2] All Garments (Code + Assets + ALL 10 Pretrained Garment Models)     ~4.9 GB
echo   [3] Code Only    (Code + Assets + SMPL, 0 heavy model checkpoints)     ~85 MB
echo ======================================================================
set /p choice="Enter choice [1, 2, or 3, default is 1]: "

if "%choice%"=="2" (
    python package_project.py --mode all
) else if "%choice%"=="3" (
    python package_project.py --mode code_only
) else (
    python package_project.py --mode recommended
)

echo.
pause

