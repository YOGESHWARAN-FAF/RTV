@echo off
title Download Pretrained RTV Garment Models
cd /d "%~dp0"

echo ======================================================================
echo           RTV Pretrained Model Downloader (from Hugging Face)
echo ======================================================================
echo This script downloads official neural models required to run RTV.
echo.
echo Select download option:
echo   [1] Core Outfits (Winter Coat + Hanfu Silk Dress + Lab Coat) ~2.1 GB [DEFAULT]
echo   [2] All Outfits  (All 8 official models)                      ~5.6 GB
echo ======================================================================
set /p choice="Enter choice [1 or 2, default is 1]: "

if "%choice%"=="2" (
    python download_models.py --mode all
) else (
    python download_models.py --mode recommended
)

echo.
pause
