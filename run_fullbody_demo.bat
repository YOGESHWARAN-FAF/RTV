@echo off
title Real-Time Virtual Try-On - Full Body Demo
cd /d "%~dp0RTV"
call "%~dp0venv\Scripts\activate.bat"
echo ========================================================
echo   Real-Time Virtual Try-On - Full Body Sequence Demo
echo ========================================================
echo Available Outfits:
echo   [0] Raw Camera (Virtual try-on off)
echo   [1] Winter Puffer Coat
echo   [2] Elegant Hanfu Silk Dress
echo   * Switch dresses anytime via sidebar or number keys [0]-[2]!
echo ========================================================
echo Mode selection:
echo   [1] Live Webcam (Auto-detect active camera)
echo   [2] Lightweight Sample Video Dataset (Offline Test)
echo ========================================================
set /p choice="Enter choice [1 or 2, default is 1]: "
if "%choice%"=="2" (
    echo Starting full body demo with sample video dataset...
    python demo.py assets/sample_video.mp4
) else (
    echo Starting full body demo with active webcam...
    python demo.py
)
pause
