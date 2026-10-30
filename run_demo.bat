@echo off
title Real-Time Virtual Try-On (RTV)
cd /d "%~dp0RTV"
call "%~dp0venv\Scripts\activate.bat"
echo ========================================================
echo        Real-Time Virtual Try-On (RTV) Launcher
echo ========================================================
echo Mode Selection:
echo   [1] Full-Body Try-On with Live Webcam (Winter Coat, Hanfu Dress) [DEFAULT]
echo   [2] Full-Body Try-On with Sample Video (Winter Coat, Hanfu Dress)
echo   [3] Upper-Body Try-On with Live Webcam (6 Wardrobe Outfits)
echo   [4] Upper-Body Try-On with Sample Video (6 Wardrobe Outfits)
echo   [5] Interactive Web Studio / Colab Interface (Gradio Browser UI + Public Link)
echo ========================================================
set /p choice="Enter choice [1, 2, 3, 4, or 5, default is 1]: "

if "%choice%"=="5" (
    echo Starting Interactive Gradio Web Studio...
    python colab_gradio_app.py --share
) else if "%choice%"=="2" (
    echo Starting Full-Body Demo with sample video...
    python demo.py assets/sample_video.mp4
) else if "%choice%"=="3" (
    echo Starting Upper-Body Demo with active webcam...
    python rtl_demo.py
) else if "%choice%"=="4" (
    echo Starting Upper-Body Demo with sample video...
    python rtl_demo.py assets/sample_video.mp4
) else (
    echo Starting Full-Body Demo with active webcam...
    python demo.py
)
pause
