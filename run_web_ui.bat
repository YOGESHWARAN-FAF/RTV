@echo off
title RTV Interactive Virtual Try-On Studio (Web / Colab UI)
cd /d "%~dp0RTV"

echo ========================================================
echo   RTV Interactive Virtual Try-On Studio (Web UI)
echo ========================================================

if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment not found. Using system python...
)

echo Starting Gradio Web Studio with public shareable URL...
python colab_gradio_app.py --share

pause
