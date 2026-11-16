@echo off
REM ============================================================
REM  Plant Doctor - one-click setup and launch for Windows
REM  Requires: Python 3.10 - 3.13 installed from python.org
REM  (tick "Add python.exe to PATH" during installation).
REM ============================================================
setlocal
cd /d "%~dp0"
title Plant Doctor

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Python was not found. Please install Python 3.11 or newer from
  echo  https://www.python.org/downloads/windows/  and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo  Creating a private Python environment (first run only, this takes a minute)...
  python -m venv .venv
  if errorlevel 1 ( echo  Could not create the environment. & pause & exit /b 1 )
)

call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_installed" (
  echo  Installing dependencies (first run only, 5-10 minutes depending on your connection)...
  python -m pip install --upgrade pip >nul
  python -m pip install -r requirements.txt
  if errorlevel 1 ( echo  Dependency installation failed. Check your internet connection and run this file again. & pause & exit /b 1 )
  echo ok> ".venv\.deps_installed"
)

if not exist "artifacts\plant_doctor_efficientnet_b0.pt" (
  echo.
  echo  The trained model file artifacts\plant_doctor_efficientnet_b0.pt is missing.
  echo  Copy it into the artifacts folder, or train one with notebooks\training_colab.ipynb.
  echo.
  pause
  exit /b 1
)

echo.
echo  Starting Plant Doctor ... your browser will open at http://127.0.0.1:7860
echo  Keep this window open while you use the application. Close it to stop.
echo.
set PLANT_DOCTOR_OPEN_BROWSER=1
python app.py
pause
