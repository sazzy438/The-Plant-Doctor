@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Plant Doctor
set "LOG=%~dp0setup_log.txt"
echo.
echo  ==========================================
echo   Plant Doctor - starting up
echo  ==========================================
echo.
echo [%date% %time%] launcher started in "%~dp0" >> "%LOG%"

if exist "%~dp0app.py" goto :check_python
echo  app.py was not found next to this file.
echo  Please extract the whole ZIP folder first: right-click the ZIP and choose "Extract All...",
echo  then open the extracted plant-doctor folder and double-click run_plant_doctor.bat there.
echo.
pause
exit /b 1

:check_python
rem Find a real Python 3.10 - 3.13. The Microsoft Store "python" stub does not count.
set "PY="
py -3 -c "import sys; assert sys.version_info[0] == 3 and 10 <= sys.version_info[1] <= 13" >nul 2>&1 && set "PY=py -3"
if defined PY goto :have_python
python -c "import sys; assert sys.version_info[0] == 3 and 10 <= sys.version_info[1] <= 13" >nul 2>&1 && set "PY=python"
if defined PY goto :have_python
echo  Python 3.10 - 3.13 was not found on this computer.
echo.
echo  1. Download Python 3.12 from https://www.python.org/downloads/windows/
echo  2. In the installer tick "Add python.exe to PATH", then click Install Now.
echo  3. Run this file again.
echo.
echo  Note: the "python" shortcut from the Microsoft Store is not enough.
echo [%date% %time%] no suitable python found >> "%LOG%"
echo.
pause
exit /b 1

:have_python
for /f "tokens=2 delims= " %%v in ('%PY% --version') do set "PYVER=%%v"
echo  Using Python %PYVER%
echo [%date% %time%] using %PY% version %PYVER% >> "%LOG%"

set "VPY=%~dp0.venv\Scripts\python.exe"
if exist "%VPY%" goto :check_deps
echo  Creating a private Python environment ... first run only, about a minute.
%PY% -m venv "%~dp0.venv"
if exist "%VPY%" goto :check_deps
echo  Could not create the environment. See setup_log.txt for details.
echo [%date% %time%] venv creation failed >> "%LOG%"
pause
exit /b 1

:check_deps
if exist "%~dp0.venv\deps_installed.txt" goto :check_model
echo  Installing dependencies ... first run only, 5-10 minutes depending on your connection.
echo.
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -r "%~dp0requirements.txt"
if %errorlevel% equ 0 goto :deps_ok
echo.
echo  Dependency installation failed. Check your internet connection, then run this file again.
echo [%date% %time%] pip install failed >> "%LOG%"
pause
exit /b 1

:deps_ok
echo ok> "%~dp0.venv\deps_installed.txt"
echo [%date% %time%] dependencies installed >> "%LOG%"

:check_model
if exist "%~dp0artifacts\plant_doctor_efficientnet_b0.pt" goto :start_app
echo.
echo  The trained model file artifacts\plant_doctor_efficientnet_b0.pt is missing.
echo  Copy it into the artifacts folder, or train one with notebooks\training_colab.ipynb.
echo [%date% %time%] model file missing >> "%LOG%"
echo.
pause
exit /b 1

:start_app
echo.
echo  Starting Plant Doctor ... your browser will open at http://127.0.0.1:7860
echo  If it does not open by itself, open that address in your browser.
echo  Keep this window open while you use the application. Close it to stop.
echo.
set "PLANT_DOCTOR_OPEN_BROWSER=1"
echo [%date% %time%] starting app >> "%LOG%"
"%VPY%" "%~dp0app.py"
echo.
echo  Plant Doctor has stopped. If it stopped unexpectedly, scroll up to read the error.
echo [%date% %time%] app exited with code %errorlevel% >> "%LOG%"
pause
endlocal
