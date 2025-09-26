@echo off
REM ===============================
REM Initialize Python venv for the app
REM ===============================

REM Get the directory of this script
set APP_DIR=%~dp0
set PYTHON_DIR=%APP_DIR%python-3.10.9
set VENV_DIR=%APP_DIR%venv

REM Check if embedded Python exists
if not exist "%PYTHON_DIR%\python.exe" (
    echo ERROR: Embedded Python not found in "%PYTHON_DIR%"
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
"%PYTHON_DIR%\python.exe" -m venv "%VENV_DIR%"

REM Activate the venv and upgrade pip
echo Activating venv and upgrading pip...
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel

echo Virtual environment setup complete!
pause