@echo off
REM ===============================================
REM Install dependencies for ALTA custom node
REM ===============================================

setlocal
set SCRIPT_DIR=%~dp0
set COMFYUI_ROOT=%SCRIPT_DIR%..\..
set VENV_PYTHON=%COMFYUI_ROOT%\.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo ❌ Could not find ComfyUI venv at: %VENV_PYTHON%
    echo Please run ComfyUI once to create the virtual environment.
    exit /b 1
)

echo 🚀 Installing ALTA node dependencies...
"%VENV_PYTHON%" -m pip install -r "%SCRIPT_DIR%requirements.txt"

if %errorlevel% equ 0 (
    echo ✅ ALTA NODES LOAD SUCCESS
) else (
    echo ❌ Installation failed
    exit /b 1
)
endlocal