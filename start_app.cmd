@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PYTHON=C:\Users\Migogla\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if not exist "%PYTHON%" (
    echo Python launcher not found: %PYTHON%
    exit /b 1
)

"%PYTHON%" "%SCRIPT_DIR%app.py"
