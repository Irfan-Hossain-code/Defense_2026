@echo off
cd /d "%~dp0.."
if not exist .venv\Scripts\activate.bat (
  echo Create venv first: python -m venv .venv
  exit /b 1
)
call .venv\Scripts\activate.bat
python main.py
pause
