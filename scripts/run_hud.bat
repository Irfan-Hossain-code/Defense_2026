@echo off
cd /d "%~dp0..\hud"
if not exist node_modules (
  call npm install
)
call npm run dev
