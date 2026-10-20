@echo off
REM ===================================================================
REM  seed-config.bat - create deploy\demo\config.bat from config.example.bat
REM
REM  Usage:
REM    seed-config.bat           create config.bat if missing
REM    seed-config.bat -Force    regenerate, overwriting config.bat
REM ===================================================================
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0seed-config.ps1" %*
exit /b %ERRORLEVEL%

