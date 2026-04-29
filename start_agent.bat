@echo off
REM Start IB Trading Agent and Dashboard in separate CMD windows.
REM Double-click this file from Explorer.

echo Starting Dashboard and Agent...

start "Dashboard Server" "%~dp0scripts\run_dashboard.bat"
timeout /t 2 /nobreak >nul
start "Trading Agent" "%~dp0scripts\run_agent.bat"

echo.
echo Both started in separate windows.
echo Dashboard: http://localhost:8888/dashboard.html
echo To stop agent: Ctrl+C in the Trading Agent window.
