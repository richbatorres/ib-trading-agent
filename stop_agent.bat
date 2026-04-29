@echo off
REM Stop IB Trading Agent gracefully
echo Stopping IB Trading Agent...

set PYTHONPATH=G:\My Drive\Documents\practice\trading agent
"C:\temp\trading-agent\venv\Scripts\python.exe" agent.py stop

echo Agent stop signal sent.
pause
