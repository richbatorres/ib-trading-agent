@echo off
set "PYTHONPATH=G:\My Drive\Documents\practice\trading agent"
cd /d "G:\My Drive\Documents\practice\trading agent"
if exist "data\agent.pid" del "data\agent.pid"
"C:\temp\trading-agent\venv\Scripts\python.exe" agent.py start
pause
