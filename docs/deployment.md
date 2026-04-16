# Deployment Guide

This guide covers running the IB Trading Agent as a persistent background service on Linux, macOS, and Windows.

## Prerequisites

- Python 3.10+ installed
- Virtual environment created and dependencies installed (`pip install -e ".[dev]"`)
- `.env` file configured (copy from `.env.example`)
- IB TWS or IB Gateway running and API connections enabled

## Watchdog Crash Limit

All deployment methods enforce a crash limit: if the agent crashes more than **5 times within 10 minutes**, restarts are stopped. This prevents infinite restart loops. Check logs to diagnose the issue, then restart manually.

---

## Linux (systemd)

1. Copy the service file:
   ```bash
   sudo cp systemd/trading-agent.service /etc/systemd/system/
   ```

2. Edit the service file to set your user and paths:
   ```bash
   sudo systemctl edit trading-agent
   ```

3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable trading-agent
   sudo systemctl start trading-agent
   ```

4. Check status and logs:
   ```bash
   sudo systemctl status trading-agent
   journalctl -u trading-agent -f
   ```

5. Stop:
   ```bash
   sudo systemctl stop trading-agent
   ```

---

## Linux / macOS / Windows (PM2)

PM2 is a cross-platform process manager for Node.js that also supports Python scripts.

1. Install PM2:
   ```bash
   npm install -g pm2
   ```

2. Start the agent:
   ```bash
   pm2 start ecosystem.config.js
   ```

3. Monitor:
   ```bash
   pm2 status
   pm2 logs ib-trading-agent
   pm2 monit
   ```

4. Auto-start on boot:
   ```bash
   pm2 startup
   pm2 save
   ```

5. Stop:
   ```bash
   pm2 stop ib-trading-agent
   ```

---

## Windows (NSSM)

NSSM (Non-Sucking Service Manager) runs any executable as a Windows service.

1. Download NSSM from https://nssm.cc/download

2. Install the service:
   ```cmd
   nssm install IBTradingAgent "C:\path\to\venv\Scripts\python.exe" "C:\path\to\agent.py" start
   nssm set IBTradingAgent AppDirectory "C:\path\to\ib-trading-agent"
   nssm set IBTradingAgent AppStdout "C:\path\to\ib-trading-agent\logs\service.log"
   nssm set IBTradingAgent AppStderr "C:\path\to\ib-trading-agent\logs\service-error.log"
   nssm set IBTradingAgent AppRestartDelay 10000
   ```

3. Start:
   ```cmd
   nssm start IBTradingAgent
   ```

4. Stop:
   ```cmd
   nssm stop IBTradingAgent
   ```

---

## macOS (launchd)

1. Create a plist file at `~/Library/LaunchAgents/com.ib-trading-agent.plist`:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
     "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
     <key>Label</key>
     <string>com.ib-trading-agent</string>
     <key>ProgramArguments</key>
     <array>
       <string>/path/to/venv/bin/python</string>
       <string>/path/to/agent.py</string>
       <string>start</string>
     </array>
     <key>WorkingDirectory</key>
     <string>/path/to/ib-trading-agent</string>
     <key>RunAtLoad</key>
     <true/>
     <key>KeepAlive</key>
     <true/>
     <key>StandardOutPath</key>
     <string>/path/to/ib-trading-agent/logs/launchd.log</string>
     <key>StandardErrorPath</key>
     <string>/path/to/ib-trading-agent/logs/launchd-error.log</string>
   </dict>
   </plist>
   ```

2. Load and start:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.ib-trading-agent.plist
   ```

3. Stop:
   ```bash
   launchctl unload ~/Library/LaunchAgents/com.ib-trading-agent.plist
   ```
