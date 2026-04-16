# Trading Agent Architecture Conventions

inclusion: always

## Tech Stack

- **Language:** Python 3.10+
- **IB Connection:** ib_insync library (async wrapper for TWS API)
- **Database:** SQLite for local deployment, PostgreSQL for server deployment
- **Scheduling:** APScheduler
- **Email:** smtplib or SendGrid
- **Async:** Use async/await and event-driven architecture throughout

## Architecture Principles

- **Event-Driven:** Use event-driven architecture. Avoid polling where streaming is available.
- **Async First:** All I/O operations must be async. Never block the event loop.
- **Latency Critical:** Market data analysis and trade decisions must execute as fast as possible.
- **State Persistence:** Save state to database after every trade and analytical decision.
- **Graceful Shutdown:** On SIGINT/SIGTERM, close all IB connections and persist state before exit.
- **Crash Recovery:** On restart, load last saved state, fetch missed market data, analyze changes, resume without manual intervention.

## Configuration

- ALL configuration in a single .env file.
- Environment switching (paper/live) via single ENVIRONMENT variable.
- Code must work identically on Windows, Mac, and Linux.
- Migration to remote server (VPS/cloud) must require zero code changes.

## CLI Interface

The agent must provide these CLI commands:
- `python agent.py start` — start the agent
- `python agent.py stop` — graceful shutdown
- `python agent.py status` — show current portfolio state
- `python agent.py report` — generate report immediately

## Logging

- Log every decision and trade to both file and console.
- Log levels: INFO (normal), WARNING (risk alerts), ERROR (errors).
- Retain logs for minimum 30 days.
- Use Python's standard logging module with rotating file handlers.

## Deployment

- Run as background service/daemon (24/7).
- Implement watchdog mechanism for auto-restart on crash.
- Support systemd (Linux), PM2, or equivalent process managers.
