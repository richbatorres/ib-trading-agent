# IB Trading Agent

Autonomous AI trading agent for Interactive Brokers. Monitors real-time market data, applies multiple trading strategies (momentum, mean reversion, trend following), enforces strict risk management, and integrates Polymarket sentiment data.

## Prerequisites

- Python 3.10+
- Interactive Brokers TWS or IB Gateway running locally
- An IB Paper Trading or Live account

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repo-url>
   cd ib-trading-agent
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure the agent:**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in your IB account details. Start with `ENVIRONMENT=paper`.

5. **Start TWS or IB Gateway** and enable API connections (Edit → Global Configuration → API → Settings).

## Usage

```bash
python agent.py start    # Start the agent
python agent.py stop     # Graceful shutdown
python agent.py status   # Show portfolio and agent state
python agent.py report   # Generate and send report now
```

## Live Dashboard

The agent includes a browser-based live log dashboard that shows today's log messages, portfolio status, trade signals, and more.

```bash
# Start the dashboard server (default port 8888)
python scripts/dashboard_server.py

# Or with a custom port
python scripts/dashboard_server.py 9000
```

Then open `http://localhost:8888/dashboard.html` in your browser.

The dashboard server filters logs server-side via `/api/log/today`, so only today's lines are sent to the browser — keeping the page fast even after weeks of accumulated logs.

When using PM2, the dashboard server starts automatically alongside the agent:
```bash
pm2 start ecosystem.config.js   # Starts both agent and dashboard server
```

## Running Tests

```bash
pytest                          # Run all tests
pytest tests/unit/              # Unit tests only
pytest tests/property/          # Property-based tests only
pytest tests/integration/       # Integration tests only
pytest tests/performance/       # Performance tests only
pytest tests/backtest/          # Backtest tests only
```
