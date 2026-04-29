"""Lightweight HTTP server for the live log dashboard.

Serves static files from the runtime logs directory and provides an
``/api/log/today`` endpoint that returns only the most recent log lines
from today — reading from the end of the file for efficiency.

The server auto-detects the log directory:
  1. ``C:/temp/trading-agent/logs`` (runtime copy on Windows)
  2. ``./logs`` relative to the project root (fallback)

Usage::

    python scripts/dashboard_server.py          # default port 8888
    python scripts/dashboard_server.py 9000     # custom port
"""

import os
import sys
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Maximum number of today's lines to return (keeps response fast)
MAX_LINES = 2000

# Auto-detect the runtime logs directory
# Prefer the log file that was most recently written to.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNTIME_LOGS = r"C:\temp\trading-agent\logs"
_WORKSPACE_LOGS = os.path.join(_PROJECT_ROOT, "logs")

_runtime_log = os.path.join(_RUNTIME_LOGS, "agent.log")
_workspace_log = os.path.join(_WORKSPACE_LOGS, "agent.log")

def _pick_logs_dir() -> str:
    """Choose the logs directory where the agent is actively writing.

    Always prefers the workspace logs directory since the agent runs
    from the workspace.  The C:\\temp copy is only used for running
    tests and may contain stale data.
    """
    workspace_exists = os.path.isfile(_workspace_log)
    if workspace_exists:
        return _WORKSPACE_LOGS

    runtime_exists = os.path.isfile(_runtime_log)
    if runtime_exists:
        return _RUNTIME_LOGS

    return _WORKSPACE_LOGS

LOGS_DIR = _pick_logs_dir()

LOG_FILE = os.path.join(LOGS_DIR, "agent.log")

# Static HTML is always served from the workspace logs/ (where dashboard.html lives)
STATIC_DIR = _WORKSPACE_LOGS


def _tail_today_lines(filepath: str, today: str, max_lines: int) -> list[str]:
    """Read the last `max_lines` lines from today by scanning backwards.

    This avoids reading the entire (potentially 100+ MB) file.
    Uses a binary reverse-read strategy: seek to the end, read chunks
    backwards, split into lines, and collect until we have enough or
    hit a line from a different date.
    """
    if not os.path.isfile(filepath):
        return []

    chunk_size = 1024 * 256  # 256 KB chunks
    lines: list[str] = []

    try:
        with open(filepath, "rb") as f:
            f.seek(0, 2)  # seek to end
            file_size = f.tell()
            if file_size == 0:
                return []

            remaining = file_size
            leftover = b""

            while remaining > 0 and len(lines) < max_lines:
                read_size = min(chunk_size, remaining)
                remaining -= read_size
                f.seek(remaining)
                chunk = f.read(read_size) + leftover

                # Split into lines (keeping line endings)
                parts = chunk.split(b"\n")

                # The first element is a partial line — save for next iteration
                leftover = parts[0]

                # Process lines from bottom to top
                for i in range(len(parts) - 1, 0, -1):
                    raw = parts[i]
                    if not raw.strip():
                        continue

                    try:
                        line = raw.decode("utf-8", errors="replace").rstrip("\r")
                    except Exception:
                        continue

                    # Check if this line starts with today's date
                    if line[:10] == today:
                        lines.append(line)
                    elif line[:4].isdigit() and line[:10] != today:
                        # Hit a line from a different date — we're done
                        # (log is chronological, so everything above is older)
                        remaining = 0
                        break
                    elif lines:
                        # Continuation line (traceback etc.) — include it
                        lines.append(line)

                    if len(lines) >= max_lines:
                        break

            # Process the very first leftover piece
            if leftover.strip() and len(lines) < max_lines:
                try:
                    line = leftover.decode("utf-8", errors="replace").rstrip("\r")
                    if line[:10] == today:
                        lines.append(line)
                except Exception:
                    pass

    except Exception:
        return []

    # Reverse to chronological order (we collected bottom-up)
    lines.reverse()
    return lines


class DashboardHandler(SimpleHTTPRequestHandler):
    """Extends SimpleHTTPRequestHandler with a today-only log endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def handle(self):
        """Override to suppress ConnectionAbortedError / BrokenPipeError.

        These occur when the browser closes the connection before the
        server finishes sending (e.g. page refresh, tab close).  They
        are harmless and should not spam the console.
        """
        try:
            super().handle()
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/api/log/today":
            self._serve_today_log()
        elif path == "/api/performance":
            self._serve_performance_data()
        elif path == "/portfolio.json":
            self._serve_file_from_runtime("portfolio.json")
        else:
            super().do_GET()

    def end_headers(self):
        """Add no-cache headers to all responses to prevent stale content."""
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _serve_today_log(self):
        """Return the last MAX_LINES log lines from today."""
        today = datetime.now().strftime("%Y-%m-%d")
        lines = _tail_today_lines(LOG_FILE, today, MAX_LINES)

        body = "\n".join(lines).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file_from_runtime(self, filename: str):
        """Serve a file from the runtime logs dir (e.g. portfolio.json)."""
        filepath = os.path.join(LOGS_DIR, filename)
        if not os.path.isfile(filepath):
            self.send_error(404, f"{filename} not found")
            return

        try:
            with open(filepath, "rb") as f:
                data = f.read()
        except Exception as exc:
            self.send_error(500, str(exc))
            return

        content_type = "application/json" if filename.endswith(".json") else "text/plain"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _serve_performance_data(self):
        """Return portfolio performance vs SPY benchmark as JSON.

        Reads portfolio_snapshots from SQLite and fetches SPY daily closes
        from Yahoo Finance. Returns JSON with:
        {
            "portfolio": [{"date": "2026-04-22", "value": 978483, "pnl_pct": 0.0}, ...],
            "benchmark": [{"date": "2026-04-22", "value": 100.0, "pnl_pct": 0.0}, ...],
            "initial_value": 978483.35
        }

        Portfolio values are sampled once per day (last snapshot of each day).
        SPY is normalized to start at 100 on the same date as the first snapshot.
        """
        import json
        import sqlite3

        db_path = os.path.join(_PROJECT_ROOT, "data", "agent.db")
        if not os.path.isfile(db_path):
            self.send_error(404, "Database not found")
            return

        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # Get daily portfolio values (last snapshot per day)
            c.execute("""
                SELECT DATE(snapshot_time) as day,
                       total_value, total_pnl_pct
                FROM portfolio_snapshots
                WHERE id IN (
                    SELECT MAX(id) FROM portfolio_snapshots
                    GROUP BY DATE(snapshot_time)
                )
                ORDER BY day
            """)
            rows = c.fetchall()
            conn.close()

            if not rows:
                self._send_json({"portfolio": [], "benchmark": [], "initial_value": 0})
                return

            initial_value = rows[0][1]
            portfolio = []
            for day, value, pnl_pct in rows:
                pct = ((value - initial_value) / initial_value * 100) if initial_value > 0 else 0
                portfolio.append({"date": day, "value": round(value, 2), "pnl_pct": round(pct, 2)})

            # Fetch SPY benchmark from Yahoo Finance
            benchmark = []
            try:
                import yfinance as yf
                start_date = rows[0][0]
                spy = yf.Ticker("SPY")
                hist = spy.history(start=start_date, interval="1d")
                if not hist.empty:
                    spy_initial = float(hist.iloc[0]["Close"])
                    for idx, row in hist.iterrows():
                        day_str = idx.strftime("%Y-%m-%d")
                        spy_close = float(row["Close"])
                        spy_pct = ((spy_close - spy_initial) / spy_initial * 100)
                        benchmark.append({
                            "date": day_str,
                            "value": round(spy_close, 2),
                            "pnl_pct": round(spy_pct, 2),
                        })
            except Exception as exc:
                # Benchmark unavailable — return portfolio only
                pass

            self._send_json({
                "portfolio": portfolio,
                "benchmark": benchmark,
                "initial_value": round(initial_value, 2),
            })

        except Exception as exc:
            self.send_error(500, str(exc))

    def _send_json(self, data):
        """Send a JSON response."""
        import json
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Suppress routine GET logs to keep the console clean."""
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard server running at http://localhost:{port}/dashboard.html")
    print(f"  Log API:    http://localhost:{port}/api/log/today")
    print(f"  Log file:   {LOG_FILE}")
    print(f"  Static dir: {STATIC_DIR}")
    print(f"  Max lines:  {MAX_LINES}")
    print("  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server")
        server.shutdown()


if __name__ == "__main__":
    main()
