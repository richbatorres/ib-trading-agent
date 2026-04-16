#!/usr/bin/env python3
"""IB Trading Agent — CLI entry point.

Usage:
    python agent.py start   — Start the agent as a background process
    python agent.py stop    — Graceful shutdown of the running agent
    python agent.py status  — Display portfolio and agent state
    python agent.py report  — Generate and send daily report immediately
    python agent.py test    — Run the test suite

Requirements: 20.1, 20.2, 20.3, 20.4, 20.5
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys

from src.config import AgentConfig
from src.logging_config import setup_logging
from src.services.state_manager import StateManager

logger = logging.getLogger(__name__)

PID_FILE = os.path.join("data", "agent.pid")


def _write_pid_file() -> None:
    """Write the current process PID to the PID file."""
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _read_pid_file() -> int | None:
    """Read the PID from the PID file. Returns None if not found."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, "r") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def _remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


async def _run_agent_loop(config: AgentConfig) -> None:
    """Main agent event loop — creates and starts the TradingAgent.

    Runs the startup sequence as async, then enters a blocking loop
    that processes both IB events and asyncio tasks.
    """
    from ib_insync import util as ib_util
    from src.agent import TradingAgent

    agent = TradingAgent(config)
    logger.info("Agent started — PID %d, environment=%s", os.getpid(), config.environment)

    try:
        await agent.initialize()
        logger.info("Agent entering main loop")
        ib = agent._connection_manager.ib
        mdt = config.market_data_type

        if mdt == "yahoo":
            # Use Yahoo Finance for market data (free, no IB subscription needed)
            from src.services.yahoo_data_provider import YahooDataProvider
            yahoo = YahooDataProvider(agent._market_data._watchlist)
            yahoo.set_tick_callback(agent._on_tick)
            logger.info("Loading historical data from Yahoo Finance...")
            loaded = yahoo.load_history()
            logger.info("Yahoo: loaded history for %d symbols", loaded)

            while ib.isConnected():
                yahoo.poll()
                ib.sleep(10)  # poll every 10 seconds
        else:
            # IB market data mode
            mdt_int = int(mdt) if mdt.isdigit() else 4
            while ib.isConnected():
                if mdt_int in (3, 4):
                    agent._market_data.poll_snapshots()
                ib.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Agent interrupted")
    finally:
        await agent.stop()
        _remove_pid_file()


def cmd_start(_args: argparse.Namespace) -> None:
    """Start the agent as a background process."""
    setup_logging()

    existing_pid = _read_pid_file()
    if existing_pid is not None:
        # Check if the process is actually running
        try:
            os.kill(existing_pid, 0)
            print(f"Agent is already running (PID {existing_pid})")
            sys.exit(1)
        except OSError:
            # Stale PID file — process is gone
            _remove_pid_file()

    config = AgentConfig.from_env()
    _write_pid_file()
    logger.info("Starting IB Trading Agent")

    try:
        asyncio.run(_run_agent_loop(config))
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user (KeyboardInterrupt)")
    finally:
        _remove_pid_file()

    sys.exit(0)


def cmd_stop(_args: argparse.Namespace) -> None:
    """Send SIGTERM to the running agent process for graceful shutdown."""
    setup_logging()

    pid = _read_pid_file()
    if pid is None:
        print("Agent is not running")
        sys.exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to agent process (PID %d)", pid)
        print(f"Stop signal sent to agent (PID {pid})")
    except ProcessLookupError:
        print("Agent is not running (stale PID file)")
        _remove_pid_file()
        sys.exit(1)
    except PermissionError:
        print(f"Permission denied sending signal to PID {pid}")
        sys.exit(1)


def cmd_status(_args: argparse.Namespace) -> None:
    """Display portfolio value, cash balance, open positions, and agent state."""
    setup_logging()
    config = AgentConfig.from_env()

    async def _show_status() -> None:
        state_manager = StateManager(config)
        await state_manager.initialize()

        try:
            # Load latest agent state
            agent_state = await state_manager.load_last_state()
            # Load latest portfolio snapshot
            snapshot = await state_manager.get_latest_portfolio_snapshot()

            # Determine operational state
            pid = _read_pid_file()
            if pid is not None:
                try:
                    os.kill(pid, 0)
                    running = True
                except OSError:
                    running = False
            else:
                running = False

            if agent_state and agent_state.state == "HALTED":
                state_label = "HALTED (hard stop active)"
            elif running:
                state_label = "RUNNING"
            else:
                state_label = "STOPPED"

            print("=" * 50)
            print("  IB Trading Agent — Status")
            print("=" * 50)
            print(f"  Agent State:    {state_label}")

            if snapshot:
                pnl_sign = "+" if snapshot.daily_pnl >= 0 else ""
                total_pnl_sign = "+" if snapshot.total_pnl >= 0 else ""
                print(f"  Portfolio Value: ${snapshot.total_value:,.2f}")
                print(f"  Cash Balance:    ${snapshot.cash_balance:,.2f}")
                print(f"  Positions Value: ${snapshot.positions_value:,.2f}")
                print(f"  Daily P&L:       {pnl_sign}${snapshot.daily_pnl:,.2f}")
                print(f"  Total P&L:       {total_pnl_sign}${snapshot.total_pnl:,.2f} ({snapshot.total_pnl_pct:+.2f}%)")
                print(f"  Open Positions:  {snapshot.num_open_positions}")
                print(f"  Hard Stop:       {'ACTIVE' if snapshot.hard_stop_active else 'Inactive'}")
                print(f"  Snapshot Time:   {snapshot.snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("  No portfolio data available.")

            if agent_state:
                print(f"  Last Heartbeat:  {agent_state.last_heartbeat.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Crash Count:     {agent_state.crash_count}")

            print("=" * 50)
        finally:
            await state_manager.close()

    asyncio.run(_show_status())


def cmd_report(_args: argparse.Namespace) -> None:
    """Generate and send the daily report immediately."""
    setup_logging()
    config = AgentConfig.from_env()

    async def _generate_and_send() -> None:
        from datetime import datetime

        from src.services.report_generator import ReportGenerator

        state_manager = StateManager(config)
        await state_manager.initialize()

        try:
            report_gen = ReportGenerator(config, state_manager)

            snapshot = await state_manager.get_latest_portfolio_snapshot()
            if snapshot is None:
                print("No portfolio data available — cannot generate report.")
                return

            trades = await state_manager.get_trades_for_date(datetime.now())

            # Build open positions list from open trades
            open_positions: list[dict] = []

            html = await report_gen.generate_report(
                portfolio=snapshot,
                trades=trades,
                open_positions=open_positions,
                polymarket_sentiment=0.0,
            )

            await report_gen.send_report(html)
            logger.info("Report generated and sent successfully")
            print("Report generated and sent.")
        finally:
            await state_manager.close()

    asyncio.run(_generate_and_send())


def cmd_test(args: argparse.Namespace) -> None:
    """Run the test suite using pytest with optional category filters."""
    pytest_args = [sys.executable, "-m", "pytest", "-v"]

    if getattr(args, "unit", False):
        print("Running unit tests...")
        pytest_args.extend(["tests/unit/", "test/test_indicators.py"])
    elif getattr(args, "integ", False):
        print("Running integration tests...")
        pytest_args.extend(["tests/integration/", "test/test_agent.py"])
    elif getattr(args, "perf", False):
        print("Running performance tests...")
        pytest_args.extend(["tests/performance/", "test/test_performance.py"])
    else:
        print("Running all tests...")

    result = subprocess.run(pytest_args, cwd=os.getcwd())
    sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="IB Trading Agent — Autonomous trading agent for Interactive Brokers",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("start", help="Start the agent as a background process")
    subparsers.add_parser("stop", help="Graceful shutdown of the running agent")
    subparsers.add_parser("status", help="Display portfolio and agent state")
    subparsers.add_parser("report", help="Generate and send daily report immediately")

    test_parser = subparsers.add_parser("test", help="Run the test suite")
    test_parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    test_parser.add_argument("--integ", action="store_true", help="Run only integration tests")
    test_parser.add_argument("--perf", action="store_true", help="Run only performance tests")

    return parser


def main() -> None:
    """Main entry point — parse args and dispatch to the appropriate command."""
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "report": cmd_report,
        "test": cmd_test,
    }

    if args.command is None or args.command not in commands:
        parser.print_help()
        sys.exit(2)

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
