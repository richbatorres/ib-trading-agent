"""Unit tests for the CLI interface (agent.py).

Tests each command's behavior and argument parsing.
Requirements: 20.1, 20.2, 20.3, 20.4, 20.5
"""

import os
import signal
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import (
    PID_FILE,
    _read_pid_file,
    _remove_pid_file,
    _write_pid_file,
    build_parser,
    cmd_report,
    cmd_start,
    cmd_status,
    cmd_stop,
    cmd_test,
    main,
)


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------

class TestPidFileHelpers:
    """Tests for PID file read/write/remove helpers."""

    def test_write_and_read_pid_file(self, tmp_path, monkeypatch):
        pid_file = str(tmp_path / "agent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)

        _write_pid_file()
        assert _read_pid_file() == os.getpid()

    def test_read_pid_file_missing(self, tmp_path, monkeypatch):
        pid_file = str(tmp_path / "nonexistent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)

        assert _read_pid_file() is None

    def test_remove_pid_file(self, tmp_path, monkeypatch):
        pid_file = str(tmp_path / "agent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)

        _write_pid_file()
        assert os.path.exists(pid_file)

        _remove_pid_file()
        assert not os.path.exists(pid_file)

    def test_remove_pid_file_missing(self, tmp_path, monkeypatch):
        """Removing a non-existent PID file should not raise."""
        pid_file = str(tmp_path / "nonexistent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)
        _remove_pid_file()  # Should not raise


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    """Tests for argparse configuration."""

    def test_valid_commands(self):
        parser = build_parser()
        for cmd in ("start", "stop", "status", "report", "test"):
            args = parser.parse_args([cmd])
            assert args.command == cmd

    def test_no_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


# ---------------------------------------------------------------------------
# Unrecognized command → usage + non-zero exit (Requirement 20.5)
# ---------------------------------------------------------------------------

class TestUnrecognizedCommand:
    """Unrecognized commands should print usage and exit non-zero."""

    def test_no_args_exits_nonzero(self):
        """Running with no arguments should exit with code 2."""
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["agent.py"]):
                main()
        assert exc_info.value.code != 0

    def test_unknown_command_exits_nonzero(self):
        """Running with an unknown command should exit non-zero."""
        # argparse itself will exit(2) for unrecognized subcommands
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["agent.py", "bogus"]):
                main()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# cmd_stop (Requirement 20.2)
# ---------------------------------------------------------------------------

class TestCmdStop:
    """Tests for the stop command."""

    def test_stop_no_pid_file(self, tmp_path, monkeypatch, capsys):
        """If no PID file exists, print 'Agent is not running' and exit 1."""
        pid_file = str(tmp_path / "agent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)
        monkeypatch.setattr("agent.setup_logging", lambda: None)

        args = build_parser().parse_args(["stop"])
        with pytest.raises(SystemExit) as exc_info:
            cmd_stop(args)
        assert exc_info.value.code == 1
        assert "not running" in capsys.readouterr().out

    def test_stop_stale_pid(self, tmp_path, monkeypatch, capsys):
        """If PID file points to a dead process, report stale and exit 1."""
        pid_file = str(tmp_path / "agent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)
        monkeypatch.setattr("agent.setup_logging", lambda: None)

        # Write a PID that doesn't exist
        with open(pid_file, "w") as f:
            f.write("999999999")

        args = build_parser().parse_args(["stop"])
        with pytest.raises(SystemExit) as exc_info:
            cmd_stop(args)
        assert exc_info.value.code == 1
        assert "not running" in capsys.readouterr().out.lower()

    def test_stop_sends_sigterm(self, tmp_path, monkeypatch):
        """If a valid PID exists, send SIGTERM."""
        pid_file = str(tmp_path / "agent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)
        monkeypatch.setattr("agent.setup_logging", lambda: None)

        # Write our own PID so os.kill(pid, SIGTERM) targets us
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        # Mock os.kill to verify it's called with SIGTERM
        with patch("os.kill") as mock_kill:
            args = build_parser().parse_args(["stop"])
            cmd_stop(args)
            mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
# cmd_status (Requirement 20.3)
# ---------------------------------------------------------------------------

class TestCmdStatus:
    """Tests for the status command."""

    def test_status_no_data(self, tmp_path, monkeypatch, capsys):
        """Status with no portfolio data should display 'No portfolio data'."""
        monkeypatch.setattr("agent.setup_logging", lambda: None)

        db_path = str(tmp_path / "test.db")
        mock_config = MagicMock()
        mock_config.db_url = f"sqlite:///{db_path}"

        monkeypatch.setattr("agent.AgentConfig.from_env", lambda: mock_config)
        # No PID file
        monkeypatch.setattr("agent.PID_FILE", str(tmp_path / "agent.pid"))

        args = build_parser().parse_args(["status"])
        cmd_status(args)

        output = capsys.readouterr().out
        assert "No portfolio data" in output
        assert "STOPPED" in output

    def test_status_with_snapshot(self, tmp_path, monkeypatch, capsys):
        """Status with portfolio data should display values."""
        from datetime import datetime

        from src.models.domain import AgentState, PortfolioSnapshot

        monkeypatch.setattr("agent.setup_logging", lambda: None)

        db_path = str(tmp_path / "test.db")
        mock_config = MagicMock()
        mock_config.db_url = f"sqlite:///{db_path}"

        monkeypatch.setattr("agent.AgentConfig.from_env", lambda: mock_config)
        monkeypatch.setattr("agent.PID_FILE", str(tmp_path / "agent.pid"))

        snapshot = PortfolioSnapshot(
            total_value=100000.0,
            cash_balance=15000.0,
            positions_value=85000.0,
            daily_pnl=1250.50,
            total_pnl=5000.0,
            total_pnl_pct=5.0,
            num_open_positions=3,
            hard_stop_active=False,
            snapshot_time=datetime(2025, 1, 15, 16, 0, 0),
        )

        agent_state = AgentState(
            state="STOPPED",
            initial_portfolio_value=95000.0,
            start_time=datetime(2025, 1, 15, 9, 30, 0),
            last_heartbeat=datetime(2025, 1, 15, 16, 0, 0),
            crash_count=0,
        )

        mock_sm = AsyncMock()
        mock_sm.initialize = AsyncMock()
        mock_sm.load_last_state = AsyncMock(return_value=agent_state)
        mock_sm.get_latest_portfolio_snapshot = AsyncMock(return_value=snapshot)
        mock_sm.close = AsyncMock()

        with patch("agent.StateManager", return_value=mock_sm):
            args = build_parser().parse_args(["status"])
            cmd_status(args)

        output = capsys.readouterr().out
        assert "$100,000.00" in output
        assert "$15,000.00" in output
        assert "3" in output
        assert "STOPPED" in output


# ---------------------------------------------------------------------------
# cmd_test (Requirement 25.1 placeholder)
# ---------------------------------------------------------------------------

class TestCmdTest:
    """Tests for the test command."""

    def test_test_runs_pytest(self, monkeypatch):
        """The test command should invoke pytest."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                args = build_parser().parse_args(["test"])
                cmd_test(args)
            assert exc_info.value.code == 0
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "-m" in call_args[0][0]
            assert "pytest" in call_args[0][0]


# ---------------------------------------------------------------------------
# cmd_start (Requirement 20.1)
# ---------------------------------------------------------------------------

class TestCmdStart:
    """Tests for the start command."""

    def test_start_already_running(self, tmp_path, monkeypatch, capsys):
        """If agent is already running, print message and exit 1."""
        pid_file = str(tmp_path / "agent.pid")
        monkeypatch.setattr("agent.PID_FILE", pid_file)
        monkeypatch.setattr("agent.setup_logging", lambda: None)

        # Write our own PID (which is running)
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        mock_config = MagicMock()
        monkeypatch.setattr("agent.AgentConfig.from_env", lambda: mock_config)

        args = build_parser().parse_args(["start"])
        with pytest.raises(SystemExit) as exc_info:
            cmd_start(args)
        assert exc_info.value.code == 1
        assert "already running" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# cmd_report (Requirement 20.4)
# ---------------------------------------------------------------------------

class TestCmdReport:
    """Tests for the report command."""

    def test_report_no_data(self, tmp_path, monkeypatch, capsys):
        """Report with no portfolio data should print a message."""
        monkeypatch.setattr("agent.setup_logging", lambda: None)

        db_path = str(tmp_path / "test.db")
        mock_config = MagicMock()
        mock_config.db_url = f"sqlite:///{db_path}"

        monkeypatch.setattr("agent.AgentConfig.from_env", lambda: mock_config)

        mock_sm = AsyncMock()
        mock_sm.initialize = AsyncMock()
        mock_sm.get_latest_portfolio_snapshot = AsyncMock(return_value=None)
        mock_sm.close = AsyncMock()

        with patch("agent.StateManager", return_value=mock_sm):
            args = build_parser().parse_args(["report"])
            cmd_report(args)

        output = capsys.readouterr().out
        assert "No portfolio data" in output
