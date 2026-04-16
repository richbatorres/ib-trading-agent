"""Unit tests for StateManager."""

import os
import tempfile
from datetime import datetime

import pytest
import pytest_asyncio

from src.config import AgentConfig
from src.models.domain import (
    AgentState,
    AnalysisState,
    PortfolioSnapshot,
    TradeRecord,
    TradeSignal,
)
from src.services.state_manager import StateManager


def _make_config(db_path: str) -> AgentConfig:
    """Create a minimal AgentConfig pointing at a temp database."""
    return AgentConfig(
        ib_account_id="DU12345",
        ib_host="127.0.0.1",
        ib_port=7497,
        environment="paper",
        email_address="test@example.com",
        email_smtp_host="smtp.example.com",
        email_smtp_port=587,
        email_smtp_user="user",
        email_smtp_password="pass",
        db_url=f"sqlite:///{db_path}",
    )


@pytest_asyncio.fixture
async def state_manager(tmp_path):
    """Provide an initialized StateManager with a temp database."""
    db_path = str(tmp_path / "test_agent.db")
    config = _make_config(db_path)
    sm = StateManager(config)
    await sm.initialize()
    yield sm
    await sm.close()


@pytest.mark.asyncio
async def test_initialize_creates_tables(state_manager):
    """All 6 tables should exist after initialize()."""
    db = state_manager._db
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in await cursor.fetchall()]
    expected = [
        "agent_state",
        "analysis_state",
        "earnings_calendar",
        "llm_calls",
        "portfolio_snapshots",
        "sqlite_sequence",
        "trades",
    ]
    assert tables == expected


@pytest.mark.asyncio
async def test_wal_mode_enabled(state_manager):
    """WAL journal mode should be active."""
    db = state_manager._db
    cursor = await db.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row[0] == "wal"


@pytest.mark.asyncio
async def test_persist_and_load_trade(state_manager):
    """Round-trip: persist a trade then verify it exists in the DB."""
    trade = TradeRecord(
        symbol="AAPL",
        direction="BUY",
        entry_price=150.0,
        quantity=10,
        stop_loss_price=142.50,
        strategy="momentum",
        signal_confidence=0.85,
        polymarket_sentiment=0.3,
        entry_time=datetime(2024, 1, 15, 10, 30, 0),
    )
    await state_manager.persist_trade(trade)

    cursor = await state_manager._db.execute("SELECT symbol, direction, entry_price, quantity, status FROM trades")
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "AAPL"
    assert row[1] == "BUY"
    assert row[2] == 150.0
    assert row[3] == 10
    assert row[4] == "OPEN"


@pytest.mark.asyncio
async def test_persist_trade_with_exit(state_manager):
    """Persist a closed trade with exit details."""
    trade = TradeRecord(
        symbol="MSFT",
        direction="SELL",
        entry_price=300.0,
        quantity=5,
        stop_loss_price=285.0,
        strategy="mean_reversion",
        signal_confidence=0.72,
        polymarket_sentiment=-0.1,
        entry_time=datetime(2024, 1, 15, 10, 0, 0),
        exit_price=310.0,
        exit_time=datetime(2024, 1, 15, 14, 0, 0),
        realized_pnl=-50.0,
        status="CLOSED",
    )
    await state_manager.persist_trade(trade)

    cursor = await state_manager._db.execute("SELECT exit_price, exit_time, realized_pnl, status FROM trades")
    row = await cursor.fetchone()
    assert row[0] == 310.0
    assert row[1] is not None
    assert row[2] == -50.0
    assert row[3] == "CLOSED"


@pytest.mark.asyncio
async def test_persist_portfolio_snapshot(state_manager):
    """Round-trip: persist a portfolio snapshot."""
    snapshot = PortfolioSnapshot(
        total_value=100000.0,
        cash_balance=15000.0,
        positions_value=85000.0,
        daily_pnl=500.0,
        total_pnl=2000.0,
        total_pnl_pct=2.0,
        num_open_positions=3,
        hard_stop_active=False,
        snapshot_time=datetime(2024, 1, 15, 12, 0, 0),
    )
    await state_manager.persist_portfolio_snapshot(snapshot)

    cursor = await state_manager._db.execute(
        "SELECT total_value, cash_balance, hard_stop_active, num_open_positions FROM portfolio_snapshots"
    )
    row = await cursor.fetchone()
    assert row[0] == 100000.0
    assert row[1] == 15000.0
    assert row[2] == 0  # False -> 0
    assert row[3] == 3


@pytest.mark.asyncio
async def test_persist_analysis_state(state_manager):
    """Round-trip: persist analysis state with JSON serialization."""
    signal = TradeSignal(
        symbol="TSLA",
        direction="BUY",
        strategy="momentum",
        confidence=0.9,
        price=250.0,
        volume=1500000.0,
        indicators={"rsi": 35.0, "macd_hist": 0.5},
        polymarket_sentiment=0.2,
        timestamp=datetime(2024, 1, 15, 11, 0, 0),
    )
    analysis = AnalysisState(
        watchlist=["AAPL", "MSFT", "TSLA"],
        active_signals=[signal],
        indicator_values={"AAPL": {"rsi": 45.0}, "MSFT": {"rsi": 55.0}},
        polymarket_sentiment=0.2,
        polymarket_last_fetch=datetime(2024, 1, 15, 10, 45, 0),
    )
    await state_manager.persist_analysis_state(analysis)

    cursor = await state_manager._db.execute(
        "SELECT watchlist_json, active_signals_json, indicator_values_json, polymarket_sentiment FROM analysis_state"
    )
    row = await cursor.fetchone()
    import json
    watchlist = json.loads(row[0])
    signals = json.loads(row[1])
    indicators = json.loads(row[2])

    assert watchlist == ["AAPL", "MSFT", "TSLA"]
    assert len(signals) == 1
    assert signals[0]["symbol"] == "TSLA"
    assert indicators["AAPL"]["rsi"] == 45.0
    assert row[3] == 0.2


@pytest.mark.asyncio
async def test_load_last_state_empty_db(state_manager):
    """load_last_state() returns None on an empty database."""
    result = await state_manager.load_last_state()
    assert result is None


@pytest.mark.asyncio
async def test_load_last_state_returns_most_recent(state_manager):
    """load_last_state() returns the most recently inserted agent state."""
    db = state_manager._db

    # Insert two states
    await db.execute(
        "INSERT INTO agent_state (state, initial_portfolio_value, start_time, last_heartbeat, crash_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("RUNNING", 100000.0, "2024-01-15T08:00:00", "2024-01-15T08:00:00", 0),
    )
    await db.execute(
        "INSERT INTO agent_state (state, initial_portfolio_value, start_time, last_heartbeat, crash_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("HALTED", 80000.0, "2024-01-15T08:00:00", "2024-01-15T14:00:00", 2),
    )
    await db.commit()

    result = await state_manager.load_last_state()
    assert result is not None
    assert result.state == "HALTED"
    assert result.initial_portfolio_value == 80000.0
    assert result.crash_count == 2


@pytest.mark.asyncio
async def test_reconcile_with_ib_none_returns_empty(state_manager):
    """reconcile_with_ib() returns empty list when ib is None."""
    discrepancies = await state_manager.reconcile_with_ib(None)
    assert discrepancies == []


@pytest.mark.asyncio
async def test_reconcile_with_ib_not_connected_returns_empty(state_manager):
    """reconcile_with_ib() returns empty list when IB is not connected."""

    class MockIB:
        def isConnected(self):
            return False

    discrepancies = await state_manager.reconcile_with_ib(MockIB())
    assert discrepancies == []


class _MockContract:
    def __init__(self, symbol: str):
        self.symbol = symbol


class _MockPosition:
    def __init__(self, symbol: str, position: float, avg_cost: float):
        self.contract = _MockContract(symbol)
        self.position = position
        self.avgCost = avg_cost


class _MockAccountValue:
    def __init__(self, tag: str, value: str, currency: str):
        self.tag = tag
        self.value = value
        self.currency = currency


class _MockIBConnected:
    """Mock IB object that reports as connected with configurable positions."""

    def __init__(self, positions=None, account_values=None):
        self._positions = positions or []
        self._account_values = account_values or []

    def isConnected(self):
        return True

    def positions(self):
        return self._positions

    def accountValues(self):
        return self._account_values


@pytest.mark.asyncio
async def test_reconcile_no_discrepancies(state_manager):
    """reconcile_with_ib() returns empty list when local matches IB."""
    # Insert a local open trade
    trade = TradeRecord(
        symbol="AAPL", direction="BUY", entry_price=150.0, quantity=10,
        stop_loss_price=142.50, strategy="momentum", signal_confidence=0.8,
        polymarket_sentiment=0.0, entry_time=datetime(2024, 1, 15, 10, 0, 0),
    )
    await state_manager.persist_trade(trade)

    # IB has the same position
    mock_ib = _MockIBConnected(
        positions=[_MockPosition("AAPL", 10.0, 150.0)],
        account_values=[],
    )
    discrepancies = await state_manager.reconcile_with_ib(mock_ib)
    assert discrepancies == []


@pytest.mark.asyncio
async def test_reconcile_quantity_discrepancy(state_manager):
    """reconcile_with_ib() detects quantity mismatch and returns discrepancy."""
    trade = TradeRecord(
        symbol="AAPL", direction="BUY", entry_price=150.0, quantity=10,
        stop_loss_price=142.50, strategy="momentum", signal_confidence=0.8,
        polymarket_sentiment=0.0, entry_time=datetime(2024, 1, 15, 10, 0, 0),
    )
    await state_manager.persist_trade(trade)

    # IB has different quantity
    mock_ib = _MockIBConnected(
        positions=[_MockPosition("AAPL", 15.0, 150.0)],
        account_values=[],
    )
    discrepancies = await state_manager.reconcile_with_ib(mock_ib)
    qty_discs = [d for d in discrepancies if d.field == "AAPL.quantity"]
    assert len(qty_discs) == 1
    assert qty_discs[0].local_value == 10
    assert qty_discs[0].ib_value == 15
    assert qty_discs[0].resolution == "updated_to_ib"


@pytest.mark.asyncio
async def test_reconcile_avg_cost_discrepancy(state_manager):
    """reconcile_with_ib() detects avg cost mismatch."""
    trade = TradeRecord(
        symbol="MSFT", direction="BUY", entry_price=300.0, quantity=5,
        stop_loss_price=285.0, strategy="momentum", signal_confidence=0.7,
        polymarket_sentiment=0.0, entry_time=datetime(2024, 1, 15, 10, 0, 0),
    )
    await state_manager.persist_trade(trade)

    mock_ib = _MockIBConnected(
        positions=[_MockPosition("MSFT", 5.0, 305.50)],
        account_values=[],
    )
    discrepancies = await state_manager.reconcile_with_ib(mock_ib)
    cost_discs = [d for d in discrepancies if d.field == "MSFT.avg_cost"]
    assert len(cost_discs) == 1
    assert cost_discs[0].local_value == 300.0
    assert cost_discs[0].ib_value == 305.50


@pytest.mark.asyncio
async def test_reconcile_position_closed_on_ib(state_manager):
    """reconcile_with_ib() detects position that exists locally but not on IB."""
    trade = TradeRecord(
        symbol="TSLA", direction="BUY", entry_price=250.0, quantity=8,
        stop_loss_price=237.50, strategy="momentum", signal_confidence=0.9,
        polymarket_sentiment=0.1, entry_time=datetime(2024, 1, 15, 10, 0, 0),
    )
    await state_manager.persist_trade(trade)

    # IB has no positions
    mock_ib = _MockIBConnected(positions=[], account_values=[])
    discrepancies = await state_manager.reconcile_with_ib(mock_ib)

    # Should detect quantity discrepancy (local=8, ib=0)
    qty_discs = [d for d in discrepancies if "quantity" in d.field]
    assert len(qty_discs) == 1
    assert qty_discs[0].ib_value == 0

    # Local trade should be marked CLOSED
    cursor = await state_manager._db.execute(
        "SELECT status FROM trades WHERE symbol = 'TSLA'"
    )
    row = await cursor.fetchone()
    assert row[0] == "CLOSED"


@pytest.mark.asyncio
async def test_reconcile_logs_offline_duration(state_manager, caplog):
    """reconcile_with_ib() logs offline duration when previous state exists."""
    import logging

    # Insert a previous agent state
    await state_manager._db.execute(
        "INSERT INTO agent_state (state, initial_portfolio_value, start_time, last_heartbeat, crash_count) "
        "VALUES (?, ?, ?, ?, ?)",
        ("RUNNING", 100000.0, "2024-01-15T08:00:00", "2024-01-15T14:00:00", 0),
    )
    await state_manager._db.commit()

    mock_ib = _MockIBConnected(positions=[], account_values=[])
    with caplog.at_level(logging.INFO):
        await state_manager.reconcile_with_ib(mock_ib)

    assert any("offline duration" in msg.lower() for msg in caplog.messages)


# --- persist_agent_state tests ---


@pytest.mark.asyncio
async def test_persist_agent_state(state_manager):
    """persist_agent_state() inserts a record and load_last_state() retrieves it."""
    agent_state = AgentState(
        state="RUNNING",
        initial_portfolio_value=100000.0,
        start_time=datetime(2024, 1, 15, 8, 0, 0),
        last_heartbeat=datetime(2024, 1, 15, 12, 0, 0),
        crash_count=0,
    )
    await state_manager.persist_agent_state(agent_state)

    loaded = await state_manager.load_last_state()
    assert loaded is not None
    assert loaded.state == "RUNNING"
    assert loaded.initial_portfolio_value == 100000.0
    assert loaded.crash_count == 0


@pytest.mark.asyncio
async def test_persist_agent_state_multiple(state_manager):
    """persist_agent_state() with multiple inserts — load_last_state returns most recent."""
    state1 = AgentState(
        state="RUNNING", initial_portfolio_value=100000.0,
        start_time=datetime(2024, 1, 15, 8, 0, 0),
        last_heartbeat=datetime(2024, 1, 15, 10, 0, 0), crash_count=0,
    )
    state2 = AgentState(
        state="HALTED", initial_portfolio_value=80000.0,
        start_time=datetime(2024, 1, 15, 8, 0, 0),
        last_heartbeat=datetime(2024, 1, 15, 14, 0, 0), crash_count=2,
    )
    await state_manager.persist_agent_state(state1)
    await state_manager.persist_agent_state(state2)

    loaded = await state_manager.load_last_state()
    assert loaded.state == "HALTED"
    assert loaded.crash_count == 2


# --- get_latest_portfolio_snapshot tests ---


@pytest.mark.asyncio
async def test_get_latest_portfolio_snapshot_empty(state_manager):
    """get_latest_portfolio_snapshot() returns None on empty table."""
    result = await state_manager.get_latest_portfolio_snapshot()
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_portfolio_snapshot(state_manager):
    """get_latest_portfolio_snapshot() returns the most recent snapshot."""
    snap1 = PortfolioSnapshot(
        total_value=100000.0, cash_balance=15000.0, positions_value=85000.0,
        daily_pnl=500.0, total_pnl=2000.0, total_pnl_pct=2.0,
        num_open_positions=3, hard_stop_active=False,
        snapshot_time=datetime(2024, 1, 15, 10, 0, 0),
    )
    snap2 = PortfolioSnapshot(
        total_value=102000.0, cash_balance=12000.0, positions_value=90000.0,
        daily_pnl=700.0, total_pnl=4000.0, total_pnl_pct=4.0,
        num_open_positions=4, hard_stop_active=False,
        snapshot_time=datetime(2024, 1, 15, 14, 0, 0),
    )
    await state_manager.persist_portfolio_snapshot(snap1)
    await state_manager.persist_portfolio_snapshot(snap2)

    result = await state_manager.get_latest_portfolio_snapshot()
    assert result is not None
    assert result.total_value == 102000.0
    assert result.num_open_positions == 4
    assert result.hard_stop_active is False


# --- get_trades_for_date tests ---


@pytest.mark.asyncio
async def test_get_trades_for_date_empty(state_manager):
    """get_trades_for_date() returns empty list when no trades exist."""
    result = await state_manager.get_trades_for_date(datetime(2024, 1, 15))
    assert result == []


@pytest.mark.asyncio
async def test_get_trades_for_date(state_manager):
    """get_trades_for_date() returns only trades for the specified date."""
    trade1 = TradeRecord(
        symbol="AAPL", direction="BUY", entry_price=150.0, quantity=10,
        stop_loss_price=142.50, strategy="momentum", signal_confidence=0.8,
        polymarket_sentiment=0.0, entry_time=datetime(2024, 1, 15, 10, 0, 0),
    )
    trade2 = TradeRecord(
        symbol="MSFT", direction="BUY", entry_price=300.0, quantity=5,
        stop_loss_price=285.0, strategy="mean_reversion", signal_confidence=0.7,
        polymarket_sentiment=0.1, entry_time=datetime(2024, 1, 16, 11, 0, 0),
    )
    await state_manager.persist_trade(trade1)
    await state_manager.persist_trade(trade2)

    result = await state_manager.get_trades_for_date(datetime(2024, 1, 15))
    assert len(result) == 1
    assert result[0].symbol == "AAPL"

    result2 = await state_manager.get_trades_for_date(datetime(2024, 1, 16))
    assert len(result2) == 1
    assert result2[0].symbol == "MSFT"


# --- persist_llm_call / get_llm_calls_today tests ---


@pytest.mark.asyncio
async def test_persist_llm_call(state_manager):
    """persist_llm_call() inserts a record into llm_calls table."""
    await state_manager.persist_llm_call(
        purpose="sentiment_interpretation",
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=200,
        total_tokens=700,
        success=True,
    )

    cursor = await state_manager._db.execute(
        "SELECT purpose, model, input_tokens, output_tokens, total_tokens, success "
        "FROM llm_calls"
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "sentiment_interpretation"
    assert row[1] == "claude-sonnet-4-6"
    assert row[2] == 500
    assert row[3] == 200
    assert row[4] == 700
    assert row[5] == 1  # True -> 1


@pytest.mark.asyncio
async def test_persist_llm_call_with_error(state_manager):
    """persist_llm_call() stores error message on failure."""
    await state_manager.persist_llm_call(
        purpose="report_generation",
        model="claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        success=False,
        error_message="API rate limit exceeded",
    )

    cursor = await state_manager._db.execute(
        "SELECT success, error_message FROM llm_calls"
    )
    row = await cursor.fetchone()
    assert row[0] == 0  # False -> 0
    assert row[1] == "API rate limit exceeded"


@pytest.mark.asyncio
async def test_get_llm_calls_today(state_manager):
    """get_llm_calls_today() counts only today's calls."""
    # Insert a call for today
    await state_manager.persist_llm_call(
        purpose="test1", model="claude-sonnet-4-6",
        input_tokens=100, output_tokens=50, total_tokens=150, success=True,
    )
    await state_manager.persist_llm_call(
        purpose="test2", model="claude-sonnet-4-6",
        input_tokens=200, output_tokens=100, total_tokens=300, success=True,
    )

    count = await state_manager.get_llm_calls_today()
    assert count == 2


@pytest.mark.asyncio
async def test_get_llm_calls_today_excludes_other_dates(state_manager):
    """get_llm_calls_today() does not count calls from other dates."""
    # Insert a call with a different date directly
    await state_manager._db.execute(
        "INSERT INTO llm_calls (purpose, model, input_tokens, output_tokens, "
        "total_tokens, success, call_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("old_call", "claude-sonnet-4-6", 100, 50, 150, 1, "2024-01-01"),
    )
    await state_manager._db.commit()

    count = await state_manager.get_llm_calls_today()
    assert count == 0


@pytest.mark.asyncio
async def test_persist_trade_raises_if_not_initialized(tmp_path):
    """Calling persist methods before initialize() raises RuntimeError."""
    db_path = str(tmp_path / "uninit.db")
    config = _make_config(db_path)
    sm = StateManager(config)

    trade = TradeRecord(
        symbol="AAPL", direction="BUY", entry_price=150.0, quantity=10,
        stop_loss_price=142.50, strategy="momentum", signal_confidence=0.8,
        polymarket_sentiment=0.0, entry_time=datetime.now(),
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        await sm.persist_trade(trade)
