"""Unit tests for the TradingAgent class (src/agent.py).

Tests component initialization, startup sequence, tick processing,
scheduled task gating, and IB event wiring.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config import AgentConfig

# All patches needed to construct TradingAgent without real dependencies
_AGENT_PATCHES = [
    "src.agent.ConnectionManager",
    "src.agent.StateManager",
    "src.agent.MarketHoursService",
    "src.agent.PolymarketService",
    "src.agent.StrategyEngine",
    "src.agent.RiskManager",
    "src.agent.OrderExecutor",
    "src.agent.MarketDataService",
    "src.agent.ReportGenerator",
    "src.agent.ShutdownHandler",
    "src.agent.WatchlistManager",
]


def _make_config() -> AgentConfig:
    return AgentConfig(
        ib_account_id="DU12345", ib_host="127.0.0.1", ib_port=7497,
        environment="paper", email_address="test@example.com",
        email_smtp_host="smtp.example.com", email_smtp_port=587,
        email_smtp_user="user", email_smtp_password="pass",
        db_url="sqlite:///data/test_agent.db",
    )


class _PatchAll:
    """Context manager that patches all agent dependencies and exposes them by name."""

    def __init__(self):
        self._patchers = {}
        self.mocks = {}

    def __enter__(self):
        for target in _AGENT_PATCHES:
            name = target.rsplit(".", 1)[-1]
            p = patch(target)
            self.mocks[name] = p.start()
            self._patchers[name] = p
        return self

    def __exit__(self, *args):
        for p in self._patchers.values():
            p.stop()


class TestTradingAgentInit:

    def test_all_components_initialized(self):
        from src.agent import TradingAgent
        with _PatchAll() as ctx:
            TradingAgent(_make_config())
            ctx.mocks["ConnectionManager"].assert_called_once()
            ctx.mocks["StateManager"].assert_called_once()
            ctx.mocks["StrategyEngine"].assert_called_once()
            ctx.mocks["RiskManager"].assert_called_once()
            ctx.mocks["ShutdownHandler"].assert_called_once()

    def test_shutdown_handler_receives_correct_deps(self):
        from src.agent import TradingAgent
        with _PatchAll() as ctx:
            TradingAgent(_make_config())
            args = ctx.mocks["ShutdownHandler"].call_args
            assert args is not None
            assert len(args[0]) == 3


class TestTradingAgentTickProcessing:

    @pytest.mark.asyncio
    async def test_process_tick_no_signal(self):
        from src.agent import TradingAgent
        with _PatchAll() as ctx:
            ctx.mocks["StrategyEngine"].return_value.process_tick = AsyncMock(return_value=None)
            agent = TradingAgent(_make_config())
            await agent._process_tick_async("AAPL", 102.0, 1200.0,
                                            np.array([100.0, 101.0, 102.0]),
                                            np.array([1000.0, 1100.0, 1200.0]), 1000.0)
            ctx.mocks["StrategyEngine"].return_value.process_tick.assert_awaited_once()
            ctx.mocks["RiskManager"].return_value.evaluate_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_tick_signal_rejected_by_risk(self):
        from src.agent import TradingAgent
        from src.models.domain import TradeSignal
        with _PatchAll() as ctx:
            signal = TradeSignal(symbol="AAPL", direction="BUY", strategy="momentum",
                                 confidence=0.8, price=150.0, volume=1e6,
                                 indicators={"rsi": 35.0}, polymarket_sentiment=0.1,
                                 timestamp=datetime.now())
            ctx.mocks["StrategyEngine"].return_value.process_tick = AsyncMock(return_value=signal)
            ctx.mocks["RiskManager"].return_value.evaluate_signal.return_value = None
            agent = TradingAgent(_make_config())
            await agent._process_tick_async("AAPL", 150.0, 1e6,
                                            np.array([100.0]), np.array([1e6]), 5e5)
            ctx.mocks["RiskManager"].return_value.evaluate_signal.assert_called_once_with(signal)
            ctx.mocks["OrderExecutor"].return_value.execute_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_tick_full_pipeline(self):
        from src.agent import TradingAgent
        from src.models.domain import ApprovedTrade, TradeSignal
        with _PatchAll() as ctx:
            signal = TradeSignal(symbol="AAPL", direction="BUY", strategy="momentum",
                                 confidence=0.8, price=150.0, volume=1e6,
                                 indicators={"rsi": 35.0}, polymarket_sentiment=0.1,
                                 timestamp=datetime.now())
            approved = ApprovedTrade(signal=signal, quantity=10,
                                     stop_loss_price=142.5, max_position_value=25000.0)
            ctx.mocks["StrategyEngine"].return_value.process_tick = AsyncMock(return_value=signal)
            ctx.mocks["RiskManager"].return_value.evaluate_signal.return_value = approved
            ctx.mocks["OrderExecutor"].return_value.execute_trade = AsyncMock()
            av_net = MagicMock(tag="NetLiquidation", currency="USD", value="100000.0")
            av_cash = MagicMock(tag="CashBalance", currency="USD", value="50000.0")
            ctx.mocks["ConnectionManager"].return_value.ib.accountValues.return_value = [av_net, av_cash]

            agent = TradingAgent(_make_config())
            await agent._process_tick_async("AAPL", 150.0, 1e6,
                                            np.array([100.0]), np.array([1e6]), 5e5)
            ctx.mocks["OrderExecutor"].return_value.execute_trade.assert_awaited_once_with(approved)
            ctx.mocks["RiskManager"].return_value.update_portfolio.assert_called_once_with(100000.0, 50000.0)


class TestScheduledTasks:

    @pytest.mark.asyncio
    async def test_portfolio_loss_check_skipped_outside_hours(self):
        from src.agent import TradingAgent
        with _PatchAll() as ctx:
            ctx.mocks["MarketHoursService"].return_value.is_market_open.return_value = False
            ctx.mocks["RiskManager"].return_value.check_portfolio_loss = AsyncMock()
            agent = TradingAgent(_make_config())
            await agent._check_portfolio_loss()
            ctx.mocks["RiskManager"].return_value.check_portfolio_loss.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_snapshot_skipped_outside_hours(self):
        from src.agent import TradingAgent
        with _PatchAll() as ctx:
            ctx.mocks["MarketHoursService"].return_value.is_market_open.return_value = False
            ctx.mocks["StateManager"].return_value.persist_portfolio_snapshot = AsyncMock()
            agent = TradingAgent(_make_config())
            await agent._take_portfolio_snapshot()
            ctx.mocks["StateManager"].return_value.persist_portfolio_snapshot.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_daily_report_skipped_without_snapshot(self):
        from src.agent import TradingAgent
        with _PatchAll() as ctx:
            ctx.mocks["StateManager"].return_value.get_latest_portfolio_snapshot = AsyncMock(return_value=None)
            ctx.mocks["ReportGenerator"].return_value.generate_report = AsyncMock()
            agent = TradingAgent(_make_config())
            await agent._generate_daily_report()
            ctx.mocks["ReportGenerator"].return_value.generate_report.assert_not_awaited()


class TestIBEventWiring:

    def test_wire_ib_events_connects_exec_and_order_status(self):
        from src.agent import TradingAgent
        with _PatchAll() as ctx:
            mock_ib = MagicMock()
            exec_event = MagicMock()
            order_event = MagicMock()
            mock_ib.execDetailsEvent = exec_event
            mock_ib.orderStatusEvent = order_event
            ctx.mocks["ConnectionManager"].return_value.ib = mock_ib

            agent = TradingAgent(_make_config())
            agent._wire_ib_events()

            assert exec_event.__iadd__.called
            assert order_event.__iadd__.called
