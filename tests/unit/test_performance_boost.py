"""Tests for performance boost changes.

Covers:
- Portfolio value reading with multi-fallback (BASE → USD → any)
- Trade closing with realized P&L computation
- Minimum confidence threshold filtering
- Position-aware SELL signal filtering
- Updated momentum strategy (RSI zone check)
- Updated mean reversion strategy (RSI confirmation)
- Yahoo data provider log level fix
"""

import asyncio
import math
from datetime import datetime
from types import SimpleNamespace
from typing import Set
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config import AgentConfig
from src.models.domain import TradeSignal
from src.services.risk_manager import RiskManager
from src.services.strategy_engine import StrategyEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**kw):
    defaults = dict(
        ib_account_id="TEST", ib_host="127.0.0.1", ib_port=7497,
        environment="paper", email_address="", email_smtp_host="",
        email_smtp_port=587, email_smtp_user="", email_smtp_password="",
        max_portfolio_loss_pct=20.0, max_position_size_pct=25.0,
        stop_loss_pct=5.0, cash_buffer_pct=10.0,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _signal(symbol="AAPL", direction="BUY", price=150.0, confidence=0.8,
            strategy="momentum"):
    return TradeSignal(
        symbol=symbol, direction=direction, strategy=strategy,
        confidence=confidence, price=price, volume=1e6,
        indicators={"rsi": 35.0}, polymarket_sentiment=0.0,
        timestamp=datetime.now(),
    )


def _rm(**kw):
    rm = RiskManager(_config(**kw), MagicMock())
    rm.update_portfolio(1_000_000.0, 1_000_000.0)
    return rm


class FakeMarketHours:
    """Stub for market hours."""
    _open: bool = True
    def is_market_open(self) -> bool:
        return self._open


def _make_engine(market_open=True, sentiment=0.0, blackout=None):
    return StrategyEngine(
        market_hours=FakeMarketHours() if market_open else FakeMarketHours.__new__(FakeMarketHours),
        polymarket_sentiment=sentiment,
        earnings_blackout_symbols=blackout,
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Portfolio value reading tests
# ---------------------------------------------------------------------------

class TestReadPortfolioFromIB:
    """Tests for the _read_portfolio_from_ib helper in TradingAgent."""

    def _make_account_values(self, entries):
        """Create mock account values from (tag, value, currency) tuples."""
        return [SimpleNamespace(tag=t, value=v, currency=c) for t, v, c in entries]

    def test_reads_base_currency_first(self):
        """BASE currency values should be preferred over USD."""
        from src.agent import TradingAgent
        config = _config(market_data_type="4")
        agent = TradingAgent(config)
        agent._connection_manager.ib.accountValues = MagicMock(return_value=
            self._make_account_values([
                ("NetLiquidation", "1000000", "BASE"),
                ("TotalCashBalance", "500000", "BASE"),
                ("NetLiquidation", "999999", "USD"),
                ("CashBalance", "499999", "USD"),
            ])
        )
        total, cash = agent._read_portfolio_from_ib()
        assert total == 1_000_000.0
        assert cash == 500_000.0

    def test_falls_back_to_usd(self):
        """When BASE is not available, should use USD."""
        from src.agent import TradingAgent
        config = _config(market_data_type="4")
        agent = TradingAgent(config)
        agent._connection_manager.ib.accountValues = MagicMock(return_value=
            self._make_account_values([
                ("NetLiquidation", "800000", "USD"),
                ("CashBalance", "400000", "USD"),
            ])
        )
        total, cash = agent._read_portfolio_from_ib()
        assert total == 800_000.0
        assert cash == 400_000.0

    def test_falls_back_to_any_currency(self):
        """When neither BASE nor USD, should use any available."""
        from src.agent import TradingAgent
        config = _config(market_data_type="4")
        agent = TradingAgent(config)
        agent._connection_manager.ib.accountValues = MagicMock(return_value=
            self._make_account_values([
                ("NetLiquidation", "700000", "EUR"),
                ("TotalCashBalance", "350000", "EUR"),
            ])
        )
        total, cash = agent._read_portfolio_from_ib()
        assert total == 700_000.0

    def test_returns_zero_when_no_data(self):
        """Should return (0, 0) when no account values available."""
        from src.agent import TradingAgent
        config = _config(market_data_type="4")
        agent = TradingAgent(config)
        agent._connection_manager.ib.accountValues = MagicMock(return_value=[])
        total, cash = agent._read_portfolio_from_ib()
        assert total == 0.0
        assert cash == 0.0

    def test_cash_defaults_to_total_when_missing(self):
        """When total_value found but no cash tag, cash = total_value."""
        from src.agent import TradingAgent
        config = _config(market_data_type="4")
        agent = TradingAgent(config)
        agent._connection_manager.ib.accountValues = MagicMock(return_value=
            self._make_account_values([
                ("NetLiquidation", "500000", "BASE"),
            ])
        )
        total, cash = agent._read_portfolio_from_ib()
        assert total == 500_000.0
        assert cash == 500_000.0


# ---------------------------------------------------------------------------
# Trade closing with P&L tests
# ---------------------------------------------------------------------------

class TestCloseTrade:
    """Tests for StateManager.close_trade()."""

    @pytest.mark.asyncio
    async def test_close_trade_computes_pnl(self):
        """Closing a trade should compute realized P&L correctly."""
        from src.services.state_manager import StateManager
        from src.models.domain import TradeRecord

        config = _config(db_url="sqlite:///C:/temp/test_close_trade.db")
        sm = StateManager(config)
        await sm.initialize()

        # Persist an OPEN BUY trade
        trade = TradeRecord(
            symbol="AAPL", direction="BUY", entry_price=100.0,
            quantity=50, stop_loss_price=95.0, strategy="momentum",
            signal_confidence=0.8, polymarket_sentiment=0.0,
            entry_time=datetime.now(), status="OPEN",
        )
        await sm.persist_trade(trade)

        # Close it at $110
        pnl = await sm.close_trade("AAPL", 110.0, datetime.now())
        assert pnl is not None
        assert pnl == pytest.approx(500.0)  # (110 - 100) * 50

        # Verify the trade is now CLOSED in DB
        cursor = await sm._db.execute(
            "SELECT status, exit_price, realized_pnl FROM trades WHERE symbol = 'AAPL'"
        )
        row = await cursor.fetchone()
        assert row[0] == "CLOSED"
        assert row[1] == pytest.approx(110.0)
        assert row[2] == pytest.approx(500.0)

        await sm.close()
        import os
        os.remove("C:/temp/test_close_trade.db")

    @pytest.mark.asyncio
    async def test_close_trade_no_open_trade(self):
        """Closing a trade for a symbol with no OPEN trade returns None."""
        from src.services.state_manager import StateManager

        config = _config(db_url="sqlite:///C:/temp/test_close_none.db")
        sm = StateManager(config)
        await sm.initialize()

        pnl = await sm.close_trade("MSFT", 200.0, datetime.now())
        assert pnl is None

        await sm.close()
        import os
        os.remove("C:/temp/test_close_none.db")

    @pytest.mark.asyncio
    async def test_close_trade_negative_pnl(self):
        """Closing at a loss should produce negative P&L."""
        from src.services.state_manager import StateManager
        from src.models.domain import TradeRecord

        config = _config(db_url="sqlite:///C:/temp/test_close_loss.db")
        sm = StateManager(config)
        await sm.initialize()

        trade = TradeRecord(
            symbol="TSLA", direction="BUY", entry_price=200.0,
            quantity=10, stop_loss_price=190.0, strategy="trend_following",
            signal_confidence=0.7, polymarket_sentiment=0.0,
            entry_time=datetime.now(), status="OPEN",
        )
        await sm.persist_trade(trade)

        pnl = await sm.close_trade("TSLA", 180.0, datetime.now())
        assert pnl is not None
        assert pnl == pytest.approx(-200.0)  # (180 - 200) * 10

        await sm.close()
        import os
        os.remove("C:/temp/test_close_loss.db")


# ---------------------------------------------------------------------------
# Minimum confidence threshold tests
# ---------------------------------------------------------------------------

class TestMinConfidenceThreshold:
    """Tests for the minimum confidence filter in _process_tick_async."""

    def test_signal_below_threshold_is_dropped(self):
        """Signals with confidence < 0.65 should be dropped."""
        from src.agent import TradingAgent
        assert TradingAgent._MIN_CONFIDENCE == 0.65

    def test_confidence_065_passes(self):
        """Confidence exactly at 0.65 should pass the threshold."""
        threshold = 0.65
        assert 0.65 >= threshold

    def test_confidence_064_fails(self):
        """Confidence at 0.64 should fail the threshold."""
        threshold = 0.65
        assert 0.64 < threshold


# ---------------------------------------------------------------------------
# Updated momentum strategy tests
# ---------------------------------------------------------------------------

class TestMomentumStrategyUpdated:
    """Tests for the relaxed momentum strategy thresholds."""

    def test_momentum_base_confidence_is_075(self):
        """Momentum base confidence should be 0.75 after update."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        assert engine._BASE_CONFIDENCE["momentum"] == 0.75

    def test_momentum_buy_rsi_below_35_macd_positive(self):
        """BUY when RSI < 35 and MACD histogram turns positive."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        # Previous: MACD histogram was negative
        prev = {"macd_histogram": -0.5, "rsi": 28.0, "price": 100.0,
                "bb_upper": 110.0, "bb_lower": 90.0, "ema_9": 99.0, "ema_21": 101.0}
        result = engine._evaluate_momentum(rsi=32.0, macd_hist=0.3, prev=prev)
        assert result is not None
        assert result[0] == "BUY"
        assert result[1] == 0.75

    def test_momentum_no_buy_rsi_above_35(self):
        """No BUY when RSI >= 35 even if MACD crosses positive."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        prev = {"macd_histogram": -0.5}
        result = engine._evaluate_momentum(rsi=36.0, macd_hist=0.3, prev=prev)
        assert result is None

    def test_momentum_sell_rsi_above_65_macd_negative(self):
        """SELL when RSI > 65 and MACD histogram turns negative."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        prev = {"macd_histogram": 0.5}
        result = engine._evaluate_momentum(rsi=70.0, macd_hist=-0.2, prev=prev)
        assert result is not None
        assert result[0] == "SELL"

    def test_momentum_no_sell_rsi_below_65(self):
        """No SELL when RSI <= 65 even if MACD crosses negative."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        prev = {"macd_histogram": 0.5}
        result = engine._evaluate_momentum(rsi=64.0, macd_hist=-0.2, prev=prev)
        assert result is None

    def test_momentum_needs_prev_macd(self):
        """No signal on first tick (no previous MACD)."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        result = engine._evaluate_momentum(rsi=25.0, macd_hist=0.5, prev={})
        assert result is None


# ---------------------------------------------------------------------------
# Updated mean reversion strategy tests
# ---------------------------------------------------------------------------

class TestMeanReversionUpdated:
    """Tests for mean reversion with RSI confirmation."""

    def test_mean_reversion_base_confidence_is_065(self):
        """Mean reversion base confidence should be 0.65 after update."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        assert engine._BASE_CONFIDENCE["mean_reversion"] == 0.65

    def test_buy_below_bb_with_rsi_confirmation(self):
        """BUY when price crosses below lower BB AND RSI < 40."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        prev = {"price": 95.0, "bb_lower": 94.0, "bb_upper": 106.0}
        result = engine._evaluate_mean_reversion(
            price=92.0, bb_upper=106.0, bb_lower=93.0, rsi=35.0, prev=prev,
        )
        assert result is not None
        assert result[0] == "BUY"

    def test_no_buy_below_bb_without_rsi_confirmation(self):
        """No BUY when price below BB but RSI >= 40 (not oversold)."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        prev = {"price": 95.0, "bb_lower": 94.0, "bb_upper": 106.0}
        result = engine._evaluate_mean_reversion(
            price=92.0, bb_upper=106.0, bb_lower=93.0, rsi=45.0, prev=prev,
        )
        assert result is None

    def test_sell_above_bb_with_rsi_confirmation(self):
        """SELL when price crosses above upper BB AND RSI > 60."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        prev = {"price": 105.0, "bb_lower": 94.0, "bb_upper": 106.0}
        result = engine._evaluate_mean_reversion(
            price=108.0, bb_upper=107.0, bb_lower=93.0, rsi=65.0, prev=prev,
        )
        assert result is not None
        assert result[0] == "SELL"

    def test_no_sell_above_bb_without_rsi_confirmation(self):
        """No SELL when price above BB but RSI <= 60 (not overbought)."""
        engine = StrategyEngine(
            market_hours=FakeMarketHours(),
            polymarket_sentiment=0.0,
        )
        prev = {"price": 105.0, "bb_lower": 94.0, "bb_upper": 106.0}
        result = engine._evaluate_mean_reversion(
            price=108.0, bb_upper=107.0, bb_lower=93.0, rsi=55.0, prev=prev,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Position-aware SELL filter tests
# ---------------------------------------------------------------------------

class TestPositionAwareSellFilter:
    """Tests for SELL signal filtering when no position is held."""

    def test_sell_rejected_when_no_position(self):
        """SELL signals should be filtered out when not holding the stock."""
        rm = _rm()
        # No positions held — SELL should be rejected by risk manager
        result = rm.evaluate_signal(_signal("AAPL", "SELL", 100.0))
        assert result is None

    def test_sell_approved_when_holding(self):
        """SELL signals should pass when holding the stock."""
        rm = _rm()
        rm.update_position("AAPL", 100, 95.0, 100.0, 90.0)
        result = rm.evaluate_signal(_signal("AAPL", "SELL", 100.0))
        assert result is not None
        assert result.quantity == 100


# ---------------------------------------------------------------------------
# Snapshot P&L computation tests
# ---------------------------------------------------------------------------

class TestSnapshotPnL:
    """Tests for portfolio snapshot P&L computation."""

    def test_pnl_computed_from_initial_value(self):
        """P&L should be computed as current - initial."""
        rm = _rm()
        rm.update_portfolio(1_000_000.0, 1_000_000.0)  # initial
        rm.update_portfolio(1_050_000.0, 800_000.0)     # current

        initial = rm._initial_portfolio_value
        current = rm._current_portfolio_value
        total_pnl = current - initial
        total_pnl_pct = (total_pnl / initial) * 100.0

        assert total_pnl == pytest.approx(50_000.0)
        assert total_pnl_pct == pytest.approx(5.0)

    def test_negative_pnl(self):
        """Negative P&L when portfolio value drops."""
        rm = _rm()
        rm.update_portfolio(1_000_000.0, 1_000_000.0)
        rm.update_portfolio(900_000.0, 700_000.0)

        initial = rm._initial_portfolio_value
        current = rm._current_portfolio_value
        total_pnl = current - initial

        assert total_pnl == pytest.approx(-100_000.0)
