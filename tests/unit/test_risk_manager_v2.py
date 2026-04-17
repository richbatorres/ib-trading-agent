"""Tests for RiskManager v2 safety checks.

Tests the new protections: no duplicate positions, no short selling,
total exposure limit, and trade cooldown.
"""
import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.config import AgentConfig
from src.models.domain import TradeSignal
from src.services.risk_manager import RiskManager


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


def _signal(symbol="AAPL", direction="BUY", price=150.0):
    return TradeSignal(
        symbol=symbol, direction=direction, strategy="momentum",
        confidence=0.8, price=price, volume=1e6,
        indicators={"rsi": 35.0}, polymarket_sentiment=0.0,
        timestamp=datetime.now(),
    )


def _rm(**kw):
    rm = RiskManager(_config(**kw), MagicMock())
    rm.update_portfolio(1_000_000.0, 1_000_000.0)
    return rm


class TestNoDuplicatePositions:
    """BUY should be rejected if we already hold the symbol."""

    def test_first_buy_approved(self):
        rm = _rm()
        result = rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        assert result is not None

    def test_second_buy_same_symbol_rejected(self):
        rm = _rm()
        # First buy succeeds
        rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        rm.update_position("AAPL", 100, 100.0, 100.0, 95.0)
        # Second buy rejected
        rm._last_trade_time.clear()  # clear cooldown for test
        result = rm.evaluate_signal(_signal("AAPL", "BUY", 105.0))
        assert result is None

    def test_buy_different_symbol_approved(self):
        rm = _rm()
        rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        rm.update_position("AAPL", 100, 100.0, 100.0, 95.0)
        rm._last_trade_time.clear()
        result = rm.evaluate_signal(_signal("MSFT", "BUY", 200.0))
        assert result is not None


class TestNoShortSelling:
    """SELL should only close existing long positions, never open shorts."""

    def test_sell_without_position_rejected(self):
        rm = _rm()
        result = rm.evaluate_signal(_signal("AAPL", "SELL", 100.0))
        assert result is None

    def test_sell_with_long_position_approved(self):
        rm = _rm()
        rm.update_position("AAPL", 100, 95.0, 100.0, 90.0)
        result = rm.evaluate_signal(_signal("AAPL", "SELL", 100.0))
        assert result is not None
        assert result.quantity == 100  # closes entire position

    def test_sell_closes_exact_quantity(self):
        rm = _rm()
        rm.update_position("AAPL", 50, 95.0, 100.0, 90.0)
        result = rm.evaluate_signal(_signal("AAPL", "SELL", 100.0))
        assert result.quantity == 50


class TestTotalExposureLimit:
    """Total position value should not exceed 90% of portfolio."""

    def test_rejects_when_total_exposure_exceeded(self):
        rm = _rm()
        # Already holding 85% of portfolio in positions
        rm.update_position("MSFT", 1000, 400.0, 400.0, 380.0)  # $400k
        rm.update_position("GOOGL", 1000, 350.0, 350.0, 332.0)  # $350k
        # Total: $750k = 75%. New $200k would make 95% > 90% limit
        rm._last_trade_time.clear()
        result = rm.evaluate_signal(_signal("AAPL", "BUY", 200.0))
        # Position size would be min(25%=$250k, cash-buffer) / 200 = ~1250 shares
        # But 750k + 250k = 1M > 900k (90%) → rejected
        assert result is None

    def test_approves_when_under_exposure_limit(self):
        rm = _rm()
        rm.update_position("MSFT", 100, 400.0, 400.0, 380.0)  # $40k = 4%
        rm._last_trade_time.clear()
        result = rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        assert result is not None


class TestTradeCooldown:
    """Minimum 60 seconds between trades on the same symbol."""

    def test_immediate_second_trade_rejected(self):
        rm = _rm()
        rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        # Immediately try again — should be rejected by cooldown
        result = rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        assert result is None

    def test_different_symbol_not_affected_by_cooldown(self):
        rm = _rm()
        rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        result = rm.evaluate_signal(_signal("MSFT", "BUY", 200.0))
        assert result is not None

    def test_trade_after_cooldown_approved(self):
        rm = _rm()
        rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        # Simulate cooldown expiry
        rm._last_trade_time["AAPL"] = time.time() - 61
        rm._open_positions.clear()  # clear position for re-buy
        result = rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
        assert result is not None


class TestCumulativeScenario:
    """Test the full scenario that caused the original bug."""

    def test_cannot_open_10_positions_in_same_symbol(self):
        """Simulates rapid-fire signals — only first should succeed."""
        rm = _rm()
        approved_count = 0
        for i in range(10):
            result = rm.evaluate_signal(_signal("AAPL", "BUY", 100.0))
            if result is not None:
                rm.update_position("AAPL", result.quantity, 100.0, 100.0, 95.0)
                approved_count += 1
        assert approved_count == 1

    def test_max_positions_limited_by_exposure(self):
        """Can't open positions totaling more than 90% of portfolio."""
        rm = _rm()
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        total_value = 0.0
        for sym in symbols:
            rm._last_trade_time.clear()
            result = rm.evaluate_signal(_signal(sym, "BUY", 200.0))
            if result is not None:
                val = result.quantity * 200.0
                total_value += val
                rm.update_position(sym, result.quantity, 200.0, 200.0, 190.0)
                rm._current_cash -= val
        # Total should not exceed 90% of $1M = $900k
        assert total_value <= 900_000.0
