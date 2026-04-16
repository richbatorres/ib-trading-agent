"""Unit tests for technical indicators and risk management.

All tests use fixed seeds and deterministic data.
Imports from src.strategies.indicators and src.services.risk_manager.
"""

import numpy as np
import pytest

from src.config import AgentConfig
from src.services.risk_manager import RiskManager
from src.services.state_manager import StateManager
from src.strategies.indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> AgentConfig:
    """Create an AgentConfig with sensible test defaults."""
    defaults = dict(
        ib_account_id="TEST",
        ib_host="127.0.0.1",
        ib_port=7497,
        environment="paper",
        email_address="test@example.com",
        email_smtp_host="smtp.example.com",
        email_smtp_port=587,
        email_smtp_user="user",
        email_smtp_password="pass",
        max_portfolio_loss_pct=20.0,
        max_position_size_pct=25.0,
        stop_loss_pct=5.0,
        cash_buffer_pct=10.0,
        db_url="sqlite:///data/test.db",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------


class TestRSI:
    """RSI indicator tests with known price sequences."""

    def test_rsi_overbought(self):
        """Monotonically rising prices must produce RSI > 70."""
        # 30 prices rising steadily — all gains, no losses
        prices = np.linspace(100.0, 150.0, num=30)
        rsi = calculate_rsi(prices, period=14)
        assert rsi > 70, f"Expected RSI > 70 for rising prices, got {rsi:.2f}"

    def test_rsi_oversold(self):
        """Monotonically falling prices must produce RSI < 30."""
        # 30 prices falling steadily — all losses, no gains
        prices = np.linspace(150.0, 100.0, num=30)
        rsi = calculate_rsi(prices, period=14)
        assert rsi < 30, f"Expected RSI < 30 for falling prices, got {rsi:.2f}"


class TestMACD:
    """MACD crossover detection with known data."""

    def test_macd_crossover(self):
        """Detect a MACD crossover: histogram sign change between two ticks.

        Build a price series that starts flat then trends up sharply so
        the fast EMA pulls above the slow EMA, producing a positive
        histogram (bullish crossover).
        """
        rng = np.random.default_rng(42)
        n = 60  # enough for MACD(12,26,9) which needs ≥35

        # Phase 1: flat prices around 100
        flat = 100.0 + rng.normal(0, 0.3, size=30)
        # Phase 2: strong uptrend
        trend = np.linspace(100.0, 120.0, num=30) + rng.normal(0, 0.3, size=30)
        prices = np.concatenate([flat, trend])

        macd_line, signal_line, histogram = calculate_macd(prices)

        # After a strong uptrend the fast EMA should be above the slow EMA
        assert macd_line > 0, f"Expected positive MACD line after uptrend, got {macd_line:.4f}"
        # The histogram should be positive (bullish crossover territory)
        assert histogram > 0, f"Expected positive histogram (bullish crossover), got {histogram:.4f}"


class TestEMACrossover:
    """9/21 EMA crossover detection with known data."""

    def test_ema_crossover(self):
        """After a strong uptrend the 9-EMA must cross above the 21-EMA."""
        rng = np.random.default_rng(42)

        # Flat then strong uptrend
        flat = 100.0 + rng.normal(0, 0.2, size=30)
        trend = np.linspace(100.0, 130.0, num=30) + rng.normal(0, 0.2, size=30)
        prices = np.concatenate([flat, trend])

        ema_9 = calculate_ema(prices, 9)
        ema_21 = calculate_ema(prices, 21)

        # The faster EMA should be above the slower EMA after a strong uptrend
        assert ema_9 > ema_21, (
            f"Expected 9-EMA ({ema_9:.4f}) > 21-EMA ({ema_21:.4f}) after uptrend"
        )


class TestBollingerBands:
    """Bollinger Bands calculation with known data."""

    def test_bollinger_bands(self):
        """Middle band = SMA, upper > middle > lower."""
        rng = np.random.default_rng(42)
        prices = 100.0 + rng.normal(0, 2.0, size=30)

        upper, middle, lower = calculate_bollinger_bands(prices, period=20, std_dev=2.0)

        # Middle band should equal the SMA of the last 20 prices
        expected_sma = float(np.mean(prices[-20:]))
        assert abs(middle - expected_sma) < 1e-10, (
            f"Middle band {middle:.6f} != SMA {expected_sma:.6f}"
        )

        # Band ordering
        assert upper > middle, f"Expected upper ({upper:.4f}) > middle ({middle:.4f})"
        assert middle > lower, f"Expected middle ({middle:.4f}) > lower ({lower:.4f})"

        # Bands should be symmetric around the middle
        assert abs((upper - middle) - (middle - lower)) < 1e-10, (
            "Bands are not symmetric around the middle"
        )


# ---------------------------------------------------------------------------
# Risk management tests
# ---------------------------------------------------------------------------


class TestStopLoss:
    """Stop-loss trigger calculation."""

    def test_stop_loss_trigger(self):
        """Stop-loss price = entry × (1 − STOP_LOSS_PCT / 100).

        With default 5% stop-loss and entry at $100, stop should be $95.
        """
        config = _make_config(stop_loss_pct=5.0)
        state_manager = unittest_mock_state_manager()
        rm = RiskManager(config, state_manager)

        entry_price = 100.0
        stop_price = rm.place_stop_loss("AAPL", entry_price, quantity=10)

        expected = entry_price * (1.0 - 5.0 / 100.0)  # 95.0
        assert stop_price == pytest.approx(expected), (
            f"Stop-loss {stop_price} != expected {expected}"
        )


class TestPortfolioHardStop:
    """Portfolio hard-stop activation at 20% loss."""

    @pytest.mark.asyncio
    async def test_portfolio_hard_stop(self):
        """Hard stop activates when portfolio drops ≥ 20%."""
        config = _make_config(max_portfolio_loss_pct=20.0)
        state_manager = unittest_mock_state_manager()
        rm = RiskManager(config, state_manager)

        # Set initial portfolio value
        rm.update_portfolio(total_value=100_000.0, cash=50_000.0)
        assert rm.is_hard_stop_active is False

        # Simulate a 20% loss
        rm.update_portfolio(total_value=80_000.0, cash=40_000.0)
        await rm.check_portfolio_loss()

        assert rm.is_hard_stop_active is True, "Hard stop should be active at 20% loss"


class TestPositionSizeLimit:
    """Position size limit enforcement at 25%."""

    def test_position_size_limit(self):
        """Position value must not exceed 25% of portfolio."""
        config = _make_config(max_position_size_pct=25.0, cash_buffer_pct=10.0)
        state_manager = unittest_mock_state_manager()
        rm = RiskManager(config, state_manager)

        rm.update_portfolio(total_value=100_000.0, cash=100_000.0)

        # At $100/share, max position = 25% of 100k / 100 = 250 shares
        shares = rm.calculate_position_size(price=100.0)
        max_value = shares * 100.0
        assert max_value <= 25_000.0, (
            f"Position value {max_value} exceeds 25% limit of 25000"
        )
        assert shares > 0, "Should allow at least some shares"


class TestCashBuffer:
    """Cash buffer enforcement at 10%."""

    def test_cash_buffer(self):
        """After buying, remaining cash must be ≥ 10% of portfolio."""
        config = _make_config(
            max_position_size_pct=100.0,  # remove position limit for this test
            cash_buffer_pct=10.0,
        )
        state_manager = unittest_mock_state_manager()
        rm = RiskManager(config, state_manager)

        rm.update_portfolio(total_value=100_000.0, cash=100_000.0)

        shares = rm.calculate_position_size(price=100.0)
        spent = shares * 100.0
        remaining_cash = 100_000.0 - spent
        buffer_required = 0.10 * 100_000.0  # 10_000

        assert remaining_cash >= buffer_required, (
            f"Remaining cash {remaining_cash} < required buffer {buffer_required}"
        )


# ---------------------------------------------------------------------------
# Lightweight mock for StateManager (avoids DB)
# ---------------------------------------------------------------------------

def unittest_mock_state_manager():
    """Return a minimal mock of StateManager that avoids any DB access."""
    from unittest.mock import MagicMock
    return MagicMock(spec=StateManager)
