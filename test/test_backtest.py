"""Backtesting smoke tests for the IB Trading Agent.

All tests use np.random.default_rng(42) for deterministic synthetic data.
External APIs are mocked.
"""

from datetime import datetime
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import AgentConfig
from src.models.domain import TradeSignal
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


def _generate_synthetic_prices(rng: np.random.Generator, days: int = 30) -> np.ndarray:
    """Generate 30 days of synthetic intraday prices (390 minutes/day).

    Uses geometric Brownian motion with mean-reverting drift to produce
    realistic-looking price data.
    """
    minutes_per_day = 390  # 9:30 - 16:00
    total_minutes = days * minutes_per_day

    # GBM parameters
    mu = 0.0001  # slight upward drift per minute
    sigma = 0.002  # volatility per minute

    # Generate log returns
    log_returns = rng.normal(mu, sigma, size=total_minutes)

    # Add mean-reversion component
    prices = np.zeros(total_minutes)
    prices[0] = 100.0  # starting price

    for i in range(1, total_minutes):
        # Mean reversion toward 100
        reversion = -0.00001 * (prices[i - 1] - 100.0)
        prices[i] = prices[i - 1] * np.exp(log_returns[i] + reversion)

    return prices


def _generate_synthetic_volumes(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate synthetic volume data with realistic patterns."""
    base_volume = 1_000_000.0
    return base_volume + rng.exponential(500_000, size=n)


def _run_backtest(
    prices: np.ndarray,
    volumes: np.ndarray,
    config: AgentConfig,
) -> Tuple[float, List[dict]]:
    """Run a simplified backtest over the price series.

    Returns (final_portfolio_pct_change, list_of_trades).
    Simulates the strategy engine logic without async or IB.
    """
    state_manager = MagicMock(spec=StateManager)
    risk_manager = RiskManager(config, state_manager)

    initial_capital = 100_000.0
    cash = initial_capital
    portfolio_value = initial_capital
    risk_manager.update_portfolio(portfolio_value, cash)

    trades: List[dict] = []
    position: Optional[dict] = None  # {symbol, quantity, entry_price}

    # Minimum data window for MACD (needs 35 prices)
    min_window = 40
    prev_indicators: dict = {}

    for i in range(min_window, len(prices)):
        window = prices[max(0, i - 200):i + 1]
        current_price = float(prices[i])
        current_volume = float(volumes[i])
        avg_volume = float(np.mean(volumes[max(0, i - 20):i + 1]))

        if len(window) < 35:
            continue

        # Calculate indicators
        try:
            rsi = calculate_rsi(window, period=14)
            macd_line, signal_line, histogram = calculate_macd(window)
            upper, middle, lower = calculate_bollinger_bands(window)
            ema_9 = calculate_ema(window, 9)
            ema_21 = calculate_ema(window, 21)
        except ValueError:
            continue

        # Simple signal logic (mirrors StrategyEngine logic)
        signal_direction = None

        # Momentum: RSI crosses above 30 + MACD histogram turns positive
        prev_rsi = prev_indicators.get("rsi")
        prev_hist = prev_indicators.get("histogram")
        prev_ema_9 = prev_indicators.get("ema_9")
        prev_ema_21 = prev_indicators.get("ema_21")

        if prev_rsi is not None and prev_hist is not None:
            if prev_rsi <= 30 and rsi > 30 and prev_hist <= 0 and histogram > 0:
                signal_direction = "BUY"
            elif prev_rsi >= 70 and rsi < 70 and prev_hist >= 0 and histogram < 0:
                signal_direction = "SELL"

        # Trend following: EMA crossover
        if signal_direction is None and prev_ema_9 is not None and prev_ema_21 is not None:
            if prev_ema_9 <= prev_ema_21 and ema_9 > ema_21:
                signal_direction = "BUY"
            elif prev_ema_9 >= prev_ema_21 and ema_9 < ema_21:
                signal_direction = "SELL"

        # Mean reversion: price crosses Bollinger Bands
        prev_price = prev_indicators.get("price")
        prev_lower = prev_indicators.get("lower")
        prev_upper = prev_indicators.get("upper")
        if signal_direction is None and prev_price is not None and prev_lower is not None:
            if prev_price >= prev_lower and current_price < lower:
                signal_direction = "BUY"
            elif prev_price is not None and prev_upper is not None:
                if prev_price <= prev_upper and current_price > upper:
                    signal_direction = "SELL"

        prev_indicators = {
            "rsi": rsi,
            "histogram": histogram,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "price": current_price,
            "upper": upper,
            "lower": lower,
        }

        # Execute trades through risk manager
        if signal_direction == "BUY" and position is None:
            # Update portfolio state
            portfolio_value = cash
            risk_manager.update_portfolio(portfolio_value, cash)

            signal = TradeSignal(
                symbol="SYN",
                direction="BUY",
                strategy="backtest",
                confidence=0.7,
                price=current_price,
                volume=current_volume,
                indicators={"rsi": rsi},
                polymarket_sentiment=0.0,
                timestamp=datetime.now(),
            )
            approved = risk_manager.evaluate_signal(signal)
            if approved is not None:
                quantity = approved.quantity
                cost = quantity * current_price
                cash -= cost
                position = {
                    "quantity": quantity,
                    "entry_price": current_price,
                    "stop_loss": approved.stop_loss_price,
                }
                trades.append({
                    "direction": "BUY",
                    "price": current_price,
                    "quantity": quantity,
                    "index": i,
                })

        elif signal_direction == "SELL" and position is not None:
            # Close position
            proceeds = position["quantity"] * current_price
            cash += proceeds
            trades.append({
                "direction": "SELL",
                "price": current_price,
                "quantity": position["quantity"],
                "index": i,
            })
            position = None

        # Check stop-loss
        elif position is not None and current_price <= position["stop_loss"]:
            proceeds = position["quantity"] * current_price
            cash += proceeds
            trades.append({
                "direction": "STOP_LOSS",
                "price": current_price,
                "quantity": position["quantity"],
                "index": i,
            })
            position = None

    # Close any remaining position at the last price
    if position is not None:
        final_price = float(prices[-1])
        proceeds = position["quantity"] * final_price
        cash += proceeds
        trades.append({
            "direction": "CLOSE",
            "price": final_price,
            "quantity": position["quantity"],
            "index": len(prices) - 1,
        })

    final_value = cash
    pct_change = ((final_value - initial_capital) / initial_capital) * 100.0
    return pct_change, trades


# ---------------------------------------------------------------------------
# Backtest smoke tests
# ---------------------------------------------------------------------------


class TestStrategyNotBankrupt:
    """Strategy must not lose more than 20% on synthetic data."""

    def test_strategy_not_bankrupt(self):
        """Run strategy on 30 days of synthetic data, verify portfolio
        loss does not exceed 20%.
        """
        rng = np.random.default_rng(42)
        prices = _generate_synthetic_prices(rng, days=30)
        volumes = _generate_synthetic_volumes(rng, len(prices))

        config = _make_config()
        pct_change, trades = _run_backtest(prices, volumes, config)

        assert pct_change > -20.0, (
            f"Strategy lost {pct_change:.2f}% which exceeds the 20% hard stop limit. "
            f"Total trades: {len(trades)}"
        )


class TestStrategyMakesTrades:
    """Strategy must generate at least some trade signals."""

    def test_strategy_makes_trades(self):
        """Run strategy on 30 days of synthetic data, verify at least
        some trade signals are generated.
        """
        rng = np.random.default_rng(42)
        prices = _generate_synthetic_prices(rng, days=30)
        volumes = _generate_synthetic_volumes(rng, len(prices))

        config = _make_config()
        pct_change, trades = _run_backtest(prices, volumes, config)

        assert len(trades) > 0, (
            "Strategy generated zero trades on 30 days of synthetic data. "
            "The strategy should detect at least some signals."
        )
