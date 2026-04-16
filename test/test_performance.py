"""Performance tests for the IB Trading Agent.

All tests use np.random.default_rng(42) for deterministic data.
External APIs are mocked where needed.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.models.domain import ApprovedTrade, TradeSignal
from src.strategies.indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


class TestIndicatorCalculationSpeed:
    """Verify all indicators calculate in < 10ms for 1000 candles."""

    def test_indicator_calculation_speed(self):
        """All four indicators must complete in < 10ms each on 1000 candles."""
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, size=1000))

        # RSI
        start = time.perf_counter()
        calculate_rsi(prices, period=14)
        rsi_ms = (time.perf_counter() - start) * 1000
        assert rsi_ms < 10, f"RSI took {rsi_ms:.2f}ms (limit: 10ms)"

        # MACD
        start = time.perf_counter()
        calculate_macd(prices, fast=12, slow=26, signal=9)
        macd_ms = (time.perf_counter() - start) * 1000
        assert macd_ms < 10, f"MACD took {macd_ms:.2f}ms (limit: 10ms)"

        # Bollinger Bands
        start = time.perf_counter()
        calculate_bollinger_bands(prices, period=20, std_dev=2.0)
        bb_ms = (time.perf_counter() - start) * 1000
        assert bb_ms < 10, f"Bollinger Bands took {bb_ms:.2f}ms (limit: 10ms)"

        # EMA (9 and 21)
        start = time.perf_counter()
        calculate_ema(prices, 9)
        calculate_ema(prices, 21)
        ema_ms = (time.perf_counter() - start) * 1000
        assert ema_ms < 10, f"EMA (9+21) took {ema_ms:.2f}ms (limit: 10ms)"


class TestSignalToOrderLatency:
    """Verify end-to-end signal processing in < 100ms."""

    @pytest.mark.asyncio
    async def test_signal_to_order_latency(self):
        """Full pipeline: indicators → signal evaluation → risk check → order
        must complete in < 100ms with mocked IB.
        """
        from datetime import datetime

        from src.config import AgentConfig
        from src.services.order_executor import OrderExecutor
        from src.services.risk_manager import RiskManager
        from src.services.state_manager import StateManager

        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, size=1000))

        # Setup mocked components
        config = AgentConfig(
            ib_account_id="TEST",
            ib_host="127.0.0.1",
            ib_port=7497,
            environment="paper",
            email_address="test@example.com",
            email_smtp_host="smtp.example.com",
            email_smtp_port=587,
            email_smtp_user="user",
            email_smtp_password="pass",
        )

        ib = MagicMock()
        ib.isConnected.return_value = True
        mock_trade = MagicMock()
        mock_trade.order.orderId = 1
        mock_trade.filledEvent = MagicMock()
        mock_trade.filledEvent.__iadd__ = MagicMock(return_value=mock_trade.filledEvent)
        ib.placeOrder.return_value = mock_trade

        state_manager = MagicMock(spec=StateManager)
        state_manager.persist_trade = AsyncMock()

        risk_manager = RiskManager(config, state_manager, ib)
        risk_manager.update_portfolio(total_value=100_000.0, cash=100_000.0)

        order_executor = OrderExecutor(ib, risk_manager, state_manager)

        # Time the full pipeline
        start = time.perf_counter()

        # 1. Calculate indicators
        rsi = calculate_rsi(prices)
        macd_line, signal_line, histogram = calculate_macd(prices)
        upper, middle, lower = calculate_bollinger_bands(prices)
        ema_9 = calculate_ema(prices, 9)
        ema_21 = calculate_ema(prices, 21)

        # 2. Create a signal
        signal = TradeSignal(
            symbol="AAPL",
            direction="BUY",
            strategy="momentum",
            confidence=0.8,
            price=float(prices[-1]),
            volume=1_500_000.0,
            indicators={"rsi": rsi, "macd_histogram": histogram},
            polymarket_sentiment=0.0,
            timestamp=datetime.now(),
        )

        # 3. Risk check
        approved = risk_manager.evaluate_signal(signal)
        assert approved is not None

        # 4. Execute trade (mocked IB)
        result = await order_executor.execute_trade(approved)
        assert result is not None

        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Signal-to-order took {elapsed_ms:.2f}ms (limit: 100ms)"


class TestNumpyVsLoopBenchmark:
    """Verify NumPy vectorized RSI is at least 10x faster than a loop."""

    def test_numpy_vs_loop_benchmark(self):
        """NumPy vectorized RSI must be ≥ 10x faster than a pure Python
        for-loop reference implementation.
        """
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.5, size=1000))
        period = 14

        # --- Reference: pure Python for-loop RSI ---
        def rsi_loop(prices_list, period):
            """Naive for-loop RSI implementation."""
            deltas = [prices_list[i] - prices_list[i - 1] for i in range(1, len(prices_list))]
            gains = [d if d > 0 else 0.0 for d in deltas]
            losses = [-d if d < 0 else 0.0 for d in deltas]

            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period

            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                return 100.0 if avg_gain > 0 else 50.0
            rs = avg_gain / avg_loss
            return 100.0 - (100.0 / (1.0 + rs))

        prices_list = prices.tolist()

        # Warm up
        calculate_rsi(prices, period)
        rsi_loop(prices_list, period)

        # Benchmark loop version (multiple iterations for stability)
        n_iters = 50
        start = time.perf_counter()
        for _ in range(n_iters):
            rsi_loop(prices_list, period)
        loop_time = (time.perf_counter() - start) / n_iters

        # Benchmark NumPy version
        start = time.perf_counter()
        for _ in range(n_iters):
            calculate_rsi(prices, period)
        numpy_time = (time.perf_counter() - start) / n_iters

        speedup = loop_time / numpy_time if numpy_time > 0 else float("inf")
        assert speedup >= 10, (
            f"NumPy RSI is only {speedup:.1f}x faster than loop "
            f"(loop={loop_time*1000:.3f}ms, numpy={numpy_time*1000:.3f}ms). "
            f"Expected ≥ 10x speedup."
        )
