"""Unit tests for src/strategies/indicators.py.

Validates RSI, MACD, Bollinger Bands, EMA, and the internal _ema_array
helper using known price sequences and edge cases.
"""

import numpy as np
import pytest

from src.strategies.indicators import (
    _ema_array,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iterative_ema(prices: list[float], period: int) -> list[float]:
    """Reference iterative EMA for verification."""
    multiplier = 2.0 / (period + 1)
    result = [float("nan")] * (period - 1)
    sma = sum(prices[:period]) / period
    result.append(sma)
    for p in prices[period:]:
        sma = p * multiplier + sma * (1 - multiplier)
        result.append(sma)
    return result


def _iterative_rsi(prices: list[float], period: int = 14) -> float:
    """Reference iterative RSI (Wilder's smoothing) for verification."""
    deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# _ema_array
# ---------------------------------------------------------------------------

class TestEmaArray:
    def test_minimum_data(self):
        """Exactly `period` prices → single valid EMA = SMA."""
        prices = np.array([1.0, 2.0, 3.0])
        result = _ema_array(prices, period=3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)  # SMA of [1,2,3]

    def test_matches_iterative(self):
        prices_list = [44.0, 44.34, 44.09, 43.61, 44.33,
                       44.83, 45.10, 45.42, 45.84, 46.08,
                       45.89, 46.03, 45.61, 46.28, 46.28]
        prices = np.array(prices_list)
        period = 10
        result = _ema_array(prices, period)
        ref = _iterative_ema(prices_list, period)
        for i in range(period - 1, len(prices)):
            assert result[i] == pytest.approx(ref[i], abs=1e-10)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Need at least"):
            _ema_array(np.array([1.0, 2.0]), period=5)


# ---------------------------------------------------------------------------
# calculate_ema
# ---------------------------------------------------------------------------

class TestCalculateEma:
    def test_basic(self):
        prices = np.array([10.0, 11.0, 12.0, 11.5, 13.0])
        ema = calculate_ema(prices, period=3)
        assert 10.0 <= ema <= 13.0

    def test_constant_prices(self):
        prices = np.full(20, 50.0)
        ema = calculate_ema(prices, period=10)
        assert ema == pytest.approx(50.0)

    def test_bounded_by_min_max(self):
        prices = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        ema = calculate_ema(prices, period=3)
        assert 5.0 <= ema <= 25.0

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            calculate_ema(np.array([1.0]), period=5)


# ---------------------------------------------------------------------------
# calculate_rsi
# ---------------------------------------------------------------------------

class TestCalculateRsi:
    def test_all_gains(self):
        """Monotonically increasing prices → RSI near 100."""
        prices = np.arange(1.0, 20.0)
        rsi = calculate_rsi(prices, period=14)
        assert rsi == pytest.approx(100.0)

    def test_all_losses(self):
        """Monotonically decreasing prices → RSI near 0."""
        prices = np.arange(20.0, 1.0, -1.0)
        rsi = calculate_rsi(prices, period=14)
        assert rsi == pytest.approx(0.0, abs=0.01)

    def test_constant_prices(self):
        """No change → RSI = 50."""
        prices = np.full(20, 100.0)
        rsi = calculate_rsi(prices, period=14)
        assert rsi == pytest.approx(50.0)

    def test_range_0_100(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100)) + 100
        rsi = calculate_rsi(prices, period=14)
        assert 0.0 <= rsi <= 100.0

    def test_matches_iterative(self):
        np.random.seed(123)
        prices_list = list(np.cumsum(np.random.randn(50)) + 100)
        prices = np.array(prices_list)
        rsi_vec = calculate_rsi(prices, period=14)
        rsi_ref = _iterative_rsi(prices_list, period=14)
        assert rsi_vec == pytest.approx(rsi_ref, abs=1e-8)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Need at least"):
            calculate_rsi(np.array([1.0, 2.0, 3.0]), period=14)

    def test_minimum_data(self):
        """Exactly period+1 prices should work."""
        prices = np.arange(1.0, 16.0)  # 15 elements, period=14
        rsi = calculate_rsi(prices, period=14)
        assert 0.0 <= rsi <= 100.0


# ---------------------------------------------------------------------------
# calculate_macd
# ---------------------------------------------------------------------------

class TestCalculateMacd:
    def test_basic_output_structure(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(50)) + 100
        macd_line, signal_line, histogram = calculate_macd(prices)
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert histogram == pytest.approx(macd_line - signal_line)

    def test_constant_prices(self):
        """Constant prices → MACD ≈ 0."""
        prices = np.full(50, 100.0)
        macd_line, signal_line, histogram = calculate_macd(prices)
        assert macd_line == pytest.approx(0.0, abs=1e-10)
        assert signal_line == pytest.approx(0.0, abs=1e-10)
        assert histogram == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Need at least"):
            calculate_macd(np.arange(1.0, 30.0))  # 29 < 26+9=35

    def test_minimum_data(self):
        """Exactly slow+signal prices should work."""
        prices = np.arange(1.0, 36.0)  # 35 elements
        macd_line, signal_line, histogram = calculate_macd(prices)
        assert isinstance(macd_line, float)


# ---------------------------------------------------------------------------
# calculate_bollinger_bands
# ---------------------------------------------------------------------------

class TestCalculateBollingerBands:
    def test_basic_ordering(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(30)) + 100
        upper, middle, lower = calculate_bollinger_bands(prices)
        assert upper > middle
        assert middle > lower

    def test_middle_is_sma(self):
        prices = np.array([10.0, 12.0, 11.0, 13.0, 14.0,
                           12.5, 11.5, 13.5, 14.5, 15.0,
                           14.0, 13.0, 12.0, 11.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0])
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        expected_sma = float(np.mean(prices[-20:]))
        assert middle == pytest.approx(expected_sma)

    def test_constant_prices(self):
        """Zero std dev → upper == middle == lower."""
        prices = np.full(20, 50.0)
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        assert upper == pytest.approx(50.0)
        assert middle == pytest.approx(50.0)
        assert lower == pytest.approx(50.0)

    def test_custom_std_dev(self):
        prices = np.arange(1.0, 21.0)
        upper1, mid1, lower1 = calculate_bollinger_bands(prices, period=20, std_dev=1.0)
        upper2, mid2, lower2 = calculate_bollinger_bands(prices, period=20, std_dev=3.0)
        assert mid1 == pytest.approx(mid2)
        assert (upper2 - mid2) > (upper1 - mid1)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="Need at least"):
            calculate_bollinger_bands(np.array([1.0, 2.0]), period=20)

    def test_minimum_data(self):
        """Exactly `period` prices should work."""
        prices = np.arange(1.0, 21.0)  # 20 elements
        upper, middle, lower = calculate_bollinger_bands(prices, period=20)
        assert upper >= middle >= lower
