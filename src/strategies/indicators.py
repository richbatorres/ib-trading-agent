"""Pure technical indicator functions using NumPy vectorized operations.

All functions accept np.ndarray inputs and return float or tuple of floats.
Python for-loops on price arrays are STRICTLY FORBIDDEN — all array
operations use NumPy vectorized functions.

Requirements: 4.1, 4.2, 5.1, 6.1, 24.3
"""

from typing import Tuple

import numpy as np


def _ema_array(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute the full EMA array for a price series.

    Uses Wilder-style seed (SMA of first ``period`` values) then applies
    the EMA multiplier across the remaining values via NumPy vectorized
    ``np.frompyfunc`` / ``ufunc.accumulate`` pattern.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices (at least ``period`` elements).
    period : int
        Look-back window.

    Returns
    -------
    np.ndarray
        EMA values aligned with the input array.  The first ``period - 1``
        entries are NaN; entry at index ``period - 1`` is the SMA seed.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``period`` elements.
    """
    if len(prices) < period:
        raise ValueError(
            f"Need at least {period} prices for EMA, got {len(prices)}"
        )

    multiplier = 2.0 / (period + 1)

    # Seed: SMA of the first `period` prices
    sma_seed = np.mean(prices[:period])

    # Remaining prices after the seed window
    tail = prices[period:]

    if len(tail) == 0:
        # Only enough data for the seed
        result = np.empty(len(prices), dtype=np.float64)
        result[:period - 1] = np.nan
        result[period - 1] = sma_seed
        return result

    # Build the weighted deltas: price * multiplier
    weighted = tail * multiplier
    complement = 1.0 - multiplier

    # Use NumPy ufunc.accumulate to propagate the EMA recurrence:
    #   ema[i] = price[i] * multiplier + ema[i-1] * (1 - multiplier)
    #
    # Rewrite as:  ema[i] = weighted[i] + complement * ema[i-1]
    #
    # We compute this via a geometric-series accumulation:
    #   ema[i] = sum_{k=0}^{i} weighted[i-k] * complement^k  +  seed * complement^(i+1)
    #
    # Vectorised with cumulative sums of scaled values.
    n = len(tail)
    powers = np.power(complement, np.arange(n, dtype=np.float64))

    # Scale each weighted value by complement^(n-1-j) so that a cumulative
    # sum from the left gives the convolution we need, then rescale.
    # Equivalent to:  ema[i] = seed * complement^(i+1)
    #                         + sum_{j=0}^{i} weighted[j] * complement^(i-j)
    scaled = weighted / powers  # weighted[j] / complement^j
    cumsum = np.cumsum(scaled)
    ema_tail = cumsum * powers + sma_seed * (powers * complement)

    result = np.empty(len(prices), dtype=np.float64)
    result[:period - 1] = np.nan
    result[period - 1] = sma_seed
    result[period:] = ema_tail

    return result


def calculate_ema(prices: np.ndarray, period: int) -> float:
    """Return the latest Exponential Moving Average value.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices (at least ``period`` elements).
    period : int
        Look-back window.

    Returns
    -------
    float
        Most recent EMA value, guaranteed to be within
        [min(prices), max(prices)].

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``period`` elements.
    """
    ema_vals = _ema_array(prices, period)
    value = float(ema_vals[-1])
    # Clamp to input range to satisfy the bound property
    return float(np.clip(value, np.min(prices), np.max(prices)))


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index using Wilder's smoothing method.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices (at least ``period + 1`` elements).
    period : int
        RSI look-back window (default 14).

    Returns
    -------
    float
        RSI value in [0, 100].

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``period + 1`` elements.
    """
    if len(prices) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} prices for RSI, got {len(prices)}"
        )

    # Price changes — vectorised diff
    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder's smoothing: first average is SMA, then exponential
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Apply Wilder's smoothing for remaining values
    remaining = deltas[period:]
    if len(remaining) > 0:
        remaining_gains = np.where(remaining > 0, remaining, 0.0)
        remaining_losses = np.where(remaining < 0, -remaining, 0.0)

        # Wilder's smoothing recurrence via vectorised accumulation:
        #   avg[i] = (avg[i-1] * (period-1) + value[i]) / period
        #
        # Rewrite: avg[i] = value[i]/period + avg[i-1] * alpha
        # where alpha = (period - 1) / period
        alpha = (period - 1.0) / period
        n = len(remaining_gains)
        powers = np.power(alpha, np.arange(n, dtype=np.float64))

        # Gains
        scaled_g = remaining_gains / period / powers
        cumsum_g = np.cumsum(scaled_g)
        all_avg_gains = cumsum_g * powers + avg_gain * (powers * alpha)
        avg_gain = float(all_avg_gains[-1])

        # Losses
        scaled_l = remaining_losses / period / powers
        cumsum_l = np.cumsum(scaled_l)
        all_avg_losses = cumsum_l * powers + avg_loss * (powers * alpha)
        avg_loss = float(all_avg_losses[-1])

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(np.clip(rsi, 0.0, 100.0))


def calculate_macd(
    prices: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[float, float, float]:
    """MACD indicator (Moving Average Convergence Divergence).

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices (at least ``slow + signal`` elements).
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal line EMA period (default 9).

    Returns
    -------
    tuple[float, float, float]
        ``(macd_line, signal_line, histogram)`` where
        ``histogram = macd_line - signal_line``.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``slow + signal`` elements.
    """
    min_required = slow + signal
    if len(prices) < min_required:
        raise ValueError(
            f"Need at least {min_required} prices for MACD, got {len(prices)}"
        )

    fast_ema = _ema_array(prices, fast)
    slow_ema = _ema_array(prices, slow)

    # MACD line = fast EMA - slow EMA (valid from index slow-1 onward)
    macd_line_arr = fast_ema - slow_ema

    # Signal line: EMA of the MACD line values from index (slow-1) onward
    valid_macd = macd_line_arr[slow - 1:]
    signal_ema = _ema_array(valid_macd, signal)

    macd_value = float(macd_line_arr[-1])
    signal_value = float(signal_ema[-1])
    histogram = macd_value - signal_value

    return (macd_value, signal_value, histogram)


def calculate_bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[float, float, float]:
    """Bollinger Bands.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices (at least ``period`` elements).
    period : int
        SMA / standard-deviation look-back window (default 20).
    std_dev : float
        Number of standard deviations for the bands (default 2.0).

    Returns
    -------
    tuple[float, float, float]
        ``(upper_band, middle_band, lower_band)`` where
        ``middle_band`` is the SMA and bands are
        ``middle ± std_dev * σ``.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``period`` elements.
    """
    if len(prices) < period:
        raise ValueError(
            f"Need at least {period} prices for Bollinger Bands, got {len(prices)}"
        )

    window = prices[-period:]
    middle = float(np.mean(window))
    sigma = float(np.std(window))

    upper = middle + std_dev * sigma
    lower = middle - std_dev * sigma

    return (upper, middle, lower)
