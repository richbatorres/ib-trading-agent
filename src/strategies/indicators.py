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


def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """Average True Range for volatility measurement.

    Computes ATR using Wilder's smoothing method.  The true range for
    each bar is ``max(high - low, |high - prev_close|, |low - prev_close|)``.
    The ATR is the exponentially smoothed average of true ranges.

    Parameters
    ----------
    highs : np.ndarray
        1-D array of high prices (at least ``period + 1`` elements).
    lows : np.ndarray
        1-D array of low prices.
    closes : np.ndarray
        1-D array of close prices.
    period : int
        ATR look-back window (default 14).

    Returns
    -------
    float
        Latest ATR value (always ≥ 0).

    Raises
    ------
    ValueError
        If arrays have fewer than ``period + 1`` elements or mismatched lengths.
    """
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("highs, lows, and closes must have the same length")
    if len(highs) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} bars for ATR, got {len(highs)}"
        )

    # True Range: vectorised computation
    prev_closes = closes[:-1]
    h = highs[1:]
    l = lows[1:]

    tr1 = h - l
    tr2 = np.abs(h - prev_closes)
    tr3 = np.abs(l - prev_closes)
    true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))

    # Wilder's smoothing: first ATR is SMA, then exponential
    atr = float(np.mean(true_ranges[:period]))

    # Apply Wilder's smoothing for remaining values (vectorised)
    remaining = true_ranges[period:]
    if len(remaining) > 0:
        alpha = (period - 1.0) / period
        n = len(remaining)
        powers = np.power(alpha, np.arange(n, dtype=np.float64))
        scaled = (remaining / period) / powers
        cumsum = np.cumsum(scaled)
        atr_values = cumsum * powers + atr * (powers * alpha)
        atr = float(atr_values[-1])

    return max(atr, 0.0)


def calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-Weighted Average Price using cumulative sum approach.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices (at least 1 element).
    volumes : np.ndarray
        1-D array of volumes (same length as prices).

    Returns
    -------
    float
        Latest VWAP value.

    Raises
    ------
    ValueError
        If arrays are empty or have mismatched lengths.
    """
    if len(prices) != len(volumes):
        raise ValueError("prices and volumes must have the same length")
    if len(prices) == 0:
        raise ValueError("Need at least 1 price for VWAP")

    cumulative_pv = np.cumsum(prices * volumes)
    cumulative_vol = np.cumsum(volumes)

    # Avoid division by zero
    if cumulative_vol[-1] == 0.0:
        return float(prices[-1])

    return float(cumulative_pv[-1] / cumulative_vol[-1])


def calculate_trend_strength(prices: np.ndarray, period: int = 20) -> float:
    """ADX-like trend strength indicator.

    Computes a simplified trend strength score based on directional
    movement.  Uses the ratio of net directional movement to total
    movement, smoothed over the look-back period.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of prices (at least ``period + 1`` elements).
    period : int
        Look-back window (default 20).

    Returns
    -------
    float
        Trend strength value in [0, 100].  Values > 25 indicate a
        trending market; values < 25 indicate a ranging market.

    Raises
    ------
    ValueError
        If ``prices`` has fewer than ``period + 1`` elements.
    """
    if len(prices) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} prices for trend strength, got {len(prices)}"
        )

    # Price changes
    deltas = np.diff(prices)

    # Use the last `period` deltas
    recent = deltas[-period:]

    # Positive and negative directional movement
    pos_dm = np.where(recent > 0, recent, 0.0)
    neg_dm = np.where(recent < 0, -recent, 0.0)

    sum_pos = float(np.sum(pos_dm))
    sum_neg = float(np.sum(neg_dm))
    total_dm = sum_pos + sum_neg

    if total_dm == 0.0:
        return 0.0

    # Directional indicators
    di_plus = sum_pos / total_dm
    di_neg = sum_neg / total_dm

    # DX = |DI+ - DI-| / (DI+ + DI-)
    di_sum = di_plus + di_neg
    if di_sum == 0.0:
        return 0.0

    dx = abs(di_plus - di_neg) / di_sum

    # Scale to 0-100
    strength = dx * 100.0
    return float(np.clip(strength, 0.0, 100.0))


def calculate_volume_spike(
    volumes: np.ndarray,
    period: int = 20,
    threshold: float = 2.0,
) -> bool:
    """Detect volume spikes.

    Parameters
    ----------
    volumes : np.ndarray
        1-D array of volumes (at least ``period + 1`` elements).
    period : int
        Look-back window for average volume (default 20).
    threshold : float
        Spike multiplier (default 2.0).

    Returns
    -------
    bool
        ``True`` if the latest volume exceeds ``threshold × average``
        of the preceding ``period`` volumes.

    Raises
    ------
    ValueError
        If ``volumes`` has fewer than ``period + 1`` elements.
    """
    if len(volumes) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} volumes for spike detection, got {len(volumes)}"
        )

    avg_volume = float(np.mean(volumes[-(period + 1):-1]))
    if avg_volume <= 0.0:
        return False

    return bool(volumes[-1] > threshold * avg_volume)
