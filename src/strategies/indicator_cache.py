"""IndicatorCache: caches indicator values with dirty flag for incremental updates.

Maintains rolling price and volume windows using collections.deque with
fixed size. Tracks which indicators need recalculation via dirty flags.
Provides NumPy array conversion for vectorized indicator calculations.

Requirements: 24.4, 24.5, 24.6
"""

import logging
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default rolling window size
_DEFAULT_WINDOW_SIZE = 100

# Indicator names that depend on price data
_PRICE_DEPENDENT_INDICATORS = frozenset({
    "rsi", "macd_histogram", "macd_line", "macd_signal",
    "bb_upper", "bb_middle", "bb_lower",
    "ema_9", "ema_21",
})

# Indicator names that depend on volume data
_VOLUME_DEPENDENT_INDICATORS = frozenset({
    "avg_volume", "volume_ratio",
})


class IndicatorCache:
    """Caches indicator values with dirty flag for incremental updates.

    Each symbol has its own rolling price/volume windows and a set of
    cached indicator values. When new price or volume data arrives, the
    dependent indicators are marked as dirty. The StrategyEngine checks
    the cache before recalculating — if the value is not dirty, the
    cached value is returned directly.
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._price_windows: Dict[str, deque] = {}
        self._volume_windows: Dict[str, deque] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty_flags: Dict[str, Dict[str, bool]] = {}

    # ------------------------------------------------------------------
    # Price / volume updates
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Append a new price to the rolling window and mark all
        price-dependent indicators as dirty.
        """
        if symbol not in self._price_windows:
            self._price_windows[symbol] = deque(maxlen=self._window_size)
            self._cache[symbol] = {}
            self._dirty_flags[symbol] = {}

        self._price_windows[symbol].append(price)

        # Mark all price-dependent indicators as dirty
        for indicator in _PRICE_DEPENDENT_INDICATORS:
            self._dirty_flags.setdefault(symbol, {})[indicator] = True

    def update_volume(self, symbol: str, volume: float) -> None:
        """Append a new volume to the rolling window and mark all
        volume-dependent indicators as dirty.
        """
        if symbol not in self._volume_windows:
            self._volume_windows[symbol] = deque(maxlen=self._window_size)

        self._volume_windows[symbol].append(volume)

        # Mark volume-dependent indicators as dirty
        for indicator in _VOLUME_DEPENDENT_INDICATORS:
            self._dirty_flags.setdefault(symbol, {})[indicator] = True

    # ------------------------------------------------------------------
    # Cache access
    # ------------------------------------------------------------------

    def get_indicator(self, symbol: str, indicator_name: str) -> Optional[Any]:
        """Return the cached value if not dirty, or None if dirty/missing.

        The caller should recalculate the indicator when None is returned
        and store the result via :meth:`set_indicator`.
        """
        if symbol not in self._cache:
            return None

        if self.is_dirty(symbol, indicator_name):
            return None

        return self._cache[symbol].get(indicator_name)

    def set_indicator(self, symbol: str, indicator_name: str, value: Any) -> None:
        """Store a calculated indicator value and clear its dirty flag."""
        self._cache.setdefault(symbol, {})[indicator_name] = value
        self._dirty_flags.setdefault(symbol, {})[indicator_name] = False

    def is_dirty(self, symbol: str, indicator_name: str) -> bool:
        """Check if an indicator needs recalculation.

        Returns True if the indicator has never been calculated or if
        its input data has changed since the last calculation.
        """
        flags = self._dirty_flags.get(symbol)
        if flags is None:
            return True
        return flags.get(indicator_name, True)

    # ------------------------------------------------------------------
    # NumPy array access
    # ------------------------------------------------------------------

    def get_prices(self, symbol: str) -> Optional[np.ndarray]:
        """Return the price window as a NumPy float64 array.

        Returns None if no price data exists for the symbol.
        """
        window = self._price_windows.get(symbol)
        if window is None or len(window) == 0:
            return None
        return np.array(window, dtype=np.float64)

    def get_volumes(self, symbol: str) -> Optional[np.ndarray]:
        """Return the volume window as a NumPy float64 array.

        Returns None if no volume data exists for the symbol.
        """
        window = self._volume_windows.get(symbol)
        if window is None or len(window) == 0:
            return None
        return np.array(window, dtype=np.float64)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_window_size(self, symbol: str) -> int:
        """Return the number of prices currently in the window for a symbol."""
        window = self._price_windows.get(symbol)
        return len(window) if window else 0

    def clear_symbol(self, symbol: str) -> None:
        """Remove all cached data for a symbol."""
        self._price_windows.pop(symbol, None)
        self._volume_windows.pop(symbol, None)
        self._cache.pop(symbol, None)
        self._dirty_flags.pop(symbol, None)

    def clear_all(self) -> None:
        """Remove all cached data for all symbols."""
        self._price_windows.clear()
        self._volume_windows.clear()
        self._cache.clear()
        self._dirty_flags.clear()
