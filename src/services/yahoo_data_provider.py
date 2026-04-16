"""Yahoo Finance data provider for paper trading.

Polls real-time (15-min delayed) price data from Yahoo Finance as a free
alternative to IB market data subscriptions. Used when MARKET_DATA_TYPE
is set to 'yahoo' in .env.

This provider fetches 1-minute bars for the last hour and feeds them
into the same pipeline as IB streaming data.
"""

import logging
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_MAX_HISTORY_LEN = 100


class YahooDataProvider:
    """Fetches price data from Yahoo Finance for paper trading.

    Polls yfinance for each watchlist symbol and dispatches tick data
    through the same callback interface as MarketDataService.
    """

    def __init__(self, watchlist: List[str]) -> None:
        self._watchlist = watchlist
        self._price_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._callback: Optional[Callable] = None
        self._last_timestamps: Dict[str, object] = {}

        # Initialize deques
        for symbol in watchlist:
            self._price_history[symbol] = deque(maxlen=_MAX_HISTORY_LEN)
            self._volume_history[symbol] = deque(maxlen=_MAX_HISTORY_LEN)

    def set_tick_callback(self, callback: Callable) -> None:
        """Register the tick callback (same interface as MarketDataService)."""
        self._callback = callback
        logger.info("Yahoo data provider: tick callback registered")

    def poll(self) -> int:
        """Fetch latest prices for all watchlist symbols.

        Returns the number of symbols successfully updated.
        """
        import yfinance as yf

        updated = 0
        for symbol in self._watchlist:
            try:
                ticker = yf.Ticker(symbol)
                # Get 1-minute bars for the last 1 day
                hist = ticker.history(period="1d", interval="1m")

                if hist.empty:
                    continue

                # Get the most recent bar
                last_bar = hist.iloc[-1]
                price = float(last_bar["Close"])
                volume = float(last_bar["Volume"])
                timestamp = hist.index[-1]

                # Skip if we already processed this timestamp
                if symbol in self._last_timestamps and self._last_timestamps[symbol] == timestamp:
                    continue

                self._last_timestamps[symbol] = timestamp
                self._price_history[symbol].append(price)
                self._volume_history[symbol].append(volume)

                if self._callback is not None:
                    prices_array = np.array(self._price_history[symbol], dtype=np.float64)
                    volumes_array = np.array(self._volume_history[symbol], dtype=np.float64)
                    avg_volume = float(np.mean(volumes_array)) if len(volumes_array) > 0 else 0.0
                    self._callback(symbol, price, volume, prices_array, volumes_array, avg_volume)

                updated += 1

            except Exception as exc:
                logger.debug("Yahoo poll failed for %s: %s", symbol, exc)

        if updated > 0:
            logger.info("Yahoo data: updated %d/%d symbols", updated, len(self._watchlist))

        return updated

    def load_history(self) -> int:
        """Load historical data to seed the indicator calculations.

        Fetches 5 days of 1-minute data to fill the rolling windows
        with enough data for MACD (needs 35+ bars).

        Returns the number of symbols loaded.
        """
        import yfinance as yf

        loaded = 0
        for symbol in self._watchlist:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", interval="1m")

                if hist.empty or len(hist) < 35:
                    logger.warning("Yahoo: insufficient history for %s (%d bars)", symbol, len(hist))
                    continue

                for _, row in hist.iterrows():
                    self._price_history[symbol].append(float(row["Close"]))
                    self._volume_history[symbol].append(float(row["Volume"]))

                loaded += 1
                logger.info("Yahoo: loaded %d bars for %s", len(hist), symbol)

            except Exception as exc:
                logger.warning("Yahoo: failed to load history for %s: %s", symbol, exc)

        return loaded
