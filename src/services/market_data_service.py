"""MarketDataService: streams and processes real-time market data from IB.

Subscribes to real-time streaming data for all watchlist symbols using
IB.reqMktData() and publishes processed tick data via a configurable callback.

Requirements: 2.1, 2.2, 2.3, 2.4
"""

import logging
from collections import deque
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from ib_insync import IB, Stock, Ticker

logger = logging.getLogger(__name__)

_MAX_HISTORY_LEN = 100


def _make_contract(symbol: str) -> Stock:
    """Create an IB Stock contract with exchange/currency based on symbol suffix.
    
    - Symbols ending in .L → SMART exchange, GBP currency (London via SMART routing)
    - Symbols ending in .T → SMART exchange, JPY currency (Tokyo via SMART routing)
    - All others → SMART exchange, USD currency (US)
    
    Uses SMART routing for all exchanges to avoid IB paper account
    restrictions on direct-routed orders (Error 10311).
    """
    upper = symbol.upper()
    if upper.endswith(".L"):
        # London — use SMART routing with GBP to avoid direct LSE routing restriction
        return Stock(symbol.replace(".L", "").replace(".l", ""), "SMART", "GBP")
    elif upper.endswith(".T"):
        # Tokyo — use SMART routing with JPY
        return Stock(symbol.replace(".T", "").replace(".t", ""), "SMART", "JPY")
    else:
        return Stock(symbol, "SMART", "USD")


class MarketDataService:
    """Streams and processes real-time market data from IB.

    Subscribes to market data for each symbol on the watchlist, maintains
    rolling price and volume windows (max 100 entries), and dispatches
    processed tick data to a registered callback.
    """

    def __init__(self, ib: IB, watchlist: List[str], market_data_type: str = "4") -> None:
        self._ib = ib
        self._watchlist = watchlist
        self._market_data_type = market_data_type
        self._contracts: Dict[str, Stock] = {}
        self._price_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._callback: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def subscribe_all(self) -> None:
        """Qualify contracts and subscribe to market data for all watchlist symbols.

        For each symbol:
        1. Create a ``Stock`` contract (SMART exchange, USD currency).
        2. Qualify it via ``IB.qualifyContractsAsync``.
        3. Subscribe with ``reqMktData`` using genericTickList='233,165'
           (Time & Sales + average volume).
        4. Initialise rolling history deques.

        Finally, wire ``pendingTickersEvent`` to :meth:`on_pending_tickers`.
        """
        # Set market data type from config:
        # "1"=real-time (paid), "3"=delayed (free), "4"=frozen delayed (free), "yahoo"=skip
        if self._market_data_type == "yahoo":
            logger.info("Market data type: yahoo — skipping IB market data subscriptions")
            # Still qualify contracts for order execution
            for symbol in self._watchlist:
                contract = _make_contract(symbol)
                qualified = await self._ib.qualifyContractsAsync(contract)
                if qualified:
                    self._contracts[symbol] = qualified[0]
                    self._price_history[symbol] = deque(maxlen=_MAX_HISTORY_LEN)
                    self._volume_history[symbol] = deque(maxlen=_MAX_HISTORY_LEN)
            logger.info("Qualified %d contracts for order execution", len(self._contracts))
            return

        mdt_int = int(self._market_data_type) if self._market_data_type.isdigit() else 4
        self._ib.reqMarketDataType(mdt_int)
        logger.info("Market data type set to %d", mdt_int)

        for symbol in self._watchlist:
            contract = _make_contract(symbol)
            qualified = await self._ib.qualifyContractsAsync(contract)
            if not qualified:
                logger.warning("Failed to qualify contract for %s — skipping", symbol)
                continue

            contract = qualified[0]
            self._contracts[symbol] = contract
            self._price_history[symbol] = deque(maxlen=_MAX_HISTORY_LEN)
            self._volume_history[symbol] = deque(maxlen=_MAX_HISTORY_LEN)

            self._ib.reqMktData(contract, genericTickList="233,165")
            logger.info("Subscribed to market data for %s", symbol)

        # Wire the pending-tickers event once after all subscriptions
        self._ib.pendingTickersEvent += self.on_pending_tickers
        logger.info(
            "MarketDataService subscribed to %d symbols", len(self._contracts)
        )

    async def unsubscribe_all(self) -> None:
        """Cancel all market data subscriptions and clear state."""
        for symbol, contract in self._contracts.items():
            self._ib.cancelMktData(contract)
            logger.info("Cancelled market data subscription for %s", symbol)

        self._contracts.clear()
        logger.info("All market data subscriptions cancelled")

    def on_pending_tickers(self, tickers: Set[Ticker]) -> None:
        """Callback for ``IB.pendingTickersEvent``.

        For each ticker whose contract symbol is tracked:
        - Extract last price and volume.
        - Append to the rolling price and volume deques.
        - If a callback is registered, invoke it with:
          ``(symbol, price, volume, prices_array, volumes_array, avg_volume)``

        Target: < 100 ms from receipt to callback dispatch.
        """
        for ticker in tickers:
            symbol = ticker.contract.symbol if ticker.contract else None
            if symbol is None or symbol not in self._contracts:
                continue

            price = ticker.last
            volume = ticker.volume

            # Skip tickers with no valid price data
            if price is None or price != price:  # NaN check
                continue

            # Use 0 for missing volume rather than skipping
            if volume is None or volume != volume:
                volume = 0.0

            self._price_history[symbol].append(price)
            self._volume_history[symbol].append(volume)

            if self._callback is not None:
                prices_array = np.array(self._price_history[symbol], dtype=np.float64)
                volumes_array = np.array(self._volume_history[symbol], dtype=np.float64)
                avg_volume = float(np.mean(volumes_array)) if len(volumes_array) > 0 else 0.0

                self._callback(symbol, price, volume, prices_array, volumes_array, avg_volume)

    def get_price_history(self, symbol: str, periods: int) -> np.ndarray:
        """Return the last *periods* prices as a NumPy array.

        Returns an empty array if the symbol is not found or there is
        insufficient data.
        """
        history = self._price_history.get(symbol)
        if history is None or len(history) == 0:
            return np.array([], dtype=np.float64)

        data = list(history)
        if periods > len(data):
            return np.array([], dtype=np.float64)

        return np.array(data[-periods:], dtype=np.float64)

    def get_volume_history(self, symbol: str, periods: int) -> np.ndarray:
        """Return the last *periods* volumes as a NumPy array.

        Returns an empty array if the symbol is not found or there is
        insufficient data.
        """
        history = self._volume_history.get(symbol)
        if history is None or len(history) == 0:
            return np.array([], dtype=np.float64)

        data = list(history)
        if periods > len(data):
            return np.array([], dtype=np.float64)

        return np.array(data[-periods:], dtype=np.float64)

    def set_tick_callback(self, callback: Callable) -> None:
        """Register a callback to receive processed tick data.

        The callback signature is::

            callback(symbol: str, price: float, volume: float,
                     prices: np.ndarray, volumes: np.ndarray,
                     avg_volume: float) -> None
        """
        self._callback = callback
        logger.info("Tick callback registered")

    def get_avg_daily_volume(self, symbol: str) -> float:
        """Return the average volume from the volume history.

        Returns 0.0 if no data is available for the symbol.
        """
        history = self._volume_history.get(symbol)
        if history is None or len(history) == 0:
            return 0.0
        return float(np.mean(np.array(history, dtype=np.float64)))

    def poll_snapshots(self) -> None:
        """Poll snapshot prices for all watchlist symbols.

        Uses reqHistoricalData for a 1-bar snapshot which works without
        market data subscriptions. Called periodically by the agent's
        main loop when streaming is not available.
        """
        for symbol, contract in self._contracts.items():
            try:
                bars = self._ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='300 S',
                    barSizeSetting='1 min',
                    whatToShow='MIDPOINT',
                    useRTH=True,
                    formatDate=1,
                )
                if not bars:
                    continue

                bar = bars[-1]  # most recent bar
                price = float(bar.close)
                volume = float(bar.volume)

                self._price_history.setdefault(symbol, deque(maxlen=_MAX_HISTORY_LEN))
                self._volume_history.setdefault(symbol, deque(maxlen=_MAX_HISTORY_LEN))

                self._price_history[symbol].append(price)
                self._volume_history[symbol].append(volume)

                if self._callback is not None:
                    prices_array = np.array(self._price_history[symbol], dtype=np.float64)
                    volumes_array = np.array(self._volume_history[symbol], dtype=np.float64)
                    avg_volume = float(np.mean(volumes_array)) if len(volumes_array) > 0 else 0.0
                    self._callback(symbol, price, volume, prices_array, volumes_array, avg_volume)

            except Exception as exc:
                logger.debug("Snapshot poll failed for %s: %s", symbol, exc)
