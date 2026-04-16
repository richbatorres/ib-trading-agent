"""Unit tests for MarketDataService."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.services.market_data_service import MarketDataService, _MAX_HISTORY_LEN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker(symbol: str, last: float, volume: float) -> SimpleNamespace:
    """Create a minimal ticker-like object."""
    contract = SimpleNamespace(symbol=symbol)
    return SimpleNamespace(contract=contract, last=last, volume=volume)


def _make_ib_mock() -> MagicMock:
    """Return a mock IB instance with the methods MarketDataService uses."""
    ib = MagicMock()
    ib.qualifyContractsAsync = AsyncMock()
    ib.reqMktData = MagicMock()
    ib.cancelMktData = MagicMock()
    ib.pendingTickersEvent = MagicMock()
    ib.pendingTickersEvent.__iadd__ = MagicMock(return_value=ib.pendingTickersEvent)
    return ib


# ---------------------------------------------------------------------------
# Tests — constructor & basic state
# ---------------------------------------------------------------------------

class TestMarketDataServiceInit:
    def test_initial_state(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL", "MSFT"])

        assert svc._watchlist == ["AAPL", "MSFT"]
        assert svc._contracts == {}
        assert svc._price_history == {}
        assert svc._volume_history == {}
        assert svc._callback is None


# ---------------------------------------------------------------------------
# Tests — subscribe_all
# ---------------------------------------------------------------------------

class TestSubscribeAll:
    @pytest.mark.asyncio
    async def test_subscribe_qualifies_and_subscribes(self):
        ib = _make_ib_mock()
        # qualifyContractsAsync returns the contract back
        mock_contract = SimpleNamespace(symbol="AAPL")
        ib.qualifyContractsAsync.return_value = [mock_contract]

        svc = MarketDataService(ib, ["AAPL"])
        await svc.subscribe_all()

        assert "AAPL" in svc._contracts
        ib.reqMktData.assert_called_once()
        # Verify genericTickList was passed
        call_kwargs = ib.reqMktData.call_args
        assert call_kwargs[1].get("genericTickList") == "233,165" or "233,165" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_subscribe_skips_unqualified_symbol(self):
        ib = _make_ib_mock()
        ib.qualifyContractsAsync.return_value = []  # qualification fails

        svc = MarketDataService(ib, ["INVALID"])
        await svc.subscribe_all()

        assert "INVALID" not in svc._contracts
        ib.reqMktData.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_initialises_history_deques(self):
        ib = _make_ib_mock()
        mock_contract = SimpleNamespace(symbol="TSLA")
        ib.qualifyContractsAsync.return_value = [mock_contract]

        svc = MarketDataService(ib, ["TSLA"])
        await svc.subscribe_all()

        assert isinstance(svc._price_history["TSLA"], deque)
        assert svc._price_history["TSLA"].maxlen == _MAX_HISTORY_LEN
        assert isinstance(svc._volume_history["TSLA"], deque)
        assert svc._volume_history["TSLA"].maxlen == _MAX_HISTORY_LEN

    @pytest.mark.asyncio
    async def test_subscribe_wires_pending_tickers_event(self):
        ib = _make_ib_mock()
        mock_contract = SimpleNamespace(symbol="AAPL")
        ib.qualifyContractsAsync.return_value = [mock_contract]

        svc = MarketDataService(ib, ["AAPL"])
        await svc.subscribe_all()

        ib.pendingTickersEvent.__iadd__.assert_called_once_with(svc.on_pending_tickers)


# ---------------------------------------------------------------------------
# Tests — unsubscribe_all
# ---------------------------------------------------------------------------

class TestUnsubscribeAll:
    @pytest.mark.asyncio
    async def test_unsubscribe_cancels_all(self):
        ib = _make_ib_mock()
        mock_contract = SimpleNamespace(symbol="AAPL")
        ib.qualifyContractsAsync.return_value = [mock_contract]

        svc = MarketDataService(ib, ["AAPL"])
        await svc.subscribe_all()
        await svc.unsubscribe_all()

        ib.cancelMktData.assert_called_once()
        assert svc._contracts == {}


# ---------------------------------------------------------------------------
# Tests — on_pending_tickers
# ---------------------------------------------------------------------------

class TestOnPendingTickers:
    def test_appends_price_and_volume(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        ticker = _make_ticker("AAPL", 150.0, 1000.0)
        svc.on_pending_tickers([ticker])

        assert list(svc._price_history["AAPL"]) == [150.0]
        assert list(svc._volume_history["AAPL"]) == [1000.0]

    def test_skips_unknown_symbol(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        ticker = _make_ticker("GOOG", 100.0, 500.0)
        svc.on_pending_tickers([ticker])

        assert len(svc._price_history["AAPL"]) == 0

    def test_skips_nan_price(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        ticker = _make_ticker("AAPL", float("nan"), 1000.0)
        svc.on_pending_tickers([ticker])

        assert len(svc._price_history["AAPL"]) == 0

    def test_skips_none_price(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        ticker = _make_ticker("AAPL", None, 1000.0)
        svc.on_pending_tickers([ticker])

        assert len(svc._price_history["AAPL"]) == 0

    def test_handles_none_volume_as_zero(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        ticker = _make_ticker("AAPL", 150.0, None)
        svc.on_pending_tickers([ticker])

        assert list(svc._price_history["AAPL"]) == [150.0]
        assert list(svc._volume_history["AAPL"]) == [0.0]

    def test_invokes_callback_with_correct_args(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        received = []
        def cb(symbol, price, volume, prices, volumes, avg_vol):
            received.append((symbol, price, volume, len(prices), len(volumes), avg_vol))

        svc.set_tick_callback(cb)

        ticker = _make_ticker("AAPL", 150.0, 2000.0)
        svc.on_pending_tickers([ticker])

        assert len(received) == 1
        sym, price, vol, n_prices, n_vols, avg = received[0]
        assert sym == "AAPL"
        assert price == 150.0
        assert vol == 2000.0
        assert n_prices == 1
        assert n_vols == 1
        assert avg == 2000.0

    def test_callback_not_called_when_not_set(self):
        """No error when callback is None."""
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        ticker = _make_ticker("AAPL", 150.0, 1000.0)
        svc.on_pending_tickers([ticker])  # should not raise

    def test_rolling_window_respects_maxlen(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._contracts["AAPL"] = SimpleNamespace(symbol="AAPL")
        svc._price_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        for i in range(_MAX_HISTORY_LEN + 20):
            ticker = _make_ticker("AAPL", float(i), float(i * 10))
            svc.on_pending_tickers([ticker])

        assert len(svc._price_history["AAPL"]) == _MAX_HISTORY_LEN
        assert len(svc._volume_history["AAPL"]) == _MAX_HISTORY_LEN
        # Oldest entries should have been evicted
        assert list(svc._price_history["AAPL"])[0] == 20.0


# ---------------------------------------------------------------------------
# Tests — get_price_history / get_volume_history
# ---------------------------------------------------------------------------

class TestGetHistory:
    def test_get_price_history_returns_last_n(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._price_history["AAPL"] = deque([1.0, 2.0, 3.0, 4.0, 5.0], maxlen=_MAX_HISTORY_LEN)

        result = svc.get_price_history("AAPL", 3)
        np.testing.assert_array_equal(result, [3.0, 4.0, 5.0])

    def test_get_price_history_empty_for_unknown_symbol(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])

        result = svc.get_price_history("GOOG", 5)
        assert len(result) == 0

    def test_get_price_history_empty_when_insufficient_data(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._price_history["AAPL"] = deque([1.0, 2.0], maxlen=_MAX_HISTORY_LEN)

        result = svc.get_price_history("AAPL", 5)
        assert len(result) == 0

    def test_get_volume_history_returns_last_n(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._volume_history["AAPL"] = deque([100, 200, 300, 400], maxlen=_MAX_HISTORY_LEN)

        result = svc.get_volume_history("AAPL", 2)
        np.testing.assert_array_equal(result, [300.0, 400.0])

    def test_get_volume_history_empty_for_unknown_symbol(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])

        result = svc.get_volume_history("GOOG", 3)
        assert len(result) == 0

    def test_get_volume_history_empty_when_insufficient_data(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._volume_history["AAPL"] = deque([100], maxlen=_MAX_HISTORY_LEN)

        result = svc.get_volume_history("AAPL", 5)
        assert len(result) == 0

    def test_returns_numpy_arrays(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._price_history["AAPL"] = deque([1.0, 2.0, 3.0], maxlen=_MAX_HISTORY_LEN)
        svc._volume_history["AAPL"] = deque([10.0, 20.0, 30.0], maxlen=_MAX_HISTORY_LEN)

        prices = svc.get_price_history("AAPL", 2)
        volumes = svc.get_volume_history("AAPL", 2)

        assert isinstance(prices, np.ndarray)
        assert isinstance(volumes, np.ndarray)
        assert prices.dtype == np.float64
        assert volumes.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests — set_tick_callback
# ---------------------------------------------------------------------------

class TestSetTickCallback:
    def test_sets_callback(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])

        def my_cb(*args):
            pass

        svc.set_tick_callback(my_cb)
        assert svc._callback is my_cb


# ---------------------------------------------------------------------------
# Tests — get_avg_daily_volume
# ---------------------------------------------------------------------------

class TestGetAvgDailyVolume:
    def test_returns_average(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._volume_history["AAPL"] = deque([100, 200, 300], maxlen=_MAX_HISTORY_LEN)

        avg = svc.get_avg_daily_volume("AAPL")
        assert avg == pytest.approx(200.0)

    def test_returns_zero_for_unknown_symbol(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])

        assert svc.get_avg_daily_volume("GOOG") == 0.0

    def test_returns_zero_for_empty_history(self):
        ib = _make_ib_mock()
        svc = MarketDataService(ib, ["AAPL"])
        svc._volume_history["AAPL"] = deque(maxlen=_MAX_HISTORY_LEN)

        assert svc.get_avg_daily_volume("AAPL") == 0.0
