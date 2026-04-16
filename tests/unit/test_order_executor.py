"""Unit tests for OrderExecutor.

Tests order placement, fill handling, stop-loss placement on fill,
trade record persistence, exec details callback, order status callback,
and cancel_all_pending.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.domain import ApprovedTrade, TradeSignal, TradeRecord
from src.services.order_executor import OrderExecutor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_signal(
    symbol: str = "AAPL",
    direction: str = "BUY",
    price: float = 150.0,
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        direction=direction,
        strategy="momentum",
        confidence=0.85,
        price=price,
        volume=1_200_000.0,
        indicators={"rsi": 35.0, "macd_hist": 0.5},
        polymarket_sentiment=0.2,
        timestamp=datetime(2024, 6, 15, 10, 30, 0),
    )


def _make_approved_trade(
    symbol: str = "AAPL",
    direction: str = "BUY",
    price: float = 150.0,
    quantity: int = 100,
    stop_loss_price: float = 142.5,
) -> ApprovedTrade:
    return ApprovedTrade(
        signal=_make_signal(symbol=symbol, direction=direction, price=price),
        quantity=quantity,
        stop_loss_price=stop_loss_price,
        max_position_value=25_000.0,
    )


def _make_ib_trade_mock(
    avg_fill_price: float = 150.0,
    order_id: int = 42,
    status: str = "Filled",
) -> MagicMock:
    """Create a mock ib_insync Trade object."""
    ib_trade = MagicMock()
    ib_trade.order.orderId = order_id
    ib_trade.orderStatus.avgFillPrice = avg_fill_price
    ib_trade.orderStatus.status = status
    ib_trade.orderStatus.filled = 100
    ib_trade.orderStatus.remaining = 0
    ib_trade.contract.symbol = "AAPL"

    # filledEvent is an eventkit-style event; mock += as a list of callbacks
    _callbacks = []

    def add_callback(self_or_cb, cb=None):
        # Handle both method-style (self, cb) and direct (cb) calls
        actual_cb = cb if cb is not None else self_or_cb
        _callbacks.append(actual_cb)
        return ib_trade.filledEvent

    ib_trade.filledEvent.__iadd__ = add_callback
    ib_trade._fill_callbacks = _callbacks
    return ib_trade


def _make_executor() -> tuple:
    """Create an OrderExecutor with mocked dependencies.

    Returns (executor, mock_ib, mock_risk_manager, mock_state_manager).
    """
    mock_ib = MagicMock()
    mock_risk_manager = MagicMock()
    mock_state_manager = MagicMock()
    mock_state_manager.persist_trade = AsyncMock()

    executor = OrderExecutor(
        ib=mock_ib,
        risk_manager=mock_risk_manager,
        state_manager=mock_state_manager,
    )
    return executor, mock_ib, mock_risk_manager, mock_state_manager


# ---------------------------------------------------------------------------
# execute_trade tests
# ---------------------------------------------------------------------------


class TestExecuteTrade:
    """Tests for the execute_trade method."""

    @pytest.mark.asyncio
    async def test_places_market_order_buy(self):
        executor, mock_ib, _, _ = _make_executor()
        ib_trade = _make_ib_trade_mock()
        mock_ib.placeOrder.return_value = ib_trade

        approved = _make_approved_trade(direction="BUY")
        result = await executor.execute_trade(approved)

        assert result is ib_trade
        mock_ib.placeOrder.assert_called_once()
        args = mock_ib.placeOrder.call_args
        contract = args[0][0]
        order = args[0][1]
        assert contract.symbol == "AAPL"
        assert order.action == "BUY"
        assert order.totalQuantity == 100

    @pytest.mark.asyncio
    async def test_places_market_order_sell(self):
        executor, mock_ib, _, _ = _make_executor()
        ib_trade = _make_ib_trade_mock()
        mock_ib.placeOrder.return_value = ib_trade

        approved = _make_approved_trade(direction="SELL")
        result = await executor.execute_trade(approved)

        assert result is ib_trade
        args = mock_ib.placeOrder.call_args
        order = args[0][1]
        assert order.action == "SELL"

    @pytest.mark.asyncio
    async def test_returns_none_on_place_order_failure(self):
        executor, mock_ib, _, _ = _make_executor()
        mock_ib.placeOrder.side_effect = Exception("IB connection lost")

        approved = _make_approved_trade()
        result = await executor.execute_trade(approved)

        assert result is None

    @pytest.mark.asyncio
    async def test_registers_fill_callback(self):
        executor, mock_ib, _, _ = _make_executor()
        ib_trade = _make_ib_trade_mock()
        mock_ib.placeOrder.return_value = ib_trade

        approved = _make_approved_trade()
        await executor.execute_trade(approved)

        # The filledEvent += should have registered a callback
        assert len(ib_trade._fill_callbacks) == 1


# ---------------------------------------------------------------------------
# _handle_fill tests
# ---------------------------------------------------------------------------


class TestHandleFill:
    """Tests for the fill handler."""

    def test_places_stop_loss_on_fill(self):
        executor, _, mock_rm, _ = _make_executor()
        ib_trade = _make_ib_trade_mock(avg_fill_price=151.0)
        approved = _make_approved_trade(quantity=100)

        executor._handle_fill(ib_trade, approved)

        mock_rm.place_stop_loss.assert_called_once_with("AAPL", 151.0, 100)

    def test_persists_trade_record_on_fill(self):
        executor, _, _, mock_sm = _make_executor()
        ib_trade = _make_ib_trade_mock(avg_fill_price=150.5)
        approved = _make_approved_trade(
            symbol="MSFT",
            direction="BUY",
            quantity=50,
            stop_loss_price=143.0,
        )

        # Need a running event loop for create_task
        loop = asyncio.new_event_loop()

        async def run():
            executor._handle_fill(ib_trade, approved)
            # Let the created task run
            await asyncio.sleep(0.01)

        loop.run_until_complete(run())
        loop.close()

        mock_sm.persist_trade.assert_called_once()
        record = mock_sm.persist_trade.call_args[0][0]
        assert isinstance(record, TradeRecord)
        assert record.symbol == "MSFT"
        assert record.direction == "BUY"
        assert record.entry_price == 150.5
        assert record.quantity == 50
        assert record.stop_loss_price == 143.0
        assert record.strategy == "momentum"
        assert record.status == "OPEN"


# ---------------------------------------------------------------------------
# _on_exec_details tests
# ---------------------------------------------------------------------------


class TestOnExecDetails:
    """Tests for the exec details callback."""

    def test_logs_fill_details(self, caplog):
        executor, _, _, _ = _make_executor()

        trade_mock = MagicMock()
        trade_mock.contract.symbol = "TSLA"

        fill_mock = MagicMock()
        fill_mock.execution.price = 245.50
        fill_mock.execution.shares = 30.0
        fill_mock.execution.exchange = "SMART"
        fill_mock.execution.execId = "exec-001"

        import logging

        with caplog.at_level(logging.INFO):
            executor._on_exec_details(trade_mock, fill_mock)

        assert "TSLA" in caplog.text
        assert "245.50" in caplog.text
        assert "30.0" in caplog.text

    def test_handles_missing_contract(self, caplog):
        executor, _, _, _ = _make_executor()

        trade_mock = MagicMock()
        trade_mock.contract = None

        fill_mock = MagicMock()
        fill_mock.execution.price = 100.0
        fill_mock.execution.shares = 10.0
        fill_mock.execution.exchange = "SMART"
        fill_mock.execution.execId = "exec-002"

        import logging

        with caplog.at_level(logging.INFO):
            executor._on_exec_details(trade_mock, fill_mock)

        assert "UNKNOWN" in caplog.text


# ---------------------------------------------------------------------------
# _on_order_status tests
# ---------------------------------------------------------------------------


class TestOnOrderStatus:
    """Tests for the order status callback."""

    def test_logs_status_change(self, caplog):
        executor, _, _, _ = _make_executor()

        trade_mock = MagicMock()
        trade_mock.contract.symbol = "GOOG"
        trade_mock.order.orderId = 99
        trade_mock.orderStatus.status = "Filled"
        trade_mock.orderStatus.filled = 50
        trade_mock.orderStatus.remaining = 0

        import logging

        with caplog.at_level(logging.INFO):
            executor._on_order_status(trade_mock)

        assert "GOOG" in caplog.text
        assert "99" in caplog.text
        assert "Filled" in caplog.text

    def test_handles_missing_contract(self, caplog):
        executor, _, _, _ = _make_executor()

        trade_mock = MagicMock()
        trade_mock.contract = None
        trade_mock.order.orderId = 100
        trade_mock.orderStatus.status = "Cancelled"
        trade_mock.orderStatus.filled = 0
        trade_mock.orderStatus.remaining = 10

        import logging

        with caplog.at_level(logging.INFO):
            executor._on_order_status(trade_mock)

        assert "UNKNOWN" in caplog.text


# ---------------------------------------------------------------------------
# cancel_all_pending tests
# ---------------------------------------------------------------------------


class TestCancelAllPending:
    """Tests for the cancel_all_pending method."""

    @pytest.mark.asyncio
    async def test_cancels_all_open_orders(self):
        executor, mock_ib, _, _ = _make_executor()

        order1 = MagicMock()
        order1.orderId = 1
        order2 = MagicMock()
        order2.orderId = 2
        order3 = MagicMock()
        order3.orderId = 3

        mock_ib.openOrders.return_value = [order1, order2, order3]

        await executor.cancel_all_pending()

        assert mock_ib.cancelOrder.call_count == 3
        mock_ib.cancelOrder.assert_any_call(order1)
        mock_ib.cancelOrder.assert_any_call(order2)
        mock_ib.cancelOrder.assert_any_call(order3)

    @pytest.mark.asyncio
    async def test_handles_no_open_orders(self, caplog):
        executor, mock_ib, _, _ = _make_executor()
        mock_ib.openOrders.return_value = []

        import logging

        with caplog.at_level(logging.INFO):
            await executor.cancel_all_pending()

        assert "No pending orders" in caplog.text
        mock_ib.cancelOrder.assert_not_called()

    @pytest.mark.asyncio
    async def test_continues_on_cancel_failure(self):
        executor, mock_ib, _, _ = _make_executor()

        order1 = MagicMock()
        order1.orderId = 1
        order2 = MagicMock()
        order2.orderId = 2

        mock_ib.openOrders.return_value = [order1, order2]
        # First cancel fails, second succeeds
        mock_ib.cancelOrder.side_effect = [Exception("cancel failed"), None]

        await executor.cancel_all_pending()

        # Both orders should have been attempted
        assert mock_ib.cancelOrder.call_count == 2
