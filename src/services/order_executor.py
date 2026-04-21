"""OrderExecutor: submits orders to IB and tracks their lifecycle.

Handles market order placement, fill callbacks, stop-loss placement on fill,
trade record persistence, and bulk order cancellation for shutdown/hard stop.

Requirements: 10.1, 10.2, 10.3, 14.1
"""

import logging
from datetime import datetime
from typing import Optional

from ib_insync import IB, Fill, MarketOrder, Stock, Trade

from src.models.domain import ApprovedTrade, TradeRecord
from src.services.risk_manager import RiskManager
from src.services.state_manager import StateManager

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Submits and tracks orders via ib_insync."""

    def __init__(
        self,
        ib: IB,
        risk_manager: RiskManager,
        state_manager: StateManager,
    ) -> None:
        self._ib = ib
        self._risk_manager = risk_manager
        self._state_manager = state_manager

    async def execute_trade(self, trade: ApprovedTrade) -> Optional[Trade]:
        """Place a market order via IB for an approved trade.

        Creates a Stock contract, submits a MarketOrder (BUY or SELL),
        and on fill immediately places a stop-loss order and persists
        the trade record.

        Returns the ib_insync Trade object, or None on failure.
        """
        symbol = trade.signal.symbol
        direction = trade.signal.direction
        quantity = trade.quantity

        # Create contract and order
        contract = Stock(symbol, "SMART", "USD")
        order = MarketOrder(direction, quantity)

        logger.info(
            "Placing %s order for %s: %d shares",
            direction,
            symbol,
            quantity,
        )

        try:
            ib_trade = self._ib.placeOrder(contract, order)
        except Exception:
            logger.exception(
                "Failed to place %s order for %s", direction, symbol
            )
            return None

        # Register fill callback to place stop-loss and persist trade
        ib_trade.filledEvent += lambda t: self._handle_fill(t, trade)

        logger.info(
            "Order submitted for %s: %s %d shares, order_id=%s",
            symbol,
            direction,
            quantity,
            ib_trade.order.orderId,
        )

        return ib_trade

    def _handle_fill(self, ib_trade: Trade, approved: ApprovedTrade) -> None:
        """Handle a trade fill: place stop-loss and persist the record.

        For BUY fills: place stop-loss and persist a new OPEN trade record.
        For SELL fills: close the matching OPEN trade with realized P&L.
        Called when the filledEvent fires on the ib_insync Trade object.
        """
        symbol = approved.signal.symbol
        fill_price = ib_trade.orderStatus.avgFillPrice
        quantity = approved.quantity
        direction = approved.signal.direction

        logger.info(
            "Order filled for %s: %s %d shares @ %.2f",
            symbol, direction, quantity, fill_price,
        )

        if direction == "BUY":
            # Place stop-loss immediately on fill
            self._risk_manager.place_stop_loss(symbol, fill_price, quantity)

            # Persist new OPEN trade record
            record = TradeRecord(
                symbol=symbol,
                direction=direction,
                entry_price=fill_price,
                quantity=quantity,
                stop_loss_price=approved.stop_loss_price,
                strategy=approved.signal.strategy,
                signal_confidence=approved.signal.confidence,
                polymarket_sentiment=approved.signal.polymarket_sentiment,
                entry_time=datetime.now(),
                status="OPEN",
            )

            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._state_manager.persist_trade(record))
            except RuntimeError:
                logger.warning(
                    "No running event loop — trade record for %s not persisted",
                    symbol,
                )

        elif direction == "SELL":
            # Close the matching OPEN trade with realized P&L
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._state_manager.close_trade(
                        symbol, fill_price, datetime.now()
                    )
                )
            except RuntimeError:
                logger.warning(
                    "No running event loop — trade close for %s not persisted",
                    symbol,
                )

    def _on_exec_details(self, trade: Trade, fill: Fill) -> None:
        """Callback for IB.execDetailsEvent.

        Records fill details and logs the execution.
        """
        contract = trade.contract
        symbol = contract.symbol if contract else "UNKNOWN"

        logger.info(
            "Execution details — symbol=%s, price=%.2f, quantity=%.1f, "
            "exchange=%s, exec_id=%s",
            symbol,
            fill.execution.price,
            fill.execution.shares,
            fill.execution.exchange,
            fill.execution.execId,
        )

    def _on_order_status(self, trade: Trade) -> None:
        """Callback for IB.orderStatusEvent.

        Logs order status changes.
        """
        status = trade.orderStatus.status
        contract = trade.contract
        symbol = contract.symbol if contract else "UNKNOWN"
        order_id = trade.order.orderId if trade.order else "N/A"

        logger.info(
            "Order status changed — symbol=%s, order_id=%s, status=%s, "
            "filled=%s, remaining=%s",
            symbol,
            order_id,
            status,
            trade.orderStatus.filled,
            trade.orderStatus.remaining,
        )

    async def cancel_all_pending(self) -> None:
        """Cancel all open orders.

        Used during graceful shutdown and hard stop.
        Iterates over all open orders and cancels each one.
        """
        open_orders = self._ib.openOrders()
        if not open_orders:
            logger.info("No pending orders to cancel")
            return

        logger.info("Cancelling %d pending order(s)", len(open_orders))

        for order in open_orders:
            try:
                self._ib.cancelOrder(order)
                logger.info(
                    "Cancelled order: order_id=%s",
                    order.orderId,
                )
            except Exception:
                logger.exception(
                    "Failed to cancel order: order_id=%s",
                    order.orderId,
                )

        logger.info("All pending orders cancellation requested")
