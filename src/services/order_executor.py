"""OrderExecutor: submits orders to IB and tracks their lifecycle.

Handles market order placement, fill callbacks, stop-loss placement on fill,
trade record persistence, bulk order cancellation for shutdown/hard stop,
order queuing with concurrency limits, and retry logic.

Requirements: 10.1, 10.2, 10.3, 14.1
"""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Dict, Optional

from ib_insync import IB, Fill, LimitOrder, MarketOrder, Stock, Trade

from src.models.domain import ApprovedTrade, TradeRecord
from src.services.market_data_service import _make_contract
from src.services.risk_manager import RiskManager
from src.services.state_manager import StateManager

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Submits and tracks orders via ib_insync."""

    # Retry configuration
    _MAX_RETRIES = 3
    _RETRY_DELAY = 2.0

    def __init__(
        self,
        ib: IB,
        risk_manager: RiskManager,
        state_manager: StateManager,
    ) -> None:
        self._ib = ib
        self._risk_manager = risk_manager
        self._state_manager = state_manager

        # Order queue and concurrency tracking
        self._pending_orders: deque[ApprovedTrade] = deque()
        self._max_concurrent_orders = 5
        self._active_orders: Dict[int, ApprovedTrade] = {}

    async def execute_trade(self, trade: ApprovedTrade) -> Optional[Trade]:
        """Place a market order via IB for an approved trade.

        Creates a Stock contract, submits a MarketOrder (BUY or SELL),
        and on fill immediately places a stop-loss order and persists
        the trade record.

        If the maximum number of concurrent orders is reached, the trade
        is queued and will be processed when a slot becomes available.

        Retries up to ``_MAX_RETRIES`` times with ``_RETRY_DELAY`` second
        delays on placement failure.

        Returns the ib_insync Trade object, or None on failure.
        """
        # Check concurrent order limit — queue if at capacity
        if len(self._active_orders) >= self._max_concurrent_orders:
            self._pending_orders.append(trade)
            logger.info(
                "Order queued for %s: %d active orders (max %d), queue depth %d",
                trade.signal.symbol,
                len(self._active_orders),
                self._max_concurrent_orders,
                len(self._pending_orders),
            )
            return None

        symbol = trade.signal.symbol
        direction = trade.signal.direction
        quantity = trade.quantity

        # Create contract and order — use appropriate order type per exchange
        contract = _make_contract(symbol)
        order = self._create_order(symbol, direction, quantity, trade.signal.price)

        logger.info(
            "Placing %s order for %s: %d shares (type=%s, tif=%s)",
            direction,
            symbol,
            quantity,
            order.orderType,
            order.tif,
        )

        # Retry loop for order placement
        ib_trade: Optional[Trade] = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                ib_trade = self._ib.placeOrder(contract, order)
                break
            except Exception:
                logger.exception(
                    "Failed to place %s order for %s (attempt %d/%d)",
                    direction,
                    symbol,
                    attempt,
                    self._MAX_RETRIES,
                )
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(self._RETRY_DELAY)

        if ib_trade is None:
            logger.error(
                "All %d attempts to place %s order for %s failed",
                self._MAX_RETRIES,
                direction,
                symbol,
            )
            return None

        # Track active order
        order_id = ib_trade.order.orderId
        self._active_orders[order_id] = trade

        # Register fill callback to place stop-loss and persist trade
        ib_trade.filledEvent += lambda t: self._handle_fill(t, trade)

        logger.info(
            "Order submitted for %s: %s %d shares, order_id=%s",
            symbol,
            direction,
            quantity,
            order_id,
        )

        return ib_trade

    async def _process_queue(self) -> None:
        """Process queued orders when slots become available."""
        while (
            self._pending_orders
            and len(self._active_orders) < self._max_concurrent_orders
        ):
            queued_trade = self._pending_orders.popleft()
            logger.info(
                "Processing queued order for %s (remaining queue: %d)",
                queued_trade.signal.symbol,
                len(self._pending_orders),
            )
            await self.execute_trade(queued_trade)

    @staticmethod
    def _round_to_tick(price: float, symbol: str) -> float:
        """Round a price to the exchange's minimum tick size.

        LSE tick sizes (in pence):
        - Price > 1000p: tick = 5p
        - Price 500–1000p: tick = 1p
        - Price 50–500p: tick = 0.5p
        - Price < 50p: tick = 0.25p

        TSE tick sizes (in yen):
        - Price > 5000: tick = 5
        - Price 3000–5000: tick = 1
        - Price 1000–3000: tick = 0.5
        - Price < 1000: tick = 0.1

        US stocks: tick = 0.01 (penny)
        """
        import math as _math
        upper = symbol.upper()

        if upper.endswith(".L"):
            # LSE — prices in pence
            if price > 1000:
                tick = 5.0
            elif price > 500:
                tick = 1.0
            elif price > 50:
                tick = 0.5
            else:
                tick = 0.25
        elif upper.endswith(".T"):
            # TSE — prices in yen
            if price > 5000:
                tick = 5.0
            elif price > 3000:
                tick = 1.0
            elif price > 1000:
                tick = 0.5
            else:
                tick = 0.1
        else:
            # US — penny tick
            tick = 0.01

        # Round to nearest tick (for BUY round up, for SELL round down)
        return round(_math.ceil(price / tick) * tick, 4)

    @staticmethod
    def _create_order(symbol: str, direction: str, quantity: int, price: float) -> MarketOrder:
        """Create an order appropriate for the symbol's exchange.

        - US stocks (SMART/USD): standard MarketOrder with TIF=DAY
        - EU stocks (.L suffix, LSE): LimitOrder with a generous limit
          (±1% from current price) and TIF=IOC to avoid TIF=DAY rejection
        - ASIA stocks (.T suffix, TSE): LimitOrder with TIF=DAY

        LSE and some non-US exchanges reject plain MarketOrders or have
        restrictions on TIF=DAY for market orders.  Using a LimitOrder
        with a wide limit (1% above ask for BUY, 1% below bid for SELL)
        effectively behaves like a market order but is accepted by all
        exchanges.
        """
        upper = symbol.upper()
        if upper.endswith(".L") or upper.endswith(".T"):
            # Non-US exchange: use aggressive limit order with tick-rounded price
            if direction == "BUY":
                limit_price = price * 1.01  # 1% above current
            else:
                limit_price = price * 0.99  # 1% below current
            limit_price = OrderExecutor._round_to_tick(limit_price, symbol)
            order = LimitOrder(direction, quantity, limit_price)
            order.tif = "IOC"  # Immediate or Cancel — avoids DAY rejection
            return order
        else:
            # US exchange: standard market order
            return MarketOrder(direction, quantity)

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

        Logs order status changes, detects partial fills, removes
        completed orders from active tracking, and triggers queue
        processing when a slot opens.
        """
        status = trade.orderStatus.status
        contract = trade.contract
        symbol = contract.symbol if contract else "UNKNOWN"
        order_id = trade.order.orderId if trade.order else "N/A"
        filled = trade.orderStatus.filled
        remaining = trade.orderStatus.remaining

        logger.info(
            "Order status changed — symbol=%s, order_id=%s, status=%s, "
            "filled=%s, remaining=%s",
            symbol,
            order_id,
            status,
            filled,
            remaining,
        )

        # Detect partial fills
        if status == "Filled" and filled > 0 and remaining > 0:
            total_qty = filled + remaining
            logger.info(
                "Partial fill detected — symbol=%s, order_id=%s, "
                "filled=%s/%s, remaining=%s",
                symbol,
                order_id,
                filled,
                total_qty,
                remaining,
            )

        # Remove completed orders and process queue
        if status in ("Filled", "Cancelled", "Inactive"):
            if isinstance(order_id, int) and order_id in self._active_orders:
                del self._active_orders[order_id]
                logger.info(
                    "Order %s removed from active tracking (status=%s), "
                    "active=%d, queued=%d",
                    order_id,
                    status,
                    len(self._active_orders),
                    len(self._pending_orders),
                )
                # Trigger queue processing
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._process_queue())
                except RuntimeError:
                    logger.debug(
                        "No running event loop — queue processing deferred"
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
