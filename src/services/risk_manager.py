"""RiskManager: enforces portfolio risk limits and manages stop-loss orders.

Validates every trade signal against risk limits in strict order:
hard stop → cash buffer → position size → stop-loss readiness.
Manages portfolio-level hard stop, per-position stop-loss placement,
trailing stop conversion, and position sizing.

Requirements: 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 11.1, 11.2, 11.3,
              12.1, 12.2, 12.3, 12.4
"""

import logging
import math
from typing import Dict, Optional

from src.config import AgentConfig
from src.models.domain import ApprovedTrade, TradeSignal
from src.services.state_manager import StateManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces portfolio risk limits and manages stop-loss orders."""

    def __init__(
        self,
        config: AgentConfig,
        state_manager: StateManager,
        ib: object = None,
    ) -> None:
        self._config = config
        self._state_manager = state_manager
        self._ib = ib

        # Portfolio-level state
        self._hard_stop_active: bool = False
        self._initial_portfolio_value: Optional[float] = None
        self._current_portfolio_value: float = 0.0
        self._current_cash: float = 0.0

        # Per-position tracking: symbol -> {quantity, entry_price, current_price, stop_loss_price}
        self._open_positions: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_signal(self, signal: TradeSignal) -> Optional[ApprovedTrade]:
        """Validate a trade signal against risk limits in strict order.

        Risk checks (order is non-negotiable per Requirement 12.4):
        1. Hard stop check — reject if hard stop is active
        2. Cash buffer check — reject if trade would reduce cash below buffer
        3. Position size check — reject if position exceeds max size
        4. Stop-loss readiness — calculate stop-loss price

        Returns an ApprovedTrade if all checks pass, or None with a
        WARNING log explaining the rejection.
        """
        # 1. Hard stop check
        if self._hard_stop_active:
            logger.warning(
                "Trade REJECTED for %s: hard stop is active — all trading halted",
                signal.symbol,
            )
            return None

        # Calculate proposed quantity
        quantity = self.calculate_position_size(signal.price)
        if quantity <= 0:
            logger.warning(
                "Trade REJECTED for %s: calculated position size is 0 "
                "(price=%.2f, portfolio=%.2f, cash=%.2f)",
                signal.symbol,
                signal.price,
                self._current_portfolio_value,
                self._current_cash,
            )
            return None

        proposed_value = quantity * signal.price

        # 2. Cash buffer check
        cash_buffer = (self._config.cash_buffer_pct / 100.0) * self._current_portfolio_value
        remaining_cash = self._current_cash - proposed_value
        if remaining_cash < cash_buffer:
            logger.warning(
                "Trade REJECTED for %s: cash buffer violation — "
                "remaining cash %.2f < required buffer %.2f "
                "(%.1f%% of portfolio %.2f), proposed value=%.2f",
                signal.symbol,
                remaining_cash,
                cash_buffer,
                self._config.cash_buffer_pct,
                self._current_portfolio_value,
                proposed_value,
            )
            return None

        # 3. Position size check
        max_position_value = (self._config.max_position_size_pct / 100.0) * self._current_portfolio_value
        if proposed_value > max_position_value:
            logger.warning(
                "Trade REJECTED for %s: position size violation — "
                "proposed value %.2f > max allowed %.2f "
                "(%.1f%% of portfolio %.2f)",
                signal.symbol,
                proposed_value,
                max_position_value,
                self._config.max_position_size_pct,
                self._current_portfolio_value,
            )
            return None

        # 4. Stop-loss readiness — calculate stop-loss price
        stop_loss_price = signal.price * (1.0 - self._config.stop_loss_pct / 100.0)

        logger.info(
            "Trade APPROVED for %s: %s %d shares @ %.2f, "
            "stop-loss=%.2f, value=%.2f",
            signal.symbol,
            signal.direction,
            quantity,
            signal.price,
            stop_loss_price,
            proposed_value,
        )

        return ApprovedTrade(
            signal=signal,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
            max_position_value=max_position_value,
        )

    async def check_portfolio_loss(self) -> None:
        """Check portfolio loss and trigger hard stop if threshold exceeded.

        Called every minute during market hours (Requirement 9.1).
        """
        if self._initial_portfolio_value is None or self._initial_portfolio_value <= 0:
            return

        loss_pct = (
            (self._initial_portfolio_value - self._current_portfolio_value)
            / self._initial_portfolio_value
            * 100.0
        )

        if loss_pct >= self._config.max_portfolio_loss_pct:
            logger.error(
                "Portfolio loss %.2f%% >= threshold %.1f%% — triggering hard stop",
                loss_pct,
                self._config.max_portfolio_loss_pct,
            )
            await self.trigger_hard_stop()

    async def trigger_hard_stop(self) -> None:
        """Activate the portfolio hard stop.

        Sets the hard stop flag, disabling all future trade execution.
        Actual position closing via IB market orders will be wired later.

        Requirements: 9.2, 9.3, 9.4
        """
        self._hard_stop_active = True
        logger.error(
            "HARD STOP ACTIVATED — all trading halted. "
            "Portfolio value: %.2f, initial: %s. "
            "Agent must be manually restarted to resume trading.",
            self._current_portfolio_value,
            f"{self._initial_portfolio_value:.2f}" if self._initial_portfolio_value else "N/A",
        )
        # IB order closing and alert email will be wired later

    def place_stop_loss(self, symbol: str, entry_price: float, quantity: int) -> float:
        """Calculate the stop-loss price for a new position.

        Returns the stop-loss price = entry_price × (1 − STOP_LOSS_PCT/100).
        Actual IB stop-loss order placement will be wired later.

        Requirement: 10.1
        """
        stop_loss_price = entry_price * (1.0 - self._config.stop_loss_pct / 100.0)
        logger.info(
            "Stop-loss calculated for %s: entry=%.2f, stop=%.2f "
            "(%.1f%% below entry), quantity=%d",
            symbol,
            entry_price,
            stop_loss_price,
            self._config.stop_loss_pct,
            quantity,
        )
        return stop_loss_price

    def upgrade_to_trailing_stop(self, symbol: str, current_price: float) -> bool:
        """Check if a position qualifies for trailing stop conversion.

        Returns True if unrealized gain exceeds 3%, indicating the fixed
        stop-loss should be converted to a trailing stop.

        Requirement: 11.1
        """
        position = self._open_positions.get(symbol)
        if position is None:
            logger.warning(
                "Cannot evaluate trailing stop for %s: no open position found",
                symbol,
            )
            return False

        entry_price = position["entry_price"]
        if entry_price <= 0:
            return False

        unrealized_gain_pct = (current_price - entry_price) / entry_price

        if unrealized_gain_pct > 0.03:
            logger.info(
                "Trailing stop eligible for %s: unrealized gain %.2f%% > 3%% "
                "(entry=%.2f, current=%.2f)",
                symbol,
                unrealized_gain_pct * 100.0,
                entry_price,
                current_price,
            )
            return True

        return False

    def calculate_position_size(self, price: float) -> int:
        """Calculate the maximum number of shares to buy.

        max_shares = min(
            MAX_POSITION_SIZE_PCT/100 × portfolio_value / price,
            (cash − CASH_BUFFER_PCT/100 × portfolio_value) / price
        )
        Returns max(0, floor(max_shares)).

        Requirement: 12.1, 12.2
        """
        if price <= 0 or self._current_portfolio_value <= 0:
            return 0

        max_by_position = (
            self._config.max_position_size_pct / 100.0
            * self._current_portfolio_value
            / price
        )

        cash_buffer = self._config.cash_buffer_pct / 100.0 * self._current_portfolio_value
        available_cash = self._current_cash - cash_buffer
        if available_cash <= 0:
            return 0

        max_by_cash = available_cash / price

        max_shares = min(max_by_position, max_by_cash)
        return max(0, math.floor(max_shares))

    def update_portfolio(self, total_value: float, cash: float) -> None:
        """Update current portfolio value and cash balance.

        Sets the initial portfolio value on first call.
        """
        self._current_portfolio_value = total_value
        self._current_cash = cash

        if self._initial_portfolio_value is None:
            self._initial_portfolio_value = total_value
            logger.info(
                "Initial portfolio value set to %.2f", total_value
            )

    def update_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        current_price: float,
        stop_loss_price: float,
    ) -> None:
        """Update or add a position in the open positions tracker."""
        self._open_positions[symbol] = {
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss_price": stop_loss_price,
        }

    def remove_position(self, symbol: str) -> None:
        """Remove a position from the open positions tracker."""
        if symbol in self._open_positions:
            del self._open_positions[symbol]
            logger.info("Position removed for %s", symbol)

    @property
    def is_hard_stop_active(self) -> bool:
        """Whether the portfolio hard stop is currently active."""
        return self._hard_stop_active
