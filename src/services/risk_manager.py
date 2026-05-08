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

    # Maximum total portfolio exposure (no margin usage)
    _MAX_TOTAL_EXPOSURE_PCT = 90.0
    # Minimum seconds between trades on the same symbol
    _TRADE_COOLDOWN_SECONDS = 60

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

        # Trade cooldown: symbol -> last trade timestamp
        self._last_trade_time: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_signal(self, signal: TradeSignal) -> Optional[ApprovedTrade]:
        """Validate a trade signal against risk limits in strict order.

        Risk checks:
        1. Hard stop — reject if active
        2. Cooldown — reject if traded this symbol recently
        3. Existing position — reject BUY if already holding, reject SELL if not holding
        4. No short selling — SELL only closes existing long positions
        5. Total exposure — reject if total positions > 90% of portfolio
        6. Cash buffer — reject if cash would drop below 10%
        7. Position size — reject if single position > 25% of portfolio
        8. Stop-loss readiness
        """
        import time as _time

        # 1. Hard stop
        if self._hard_stop_active:
            logger.warning("Trade REJECTED for %s: hard stop active", signal.symbol)
            return None

        # 2. Cooldown check
        last_trade = self._last_trade_time.get(signal.symbol, 0)
        if _time.time() - last_trade < self._TRADE_COOLDOWN_SECONDS:
            logger.info(
                "Trade REJECTED for %s: cooldown active (%.0fs remaining)",
                signal.symbol,
                self._TRADE_COOLDOWN_SECONDS - (_time.time() - last_trade),
            )
            return None

        # 3. Existing position check
        existing = self._open_positions.get(signal.symbol)

        if signal.direction == "BUY":
            # Reject BUY if we already hold this symbol
            if existing and existing.get("quantity", 0) > 0:
                logger.info(
                    "Trade REJECTED for %s: already holding %d shares",
                    signal.symbol,
                    existing["quantity"],
                )
                return None
        elif signal.direction == "SELL":
            # SELL only closes existing long positions — no short selling
            if not existing or existing.get("quantity", 0) <= 0:
                logger.info(
                    "Trade REJECTED for %s: no long position to sell (no short selling)",
                    signal.symbol,
                )
                return None
            # Close the existing position
            quantity = existing["quantity"]
            stop_loss_price = 0.0  # no stop-loss needed for closing
            self._last_trade_time[signal.symbol] = _time.time()
            logger.info(
                "Trade APPROVED for %s: SELL (close) %d shares @ %.2f",
                signal.symbol, quantity, signal.price,
            )
            return ApprovedTrade(
                signal=signal,
                quantity=quantity,
                stop_loss_price=stop_loss_price,
                max_position_value=quantity * signal.price,
            )

        # --- BUY path continues ---

        # Calculate proposed quantity — use volatility-adjusted sizing if ATR available
        atr_value = signal.indicators.get("atr") if signal.indicators else None
        if atr_value is not None and atr_value > 0:
            quantity = self.calculate_volatility_adjusted_size(signal.price, atr_value)
            logger.info(
                "Using volatility-adjusted sizing for %s: ATR=%.2f, shares=%d",
                signal.symbol, atr_value, quantity,
            )
        else:
            quantity = self.calculate_position_size(signal.price)
        if quantity <= 0:
            logger.warning(
                "Trade REJECTED for %s: position size is 0 (price=%.2f, portfolio=%.2f, cash=%.2f)",
                signal.symbol, signal.price, self._current_portfolio_value, self._current_cash,
            )
            return None

        proposed_value = quantity * signal.price

        # 5. Total exposure check — reduce quantity to fit within limit
        total_positions_value = sum(
            abs(p.get("quantity", 0) * p.get("current_price", p.get("entry_price", 0)))
            for p in self._open_positions.values()
        )
        max_total_exposure = (self._MAX_TOTAL_EXPOSURE_PCT / 100.0) * self._current_portfolio_value
        available_exposure = max_total_exposure - total_positions_value
        if available_exposure <= 0:
            logger.warning(
                "Trade REJECTED for %s: portfolio fully allocated (exposure %.2f >= max %.2f)",
                signal.symbol, total_positions_value, max_total_exposure,
            )
            return None
        if proposed_value > available_exposure:
            # Reduce quantity to fit within remaining exposure
            reduced_quantity = math.floor(available_exposure / signal.price)
            if reduced_quantity <= 0:
                logger.warning(
                    "Trade REJECTED for %s: insufficient exposure headroom (available %.2f, price %.2f)",
                    signal.symbol, available_exposure, signal.price,
                )
                return None
            logger.info(
                "Reducing position for %s: %d → %d shares to fit exposure limit "
                "(available %.2f of max %.2f)",
                signal.symbol, quantity, reduced_quantity,
                available_exposure, max_total_exposure,
            )
            quantity = reduced_quantity
            proposed_value = quantity * signal.price

        # 6. Cash buffer check — reduce quantity to preserve buffer
        cash_buffer = (self._config.cash_buffer_pct / 100.0) * self._current_portfolio_value
        available_cash_for_trade = self._current_cash - cash_buffer
        if available_cash_for_trade <= 0:
            logger.warning(
                "Trade REJECTED for %s: no cash available above buffer (cash %.2f, buffer %.2f)",
                signal.symbol, self._current_cash, cash_buffer,
            )
            return None
        if proposed_value > available_cash_for_trade:
            reduced_quantity = math.floor(available_cash_for_trade / signal.price)
            if reduced_quantity <= 0:
                logger.warning(
                    "Trade REJECTED for %s: insufficient cash above buffer (available %.2f, price %.2f)",
                    signal.symbol, available_cash_for_trade, signal.price,
                )
                return None
            logger.info(
                "Reducing position for %s: %d → %d shares to preserve cash buffer",
                signal.symbol, quantity, reduced_quantity,
            )
            quantity = reduced_quantity
            proposed_value = quantity * signal.price

        # 7. Position size check
        max_position_value = (self._config.max_position_size_pct / 100.0) * self._current_portfolio_value
        if proposed_value > max_position_value:
            logger.warning(
                "Trade REJECTED for %s: position size %.2f > max %.2f",
                signal.symbol, proposed_value, max_position_value,
            )
            return None

        # 8. Stop-loss readiness
        stop_loss_price = signal.price * (1.0 - self._config.stop_loss_pct / 100.0)

        # Record cooldown
        self._last_trade_time[signal.symbol] = _time.time()

        logger.info(
            "Trade APPROVED for %s: BUY %d shares @ %.2f, stop=%.2f, value=%.2f",
            signal.symbol, quantity, signal.price, stop_loss_price, proposed_value,
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
        import time as _time
        existing = self._open_positions.get(symbol)
        self._open_positions[symbol] = {
            "quantity": quantity,
            "entry_price": entry_price,
            "current_price": current_price,
            "stop_loss_price": stop_loss_price,
            "entry_time": existing["entry_time"] if existing and "entry_time" in existing else _time.time(),
        }

    def remove_position(self, symbol: str) -> None:
        """Remove a position from the open positions tracker."""
        if symbol in self._open_positions:
            del self._open_positions[symbol]
            logger.info("Position removed for %s", symbol)

    def get_positions_to_exit(self) -> list:
        """Check all positions for exit conditions.

        Returns a list of (symbol, reason) tuples for positions that
        should be closed based on:
        1. Age: position held longer than MAX_POSITION_AGE_DAYS
        2. Profit target: unrealized gain >= PROFIT_TARGET_PCT
        3. Loss exit: unrealized loss >= MAX_LOSS_EXIT_PCT (early exit before stop-loss)
        """
        import time as _time

        exits = []
        now = _time.time()
        max_age_seconds = self._config.max_position_age_days * 86400

        for symbol, pos in self._open_positions.items():
            entry_price = pos.get("entry_price", 0)
            current_price = pos.get("current_price", entry_price)
            entry_time = pos.get("entry_time", now)

            if entry_price <= 0:
                continue

            # 1. Age check
            age_seconds = now - entry_time
            if age_seconds > max_age_seconds:
                age_days = age_seconds / 86400
                exits.append((symbol, f"age={age_days:.1f}d > max={self._config.max_position_age_days}d"))
                continue

            # 2. Profit target
            pnl_pct = (current_price - entry_price) / entry_price * 100
            if pnl_pct >= self._config.profit_target_pct:
                exits.append((symbol, f"profit={pnl_pct:.1f}% >= target={self._config.profit_target_pct}%"))
                continue

            # 3. Loss exit (early exit before stop-loss triggers)
            if pnl_pct <= -self._config.max_loss_exit_pct:
                exits.append((symbol, f"loss={pnl_pct:.1f}% <= max_loss=-{self._config.max_loss_exit_pct}%"))

        return exits

    def calculate_volatility_adjusted_size(self, price: float, atr: float) -> int:
        """Calculate position size adjusted for volatility using ATR.

        Risk per trade = 1% of portfolio value.
        Shares = risk_amount / (2 × ATR), then capped by existing
        position size and cash buffer limits.

        Parameters
        ----------
        price : float
            Current price of the instrument.
        atr : float
            Current Average True Range value.

        Returns
        -------
        int
            Number of shares to trade (≥ 0).
        """
        if price <= 0 or atr <= 0 or self._current_portfolio_value <= 0:
            return 0

        # Risk 1% of portfolio per trade
        risk_amount = 0.01 * self._current_portfolio_value
        # Shares based on volatility: risk_amount / (2 * ATR)
        vol_shares = risk_amount / (2.0 * atr)

        # Cap by existing position size and cash buffer limits
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

        max_shares = min(vol_shares, max_by_position, max_by_cash)
        return max(0, math.floor(max_shares))

    def check_circuit_breaker(
        self, symbol: str, price: float, prev_price: float
    ) -> bool:
        """Return True if trade should be blocked due to anomalous price movement.

        Blocks if price moved more than 10% in a single tick (flash crash
        protection).

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        price : float
            Current tick price.
        prev_price : float
            Previous tick price.

        Returns
        -------
        bool
            ``True`` if the trade should be blocked.
        """
        if prev_price <= 0:
            return False

        change_pct = abs(price - prev_price) / prev_price * 100.0
        if change_pct > 10.0:
            logger.warning(
                "Circuit breaker triggered for %s: price moved %.2f%% "
                "in a single tick (%.2f → %.2f)",
                symbol, change_pct, prev_price, price,
            )
            return True

        return False

    @property
    def is_hard_stop_active(self) -> bool:
        """Whether the portfolio hard stop is currently active."""
        return self._hard_stop_active
