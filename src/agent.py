"""TradingAgent: main orchestrator that wires all components together.

Initializes all services with proper dependency injection, sets up the
APScheduler for periodic tasks, wires IB event callbacks, and runs the
ib_insync event loop as the main asyncio loop.

Requirements: 1.1, 2.1, 2.3, 3.3, 3.4, 4.1, 13.1, 14.4, 18.1, 23.1, 23.2, 23.4, 24.8
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Tuple

import nest_asyncio
import numpy as np

# Allow nested event loops — required for ib_insync's sleep() inside asyncio.run()
nest_asyncio.apply()

from ib_insync import util as ib_util
# Patch asyncio to work with ib_insync
ib_util.patchAsyncio()
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.config import AgentConfig
from src.logging_config import setup_logging
from src.models.domain import AgentState, PortfolioSnapshot
from src.services.connection_manager import ConnectionManager
from src.services.market_data_service import MarketDataService
from src.services.market_hours_service import MarketHoursService
from src.services.market_screener import MarketScreener
from src.services.order_executor import OrderExecutor
from src.services.polymarket_service import PolymarketService
from src.services.report_generator import ReportGenerator
from src.services.risk_manager import RiskManager
from src.services.shutdown_handler import ShutdownHandler
from src.services.state_manager import StateManager
from src.services.strategy_engine import StrategyEngine
from src.services.watchlist_manager import WatchlistManager

logger = logging.getLogger(__name__)

# Fallback watchlist — used if screener fails
_FALLBACK_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "V", "JNJ",
]


class TradingAgent:
    """Main trading agent that wires all components together.

    Manages the full lifecycle: initialization, startup, event loop,
    tick processing, scheduled tasks, and shutdown.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._initialized = False
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._latency_samples: list = []  # signal-to-order latency in ms

        # Core services — initialized in constructor with dependency injection
        self._connection_manager = ConnectionManager(config)
        self._state_manager = StateManager(config)

        # IB instance from connection manager
        ib = self._connection_manager.ib

        # Market services
        self._market_hours = MarketHoursService(ib)
        self._polymarket = PolymarketService()

        # Strategy engine — uses market hours for filtering, polymarket for sentiment
        self._strategy_engine = StrategyEngine(
            market_hours=self._market_hours,
            polymarket_sentiment=self._polymarket.sentiment_score,
            earnings_blackout_symbols=set(),
            market_data_type=config.market_data_type,
        )

        # Risk management
        self._risk_manager = RiskManager(config, self._state_manager, ib)

        # Order execution
        self._order_executor = OrderExecutor(ib, self._risk_manager, self._state_manager)

        # Watchlist manager — filters candidates by volume and earnings
        self._watchlist_manager = WatchlistManager(ib)

        # Market screener — scans S&P 500 daily for top candidates
        self._screener = MarketScreener(top_n=30)

        # Market data — subscribes to watchlist symbols (placeholder, updated after screening)
        self._market_data = MarketDataService(ib, _FALLBACK_WATCHLIST, market_data_type=config.market_data_type)

        # Reporting
        self._report_generator = ReportGenerator(config, self._state_manager)

        # Shutdown handler
        self._shutdown_handler = ShutdownHandler(
            self._order_executor, self._state_manager, self._connection_manager
        )

    # Minimum signal confidence to execute a trade
    _MIN_CONFIDENCE = 0.65

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Full startup + blocking event loop. Use initialize() + ib.run() for more control."""
        await self.initialize()
        # Blocking event loop — processes IB events
        logger.info("TradingAgent fully started — entering event loop")
        try:
            while True:
                await asyncio.sleep(0.1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Agent loop interrupted")

    async def initialize(self) -> None:
        """Execute the full startup sequence (non-blocking, idempotent).

        Safe to call multiple times — skips already-completed steps.
        Used for initial startup and reconnection after crashes.

        Steps:
        a. Setup logging
        b. Initialize StateManager (create tables)
        c. Connect to IB via ConnectionManager
        d. Check for crash recovery (load_last_state, reconcile_with_ib)
        e. Update market hours schedule
        f. Subscribe to market data
        g. Setup APScheduler with periodic tasks
        h. Wire IB event callbacks
        i. Setup signal handlers via ShutdownHandler
        j. Persist RUNNING agent state
        k. Start the event loop (ib.run())
        """
        # a. Setup logging (idempotent)
        setup_logging()

        if self._initialized:
            # Reconnection path: just reconnect IB and re-sync
            logger.info("Re-initializing after disconnect...")
            if not self._connection_manager.is_connected():
                await self._connection_manager.connect()
                self._connection_manager.ib.sleep(3)
                logger.info("Reconnected to IB")
            return

        logger.info("TradingAgent starting — environment=%s", self._config.environment)

        # b. Initialize StateManager
        await self._state_manager.initialize()
        logger.info("StateManager initialized")

        # c. Connect to IB
        await self._connection_manager.connect()
        # Wait for IB to sync account data (positions, account values)
        self._connection_manager.ib.sleep(3)
        logger.info("Connected to IB (account data synced)")

        # d. Crash recovery
        last_state = await self._state_manager.load_last_state()
        if last_state is not None:
            logger.info(
                "Previous agent state found: %s (crash_count=%d)",
                last_state.state,
                last_state.crash_count,
            )
            ib = self._connection_manager.ib
            discrepancies = await self._state_manager.reconcile_with_ib(ib)
            if discrepancies:
                logger.warning(
                    "State reconciliation found %d discrepancies", len(discrepancies)
                )
        else:
            logger.info("No previous agent state — fresh start")

        # e. Update market hours schedule
        await self._market_hours.update_schedule()
        logger.info("Market hours schedule updated")

        # e1. Initialize portfolio from IB account values
        total_value, cash = self._read_portfolio_from_ib()
        if total_value > 0:
            self._risk_manager.update_portfolio(total_value, cash)
            logger.info(
                "Portfolio initialized from IB: value=%.2f, cash=%.2f",
                total_value, cash,
            )
        else:
            logger.warning("Could not load portfolio from IB — setting default 1M")
            self._risk_manager.update_portfolio(1_000_000.0, 1_000_000.0)

        # e2. Run market screener to select today's candidates
        logger.info("Running market screener...")
        try:
            candidates = self._screener.screen()
            if not candidates:
                candidates = _FALLBACK_WATCHLIST
                logger.warning("Screener returned no candidates — using fallback")
        except Exception as exc:
            logger.warning("Screener failed: %s — using fallback", exc)
            candidates = _FALLBACK_WATCHLIST

        # e3. Build filtered watchlist and check earnings blackout
        watchlist = await self._watchlist_manager.build_watchlist(candidates)
        blackout = await self._watchlist_manager.update_earnings_blackout(watchlist)
        self._market_data._watchlist = watchlist
        self._strategy_engine.earnings_blackout_symbols = blackout
        logger.info(
            "Watchlist ready: %d symbols, %d in earnings blackout",
            len(watchlist),
            len(blackout),
        )

        # f. Subscribe to market data and wire tick callback
        self._market_data.set_tick_callback(self._on_tick)
        await self._market_data.subscribe_all()
        logger.info("Market data subscriptions active")

        # g. Setup APScheduler
        self._setup_scheduler()
        logger.info("Scheduler configured and started")

        # h. Wire IB event callbacks
        self._wire_ib_events()
        logger.info("IB event callbacks wired")

        # i. Setup signal handlers
        loop = asyncio.get_running_loop()
        self._shutdown_handler.setup_signal_handlers(loop)
        logger.info("Signal handlers registered")

        # j. Persist RUNNING agent state
        now = datetime.now()
        agent_state = AgentState(
            state="RUNNING",
            initial_portfolio_value=None,
            start_time=now,
            last_heartbeat=now,
            crash_count=last_state.crash_count if last_state else 0,
        )
        await self._state_manager.persist_agent_state(agent_state)
        logger.info("Agent state persisted as RUNNING")
        logger.info("TradingAgent initialization complete")
        self._initialized = True

    async def stop(self) -> None:
        """Delegate shutdown to the ShutdownHandler."""
        logger.info("TradingAgent stop requested")
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler shut down")
        await self._polymarket.close()
        await self._shutdown_handler.shutdown()

    # ------------------------------------------------------------------
    # IB account helpers
    # ------------------------------------------------------------------

    def _read_portfolio_from_ib(self) -> Tuple[float, float]:
        """Read portfolio total value and cash from IB account values.

        Uses a multi-fallback chain to handle different IB account
        currency reporting (BASE → USD → any):
        1. Try ``currency == "BASE"`` first (paper accounts)
        2. Fallback to ``currency == "USD"``
        3. Fallback to any available currency

        Returns ``(total_value, cash)``. Both are 0.0 if unavailable.
        """
        ib = self._connection_manager.ib
        account_values = ib.accountValues()

        total_value = 0.0
        cash = 0.0

        # Pass 1: prefer BASE currency
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "BASE":
                total_value = float(av.value)
            elif av.tag == "TotalCashBalance" and av.currency == "BASE":
                cash = float(av.value)

        # Pass 2: fallback to USD
        if total_value == 0:
            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    total_value = float(av.value)
                elif av.tag == "CashBalance" and av.currency == "USD":
                    cash = float(av.value)

        # Pass 3: fallback to any currency
        if total_value == 0:
            for av in account_values:
                if av.tag == "NetLiquidation":
                    try:
                        total_value = float(av.value)
                    except (ValueError, TypeError):
                        continue
                    break

        # If we found total_value but no cash, assume all cash
        if total_value > 0 and cash == 0:
            cash = total_value

        return total_value, cash

    # ------------------------------------------------------------------
    # Tick processing
    # ------------------------------------------------------------------

    def _on_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        prices: np.ndarray,
        volumes: np.ndarray,
        avg_volume: float,
    ) -> None:
        """Tick callback from MarketDataService.

        Processes the tick through the signal pipeline:
        1. StrategyEngine.process_tick() → optional TradeSignal
        2. RiskManager.evaluate_signal() → optional ApprovedTrade
        3. OrderExecutor.execute_trade() → submit order
        4. Update RiskManager portfolio state

        Runs the async pipeline on the current event loop.
        """
        # Update polymarket sentiment on the strategy engine before processing
        self._strategy_engine.polymarket_sentiment = self._polymarket.sentiment_score

        # Update current price for existing positions
        if symbol in self._risk_manager._open_positions:
            self._risk_manager._open_positions[symbol]["current_price"] = price

        # Schedule the async tick processing on the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._process_tick_async(symbol, price, volume, prices, volumes, avg_volume)
            )
        except RuntimeError:
            logger.warning("No running event loop — tick for %s dropped", symbol)

    async def _process_tick_async(
        self,
        symbol: str,
        price: float,
        volume: float,
        prices: np.ndarray,
        volumes: np.ndarray,
        avg_volume: float,
    ) -> None:
        """Async tick processing pipeline with latency profiling."""
        import time as _time

        tick_start = _time.perf_counter()
        try:
            # 1. Generate signal from strategy engine
            signal = await self._strategy_engine.process_tick(
                symbol=symbol,
                price=price,
                volume=volume,
                prices=prices,
                volumes=volumes,
                avg_daily_volume=avg_volume,
            )

            if signal is None:
                return

            # 1b. Minimum confidence filter
            if signal.confidence < self._MIN_CONFIDENCE:
                logger.debug(
                    "Signal dropped for %s: confidence %.2f < minimum %.2f",
                    signal.symbol, signal.confidence, self._MIN_CONFIDENCE,
                )
                return

            # 1c. Position-aware SELL filter — skip SELL if not holding
            if signal.direction == "SELL":
                pos = self._risk_manager._open_positions.get(signal.symbol)
                if not pos or pos.get("quantity", 0) <= 0:
                    logger.debug(
                        "SELL signal dropped for %s: no open position",
                        signal.symbol,
                    )
                    return

            logger.info(
                "Signal generated: %s %s (strategy=%s, confidence=%.2f)",
                signal.direction,
                signal.symbol,
                signal.strategy,
                signal.confidence,
            )

            # 2. Evaluate signal against risk limits
            approved = self._risk_manager.evaluate_signal(signal)
            if approved is None:
                return

            # 3. Execute the trade
            await self._order_executor.execute_trade(approved)

            # 3b. Update position tracking in RiskManager
            if signal.direction == "BUY":
                self._risk_manager.update_position(
                    signal.symbol, approved.quantity, signal.price,
                    signal.price, approved.stop_loss_price,
                )
            elif signal.direction == "SELL":
                self._risk_manager.remove_position(signal.symbol)

            # Measure signal-to-order latency
            latency_ms = (_time.perf_counter() - tick_start) * 1000
            self._latency_samples.append(latency_ms)
            if latency_ms > 100:
                logger.warning(
                    "Signal-to-order latency %.1fms > 100ms target for %s",
                    latency_ms,
                    symbol,
                )
            else:
                logger.debug(
                    "Signal-to-order latency %.1fms for %s", latency_ms, symbol
                )

            # 4. Update risk manager portfolio state
            total_value, cash = self._read_portfolio_from_ib()
            if total_value > 0:
                self._risk_manager.update_portfolio(total_value, cash)

        except Exception:
            logger.exception("Error processing tick for %s", symbol)

    # ------------------------------------------------------------------
    # Scheduled tasks
    # ------------------------------------------------------------------

    def _setup_scheduler(self) -> None:
        """Configure and start APScheduler with all periodic tasks."""
        self._scheduler = AsyncIOScheduler()

        # Polymarket updates every 15 minutes
        self._scheduler.add_job(
            self._polymarket.update,
            IntervalTrigger(minutes=15),
            id="polymarket_update",
            name="Polymarket sentiment update",
            replace_existing=True,
        )

        # Portfolio loss checks every 1 minute during market hours
        self._scheduler.add_job(
            self._check_portfolio_loss,
            IntervalTrigger(minutes=1),
            id="portfolio_loss_check",
            name="Portfolio loss check",
            replace_existing=True,
        )

        # Portfolio snapshots every 5 minutes during market hours
        self._scheduler.add_job(
            self._take_portfolio_snapshot,
            IntervalTrigger(minutes=5),
            id="portfolio_snapshot",
            name="Portfolio snapshot",
            replace_existing=True,
        )

        # Daily report at 18:00 ET
        self._scheduler.add_job(
            self._generate_daily_report,
            CronTrigger(hour=18, minute=0, timezone="America/New_York"),
            id="daily_report",
            name="Daily report generation",
            replace_existing=True,
        )

        # Weekly performance report
        self._scheduler.add_job(
            self._generate_weekly_performance_report,
            CronTrigger(day_of_week="mon", hour=8, minute=0, timezone="America/New_York"),
            id="weekly_performance",
            name="Weekly performance report",
            replace_existing=True,
        )

        # Daily market screening at 9:00 ET (30 min before market open)
        self._scheduler.add_job(
            self._run_daily_screening,
            CronTrigger(hour=9, minute=0, timezone="America/New_York"),
            id="daily_screening",
            name="Daily market screening",
            replace_existing=True,
        )

        self._scheduler.start()

    async def _check_portfolio_loss(self) -> None:
        """Check portfolio loss — only during market hours."""
        if not self._market_hours.is_market_open():
            return

        # Update portfolio values from IB before checking
        total_value, cash = self._read_portfolio_from_ib()
        if total_value > 0:
            self._risk_manager.update_portfolio(total_value, cash)

        await self._risk_manager.check_portfolio_loss()

    async def _take_portfolio_snapshot(self) -> None:
        """Take a portfolio snapshot — only during market hours."""
        if not self._market_hours.is_market_open():
            return

        total_value, cash = self._read_portfolio_from_ib()
        if total_value <= 0:
            logger.debug("Portfolio snapshot skipped — could not read IB values")
            return

        ib = self._connection_manager.ib
        positions_value = total_value - cash
        num_positions = len(ib.positions())

        # Compute P&L from initial portfolio value
        initial = self._risk_manager._initial_portfolio_value
        total_pnl = 0.0
        total_pnl_pct = 0.0
        if initial and initial > 0:
            total_pnl = total_value - initial
            total_pnl_pct = (total_pnl / initial) * 100.0

        # Update risk manager with fresh values
        self._risk_manager.update_portfolio(total_value, cash)

        snapshot = PortfolioSnapshot(
            total_value=total_value,
            cash_balance=cash,
            positions_value=positions_value,
            daily_pnl=total_pnl,  # Approximation: total P&L since start
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            num_open_positions=num_positions,
            hard_stop_active=self._risk_manager.is_hard_stop_active,
            snapshot_time=datetime.now(),
        )

        await self._state_manager.persist_portfolio_snapshot(snapshot)
        logger.info(
            "Portfolio snapshot: value=%.2f, cash=%.2f, positions=%d, pnl=%.2f (%.2f%%)",
            total_value, cash, num_positions, total_pnl, total_pnl_pct,
        )

        # Export portfolio status as JSON for the dashboard
        self._export_portfolio_json(ib)

    def _export_portfolio_json(self, ib) -> None:
        """Write portfolio status to logs/portfolio.json for the dashboard."""
        import json as _json

        try:
            positions = []
            for item in ib.portfolio():
                positions.append({
                    "symbol": item.contract.symbol,
                    "quantity": int(item.position),
                    "avgCost": round(item.averageCost, 2),
                    "marketPrice": round(item.marketPrice, 2),
                    "marketValue": round(item.marketValue, 0),
                    "unrealizedPnL": round(item.unrealizedPNL, 2),
                    "realizedPnL": round(item.realizedPNL, 2),
                    "type": "LONG" if item.position > 0 else "SHORT",
                })

            account = {}
            wanted_tags = {
                "NetLiquidation", "TotalCashBalance",
                "UnrealizedPnL", "RealizedPnL", "GrossPositionValue",
            }
            # First pass: prefer BASE currency
            for av in ib.accountValues():
                if av.currency == "BASE" and av.tag in wanted_tags:
                    account[av.tag] = float(av.value)
            # Second pass: fill missing tags from USD
            for av in ib.accountValues():
                if av.tag in wanted_tags and av.tag not in account:
                    if av.currency == "USD":
                        account[av.tag] = float(av.value)
            # Fallback: fill missing tags from any currency
            for av in ib.accountValues():
                if av.tag in wanted_tags and av.tag not in account:
                    try:
                        account[av.tag] = float(av.value)
                    except (ValueError, TypeError):
                        pass

            fills_today = len(ib.fills())

            data = {
                "timestamp": datetime.now().isoformat(),
                "account": account,
                "positions": positions,
                "fillsToday": fills_today,
                "hardStopActive": self._risk_manager.is_hard_stop_active,
                "openPositionCount": len(positions),
            }

            with open("logs/portfolio.json", "w") as f:
                _json.dump(data, f, indent=2)

        except Exception as exc:
            logger.debug("Failed to export portfolio JSON: %s", exc)

    async def _generate_daily_report(self) -> None:
        """Generate and send the daily report.

        Fetches the latest portfolio snapshot and today's trades,
        then generates and sends the HTML report.
        """
        logger.info("Generating daily report")

        try:
            snapshot = await self._state_manager.get_latest_portfolio_snapshot()
            if snapshot is None:
                logger.warning("No portfolio snapshot available — skipping daily report")
                return

            trades = await self._state_manager.get_trades_for_date(datetime.now())

            # Build open positions from IB
            ib = self._connection_manager.ib
            open_positions = []
            for pos in ib.positions():
                open_positions.append({
                    "symbol": pos.contract.symbol,
                    "quantity": int(pos.position),
                    "avg_cost": pos.avgCost,
                    "current_price": 0.0,  # Would need market data lookup
                    "unrealized_pnl": 0.0,
                })

            html = await self._report_generator.generate_report(
                portfolio=snapshot,
                trades=trades,
                open_positions=open_positions,
                polymarket_sentiment=self._polymarket.sentiment_score,
            )

            await self._report_generator.send_report(html)
            logger.info("Daily report generated and sent")

        except Exception:
            logger.exception("Failed to generate daily report")

    async def _run_daily_screening(self) -> None:
        """Run market screener and update the watchlist for today.

        Called daily at 9:00 ET (30 min before market open).
        Scans S&P 500, selects top candidates, updates MarketDataService.
        """
        logger.info("Running daily market screening...")
        try:
            candidates = self._screener.screen()
            if candidates:
                self._market_data._watchlist = candidates
                logger.info("Watchlist updated: %d candidates for today", len(candidates))
                logger.info("Top 5: %s", candidates[:5])
            else:
                logger.warning("Screening returned no candidates — keeping current watchlist")
        except Exception:
            logger.exception("Daily screening failed — keeping current watchlist")

    async def _generate_weekly_performance_report(self) -> None:
        """Generate weekly performance report with latency statistics."""
        if not self._latency_samples:
            logger.info("Weekly performance report — no latency data collected")
            return

        samples = self._latency_samples
        avg_latency = sum(samples) / len(samples)
        min_latency = min(samples)
        max_latency = max(samples)
        over_100ms = sum(1 for s in samples if s > 100)

        logger.info(
            "Weekly performance report — signals processed: %d, "
            "latency avg=%.1fms min=%.1fms max=%.1fms, "
            "over 100ms target: %d (%.1f%%)",
            len(samples),
            avg_latency,
            min_latency,
            max_latency,
            over_100ms,
            (over_100ms / len(samples)) * 100 if samples else 0,
        )

        # Reset samples for next week
        self._latency_samples.clear()

    # ------------------------------------------------------------------
    # IB event wiring
    # ------------------------------------------------------------------

    def _wire_ib_events(self) -> None:
        """Wire IB event callbacks to the appropriate service handlers.

        - pendingTickersEvent → MarketDataService.on_pending_tickers
        - execDetailsEvent → OrderExecutor._on_exec_details
        - orderStatusEvent → OrderExecutor._on_order_status
        - disconnectedEvent → ConnectionManager._on_disconnected
        """
        ib = self._connection_manager.ib

        # Note: pendingTickersEvent is already wired in MarketDataService.subscribe_all()
        # We wire the remaining events here.

        ib.execDetailsEvent += self._order_executor._on_exec_details
        ib.orderStatusEvent += self._order_executor._on_order_status
        # disconnectedEvent is already wired in ConnectionManager.connect()

        logger.info(
            "IB events wired: execDetailsEvent → OrderExecutor, "
            "orderStatusEvent → OrderExecutor"
        )
