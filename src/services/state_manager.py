"""State persistence manager using aiosqlite.

Handles all database operations: table creation, trade persistence,
portfolio snapshots, analysis state, agent state, and crash recovery.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

import aiosqlite

from src.config import AgentConfig
from src.models.domain import (
    AgentState,
    AnalysisState,
    Discrepancy,
    PortfolioSnapshot,
    TradeRecord,
)

logger = logging.getLogger(__name__)


class StateManager:
    """Persists agent state to SQLite using aiosqlite for async access."""

    def __init__(self, config: AgentConfig) -> None:
        self._db_url = config.db_url
        self._db_path = self._parse_db_path(config.db_url)
        self._db: Optional[aiosqlite.Connection] = None

    @staticmethod
    def _parse_db_path(db_url: str) -> str:
        """Extract file path from sqlite:/// URL."""
        prefix = "sqlite:///"
        if db_url.startswith(prefix):
            return db_url[len(prefix):]
        return db_url

    async def initialize(self) -> None:
        """Create all tables and enable WAL mode."""
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL')),
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity INTEGER NOT NULL,
                stop_loss_price REAL NOT NULL,
                strategy TEXT NOT NULL,
                signal_confidence REAL,
                polymarket_sentiment REAL,
                status TEXT NOT NULL DEFAULT 'OPEN'
                    CHECK(status IN ('OPEN', 'CLOSED', 'STOPPED_OUT')),
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                realized_pnl REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                positions_value REAL NOT NULL,
                daily_pnl REAL,
                total_pnl REAL,
                total_pnl_pct REAL,
                num_open_positions INTEGER NOT NULL,
                hard_stop_active INTEGER NOT NULL DEFAULT 0,
                snapshot_time TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS analysis_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watchlist_json TEXT NOT NULL,
                active_signals_json TEXT NOT NULL,
                indicator_values_json TEXT NOT NULL,
                polymarket_sentiment REAL,
                polymarket_last_fetch TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                earnings_date TEXT NOT NULL,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(symbol, earnings_date)
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state TEXT NOT NULL CHECK(state IN ('RUNNING', 'STOPPED', 'HALTED')),
                initial_portfolio_value REAL,
                start_time TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                crash_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                purpose TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                success INTEGER NOT NULL DEFAULT 1,
                error_message TEXT,
                call_date TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await self._db.commit()
        logger.info("StateManager initialized — all tables created with WAL mode")

    async def persist_trade(self, trade: TradeRecord) -> None:
        """Save a trade record to the trades table."""
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        await self._db.execute(
            """
            INSERT INTO trades (
                symbol, direction, entry_price, exit_price, quantity,
                stop_loss_price, strategy, signal_confidence,
                polymarket_sentiment, status, entry_time, exit_time,
                realized_pnl, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.symbol,
                trade.direction,
                trade.entry_price,
                trade.exit_price,
                trade.quantity,
                trade.stop_loss_price,
                trade.strategy,
                trade.signal_confidence,
                trade.polymarket_sentiment,
                trade.status,
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.realized_pnl,
                datetime.now().isoformat(),
            ),
        )
        await self._db.commit()
        logger.info("Persisted trade: %s %s %s @ %.2f", trade.direction, trade.quantity, trade.symbol, trade.entry_price)

    async def persist_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Save a portfolio snapshot to the portfolio_snapshots table."""
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        await self._db.execute(
            """
            INSERT INTO portfolio_snapshots (
                total_value, cash_balance, positions_value, daily_pnl,
                total_pnl, total_pnl_pct, num_open_positions,
                hard_stop_active, snapshot_time, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.total_value,
                snapshot.cash_balance,
                snapshot.positions_value,
                snapshot.daily_pnl,
                snapshot.total_pnl,
                snapshot.total_pnl_pct,
                snapshot.num_open_positions,
                1 if snapshot.hard_stop_active else 0,
                snapshot.snapshot_time.isoformat(),
                datetime.now().isoformat(),
            ),
        )
        await self._db.commit()
        logger.info("Persisted portfolio snapshot: value=%.2f, pnl=%.2f", snapshot.total_value, snapshot.daily_pnl)

    async def persist_analysis_state(self, state: AnalysisState) -> None:
        """Save analysis state, serializing lists/dicts to JSON."""
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        # Serialize active signals — convert TradeSignal dataclasses to dicts
        signals_data = []
        for sig in state.active_signals:
            signals_data.append({
                "symbol": sig.symbol,
                "direction": sig.direction,
                "strategy": sig.strategy,
                "confidence": sig.confidence,
                "price": sig.price,
                "volume": sig.volume,
                "indicators": sig.indicators,
                "polymarket_sentiment": sig.polymarket_sentiment,
                "timestamp": sig.timestamp.isoformat(),
            })

        await self._db.execute(
            """
            INSERT INTO analysis_state (
                watchlist_json, active_signals_json, indicator_values_json,
                polymarket_sentiment, polymarket_last_fetch, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                json.dumps(state.watchlist),
                json.dumps(signals_data),
                json.dumps(state.indicator_values),
                state.polymarket_sentiment,
                state.polymarket_last_fetch.isoformat() if state.polymarket_last_fetch else None,
                datetime.now().isoformat(),
            ),
        )
        await self._db.commit()
        logger.info("Persisted analysis state: %d watchlist symbols, %d active signals", len(state.watchlist), len(state.active_signals))

    async def load_last_state(self) -> Optional[AgentState]:
        """Load the most recent agent state for crash recovery.

        Returns None if no state exists in the database.
        """
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        cursor = await self._db.execute(
            "SELECT state, initial_portfolio_value, start_time, last_heartbeat, crash_count "
            "FROM agent_state ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            logger.info("No previous agent state found in database")
            return None

        agent_state = AgentState(
            state=row[0],
            initial_portfolio_value=row[1],
            start_time=datetime.fromisoformat(row[2]),
            last_heartbeat=datetime.fromisoformat(row[3]),
            crash_count=row[4],
        )
        logger.info("Loaded last agent state: %s (crash_count=%d)", agent_state.state, agent_state.crash_count)
        return agent_state

    async def reconcile_with_ib(self, ib: object) -> List[Discrepancy]:
        """Compare persisted state with live IB account state.

        Loads the last agent state and latest portfolio snapshot, then
        compares local open positions with live IB positions. For each
        discrepancy found, updates local state to match IB and logs at
        WARNING level. Returns empty list if IB is not connected.
        """
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        # Handle case where IB is not connected
        if ib is None or not hasattr(ib, "isConnected") or not ib.isConnected():
            logger.info("State reconciliation skipped — IB not connected")
            return []

        discrepancies: List[Discrepancy] = []

        # Load last agent state to compute offline duration
        last_state = await self.load_last_state()
        recovery_start = datetime.now()
        if last_state is not None:
            offline_duration = recovery_start - last_state.last_heartbeat
            logger.info(
                "Crash recovery started — offline duration: %s",
                offline_duration,
            )
        else:
            logger.info("Crash recovery started — no previous state found (fresh start)")

        # Build a map of local open positions from the trades table
        cursor = await self._db.execute(
            "SELECT id, symbol, quantity, entry_price, direction "
            "FROM trades WHERE status = 'OPEN'"
        )
        local_positions: dict = {}
        async for row in cursor:
            trade_id, symbol, quantity, entry_price, direction = row
            # Aggregate by symbol (net quantity: BUY positive, SELL negative)
            if symbol not in local_positions:
                local_positions[symbol] = {
                    "quantity": 0,
                    "avg_cost": 0.0,
                    "trade_ids": [],
                }
            sign = 1 if direction == "BUY" else -1
            local_positions[symbol]["quantity"] += sign * quantity
            local_positions[symbol]["avg_cost"] = entry_price
            local_positions[symbol]["trade_ids"].append(trade_id)

        # Get live IB positions
        ib_positions = ib.positions()
        ib_position_map: dict = {}
        for pos in ib_positions:
            symbol = pos.contract.symbol
            ib_position_map[symbol] = {
                "quantity": int(pos.position),
                "avg_cost": pos.avgCost,
            }

        # Compare local positions with IB positions
        all_symbols = set(local_positions.keys()) | set(ib_position_map.keys())
        for symbol in sorted(all_symbols):
            local = local_positions.get(symbol)
            ib_pos = ib_position_map.get(symbol)

            local_qty = local["quantity"] if local else 0
            ib_qty = ib_pos["quantity"] if ib_pos else 0

            local_cost = local["avg_cost"] if local else 0.0
            ib_cost = ib_pos["avg_cost"] if ib_pos else 0.0

            # Check quantity discrepancy
            if local_qty != ib_qty:
                disc = Discrepancy(
                    field=f"{symbol}.quantity",
                    local_value=local_qty,
                    ib_value=ib_qty,
                    resolution="updated_to_ib",
                )
                discrepancies.append(disc)
                logger.warning(
                    "Discrepancy: %s quantity local=%d ib=%d — updated to IB",
                    symbol,
                    local_qty,
                    ib_qty,
                )

            # Check average cost discrepancy (use tolerance for float comparison)
            if abs(local_cost - ib_cost) > 0.01:
                disc = Discrepancy(
                    field=f"{symbol}.avg_cost",
                    local_value=local_cost,
                    ib_value=ib_cost,
                    resolution="updated_to_ib",
                )
                discrepancies.append(disc)
                logger.warning(
                    "Discrepancy: %s avg_cost local=%.2f ib=%.2f — updated to IB",
                    symbol,
                    local_cost,
                    ib_cost,
                )

        # Update local state to match IB for symbols with discrepancies
        for symbol in sorted(all_symbols):
            ib_pos = ib_position_map.get(symbol)
            local = local_positions.get(symbol)

            if ib_pos is None and local is not None:
                # Position closed on IB side — mark local trades as CLOSED
                for trade_id in local["trade_ids"]:
                    await self._db.execute(
                        "UPDATE trades SET status = 'CLOSED', exit_time = ? WHERE id = ?",
                        (datetime.now().isoformat(), trade_id),
                    )
            elif ib_pos is not None and local is not None:
                # Update local avg cost to match IB
                for trade_id in local["trade_ids"]:
                    await self._db.execute(
                        "UPDATE trades SET entry_price = ? WHERE id = ?",
                        (ib_pos["avg_cost"], trade_id),
                    )

        if discrepancies:
            await self._db.commit()

        # Compare account values if available
        try:
            account_values = ib.accountValues()
            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    ib_net_liq = float(av.value)
                    snapshot = await self.get_latest_portfolio_snapshot()
                    if snapshot is not None:
                        local_value = snapshot.total_value
                        if abs(local_value - ib_net_liq) > 1.0:
                            disc = Discrepancy(
                                field="portfolio.total_value",
                                local_value=local_value,
                                ib_value=ib_net_liq,
                                resolution="updated_to_ib",
                            )
                            discrepancies.append(disc)
                            logger.warning(
                                "Discrepancy: portfolio total_value local=%.2f ib=%.2f — updated to IB",
                                local_value,
                                ib_net_liq,
                            )
                    break
        except Exception:
            logger.debug("Could not compare account values — skipping")

        logger.info(
            "State reconciliation complete — %d discrepancies found",
            len(discrepancies),
        )
        return discrepancies

    async def persist_agent_state(self, state: AgentState) -> None:
        """Insert agent state record into agent_state table.

        Used for tracking operational state across restarts.
        """
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        await self._db.execute(
            """
            INSERT INTO agent_state (
                state, initial_portfolio_value, start_time,
                last_heartbeat, crash_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                state.state,
                state.initial_portfolio_value,
                state.start_time.isoformat(),
                state.last_heartbeat.isoformat(),
                state.crash_count,
                datetime.now().isoformat(),
            ),
        )
        await self._db.commit()
        logger.info(
            "Persisted agent state: %s (crash_count=%d)",
            state.state,
            state.crash_count,
        )

    async def get_latest_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Query the most recent portfolio snapshot.

        Returns PortfolioSnapshot or None if no snapshots exist.
        """
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        cursor = await self._db.execute(
            "SELECT total_value, cash_balance, positions_value, daily_pnl, "
            "total_pnl, total_pnl_pct, num_open_positions, hard_stop_active, "
            "snapshot_time FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        return PortfolioSnapshot(
            total_value=row[0],
            cash_balance=row[1],
            positions_value=row[2],
            daily_pnl=row[3],
            total_pnl=row[4],
            total_pnl_pct=row[5],
            num_open_positions=row[6],
            hard_stop_active=bool(row[7]),
            snapshot_time=datetime.fromisoformat(row[8]),
        )

    async def get_trades_for_date(self, date: datetime) -> List[TradeRecord]:
        """Query all trades for a specific date.

        Used by ReportGenerator for daily reports.
        """
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        date_str = date.strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT symbol, direction, entry_price, quantity, stop_loss_price, "
            "strategy, signal_confidence, polymarket_sentiment, entry_time, "
            "exit_price, exit_time, realized_pnl, status "
            "FROM trades WHERE entry_time LIKE ? || '%' ORDER BY entry_time",
            (date_str,),
        )
        trades: List[TradeRecord] = []
        async for row in cursor:
            trades.append(
                TradeRecord(
                    symbol=row[0],
                    direction=row[1],
                    entry_price=row[2],
                    quantity=row[3],
                    stop_loss_price=row[4],
                    strategy=row[5],
                    signal_confidence=row[6],
                    polymarket_sentiment=row[7],
                    entry_time=datetime.fromisoformat(row[8]),
                    exit_price=row[9],
                    exit_time=datetime.fromisoformat(row[10]) if row[10] else None,
                    realized_pnl=row[11],
                    status=row[12],
                )
            )
        return trades

    async def persist_llm_call(
        self,
        purpose: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """Insert LLM call record into llm_calls table.

        Used by LLMService for cost tracking.
        """
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        await self._db.execute(
            """
            INSERT INTO llm_calls (
                purpose, model, input_tokens, output_tokens, total_tokens,
                success, error_message, call_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                purpose,
                model,
                input_tokens,
                output_tokens,
                total_tokens,
                1 if success else 0,
                error_message,
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().isoformat(),
            ),
        )
        await self._db.commit()
        logger.info(
            "Persisted LLM call: purpose=%s model=%s tokens=%d (in=%d out=%d) success=%s",
            purpose,
            model,
            total_tokens,
            input_tokens,
            output_tokens,
            success,
        )

    async def get_llm_calls_today(self) -> int:
        """Count LLM calls for today's date.

        Used by LLMService for daily limit enforcement.
        """
        if self._db is None:
            raise RuntimeError("StateManager not initialized")

        today_str = datetime.now().strftime("%Y-%m-%d")
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM llm_calls WHERE call_date = ?",
            (today_str,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
            logger.info("StateManager database connection closed")
