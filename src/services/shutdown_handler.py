"""Graceful shutdown handler for the IB Trading Agent.

Handles SIGINT/SIGTERM signals and executes a strict shutdown sequence:
cancel pending orders → persist state → disconnect from IB.
Enforces a configurable timeout (default 30 seconds).

Requirements: 15.1, 15.2, 15.3, 15.4
"""

import asyncio
import logging
import platform
import signal
import sys
from datetime import datetime
from typing import Optional

from src.models.domain import AgentState
from src.services.connection_manager import ConnectionManager
from src.services.order_executor import OrderExecutor
from src.services.state_manager import StateManager

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """Manages graceful shutdown of the trading agent.

    Executes a strict shutdown sequence within a configurable timeout:
    1. Cancel all pending orders via OrderExecutor
    2. Persist current agent state as STOPPED via StateManager
    3. Disconnect from IB via ConnectionManager

    If the timeout is exceeded, force-closes all connections and exits
    with code 1.
    """

    def __init__(
        self,
        order_executor: OrderExecutor,
        state_manager: StateManager,
        connection_manager: ConnectionManager,
        timeout: int = 30,
    ) -> None:
        self._order_executor = order_executor
        self._state_manager = state_manager
        self._connection_manager = connection_manager
        self._timeout = timeout
        self._shutting_down = False

    async def shutdown(self) -> int:
        """Execute the graceful shutdown sequence.

        Returns 0 on success, 1 if the timeout was exceeded.
        """
        if self._shutting_down:
            logger.warning("Shutdown already in progress — ignoring duplicate request")
            return 0

        self._shutting_down = True
        logger.info("Graceful shutdown initiated — timeout=%d seconds", self._timeout)

        try:
            exit_code = await asyncio.wait_for(
                self._execute_shutdown_sequence(),
                timeout=self._timeout,
            )
            return exit_code
        except asyncio.TimeoutError:
            logger.error(
                "Graceful shutdown exceeded %d-second timeout — force-closing connections",
                self._timeout,
            )
            await self._force_close()
            return 1
        finally:
            self._shutting_down = False

    async def _execute_shutdown_sequence(self) -> int:
        """Run the three shutdown steps in strict order.

        Each step is wrapped in its own try/except so that a failure
        in one step does not prevent the remaining steps from executing.

        Returns 0 on success.
        """
        # Step 1: Cancel all pending orders
        logger.info("Shutdown step 1/3: Cancelling all pending orders")
        try:
            await self._order_executor.cancel_all_pending()
            logger.info("Shutdown step 1/3 complete: Pending orders cancelled")
        except Exception:
            logger.exception("Shutdown step 1/3 failed: Error cancelling pending orders")

        # Step 2: Persist current state as STOPPED
        logger.info("Shutdown step 2/3: Persisting agent state as STOPPED")
        try:
            now = datetime.now()
            agent_state = AgentState(
                state="STOPPED",
                initial_portfolio_value=None,
                start_time=now,
                last_heartbeat=now,
                crash_count=0,
            )
            await self._state_manager.persist_agent_state(agent_state)
            logger.info("Shutdown step 2/3 complete: Agent state persisted as STOPPED")
        except Exception:
            logger.exception("Shutdown step 2/3 failed: Error persisting agent state")

        # Step 3: Disconnect from IB
        logger.info("Shutdown step 3/3: Disconnecting from IB")
        try:
            await self._connection_manager.disconnect()
            logger.info("Shutdown step 3/3 complete: Disconnected from IB")
        except Exception:
            logger.exception("Shutdown step 3/3 failed: Error disconnecting from IB")

        logger.info("Graceful shutdown completed successfully")
        return 0

    async def _force_close(self) -> None:
        """Force-close all connections when timeout is exceeded.

        Best-effort attempt to disconnect — exceptions are logged but
        do not propagate.
        """
        logger.warning("Force-closing all connections")
        try:
            await self._connection_manager.disconnect()
        except Exception:
            logger.exception("Error during force-close disconnect")

    def setup_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register SIGINT and SIGTERM handlers on the event loop.

        On Unix systems, uses ``loop.add_signal_handler()`` for proper
        async signal handling. On Windows, falls back to ``signal.signal()``
        since ``add_signal_handler`` is not supported.

        Parameters
        ----------
        loop : asyncio.AbstractEventLoop
            The running event loop to register signal handlers on.
        """
        if platform.system() != "Windows":
            # Unix: use loop.add_signal_handler for proper async handling
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.ensure_future(
                        self._handle_signal(s, loop)
                    ),
                )
            logger.info(
                "Signal handlers registered via loop.add_signal_handler (SIGINT, SIGTERM)"
            )
        else:
            # Windows: fallback to signal.signal()
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(
                    sig,
                    lambda signum, frame: asyncio.ensure_future(
                        self._handle_signal(signum, loop)
                    ),
                )
            logger.info(
                "Signal handlers registered via signal.signal (Windows fallback)"
            )

    async def _handle_signal(
        self,
        sig: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Handle a received signal by running shutdown and exiting.

        Parameters
        ----------
        sig : int
            The signal number received.
        loop : asyncio.AbstractEventLoop
            The event loop (used to schedule stop if needed).
        """
        sig_name = signal.Signals(sig).name
        logger.info("Received signal %s — initiating shutdown", sig_name)

        exit_code = await self.shutdown()

        logger.info("Exiting with code %d", exit_code)
        sys.exit(exit_code)
