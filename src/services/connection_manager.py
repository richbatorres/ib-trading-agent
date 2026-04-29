"""ConnectionManager: wraps ib_insync.IB for connection lifecycle management.

Handles connecting to IB TWS/Gateway, environment validation (paper vs live
port mapping), automatic reconnection with backoff, and IB event wiring.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import asyncio
import logging
import random

from ib_insync import IB

from src.config import AgentConfig

logger = logging.getLogger(__name__)

# Valid port mappings per environment
_PAPER_PORTS = {7497, 4002}  # TWS paper, Gateway paper
_LIVE_PORTS = {7496, 4001}   # TWS live, Gateway live


def _generate_client_id() -> int:
    """Generate a random clientId (1-999) to avoid stale connection conflicts.

    IB Gateway keeps old connections alive for ~30-60s after ungraceful
    disconnect.  Using a random clientId on each startup avoids the
    'clientId already in use' error.
    """
    return random.randint(1, 999)


class ConnectionManager:
    """Manages IB API connection via ib_insync.

    Wraps the ``ib_insync.IB`` instance and handles connection lifecycle,
    reconnection with exponential back-off, and environment validation
    (paper mode ports vs live mode ports).
    """

    def __init__(self, config: AgentConfig) -> None:
        self._ib = IB()
        self._config = config
        self._client_id: int = _generate_client_id()
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 5
        self._reconnect_interval: int = 30  # seconds between reconnect attempts
        self._waiting_interval: int = 60    # seconds between retries after max failures
        self._reconnecting: bool = False
        self._events_wired: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to IB TWS/Gateway using config parameters.

        Validates the environment setting before connecting:
        - Paper mode: port must be 7497 (TWS) or 4002 (Gateway).
        - Live mode: port must be 7496 (TWS) or 4001 (Gateway).

        Wires IB event handlers for disconnection, reconnection, and errors.

        Raises
        ------
        ConnectionError
            If environment/port validation fails.
        """
        self._validate_environment()

        # Wire event handlers only once (avoid duplicate handlers on reconnect)
        if not self._events_wired:
            self._ib.disconnectedEvent += self._on_disconnected
            self._ib.connectedEvent += self._on_connected
            self._ib.errorEvent += self._on_error
            self._events_wired = True

        # Disconnect any stale connection before reconnecting
        if self._ib.isConnected():
            self._ib.disconnect()
            await asyncio.sleep(1)

        logger.info(
            "Connecting to IB %s at %s:%d (clientId=%d, environment=%s)",
            "TWS/Gateway",
            self._config.ib_host,
            self._config.ib_port,
            self._client_id,
            self._config.environment,
        )

        await self._ib.connectAsync(
            host=self._config.ib_host,
            port=self._config.ib_port,
            clientId=self._client_id,
            readonly=False,
        )

        logger.info(
            "Connected to IB successfully (account=%s, environment=%s)",
            self._config.ib_account_id,
            self._config.environment,
        )

    async def disconnect(self) -> None:
        """Gracefully disconnect from IB."""
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB gracefully")
        else:
            logger.info("Disconnect called but IB was not connected")

    def is_connected(self) -> bool:
        """Return current IB connection status."""
        return self._ib.isConnected()

    @property
    def ib(self) -> IB:
        """Return the underlying ib_insync.IB instance."""
        return self._ib

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_disconnected(self) -> None:
        """Handle IB disconnection event.

        Attempts reconnection up to ``_max_reconnect_attempts`` times with
        ``_reconnect_interval`` second delays. After exhausting attempts,
        logs an ERROR and enters a waiting state that retries every
        ``_waiting_interval`` seconds.

        Each reconnection attempt uses a fresh random clientId to avoid
        'clientId already in use' errors from stale IB Gateway sessions.

        Requirement: 1.2, 1.3
        """
        if self._reconnecting:
            return

        self._reconnecting = True
        logger.warning("IB connection lost — starting reconnection sequence")

        try:
            # Phase 1: up to _max_reconnect_attempts at _reconnect_interval
            while self._reconnect_attempts < self._max_reconnect_attempts:
                self._reconnect_attempts += 1
                logger.info(
                    "Reconnection attempt %d/%d in %d seconds",
                    self._reconnect_attempts,
                    self._max_reconnect_attempts,
                    self._reconnect_interval,
                )
                await asyncio.sleep(self._reconnect_interval)

                try:
                    # Fresh clientId to avoid stale connection conflicts
                    self._client_id = _generate_client_id()
                    if self._ib.isConnected():
                        self._ib.disconnect()
                        await asyncio.sleep(1)
                    await self._ib.connectAsync(
                        host=self._config.ib_host,
                        port=self._config.ib_port,
                        clientId=self._client_id,
                        readonly=False,
                    )
                    # Success — counter is reset in _on_connected
                    return
                except Exception as exc:
                    logger.warning(
                        "Reconnection attempt %d failed: %s",
                        self._reconnect_attempts,
                        exc,
                    )

            # Phase 2: all initial attempts exhausted
            logger.error(
                "Failed to reconnect after %d attempts — entering waiting state "
                "(retrying every %d seconds). Alert email should be sent.",
                self._max_reconnect_attempts,
                self._waiting_interval,
            )

            # Continuous retry at longer interval
            while True:
                await asyncio.sleep(self._waiting_interval)
                try:
                    self._client_id = _generate_client_id()
                    if self._ib.isConnected():
                        self._ib.disconnect()
                        await asyncio.sleep(1)
                    await self._ib.connectAsync(
                        host=self._config.ib_host,
                        port=self._config.ib_port,
                        clientId=self._client_id,
                        readonly=False,
                    )
                    # Success — counter is reset in _on_connected
                    return
                except Exception as exc:
                    logger.error(
                        "Waiting-state reconnection failed: %s", exc
                    )
        finally:
            self._reconnecting = False

    def _on_connected(self) -> None:
        """Handle IB reconnection event.

        Resets the reconnect attempt counter and logs the event.
        """
        self._reconnect_attempts = 0
        logger.info(
            "IB connection (re)established (account=%s)",
            self._config.ib_account_id,
        )

    def _on_error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        contract: object = None,
    ) -> None:
        """Handle IB error events at the appropriate log level.

        - Error code 1100 (connectivity lost) → WARNING
        - Error code 1102 (connectivity restored) → INFO
        - All other errors → WARNING
        """
        if errorCode == 1100:
            logger.warning(
                "IB error %d (connectivity lost): %s (reqId=%d)",
                errorCode,
                errorString,
                reqId,
            )
        elif errorCode == 1102:
            logger.info(
                "IB error %d (connectivity restored): %s (reqId=%d)",
                errorCode,
                errorString,
                reqId,
            )
        else:
            logger.warning(
                "IB error %d: %s (reqId=%d, contract=%s)",
                errorCode,
                errorString,
                reqId,
                contract,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_environment(self) -> None:
        """Validate that the configured port matches the environment.

        Paper mode must use port 7497 (TWS) or 4002 (Gateway).
        Live mode must use port 7496 (TWS) or 4001 (Gateway).

        Raises
        ------
        ConnectionError
            If the port does not match the environment.
        """
        port = self._config.ib_port
        env = self._config.environment

        if env == "paper" and port not in _PAPER_PORTS:
            raise ConnectionError(
                f"Paper environment requires port 7497 (TWS) or 4002 (Gateway), "
                f"got {port}"
            )

        if env == "live" and port not in _LIVE_PORTS:
            raise ConnectionError(
                f"Live environment requires port 7496 (TWS) or 4001 (Gateway), "
                f"got {port}"
            )

        logger.info(
            "Environment validation passed: %s mode, port %d", env, port
        )
