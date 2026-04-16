"""Unit tests for ConnectionManager.

Tests connection with paper vs live port selection, environment validation,
reconnection logic, error event handling, and connection status.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AgentConfig
from src.services.connection_manager import ConnectionManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> AgentConfig:
    """Create an AgentConfig with sensible defaults for testing."""
    defaults = dict(
        ib_account_id="DU12345",
        ib_host="127.0.0.1",
        ib_port=7497,
        environment="paper",
        email_address="test@example.com",
        email_smtp_host="smtp.example.com",
        email_smtp_port=587,
        email_smtp_user="user",
        email_smtp_password="pass",
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _make_connection_manager(**config_overrides) -> ConnectionManager:
    """Create a ConnectionManager with a mocked IB instance."""
    config = _make_config(**config_overrides)
    cm = ConnectionManager(config)
    # Replace the real IB instance with a mock
    mock_ib = MagicMock()
    mock_ib.connectAsync = AsyncMock()
    mock_ib.isConnected = MagicMock(return_value=False)
    mock_ib.disconnect = MagicMock()
    mock_ib.disconnectedEvent = MagicMock()
    mock_ib.connectedEvent = MagicMock()
    mock_ib.errorEvent = MagicMock()
    # Make += work for event wiring
    mock_ib.disconnectedEvent.__iadd__ = MagicMock(return_value=mock_ib.disconnectedEvent)
    mock_ib.connectedEvent.__iadd__ = MagicMock(return_value=mock_ib.connectedEvent)
    mock_ib.errorEvent.__iadd__ = MagicMock(return_value=mock_ib.errorEvent)
    cm._ib = mock_ib
    return cm


# ---------------------------------------------------------------------------
# Environment validation tests
# ---------------------------------------------------------------------------


class TestEnvironmentValidation:
    """Tests for environment/port validation."""

    def test_paper_mode_accepts_tws_port(self):
        cm = _make_connection_manager(environment="paper", ib_port=7497)
        # Should not raise
        cm._validate_environment()

    def test_paper_mode_accepts_gateway_port(self):
        cm = _make_connection_manager(environment="paper", ib_port=4002)
        cm._validate_environment()

    def test_paper_mode_rejects_live_tws_port(self):
        cm = _make_connection_manager(environment="paper", ib_port=7496)
        with pytest.raises(ConnectionError, match="Paper environment"):
            cm._validate_environment()

    def test_paper_mode_rejects_live_gateway_port(self):
        cm = _make_connection_manager(environment="paper", ib_port=4001)
        with pytest.raises(ConnectionError, match="Paper environment"):
            cm._validate_environment()

    def test_live_mode_accepts_tws_port(self):
        cm = _make_connection_manager(environment="live", ib_port=7496)
        cm._validate_environment()

    def test_live_mode_accepts_gateway_port(self):
        cm = _make_connection_manager(environment="live", ib_port=4001)
        cm._validate_environment()

    def test_live_mode_rejects_paper_tws_port(self):
        cm = _make_connection_manager(environment="live", ib_port=7497)
        with pytest.raises(ConnectionError, match="Live environment"):
            cm._validate_environment()

    def test_live_mode_rejects_paper_gateway_port(self):
        cm = _make_connection_manager(environment="live", ib_port=4002)
        with pytest.raises(ConnectionError, match="Live environment"):
            cm._validate_environment()

    def test_paper_mode_rejects_arbitrary_port(self):
        cm = _make_connection_manager(environment="paper", ib_port=9999)
        with pytest.raises(ConnectionError, match="Paper environment"):
            cm._validate_environment()

    def test_live_mode_rejects_arbitrary_port(self):
        cm = _make_connection_manager(environment="live", ib_port=9999)
        with pytest.raises(ConnectionError, match="Live environment"):
            cm._validate_environment()


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestConnect:
    """Tests for the connect() method."""

    @pytest.mark.asyncio
    async def test_connect_calls_ib_connect_async(self):
        cm = _make_connection_manager(environment="paper", ib_port=7497)
        await cm.connect()

        cm._ib.connectAsync.assert_called_once_with(
            host="127.0.0.1",
            port=7497,
            clientId=1,
            readonly=False,
        )

    @pytest.mark.asyncio
    async def test_connect_wires_event_handlers(self):
        cm = _make_connection_manager(environment="paper", ib_port=7497)
        await cm.connect()

        cm._ib.disconnectedEvent.__iadd__.assert_called_once()
        cm._ib.connectedEvent.__iadd__.assert_called_once()
        cm._ib.errorEvent.__iadd__.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_raises_on_invalid_environment(self):
        cm = _make_connection_manager(environment="paper", ib_port=7496)
        with pytest.raises(ConnectionError):
            await cm.connect()

    @pytest.mark.asyncio
    async def test_connect_with_live_environment(self):
        cm = _make_connection_manager(environment="live", ib_port=7496)
        await cm.connect()

        cm._ib.connectAsync.assert_called_once_with(
            host="127.0.0.1",
            port=7496,
            clientId=1,
            readonly=False,
        )


# ---------------------------------------------------------------------------
# Disconnect tests
# ---------------------------------------------------------------------------


class TestDisconnect:
    """Tests for the disconnect() method."""

    @pytest.mark.asyncio
    async def test_disconnect_when_connected(self):
        cm = _make_connection_manager()
        cm._ib.isConnected.return_value = True

        await cm.disconnect()
        cm._ib.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        cm = _make_connection_manager()
        cm._ib.isConnected.return_value = False

        await cm.disconnect()
        cm._ib.disconnect.assert_not_called()


# ---------------------------------------------------------------------------
# Connection status tests
# ---------------------------------------------------------------------------


class TestIsConnected:
    """Tests for the is_connected() method."""

    def test_is_connected_returns_true(self):
        cm = _make_connection_manager()
        cm._ib.isConnected.return_value = True
        assert cm.is_connected() is True

    def test_is_connected_returns_false(self):
        cm = _make_connection_manager()
        cm._ib.isConnected.return_value = False
        assert cm.is_connected() is False


# ---------------------------------------------------------------------------
# IB property tests
# ---------------------------------------------------------------------------


class TestIBProperty:
    """Tests for the ib property."""

    def test_ib_property_returns_ib_instance(self):
        cm = _make_connection_manager()
        assert cm.ib is cm._ib


# ---------------------------------------------------------------------------
# Reconnection tests
# ---------------------------------------------------------------------------


class TestReconnection:
    """Tests for the _on_disconnected() reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_succeeds_on_first_attempt(self):
        cm = _make_connection_manager()
        cm._reconnect_interval = 0  # No delay for tests

        # connectAsync succeeds on first call
        cm._ib.connectAsync = AsyncMock()

        await cm._on_disconnected()

        # _on_connected is not called automatically by the mock,
        # so the counter reflects the attempt count (1), not reset (0).
        assert cm._reconnect_attempts == 1
        cm._ib.connectAsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_succeeds_after_failures(self):
        cm = _make_connection_manager()
        cm._reconnect_interval = 0

        # Fail twice, then succeed
        cm._ib.connectAsync = AsyncMock(
            side_effect=[ConnectionError("fail"), ConnectionError("fail"), None]
        )

        await cm._on_disconnected()

        assert cm._ib.connectAsync.call_count == 3
        assert cm._reconnect_attempts == 3

    @pytest.mark.asyncio
    async def test_reconnect_enters_waiting_state_after_max_attempts(self):
        cm = _make_connection_manager()
        cm._reconnect_interval = 0
        cm._waiting_interval = 0
        cm._max_reconnect_attempts = 3

        # Fail 3 times (exhaust initial attempts), then succeed on waiting retry
        cm._ib.connectAsync = AsyncMock(
            side_effect=[
                ConnectionError("fail 1"),
                ConnectionError("fail 2"),
                ConnectionError("fail 3"),
                None,  # Succeeds in waiting state
            ]
        )

        await cm._on_disconnected()

        assert cm._ib.connectAsync.call_count == 4

    @pytest.mark.asyncio
    async def test_reconnect_guard_prevents_concurrent_reconnection(self):
        cm = _make_connection_manager()
        cm._reconnecting = True
        cm._ib.connectAsync = AsyncMock()

        await cm._on_disconnected()

        # Should return immediately without attempting reconnection
        cm._ib.connectAsync.assert_not_called()


# ---------------------------------------------------------------------------
# _on_connected tests
# ---------------------------------------------------------------------------


class TestOnConnected:
    """Tests for the _on_connected() event handler."""

    def test_on_connected_resets_reconnect_counter(self):
        cm = _make_connection_manager()
        cm._reconnect_attempts = 4

        cm._on_connected()

        assert cm._reconnect_attempts == 0


# ---------------------------------------------------------------------------
# Error event tests
# ---------------------------------------------------------------------------


class TestOnError:
    """Tests for the _on_error() event handler."""

    def test_error_1100_connectivity_lost(self, caplog):
        cm = _make_connection_manager()
        import logging

        with caplog.at_level(logging.WARNING):
            cm._on_error(reqId=1, errorCode=1100, errorString="Connectivity lost")

        assert "connectivity lost" in caplog.text.lower()

    def test_error_1102_connectivity_restored(self, caplog):
        cm = _make_connection_manager()
        import logging

        with caplog.at_level(logging.INFO):
            cm._on_error(reqId=1, errorCode=1102, errorString="Connectivity restored")

        assert "connectivity restored" in caplog.text.lower()

    def test_generic_error_logged_as_warning(self, caplog):
        cm = _make_connection_manager()
        import logging

        with caplog.at_level(logging.WARNING):
            cm._on_error(reqId=5, errorCode=200, errorString="No security definition")

        assert "200" in caplog.text
        assert "No security definition" in caplog.text

    def test_error_with_contract(self, caplog):
        cm = _make_connection_manager()
        import logging

        mock_contract = MagicMock()
        mock_contract.__str__ = MagicMock(return_value="Stock(AAPL)")

        with caplog.at_level(logging.WARNING):
            cm._on_error(
                reqId=10,
                errorCode=321,
                errorString="Error validating request",
                contract=mock_contract,
            )

        assert "321" in caplog.text
