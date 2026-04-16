"""Unit tests for ShutdownHandler.

Requirements: 15.1, 15.2, 15.3, 15.4
"""

import asyncio
import platform
import signal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.domain import AgentState
from src.services.shutdown_handler import ShutdownHandler


@pytest.fixture
def mock_order_executor():
    e = MagicMock()
    e.cancel_all_pending = AsyncMock()
    return e


@pytest.fixture
def mock_state_manager():
    m = MagicMock()
    m.persist_agent_state = AsyncMock()
    return m


@pytest.fixture
def mock_connection_manager():
    c = MagicMock()
    c.disconnect = AsyncMock()
    return c


@pytest.fixture
def handler(mock_order_executor, mock_state_manager, mock_connection_manager):
    return ShutdownHandler(
        order_executor=mock_order_executor,
        state_manager=mock_state_manager,
        connection_manager=mock_connection_manager,
        timeout=30,
    )


class TestShutdownSequence:

    @pytest.mark.asyncio
    async def test_shutdown_calls_steps_in_order(
        self, handler, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        call_order = []
        mock_order_executor.cancel_all_pending = AsyncMock(side_effect=lambda: call_order.append("cancel"))
        mock_state_manager.persist_agent_state = AsyncMock(side_effect=lambda s: call_order.append("persist"))
        mock_connection_manager.disconnect = AsyncMock(side_effect=lambda: call_order.append("disconnect"))

        exit_code = await handler.shutdown()
        assert exit_code == 0
        assert call_order == ["cancel", "persist", "disconnect"]

    @pytest.mark.asyncio
    async def test_shutdown_persists_stopped_state(self, handler, mock_state_manager):
        await handler.shutdown()
        mock_state_manager.persist_agent_state.assert_called_once()
        state = mock_state_manager.persist_agent_state.call_args[0][0]
        assert isinstance(state, AgentState)
        assert state.state == "STOPPED"

    @pytest.mark.asyncio
    async def test_shutdown_cancels_all_pending_orders(self, handler, mock_order_executor):
        await handler.shutdown()
        mock_order_executor.cancel_all_pending.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_from_ib(self, handler, mock_connection_manager):
        await handler.shutdown()
        mock_connection_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_continues_on_cancel_failure(
        self, handler, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        mock_order_executor.cancel_all_pending = AsyncMock(side_effect=Exception("fail"))
        exit_code = await handler.shutdown()
        assert exit_code == 0
        mock_state_manager.persist_agent_state.assert_called_once()
        mock_connection_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_continues_on_persist_failure(
        self, handler, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        mock_state_manager.persist_agent_state = AsyncMock(side_effect=Exception("fail"))
        exit_code = await handler.shutdown()
        assert exit_code == 0
        mock_order_executor.cancel_all_pending.assert_called_once()
        mock_connection_manager.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_continues_on_disconnect_failure(
        self, handler, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        mock_connection_manager.disconnect = AsyncMock(side_effect=Exception("fail"))
        exit_code = await handler.shutdown()
        assert exit_code == 0
        mock_order_executor.cancel_all_pending.assert_called_once()
        mock_state_manager.persist_agent_state.assert_called_once()


class TestTimeoutEnforcement:

    @pytest.mark.asyncio
    async def test_timeout_returns_exit_code_1(
        self, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        async def slow_cancel():
            await asyncio.sleep(5)

        mock_order_executor.cancel_all_pending = AsyncMock(side_effect=slow_cancel)
        h = ShutdownHandler(
            order_executor=mock_order_executor,
            state_manager=mock_state_manager,
            connection_manager=mock_connection_manager,
            timeout=1,
        )
        exit_code = await h.shutdown()
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_timeout_force_closes_connections(
        self, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        async def slow_cancel():
            await asyncio.sleep(5)

        mock_order_executor.cancel_all_pending = AsyncMock(side_effect=slow_cancel)
        h = ShutdownHandler(
            order_executor=mock_order_executor,
            state_manager=mock_state_manager,
            connection_manager=mock_connection_manager,
            timeout=1,
        )
        await h.shutdown()
        mock_connection_manager.disconnect.assert_called_once()

    def test_default_timeout_is_30_seconds(
        self, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        h = ShutdownHandler(
            order_executor=mock_order_executor,
            state_manager=mock_state_manager,
            connection_manager=mock_connection_manager,
        )
        assert h._timeout == 30


class TestExitCodes:

    @pytest.mark.asyncio
    async def test_success_returns_0(self, handler):
        assert await handler.shutdown() == 0

    @pytest.mark.asyncio
    async def test_timeout_returns_1(
        self, mock_order_executor, mock_state_manager, mock_connection_manager
    ):
        async def slow():
            await asyncio.sleep(5)

        mock_order_executor.cancel_all_pending = AsyncMock(side_effect=slow)
        h = ShutdownHandler(
            order_executor=mock_order_executor,
            state_manager=mock_state_manager,
            connection_manager=mock_connection_manager,
            timeout=1,
        )
        assert await h.shutdown() == 1


class TestSignalHandlers:

    def test_setup_signal_handlers_unix(self, handler):
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        with patch("src.services.shutdown_handler.platform") as mp:
            mp.system.return_value = "Linux"
            handler.setup_signal_handlers(loop)
        assert loop.add_signal_handler.call_count == 2
        sigs = [c[0][0] for c in loop.add_signal_handler.call_args_list]
        assert signal.SIGINT in sigs
        assert signal.SIGTERM in sigs

    def test_setup_signal_handlers_windows(self, handler):
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        with (
            patch("src.services.shutdown_handler.platform") as mp,
            patch("src.services.shutdown_handler.signal.signal") as ms,
        ):
            mp.system.return_value = "Windows"
            handler.setup_signal_handlers(loop)
        assert ms.call_count == 2
        sigs = [c[0][0] for c in ms.call_args_list]
        assert signal.SIGINT in sigs
        assert signal.SIGTERM in sigs
