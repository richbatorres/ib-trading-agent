"""Unit tests for MarketHoursService.

Tests market open/close detection at boundary times, next_market_open/close
calculations, weekend skipping, IB schedule parsing, and fallback to defaults.

Requirements: 3.1, 3.2, 3.3, 3.4
"""

from datetime import datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.services.market_hours_service import MarketHoursService

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(ib=None) -> MarketHoursService:
    """Create a MarketHoursService, optionally with a mocked IB."""
    return MarketHoursService(ib=ib)


def _dt(year, month, day, hour, minute) -> datetime:
    """Create an ET-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


# ---------------------------------------------------------------------------
# is_market_open — default schedule (no IB)
# ---------------------------------------------------------------------------


class TestIsMarketOpenDefault:
    """Tests for is_market_open() using default NYSE hours."""

    def test_open_during_regular_hours_weekday(self):
        """Market should be open at 10:00 on a Wednesday."""
        svc = _make_service()
        # Wednesday 2025-01-08 10:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is True

    def test_closed_before_open_weekday(self):
        """Market should be closed at 9:00 on a Wednesday."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 9, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is False

    def test_closed_after_close_weekday(self):
        """Market should be closed at 16:01 on a Wednesday."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 16, 1)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is False

    def test_open_at_exact_open_time(self):
        """Market should be open at exactly 9:30."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 9, 30)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is True

    def test_closed_at_exact_close_time(self):
        """Market should be closed at exactly 16:00 (close is exclusive)."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 16, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is False

    def test_closed_on_saturday(self):
        """Market should be closed on Saturday even during normal hours."""
        svc = _make_service()
        # Saturday 2025-01-11 10:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 11, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is False

    def test_closed_on_sunday(self):
        """Market should be closed on Sunday."""
        svc = _make_service()
        # Sunday 2025-01-12 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 12, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is False


# ---------------------------------------------------------------------------
# is_market_open — loaded schedule
# ---------------------------------------------------------------------------


class TestIsMarketOpenLoaded:
    """Tests for is_market_open() with a loaded schedule."""

    def test_open_within_loaded_schedule(self):
        svc = _make_service()
        svc._schedule_loaded = True
        svc._market_open_time = _dt(2025, 1, 8, 9, 30)
        svc._market_close_time = _dt(2025, 1, 8, 16, 0)

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is True

    def test_closed_outside_loaded_schedule(self):
        svc = _make_service()
        svc._schedule_loaded = True
        svc._market_open_time = _dt(2025, 1, 8, 9, 30)
        svc._market_close_time = _dt(2025, 1, 8, 16, 0)

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 17, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_market_open() is False

    def test_holiday_closed(self):
        """When schedule is loaded but times are None (holiday), market is closed."""
        svc = _make_service()
        svc._schedule_loaded = True
        svc._market_open_time = None
        svc._market_close_time = None

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            # Falls back to default check
            assert svc.is_market_open() is True  # Wednesday 12:00 is within defaults


# ---------------------------------------------------------------------------
# next_market_open
# ---------------------------------------------------------------------------


class TestNextMarketOpen:
    """Tests for next_market_open()."""

    def test_before_open_returns_today(self):
        """Before 9:30 on a weekday, returns today's open."""
        svc = _make_service()
        # Wednesday 2025-01-08 8:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 8, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_open()
            assert result.hour == 9
            assert result.minute == 30
            assert result.day == 8

    def test_after_close_returns_tomorrow(self):
        """After 16:00 on a weekday, returns next weekday's open."""
        svc = _make_service()
        # Wednesday 2025-01-08 17:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 17, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_open()
            assert result.day == 9  # Thursday
            assert result.hour == 9
            assert result.minute == 30

    def test_friday_after_close_returns_monday(self):
        """After close on Friday, returns Monday's open."""
        svc = _make_service()
        # Friday 2025-01-10 17:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 10, 17, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_open()
            assert result.day == 13  # Monday
            assert result.weekday() == 0  # Monday

    def test_saturday_returns_monday(self):
        """On Saturday, returns Monday's open."""
        svc = _make_service()
        # Saturday 2025-01-11 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 11, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_open()
            assert result.day == 13  # Monday
            assert result.weekday() == 0

    def test_sunday_returns_monday(self):
        """On Sunday, returns Monday's open."""
        svc = _make_service()
        # Sunday 2025-01-12 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 12, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_open()
            assert result.day == 13  # Monday

    def test_during_market_hours_returns_next_day(self):
        """During market hours, returns next weekday's open."""
        svc = _make_service()
        # Wednesday 2025-01-08 12:00 ET (market is open)
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_open()
            assert result.day == 9  # Thursday


# ---------------------------------------------------------------------------
# next_market_close
# ---------------------------------------------------------------------------


class TestNextMarketClose:
    """Tests for next_market_close()."""

    def test_before_close_returns_today(self):
        """Before 16:00 on a weekday, returns today's close."""
        svc = _make_service()
        # Wednesday 2025-01-08 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_close()
            assert result.hour == 16
            assert result.minute == 0
            assert result.day == 8

    def test_after_close_returns_tomorrow(self):
        """After 16:00 on a weekday, returns next weekday's close."""
        svc = _make_service()
        # Wednesday 2025-01-08 17:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 17, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_close()
            assert result.day == 9  # Thursday
            assert result.hour == 16

    def test_friday_after_close_returns_monday(self):
        """After close on Friday, returns Monday's close."""
        svc = _make_service()
        # Friday 2025-01-10 17:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 10, 17, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_close()
            assert result.day == 13  # Monday
            assert result.weekday() == 0

    def test_saturday_returns_monday(self):
        """On Saturday, returns Monday's close."""
        svc = _make_service()
        # Saturday 2025-01-11 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 11, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_close()
            assert result.day == 13  # Monday

    def test_before_open_returns_today_close(self):
        """Before market open on a weekday, today's close is still ahead."""
        svc = _make_service()
        # Wednesday 2025-01-08 8:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 8, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.next_market_close()
            assert result.day == 8
            assert result.hour == 16


# ---------------------------------------------------------------------------
# update_schedule — IB integration
# ---------------------------------------------------------------------------


class TestUpdateSchedule:
    """Tests for update_schedule() with mocked IB."""

    @pytest.mark.asyncio
    async def test_fallback_to_defaults_when_no_ib(self):
        """Without IB, should use default NYSE hours."""
        svc = _make_service(ib=None)

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await svc.update_schedule()

        assert svc._schedule_loaded is True
        assert svc._market_open_time is not None
        assert svc._market_close_time is not None
        assert svc._market_open_time.hour == 9
        assert svc._market_open_time.minute == 30
        assert svc._market_close_time.hour == 16
        assert svc._market_close_time.minute == 0

    @pytest.mark.asyncio
    async def test_fallback_when_ib_not_connected(self):
        """When IB is provided but not connected, should use defaults."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = False
        svc = _make_service(ib=mock_ib)

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await svc.update_schedule()

        assert svc._schedule_loaded is True

    @pytest.mark.asyncio
    async def test_parses_liquid_hours_from_ib(self):
        """Should parse liquidHours from IB contract details."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        mock_details = MagicMock()
        mock_details.liquidHours = "20250108:0930-20250108:1600;20250109:0930-20250109:1600"
        mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[mock_details])

        svc = _make_service(ib=mock_ib)

        # Patch only datetime.now, not the whole datetime class,
        # so that datetime.strptime still works.
        from datetime import datetime as real_datetime
        with patch(
            "src.services.market_hours_service.datetime",
            wraps=real_datetime,
        ) as mock_dt:
            mock_dt.now = MagicMock(return_value=_dt(2025, 1, 8, 10, 0))
            await svc.update_schedule()

        assert svc._schedule_loaded is True
        assert svc._market_open_time.hour == 9
        assert svc._market_open_time.minute == 30
        assert svc._market_close_time.hour == 16
        assert svc._market_close_time.minute == 0

    @pytest.mark.asyncio
    async def test_handles_closed_holiday(self):
        """Should detect CLOSED days in liquidHours."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        mock_details = MagicMock()
        mock_details.liquidHours = "20250108:CLOSED;20250109:0930-20250109:1600"
        mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[mock_details])

        svc = _make_service(ib=mock_ib)

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await svc.update_schedule()

        assert svc._schedule_loaded is True
        assert svc._market_open_time is None
        assert svc._market_close_time is None

    @pytest.mark.asyncio
    async def test_fallback_on_ib_error(self):
        """Should fall back to defaults when IB raises an exception."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.reqContractDetailsAsync = AsyncMock(
            side_effect=Exception("IB connection error")
        )

        svc = _make_service(ib=mock_ib)

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await svc.update_schedule()

        assert svc._schedule_loaded is True
        assert svc._market_open_time.hour == 9
        assert svc._market_open_time.minute == 30

    @pytest.mark.asyncio
    async def test_fallback_on_empty_contract_details(self):
        """Should fall back to defaults when IB returns empty details."""
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[])

        svc = _make_service(ib=mock_ib)

        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await svc.update_schedule()

        assert svc._schedule_loaded is True
        assert svc._market_open_time.hour == 9


# ---------------------------------------------------------------------------
# _parse_liquid_hours edge cases
# ---------------------------------------------------------------------------


class TestParseLiquidHours:
    """Tests for _parse_liquid_hours() edge cases."""

    def test_empty_string(self):
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            svc._parse_liquid_hours("")

        # Falls back to defaults
        assert svc._schedule_loaded is True

    def test_no_matching_date(self):
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            svc._parse_liquid_hours("20250109:0930-20250109:1600")

        # No match for today (Jan 8), falls back to defaults
        assert svc._schedule_loaded is True
        assert svc._market_open_time.hour == 9
        assert svc._market_open_time.minute == 30

    def test_malformed_segment_skipped(self):
        svc = _make_service()
        from datetime import datetime as real_datetime
        with patch(
            "src.services.market_hours_service.datetime",
            wraps=real_datetime,
        ) as mock_dt:
            mock_dt.now = MagicMock(return_value=_dt(2025, 1, 8, 10, 0))
            svc._parse_liquid_hours(
                "GARBAGE;20250108:0930-20250108:1600"
            )

        assert svc._schedule_loaded is True
        assert svc._market_open_time.hour == 9
        assert svc._market_open_time.minute == 30


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for MarketHoursService constructor."""

    def test_default_state(self):
        svc = _make_service()
        assert svc._is_market_open is False
        assert svc._market_open_time is None
        assert svc._market_close_time is None
        assert svc._schedule_loaded is False
        assert svc._ib is None

    def test_with_ib(self):
        mock_ib = MagicMock()
        svc = _make_service(ib=mock_ib)
        assert svc._ib is mock_ib
