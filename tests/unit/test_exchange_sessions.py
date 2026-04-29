"""Unit tests for multi-exchange session management.

Tests the ExchangeSession dataclass, EXCHANGE_SESSIONS registry,
and the new session-aware methods on MarketHoursService:
- get_active_sessions()
- is_session_active()
- get_session_for_symbol()

Requirements: 3.1, 3.2, 3.3, 3.4
"""

from datetime import datetime, time
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.services.market_hours_service import (
    EXCHANGE_SESSIONS,
    ExchangeSession,
    MarketHoursService,
)

_ET = ZoneInfo("America/New_York")
_LT = ZoneInfo("Europe/London")
_TT = ZoneInfo("Asia/Tokyo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(ib=None) -> MarketHoursService:
    """Create a MarketHoursService, optionally with a mocked IB."""
    return MarketHoursService(ib=ib)


def _dt_et(year, month, day, hour, minute) -> datetime:
    """Create an ET-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=_ET)


# ---------------------------------------------------------------------------
# ExchangeSession dataclass
# ---------------------------------------------------------------------------


class TestExchangeSession:
    """Tests for the ExchangeSession frozen dataclass."""

    def test_create_session(self):
        session = ExchangeSession(
            name="TEST",
            exchange="TEST_EX",
            timezone="America/New_York",
            open_time=time(9, 0),
            close_time=time(17, 0),
            reference_symbol="TEST",
        )
        assert session.name == "TEST"
        assert session.exchange == "TEST_EX"
        assert session.timezone == "America/New_York"
        assert session.open_time == time(9, 0)
        assert session.close_time == time(17, 0)
        assert session.reference_symbol == "TEST"

    def test_frozen_immutable(self):
        """ExchangeSession should be immutable (frozen=True)."""
        session = EXCHANGE_SESSIONS["US"]
        with pytest.raises(AttributeError):
            session.name = "CHANGED"


# ---------------------------------------------------------------------------
# EXCHANGE_SESSIONS registry
# ---------------------------------------------------------------------------


class TestExchangeSessionsRegistry:
    """Tests for the pre-defined EXCHANGE_SESSIONS dict."""

    def test_contains_us_session(self):
        assert "US" in EXCHANGE_SESSIONS
        us = EXCHANGE_SESSIONS["US"]
        assert us.exchange == "NYSE/NASDAQ"
        assert us.timezone == "America/New_York"
        assert us.open_time == time(9, 30)
        assert us.close_time == time(16, 0)
        assert us.reference_symbol == "SPY"

    def test_contains_eu_session(self):
        assert "EU" in EXCHANGE_SESSIONS
        eu = EXCHANGE_SESSIONS["EU"]
        assert eu.exchange == "LSE/Eurex"
        assert eu.timezone == "Europe/London"
        assert eu.open_time == time(8, 0)
        assert eu.close_time == time(16, 30)
        assert eu.reference_symbol == "VOD"

    def test_contains_asia_session(self):
        assert "ASIA" in EXCHANGE_SESSIONS
        asia = EXCHANGE_SESSIONS["ASIA"]
        assert asia.exchange == "TSE"
        assert asia.timezone == "Asia/Tokyo"
        assert asia.open_time == time(9, 0)
        assert asia.close_time == time(15, 0)
        assert asia.reference_symbol == "7203"

    def test_contains_us_premarket(self):
        assert "US_PREMARKET" in EXCHANGE_SESSIONS
        pre = EXCHANGE_SESSIONS["US_PREMARKET"]
        assert pre.open_time == time(4, 0)
        assert pre.close_time == time(9, 30)

    def test_contains_us_afterhours(self):
        assert "US_AFTERHOURS" in EXCHANGE_SESSIONS
        after = EXCHANGE_SESSIONS["US_AFTERHOURS"]
        assert after.open_time == time(16, 0)
        assert after.close_time == time(20, 0)

    def test_all_five_sessions_present(self):
        expected = {"US", "EU", "ASIA", "US_PREMARKET", "US_AFTERHOURS"}
        assert set(EXCHANGE_SESSIONS.keys()) == expected


# ---------------------------------------------------------------------------
# is_session_active
# ---------------------------------------------------------------------------


class TestIsSessionActive:
    """Tests for is_session_active()."""

    def test_us_active_during_regular_hours(self):
        """US session should be active at 10:00 ET on a Wednesday."""
        svc = _make_service()
        # Wednesday 2025-01-08 10:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_session_active("US") is True

    def test_us_inactive_before_open(self):
        """US session should be inactive at 8:00 ET on a Wednesday."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 8, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_session_active("US") is False

    def test_us_inactive_on_weekend(self):
        """US session should be inactive on Saturday."""
        svc = _make_service()
        # Saturday 2025-01-11 10:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 11, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_session_active("US") is False

    def test_unknown_session_returns_false(self):
        """Unknown session name should return False and log a warning."""
        svc = _make_service()
        assert svc.is_session_active("MARS") is False

    def test_premarket_active_at_5am(self):
        """US_PREMARKET should be active at 5:00 ET on a weekday."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 5, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_session_active("US_PREMARKET") is True

    def test_premarket_inactive_at_10am(self):
        """US_PREMARKET should be inactive at 10:00 ET (regular hours)."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_session_active("US_PREMARKET") is False

    def test_afterhours_active_at_18(self):
        """US_AFTERHOURS should be active at 18:00 ET on a weekday."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 18, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_session_active("US_AFTERHOURS") is True

    def test_afterhours_inactive_at_21(self):
        """US_AFTERHOURS should be inactive at 21:00 ET."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 21, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert svc.is_session_active("US_AFTERHOURS") is False


# ---------------------------------------------------------------------------
# get_active_sessions
# ---------------------------------------------------------------------------


class TestGetActiveSessions:
    """Tests for get_active_sessions()."""

    def test_us_regular_hours_returns_us(self):
        """During US regular hours, at least US should be active."""
        svc = _make_service()
        # Wednesday 2025-01-08 10:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 10, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            active = svc.get_active_sessions()
            assert "US" in active

    def test_weekend_returns_empty(self):
        """On a weekend, no sessions should be active."""
        svc = _make_service()
        # Saturday 2025-01-11 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 11, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            active = svc.get_active_sessions()
            assert active == []

    def test_returns_list_type(self):
        """get_active_sessions should always return a list."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 11, 12, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = svc.get_active_sessions()
            assert isinstance(result, list)

    def test_premarket_active_early_morning(self):
        """At 5:00 ET on a weekday, US_PREMARKET should be in active list."""
        svc = _make_service()
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = _dt_et(2025, 1, 8, 5, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            active = svc.get_active_sessions()
            assert "US_PREMARKET" in active
            assert "US" not in active  # Regular US not open yet


# ---------------------------------------------------------------------------
# get_session_for_symbol
# ---------------------------------------------------------------------------


class TestGetSessionForSymbol:
    """Tests for get_session_for_symbol()."""

    def test_us_symbol_default(self):
        """Plain symbols like AAPL should map to US."""
        svc = _make_service()
        assert svc.get_session_for_symbol("AAPL") == "US"

    def test_us_symbol_msft(self):
        svc = _make_service()
        assert svc.get_session_for_symbol("MSFT") == "US"

    def test_london_symbol(self):
        """Symbols ending in .L should map to EU."""
        svc = _make_service()
        assert svc.get_session_for_symbol("VOD.L") == "EU"

    def test_london_symbol_lowercase(self):
        """Case-insensitive: vod.l should also map to EU."""
        svc = _make_service()
        assert svc.get_session_for_symbol("vod.l") == "EU"

    def test_tokyo_symbol(self):
        """Symbols ending in .T should map to ASIA."""
        svc = _make_service()
        assert svc.get_session_for_symbol("7203.T") == "ASIA"

    def test_tokyo_symbol_lowercase(self):
        svc = _make_service()
        assert svc.get_session_for_symbol("7203.t") == "ASIA"

    def test_empty_symbol_returns_none(self):
        """Empty string should return None."""
        svc = _make_service()
        assert svc.get_session_for_symbol("") is None

    def test_none_like_empty(self):
        """None-ish empty symbol returns None."""
        svc = _make_service()
        assert svc.get_session_for_symbol("") is None

    def test_symbol_with_dot_but_not_l_or_t(self):
        """Symbols like BRK.B should default to US."""
        svc = _make_service()
        assert svc.get_session_for_symbol("BRK.B") == "US"


# ---------------------------------------------------------------------------
# _is_session_active_now (static method)
# ---------------------------------------------------------------------------


class TestIsSessionActiveNow:
    """Tests for the internal _is_session_active_now static method."""

    def test_within_session_hours(self):
        """Should return True when current time is within session hours."""
        session = ExchangeSession(
            name="TEST",
            exchange="TEST",
            timezone="America/New_York",
            open_time=time(9, 0),
            close_time=time(17, 0),
            reference_symbol="TEST",
        )
        # Wednesday 2025-01-08 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 1, 8, 12, 0, tzinfo=_ET
            )
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert MarketHoursService._is_session_active_now(session) is True

    def test_outside_session_hours(self):
        """Should return False when current time is outside session hours."""
        session = ExchangeSession(
            name="TEST",
            exchange="TEST",
            timezone="America/New_York",
            open_time=time(9, 0),
            close_time=time(17, 0),
            reference_symbol="TEST",
        )
        # Wednesday 2025-01-08 20:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 1, 8, 20, 0, tzinfo=_ET
            )
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert MarketHoursService._is_session_active_now(session) is False

    def test_weekend_returns_false(self):
        """Should return False on weekends regardless of time."""
        session = ExchangeSession(
            name="TEST",
            exchange="TEST",
            timezone="America/New_York",
            open_time=time(0, 0),
            close_time=time(23, 59),
            reference_symbol="TEST",
        )
        # Saturday 2025-01-11 12:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 1, 11, 12, 0, tzinfo=_ET
            )
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert MarketHoursService._is_session_active_now(session) is False

    def test_at_exact_open_time(self):
        """Should return True at exactly the open time (inclusive)."""
        session = EXCHANGE_SESSIONS["US"]
        # Wednesday 2025-01-08 9:30 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 1, 8, 9, 30, tzinfo=_ET
            )
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert MarketHoursService._is_session_active_now(session) is True

    def test_at_exact_close_time(self):
        """Should return False at exactly the close time (exclusive)."""
        session = EXCHANGE_SESSIONS["US"]
        # Wednesday 2025-01-08 16:00 ET
        with patch(
            "src.services.market_hours_service.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = datetime(
                2025, 1, 8, 16, 0, tzinfo=_ET
            )
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert MarketHoursService._is_session_active_now(session) is False
