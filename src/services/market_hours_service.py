"""MarketHoursService: tracks NYSE/NASDAQ regular trading hours.

Determines whether the market is open using exchange calendar data from IB
(via reqContractDetails for a reference contract). Falls back to default
NYSE hours (9:30-16:00 ET) when IB is unavailable.

Also provides multi-exchange session awareness for US, EU, and ASIA
sessions (including US pre-market and after-hours).

Requirements: 3.1, 3.2, 3.3, 3.4
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from ib_insync import IB, Stock

logger = logging.getLogger(__name__)

# Default NYSE regular trading hours
_DEFAULT_OPEN = time(9, 30)
_DEFAULT_CLOSE = time(16, 0)

# Eastern Time zone
_ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------------
# Multi-exchange session definitions
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ExchangeSession:
    """Defines a trading session for an exchange."""

    name: str  # e.g. "US", "EU", "ASIA"
    exchange: str  # e.g. "NYSE", "LSE", "TSE"
    timezone: str  # e.g. "America/New_York", "Europe/London", "Asia/Tokyo"
    open_time: time  # local open time
    close_time: time  # local close time
    reference_symbol: str  # IB contract for schedule lookup, e.g. "SPY", "VOD", "7203"


# Pre-defined exchange sessions
EXCHANGE_SESSIONS = {
    "US": ExchangeSession(
        name="US",
        exchange="NYSE/NASDAQ",
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
        reference_symbol="SPY",
    ),
    "EU": ExchangeSession(
        name="EU",
        exchange="LSE/Eurex",
        timezone="Europe/London",
        open_time=time(8, 0),
        close_time=time(16, 30),
        reference_symbol="VOD",
    ),
    "ASIA": ExchangeSession(
        name="ASIA",
        exchange="TSE",
        timezone="Asia/Tokyo",
        open_time=time(9, 0),
        close_time=time(15, 0),
        reference_symbol="7203",
    ),
    "US_PREMARKET": ExchangeSession(
        name="US_PREMARKET",
        exchange="NYSE/NASDAQ",
        timezone="America/New_York",
        open_time=time(4, 0),
        close_time=time(9, 30),
        reference_symbol="SPY",
    ),
    "US_AFTERHOURS": ExchangeSession(
        name="US_AFTERHOURS",
        exchange="NYSE/NASDAQ",
        timezone="America/New_York",
        open_time=time(16, 0),
        close_time=time(20, 0),
        reference_symbol="SPY",
    ),
}


class MarketHoursService:
    """Tracks NYSE/NASDAQ regular trading hours.

    Uses IB contract details (liquidHours) for a reference contract (SPY)
    to determine the actual exchange schedule, including holidays. Falls
    back to default NYSE hours when IB is not available.
    """

    def __init__(self, ib: Optional[IB] = None) -> None:
        self._ib = ib
        self._is_market_open: bool = False
        self._market_open_time: Optional[datetime] = None
        self._market_close_time: Optional[datetime] = None
        self._schedule_loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update_schedule(self) -> None:
        """Fetch trading schedule from IB and update open/close times.

        Uses ``reqContractDetails()`` for a reference SPY contract to
        obtain ``liquidHours``. If IB is unavailable or the fetch fails,
        falls back to default NYSE hours (9:30-16:00 ET).
        """
        if self._ib is not None and self._ib.isConnected():
            try:
                await self._fetch_schedule_from_ib()
                return
            except Exception as exc:
                logger.warning(
                    "Failed to fetch schedule from IB, using defaults: %s", exc
                )

        self._apply_default_schedule()

    def is_market_open(self) -> bool:
        """Return True if the current time is within regular market hours.

        If the schedule has not been loaded, uses default NYSE hours
        (9:30-16:00 ET on weekdays).
        """
        now = datetime.now(_ET)

        if not self._schedule_loaded:
            return self._is_default_market_hours(now)

        if self._market_open_time is None or self._market_close_time is None:
            return self._is_default_market_hours(now)

        return self._market_open_time <= now < self._market_close_time

    def next_market_open(self) -> datetime:
        """Return the datetime of the next market open.

        - If currently before today's open, returns today's open.
        - If currently after today's close (or during market hours),
          returns the next weekday's open.
        """
        now = datetime.now(_ET)
        open_time, _ = self._get_effective_times()

        today_open = self._make_datetime(now, open_time)

        if now < today_open and self._is_weekday(now):
            return today_open

        # Move to next weekday
        return self._next_weekday_open(now, open_time)

    def next_market_close(self) -> datetime:
        """Return the datetime of the next market close.

        - If currently before today's close, returns today's close.
        - If currently after today's close, returns the next weekday's close.
        """
        now = datetime.now(_ET)
        _, close_time = self._get_effective_times()

        today_close = self._make_datetime(now, close_time)

        if now < today_close and self._is_weekday(now):
            return today_close

        # Move to next weekday
        return self._next_weekday_close(now, close_time)

    def get_active_sessions(self) -> list[str]:
        """Return names of all currently active exchange sessions.

        Checks each session in EXCHANGE_SESSIONS against current time
        in that session's timezone. Returns list of active session names.
        Only considers weekdays (Mon–Fri) in each session's timezone.
        """
        active: list[str] = []
        for name, session in EXCHANGE_SESSIONS.items():
            if self._is_session_active_now(session):
                active.append(name)
        return active

    def is_session_active(self, session_name: str) -> bool:
        """Check if a specific exchange session is currently active.

        Parameters
        ----------
        session_name : str
            Session name from EXCHANGE_SESSIONS (e.g. "US", "EU", "ASIA").

        Returns
        -------
        bool
            True if the session is currently within trading hours.
            False if the session name is unknown or outside trading hours.
        """
        session = EXCHANGE_SESSIONS.get(session_name)
        if session is None:
            logger.warning("Unknown session name: %s", session_name)
            return False
        return self._is_session_active_now(session)

    def get_session_for_symbol(self, symbol: str) -> Optional[str]:
        """Determine which session a symbol belongs to based on exchange.

        Simple heuristic:
        - Symbols ending in .L → EU (London)
        - Symbols ending in .T → ASIA (Tokyo)
        - All others → US (default)

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. "AAPL", "VOD.L", "7203.T").

        Returns
        -------
        Optional[str]
            Session name ("US", "EU", "ASIA") or None if no match.
        """
        if not symbol:
            return None
        upper = symbol.upper()
        if upper.endswith(".L"):
            return "EU"
        if upper.endswith(".T"):
            return "ASIA"
        return "US"

    # ------------------------------------------------------------------
    # Internal: session helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_session_active_now(session: ExchangeSession) -> bool:
        """Check if the given session is active right now.

        Considers the session's local timezone and only returns True
        on weekdays (Mon–Fri).
        """
        tz = ZoneInfo(session.timezone)
        now = datetime.now(tz)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        current_time = now.time()
        return session.open_time <= current_time < session.close_time

    # ------------------------------------------------------------------
    # Internal: IB schedule fetching
    # ------------------------------------------------------------------

    async def _fetch_schedule_from_ib(self) -> None:
        """Fetch liquidHours from IB for a reference SPY contract."""
        contract = Stock("SPY", "SMART", "USD")
        details_list = await self._ib.reqContractDetailsAsync(contract)

        if not details_list:
            raise ValueError("No contract details returned for SPY")

        details = details_list[0]
        liquid_hours = details.liquidHours

        if not liquid_hours:
            raise ValueError("liquidHours is empty for SPY")

        self._parse_liquid_hours(liquid_hours)

    def _parse_liquid_hours(self, liquid_hours: str) -> None:
        """Parse the liquidHours string from IB ContractDetails.

        The format is semicolon-separated segments like:
        ``20240102:0930-20240102:1600;20240103:0930-20240103:1600``

        Each segment is ``YYYYMMDD:HHMM-YYYYMMDD:HHMM``.
        We find today's segment to extract open/close times.
        Segments with ``CLOSED`` indicate holidays.
        """
        now = datetime.now(_ET)
        today_str = now.strftime("%Y%m%d")

        segments = liquid_hours.split(";")
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Skip closed days
            if "CLOSED" in segment.upper():
                if today_str in segment:
                    logger.info(
                        "Market is closed today (holiday): %s", segment
                    )
                    self._market_open_time = None
                    self._market_close_time = None
                    self._schedule_loaded = True
                    return
                continue

            if "-" not in segment:
                continue

            try:
                open_part, close_part = segment.split("-")
                open_date_str, open_time_str = open_part.split(":")
                close_date_str, close_time_str = close_part.split(":")

                if open_date_str != today_str:
                    continue

                open_dt = datetime.strptime(
                    f"{open_date_str}{open_time_str}", "%Y%m%d%H%M"
                ).replace(tzinfo=_ET)

                close_dt = datetime.strptime(
                    f"{close_date_str}{close_time_str}", "%Y%m%d%H%M"
                ).replace(tzinfo=_ET)

                self._market_open_time = open_dt
                self._market_close_time = close_dt
                self._schedule_loaded = True

                logger.info(
                    "Market schedule loaded from IB: open=%s, close=%s",
                    open_dt.strftime("%Y-%m-%d %H:%M %Z"),
                    close_dt.strftime("%Y-%m-%d %H:%M %Z"),
                )
                return
            except (ValueError, IndexError) as exc:
                logger.warning(
                    "Failed to parse liquidHours segment '%s': %s",
                    segment,
                    exc,
                )
                continue

        # No matching segment for today — fall back to defaults
        logger.info(
            "No schedule found for today in liquidHours, using defaults"
        )
        self._apply_default_schedule()

    # ------------------------------------------------------------------
    # Internal: default schedule
    # ------------------------------------------------------------------

    def _apply_default_schedule(self) -> None:
        """Apply default NYSE hours (9:30-16:00 ET) for today."""
        now = datetime.now(_ET)
        self._market_open_time = self._make_datetime(now, _DEFAULT_OPEN)
        self._market_close_time = self._make_datetime(now, _DEFAULT_CLOSE)
        self._schedule_loaded = True

        logger.info(
            "Using default NYSE schedule: open=%s, close=%s",
            self._market_open_time.strftime("%Y-%m-%d %H:%M %Z"),
            self._market_close_time.strftime("%Y-%m-%d %H:%M %Z"),
        )

    # ------------------------------------------------------------------
    # Internal: time helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_default_market_hours(now: datetime) -> bool:
        """Check if ``now`` falls within default NYSE hours on a weekday."""
        if not MarketHoursService._is_weekday(now):
            return False
        current_time = now.time()
        return _DEFAULT_OPEN <= current_time < _DEFAULT_CLOSE

    @staticmethod
    def _is_weekday(dt: datetime) -> bool:
        """Return True if ``dt`` is Monday–Friday (weekday 0–4)."""
        return dt.weekday() < 5

    @staticmethod
    def _make_datetime(reference: datetime, t: time) -> datetime:
        """Combine the date from ``reference`` with time ``t`` in ET."""
        return reference.replace(
            hour=t.hour,
            minute=t.minute,
            second=0,
            microsecond=0,
        )

    def _get_effective_times(self) -> tuple[time, time]:
        """Return the effective (open_time, close_time) as time objects.

        Uses loaded schedule times if available, otherwise defaults.
        """
        if (
            self._schedule_loaded
            and self._market_open_time is not None
            and self._market_close_time is not None
        ):
            return (
                self._market_open_time.time(),
                self._market_close_time.time(),
            )
        return _DEFAULT_OPEN, _DEFAULT_CLOSE

    @staticmethod
    def _next_weekday_open(now: datetime, open_time: time) -> datetime:
        """Return the open datetime for the next weekday after ``now``."""
        candidate = now + timedelta(days=1)
        while candidate.weekday() >= 5:  # skip Saturday (5) and Sunday (6)
            candidate += timedelta(days=1)
        return MarketHoursService._make_datetime(candidate, open_time)

    @staticmethod
    def _next_weekday_close(now: datetime, close_time: time) -> datetime:
        """Return the close datetime for the next weekday after ``now``."""
        candidate = now + timedelta(days=1)
        while candidate.weekday() >= 5:  # skip Saturday (5) and Sunday (6)
            candidate += timedelta(days=1)
        return MarketHoursService._make_datetime(candidate, close_time)
