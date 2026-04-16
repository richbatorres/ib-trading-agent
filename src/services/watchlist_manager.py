"""WatchlistManager: filters watchlist candidates by volume and earnings blackout.

Qualifies stock contracts via IB, checks average daily volume against a
configurable threshold, and identifies symbols in earnings blackout windows.

Requirements: 2.1, 5.2, 13.3
"""

import logging
from typing import List, Optional, Set

from ib_insync import IB, Stock

logger = logging.getLogger(__name__)


class WatchlistManager:
    """Builds and maintains a filtered watchlist of tradeable symbols.

    Filters candidates by average daily volume (via IB fundamental data)
    and tracks earnings blackout windows to suppress trading around
    earnings announcements.
    """

    def __init__(self, ib: IB, min_avg_volume: float = 500_000) -> None:
        self._ib = ib
        self._min_avg_volume = min_avg_volume

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_watchlist(self, candidates: List[str]) -> List[str]:
        """Filter *candidates* by average daily volume via IB fundamental data.

        For each candidate symbol:
        1. Create and qualify a ``Stock`` contract on SMART/USD.
        2. Request fundamental data (``ReportSnapshot``) to extract the
           average daily volume.
        3. Include the symbol only if avg volume exceeds
           ``min_avg_volume``.
        4. If IB data is unavailable (qualification failure, missing
           fundamental data, or parsing error), include the symbol by
           default so it is not silently dropped.

        Parameters
        ----------
        candidates : List[str]
            Ticker symbols to evaluate.

        Returns
        -------
        List[str]
            Symbols that passed the volume filter (order preserved).
        """
        accepted: List[str] = []

        for symbol in candidates:
            contract = Stock(symbol, "SMART", "USD")
            try:
                qualified = await self._ib.qualifyContractsAsync(contract)
            except Exception:
                logger.warning(
                    "Failed to qualify %s — including by default", symbol
                )
                accepted.append(symbol)
                continue

            if not qualified:
                logger.warning(
                    "No qualified contract for %s — including by default", symbol
                )
                accepted.append(symbol)
                continue

            contract = qualified[0]
            avg_volume = await self._get_avg_daily_volume(symbol, contract)

            if avg_volume is None:
                # IB data unavailable — include by default
                logger.info(
                    "No volume data for %s — including by default", symbol
                )
                accepted.append(symbol)
            elif avg_volume > self._min_avg_volume:
                logger.info(
                    "Including %s (avg volume %.0f > %.0f)",
                    symbol,
                    avg_volume,
                    self._min_avg_volume,
                )
                accepted.append(symbol)
            else:
                logger.info(
                    "Filtered out %s (avg volume %.0f <= %.0f)",
                    symbol,
                    avg_volume,
                    self._min_avg_volume,
                )

        logger.info(
            "Watchlist built: %d/%d candidates accepted", len(accepted), len(candidates)
        )
        return accepted

    async def update_earnings_blackout(self, watchlist: List[str]) -> Set[str]:
        """Identify symbols in an earnings blackout window.

        A symbol is blacked out if its next earnings date falls within
        2 trading days before or 1 trading day after the current date.

        This is a stub implementation that returns an empty set.  Full IB
        wiring (via ``reqFundamentalData`` with ``CalendarReport``) will
        be added later.

        Parameters
        ----------
        watchlist : List[str]
            Symbols to check for upcoming earnings.

        Returns
        -------
        Set[str]
            Symbols currently in an earnings blackout window.
        """
        blackout: Set[str] = set()

        # ------------------------------------------------------------------
        # Stub: full implementation will use IB reqFundamentalData with
        # reportType='CalendarReport' to fetch the next earnings date for
        # each symbol and compare against the current date ± the blackout
        # window (2 trading days before, 1 trading day after).
        # ------------------------------------------------------------------

        if blackout:
            logger.info("Earnings blackout symbols: %s", blackout)
        else:
            logger.info("No symbols in earnings blackout window")

        return blackout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_avg_daily_volume(
        self, symbol: str, contract: Stock
    ) -> Optional[float]:
        """Request fundamental data from IB and extract average daily volume.

        Returns ``None`` when the data is unavailable or cannot be parsed.
        """
        try:
            xml_data = await self._ib.reqFundamentalDataAsync(
                contract, reportType="ReportSnapshot"
            )
        except Exception:
            logger.warning(
                "reqFundamentalData failed for %s — volume unknown", symbol
            )
            return None

        if not xml_data:
            return None

        return self._parse_avg_volume(xml_data)

    @staticmethod
    def _parse_avg_volume(xml_data: str) -> Optional[float]:
        """Extract average daily volume from an IB ReportSnapshot XML string.

        Looks for the ``<AvgDailyVolume>`` element.  Returns ``None`` if
        the element is missing or the value cannot be converted to float.
        """
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError:
            return None

        # The tag may appear at various depths; search the whole tree.
        elem = root.find(".//AvgDailyVolume")
        if elem is not None and elem.text:
            try:
                return float(elem.text)
            except ValueError:
                return None

        return None
