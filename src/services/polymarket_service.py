"""PolymarketService: fetches and scores Polymarket prediction market data.

Fetches prediction market data from the Polymarket Gamma API and computes
a sentiment score from -1.0 (strongly bearish) to +1.0 (strongly bullish).
The sentiment score is used as a secondary weighting factor in the
StrategyEngine — it adjusts signal confidence but never triggers a trade
on its own.

Requirements: 13.1, 13.2, 13.3, 13.4
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import List, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Polymarket Gamma API base URL (public, no auth required)
GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Tags relevant to macroeconomic / political market sentiment
RELEVANT_TAGS = [
    "economics",
    "politics",
    "fed",
    "inflation",
    "recession",
    "interest-rates",
]

# Keywords that indicate bearish market conditions
_BEARISH_KEYWORDS = [
    "recession",
    "crash",
    "decline",
    "downturn",
    "default",
    "shutdown",
    "layoff",
    "unemployment",
    "rate hike",
    "rate increase",
    "bear market",
    "inflation rise",
    "inflation increase",
    "stagflation",
    "debt ceiling",
    "tariff",
    "war",
    "sanctions",
]

# Keywords that indicate bullish market conditions
_BULLISH_KEYWORDS = [
    "growth",
    "recovery",
    "rally",
    "bull market",
    "rate cut",
    "rate decrease",
    "stimulus",
    "expansion",
    "employment",
    "job growth",
    "gdp growth",
    "soft landing",
    "inflation decrease",
    "inflation drop",
    "trade deal",
    "peace",
]


class PolymarketService:
    """Fetches and scores Polymarket prediction market data.

    Polls the Polymarket Gamma API every 15 minutes (driven by the
    scheduler) and computes a sentiment score in [-1.0, +1.0].
    """

    def __init__(self) -> None:
        self._sentiment_score: float = 0.0
        self._last_fetch: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_markets(self) -> List[dict]:
        """Fetch active markets from the Polymarket Gamma API.

        Queries ``GET /markets?active=true&closed=false&limit=100``
        for each tag in :data:`RELEVANT_TAGS`. No authentication is
        required.

        Returns:
            A list of market dicts. Returns an empty list on error.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        all_markets: List[dict] = []
        seen_ids: set = set()

        for tag in RELEVANT_TAGS:
            url = f"{GAMMA_API_BASE}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": "100",
                "tag": tag,
            }
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "Polymarket API returned status %d for tag '%s'",
                            resp.status,
                            tag,
                        )
                        continue

                    markets = await resp.json()
                    if not isinstance(markets, list):
                        logger.warning(
                            "Unexpected response format for tag '%s': %s",
                            tag,
                            type(markets).__name__,
                        )
                        continue

                    for market in markets:
                        market_id = market.get("id")
                        if market_id and market_id not in seen_ids:
                            seen_ids.add(market_id)
                            all_markets.append(market)

            except Exception as exc:
                logger.warning(
                    "Failed to fetch Polymarket markets for tag '%s': %s",
                    tag,
                    exc,
                )
                continue

        logger.info(
            "Fetched %d unique markets from Polymarket across %d tags",
            len(all_markets),
            len(RELEVANT_TAGS),
        )
        return all_markets

    def compute_sentiment(self, markets: List[dict]) -> float:
        """Compute an aggregate sentiment score from market data.

        Aggregates market probabilities into a score from -1.0 (strongly
        bearish) to +1.0 (strongly bullish). Each market is weighted by
        its trading volume and recency.

        For each market:
        - Extract ``outcomePrices``, ``volume``, and ``question``.
        - Classify the market as bullish or bearish based on question
          keywords.
        - Bullish markets push the score toward +1.0.
        - Bearish markets (recession, rate hikes, etc.) push toward -1.0.

        Args:
            markets: List of market dicts from the Gamma API.

        Returns:
            A float clamped to [-1.0, +1.0]. Returns 0.0 if no markets.
        """
        if not markets:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for market in markets:
            try:
                score = self._score_market(market)
                if score is None:
                    continue

                sentiment_value, weight = score
                weighted_sum += sentiment_value * weight
                total_weight += weight
            except Exception as exc:
                logger.warning(
                    "Error scoring market '%s': %s",
                    market.get("question", "unknown"),
                    exc,
                )
                continue

        if total_weight == 0.0:
            return 0.0

        raw_score = weighted_sum / total_weight

        # Clamp to [-1.0, +1.0]
        return max(-1.0, min(1.0, raw_score))

    async def update(self) -> None:
        """Fetch markets and update the sentiment score.

        Called every 15 minutes by the scheduler. On API failure, keeps
        the last score and logs a WARNING with the last fetch timestamp.
        """
        try:
            markets = await self.fetch_markets()

            if markets:
                self._sentiment_score = self.compute_sentiment(markets)
                self._last_fetch = datetime.now(timezone.utc)
                logger.info(
                    "Polymarket sentiment updated: score=%.4f, markets=%d",
                    self._sentiment_score,
                    len(markets),
                )
            else:
                logger.warning(
                    "No markets fetched from Polymarket; "
                    "keeping last sentiment score=%.4f (last fetch: %s)",
                    self._sentiment_score,
                    self._last_fetch,
                )
        except Exception as exc:
            logger.warning(
                "Polymarket update failed: %s; "
                "keeping last sentiment score=%.4f (last fetch: %s)",
                exc,
                self._sentiment_score,
                self._last_fetch,
            )

    @property
    def sentiment_score(self) -> float:
        """Current sentiment score in [-1.0, +1.0]."""
        return self._sentiment_score

    @property
    def last_fetch_time(self) -> Optional[datetime]:
        """Timestamp of the last successful fetch (UTC)."""
        return self._last_fetch

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("Polymarket HTTP session closed")

    # ------------------------------------------------------------------
    # Internal: market scoring
    # ------------------------------------------------------------------

    def _score_market(self, market: dict) -> Optional[tuple]:
        """Score a single market dict.

        Returns:
            A tuple of ``(sentiment_value, weight)`` where
            ``sentiment_value`` is in [-1.0, +1.0] and ``weight``
            reflects volume and recency. Returns ``None`` if the
            market cannot be scored.
        """
        question = market.get("question", "")
        if not question:
            return None

        # Parse outcome prices — may be a JSON string or list
        outcome_prices = market.get("outcomePrices")
        if outcome_prices is None:
            return None

        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (ValueError, TypeError):
                return None

        if not isinstance(outcome_prices, list) or len(outcome_prices) == 0:
            return None

        # The first outcome price is typically the "Yes" probability
        try:
            yes_probability = float(outcome_prices[0])
        except (ValueError, TypeError, IndexError):
            return None

        # Determine volume weight
        volume_raw = market.get("volume", 0)
        try:
            volume = float(volume_raw)
        except (ValueError, TypeError):
            volume = 0.0

        if volume <= 0:
            return None

        # Volume weight: log-scale to avoid huge markets dominating
        volume_weight = math.log1p(volume)

        # Recency weight: more recent markets get higher weight
        recency_weight = self._recency_weight(market)

        weight = volume_weight * recency_weight

        # Classify market direction based on question keywords
        question_lower = question.lower()
        direction = self._classify_direction(question_lower)

        if direction == 0:
            # Neutral / unclassifiable — skip
            return None

        # For bearish markets: high "Yes" probability means bearish sentiment
        # For bullish markets: high "Yes" probability means bullish sentiment
        # Map yes_probability [0, 1] to sentiment contribution
        # direction is +1 (bullish) or -1 (bearish)
        sentiment_value = direction * (2.0 * yes_probability - 1.0)

        return (sentiment_value, weight)

    @staticmethod
    def _classify_direction(question_lower: str) -> int:
        """Classify a market question as bullish (+1), bearish (-1), or
        neutral (0) based on keyword matching.
        """
        bearish_count = sum(
            1 for kw in _BEARISH_KEYWORDS if kw in question_lower
        )
        bullish_count = sum(
            1 for kw in _BULLISH_KEYWORDS if kw in question_lower
        )

        if bearish_count > bullish_count:
            return -1
        elif bullish_count > bearish_count:
            return 1
        else:
            return 0

    @staticmethod
    def _recency_weight(market: dict) -> float:
        """Compute a recency weight for a market.

        Markets updated more recently get higher weight. Uses the
        ``endDate`` or ``updatedAt`` field if available.

        Returns a weight in (0.0, 1.0].
        """
        # Try endDate first, then updatedAt
        date_str = market.get("endDate") or market.get("updatedAt")
        if not date_str:
            return 0.5  # default weight for markets without date info

        try:
            # Parse ISO format datetime
            if isinstance(date_str, str):
                # Handle various ISO formats
                date_str = date_str.replace("Z", "+00:00")
                market_dt = datetime.fromisoformat(date_str)
                if market_dt.tzinfo is None:
                    market_dt = market_dt.replace(tzinfo=timezone.utc)
            else:
                return 0.5

            now = datetime.now(timezone.utc)
            age_days = (now - market_dt).total_seconds() / 86400.0

            # Exponential decay: half-life of 7 days
            weight = 2.0 ** (-age_days / 7.0)
            return max(0.01, min(1.0, weight))

        except (ValueError, TypeError):
            return 0.5
