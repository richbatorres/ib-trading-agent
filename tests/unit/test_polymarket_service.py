"""Unit tests for PolymarketService.

Tests sentiment computation with known market data, API failure fallback,
session management, and market scoring logic.

Requirements: 13.1, 13.2, 13.3, 13.4
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.polymarket_service import (
    GAMMA_API_BASE,
    RELEVANT_TAGS,
    PolymarketService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_market(
    question: str,
    outcome_prices: list | str | None = None,
    volume: float = 1000.0,
    market_id: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Create a mock market dict matching the Gamma API shape."""
    market = {
        "question": question,
        "volume": volume,
        "id": market_id or f"id-{hash(question) % 10000}",
    }
    if outcome_prices is not None:
        if isinstance(outcome_prices, list):
            market["outcomePrices"] = json.dumps(outcome_prices)
        else:
            market["outcomePrices"] = outcome_prices
    if end_date is not None:
        market["endDate"] = end_date
    return market


def _make_bearish_market(
    probability: float = 0.7, volume: float = 5000.0
) -> dict:
    """Create a market with bearish keywords and given probability."""
    return _make_market(
        question="Will there be a recession in 2025?",
        outcome_prices=[probability, 1.0 - probability],
        volume=volume,
        market_id="bearish-1",
    )


def _make_bullish_market(
    probability: float = 0.7, volume: float = 5000.0
) -> dict:
    """Create a market with bullish keywords and given probability."""
    return _make_market(
        question="Will there be GDP growth above 3% in 2025?",
        outcome_prices=[probability, 1.0 - probability],
        volume=volume,
        market_id="bullish-1",
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for PolymarketService constructor."""

    def test_default_state(self):
        svc = PolymarketService()
        assert svc.sentiment_score == 0.0
        assert svc.last_fetch_time is None
        assert svc._session is None

    def test_sentiment_score_property(self):
        svc = PolymarketService()
        svc._sentiment_score = 0.5
        assert svc.sentiment_score == 0.5

    def test_last_fetch_time_property(self):
        svc = PolymarketService()
        now = datetime.now(timezone.utc)
        svc._last_fetch = now
        assert svc.last_fetch_time == now


# ---------------------------------------------------------------------------
# compute_sentiment
# ---------------------------------------------------------------------------


class TestComputeSentiment:
    """Tests for compute_sentiment() with known market data."""

    def test_empty_markets_returns_zero(self):
        """No markets should produce a neutral score of 0.0."""
        svc = PolymarketService()
        assert svc.compute_sentiment([]) == 0.0

    def test_single_bearish_market_high_probability(self):
        """A bearish market with high 'Yes' probability should produce
        a negative sentiment score."""
        svc = PolymarketService()
        markets = [_make_bearish_market(probability=0.8)]
        score = svc.compute_sentiment(markets)
        assert -1.0 <= score <= 0.0
        assert score < 0  # should be negative (bearish)

    def test_single_bullish_market_high_probability(self):
        """A bullish market with high 'Yes' probability should produce
        a positive sentiment score."""
        svc = PolymarketService()
        markets = [_make_bullish_market(probability=0.8)]
        score = svc.compute_sentiment(markets)
        assert 0.0 <= score <= 1.0
        assert score > 0  # should be positive (bullish)

    def test_bearish_market_low_probability_is_bullish(self):
        """A bearish market with low 'Yes' probability (recession unlikely)
        should produce a positive sentiment score."""
        svc = PolymarketService()
        markets = [_make_bearish_market(probability=0.2)]
        score = svc.compute_sentiment(markets)
        assert score > 0  # recession unlikely → bullish

    def test_score_clamped_to_bounds(self):
        """Score should always be in [-1.0, +1.0]."""
        svc = PolymarketService()
        # Many strongly bearish markets
        markets = [
            _make_market(
                question=f"Will recession happen scenario {i}?",
                outcome_prices=[0.95, 0.05],
                volume=100000.0,
                market_id=f"bear-{i}",
            )
            for i in range(20)
        ]
        score = svc.compute_sentiment(markets)
        assert -1.0 <= score <= 1.0

    def test_mixed_markets_produce_intermediate_score(self):
        """A mix of bullish and bearish markets should produce a score
        between the extremes."""
        svc = PolymarketService()
        markets = [
            _make_bearish_market(probability=0.7, volume=5000.0),
            _make_bullish_market(probability=0.7, volume=5000.0),
        ]
        score = svc.compute_sentiment(markets)
        assert -1.0 <= score <= 1.0

    def test_volume_weighting(self):
        """Higher volume markets should have more influence on the score."""
        svc = PolymarketService()
        # High-volume bearish vs low-volume bullish
        markets = [
            _make_market(
                question="Will there be a recession?",
                outcome_prices=[0.8, 0.2],
                volume=1000000.0,
                market_id="high-vol-bear",
            ),
            _make_market(
                question="Will there be GDP growth?",
                outcome_prices=[0.8, 0.2],
                volume=100.0,
                market_id="low-vol-bull",
            ),
        ]
        score = svc.compute_sentiment(markets)
        # Bearish market has much higher volume, so score should lean negative
        assert score < 0

    def test_markets_without_outcome_prices_skipped(self):
        """Markets missing outcomePrices should be skipped."""
        svc = PolymarketService()
        markets = [
            _make_market(
                question="Will there be a recession?",
                outcome_prices=None,
                volume=5000.0,
            ),
        ]
        score = svc.compute_sentiment(markets)
        assert score == 0.0

    def test_markets_with_zero_volume_skipped(self):
        """Markets with zero volume should be skipped."""
        svc = PolymarketService()
        markets = [
            _make_bearish_market(probability=0.9, volume=0.0),
        ]
        # Volume is 0, so market is skipped → returns 0.0
        # The _score_market method returns None for volume <= 0
        score = svc.compute_sentiment(markets)
        assert score == 0.0

    def test_neutral_question_skipped(self):
        """Markets with no bullish/bearish keywords should be skipped."""
        svc = PolymarketService()
        markets = [
            _make_market(
                question="Will team X win the championship?",
                outcome_prices=[0.6, 0.4],
                volume=5000.0,
            ),
        ]
        score = svc.compute_sentiment(markets)
        assert score == 0.0

    def test_outcome_prices_as_json_string(self):
        """outcomePrices as a JSON string should be parsed correctly."""
        svc = PolymarketService()
        market = {
            "question": "Will there be a recession?",
            "outcomePrices": '[0.7, 0.3]',
            "volume": 5000.0,
            "id": "json-str-1",
        }
        score = svc.compute_sentiment([market])
        assert score < 0  # bearish

    def test_outcome_prices_as_list(self):
        """outcomePrices as a native list should work correctly."""
        svc = PolymarketService()
        market = {
            "question": "Will there be a recession?",
            "outcomePrices": [0.7, 0.3],
            "volume": 5000.0,
            "id": "list-1",
        }
        score = svc.compute_sentiment([market])
        assert score < 0  # bearish


# ---------------------------------------------------------------------------
# _classify_direction
# ---------------------------------------------------------------------------


class TestClassifyDirection:
    """Tests for the internal _classify_direction method."""

    def test_bearish_keywords(self):
        assert PolymarketService._classify_direction("will there be a recession") == -1
        assert PolymarketService._classify_direction("rate hike in 2025") == -1
        assert PolymarketService._classify_direction("market crash prediction") == -1

    def test_bullish_keywords(self):
        assert PolymarketService._classify_direction("gdp growth above 3%") == 1
        assert PolymarketService._classify_direction("rate cut expected") == 1
        assert PolymarketService._classify_direction("economic recovery") == 1

    def test_neutral_question(self):
        assert PolymarketService._classify_direction("will it rain tomorrow") == 0

    def test_mixed_keywords_bearish_wins(self):
        """When more bearish keywords than bullish, classify as bearish."""
        # "recession" and "crash" are bearish, "growth" is bullish
        assert (
            PolymarketService._classify_direction(
                "recession and crash despite growth"
            )
            == -1
        )

    def test_mixed_keywords_bullish_wins(self):
        """When more bullish keywords than bearish, classify as bullish."""
        # "growth" and "recovery" are bullish, "recession" is bearish
        assert (
            PolymarketService._classify_direction(
                "growth and recovery after recession"
            )
            == 1
        )


# ---------------------------------------------------------------------------
# _recency_weight
# ---------------------------------------------------------------------------


class TestRecencyWeight:
    """Tests for the internal _recency_weight method."""

    def test_recent_market_high_weight(self):
        """A market ending today should have high weight."""
        now = datetime.now(timezone.utc)
        market = {"endDate": now.isoformat()}
        weight = PolymarketService._recency_weight(market)
        assert weight > 0.9

    def test_old_market_low_weight(self):
        """A market ending 30 days ago should have low weight."""
        from datetime import timedelta

        old = datetime.now(timezone.utc) - timedelta(days=30)
        market = {"endDate": old.isoformat()}
        weight = PolymarketService._recency_weight(market)
        assert weight < 0.1

    def test_no_date_returns_default(self):
        """A market with no date info should get default weight."""
        market = {}
        weight = PolymarketService._recency_weight(market)
        assert weight == 0.5

    def test_uses_updated_at_fallback(self):
        """Should use updatedAt when endDate is not available."""
        now = datetime.now(timezone.utc)
        market = {"updatedAt": now.isoformat()}
        weight = PolymarketService._recency_weight(market)
        assert weight > 0.9

    def test_invalid_date_returns_default(self):
        """Invalid date string should return default weight."""
        market = {"endDate": "not-a-date"}
        weight = PolymarketService._recency_weight(market)
        assert weight == 0.5


# ---------------------------------------------------------------------------
# fetch_markets
# ---------------------------------------------------------------------------


class TestFetchMarkets:
    """Tests for fetch_markets() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_fetches_from_all_tags(self):
        """Should make one request per relevant tag."""
        svc = PolymarketService()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[
                {"id": "m1", "question": "Test market", "volume": 1000}
            ]
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        svc._session = mock_session

        markets = await svc.fetch_markets()

        # Should have called get() once per tag
        assert mock_session.get.call_count == len(RELEVANT_TAGS)

        # All calls should use the correct base URL
        for call in mock_session.get.call_args_list:
            assert call[0][0] == f"{GAMMA_API_BASE}/markets"

    @pytest.mark.asyncio
    async def test_deduplicates_markets(self):
        """Markets with the same ID from different tags should be
        deduplicated."""
        svc = PolymarketService()

        # Same market returned for every tag
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[
                {"id": "same-id", "question": "Duplicate market", "volume": 1000}
            ]
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        svc._session = mock_session

        markets = await svc.fetch_markets()

        # Should only have 1 unique market despite multiple tags
        assert len(markets) == 1

    @pytest.mark.asyncio
    async def test_handles_api_error_gracefully(self):
        """Should return empty list and log warning on HTTP error."""
        svc = PolymarketService()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        svc._session = mock_session

        markets = await svc.fetch_markets()
        assert markets == []

    @pytest.mark.asyncio
    async def test_handles_network_exception(self):
        """Should return empty list on network exception."""
        svc = PolymarketService()

        mock_response = AsyncMock()
        mock_response.__aenter__ = AsyncMock(
            side_effect=Exception("Connection refused")
        )
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        svc._session = mock_session

        markets = await svc.fetch_markets()
        assert markets == []

    @pytest.mark.asyncio
    async def test_creates_session_if_none(self):
        """Should create a new aiohttp session if none exists."""
        svc = PolymarketService()
        assert svc._session is None

        with patch("src.services.polymarket_service.aiohttp.ClientSession") as mock_cls:
            mock_session = MagicMock()
            mock_session.closed = False

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[])
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_session.get = MagicMock(return_value=mock_response)

            mock_cls.return_value = mock_session

            await svc.fetch_markets()
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    """Tests for update() — the scheduler entry point."""

    @pytest.mark.asyncio
    async def test_updates_score_on_success(self):
        """Should update sentiment score and last fetch time on success."""
        svc = PolymarketService()

        markets = [_make_bearish_market(probability=0.8)]
        with patch.object(svc, "fetch_markets", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = markets
            await svc.update()

        assert svc.sentiment_score != 0.0
        assert svc.last_fetch_time is not None

    @pytest.mark.asyncio
    async def test_keeps_last_score_on_empty_fetch(self):
        """Should keep last score when no markets are fetched."""
        svc = PolymarketService()
        svc._sentiment_score = -0.5
        original_fetch = svc._last_fetch

        with patch.object(svc, "fetch_markets", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []
            await svc.update()

        assert svc.sentiment_score == -0.5
        assert svc.last_fetch_time == original_fetch

    @pytest.mark.asyncio
    async def test_keeps_last_score_on_exception(self):
        """Should keep last score and log warning on API failure."""
        svc = PolymarketService()
        svc._sentiment_score = 0.3
        svc._last_fetch = datetime(2025, 1, 1, tzinfo=timezone.utc)

        with patch.object(svc, "fetch_markets", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("API timeout")
            await svc.update()

        assert svc.sentiment_score == 0.3
        assert svc.last_fetch_time == datetime(2025, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    """Tests for close() — session cleanup."""

    @pytest.mark.asyncio
    async def test_closes_open_session(self):
        """Should close the aiohttp session."""
        svc = PolymarketService()
        mock_session = AsyncMock()
        mock_session.closed = False
        svc._session = mock_session

        await svc.close()

        mock_session.close.assert_called_once()
        assert svc._session is None

    @pytest.mark.asyncio
    async def test_noop_when_no_session(self):
        """Should not raise when no session exists."""
        svc = PolymarketService()
        await svc.close()  # should not raise

    @pytest.mark.asyncio
    async def test_noop_when_session_already_closed(self):
        """Should not close an already-closed session."""
        svc = PolymarketService()
        mock_session = AsyncMock()
        mock_session.closed = True
        svc._session = mock_session

        await svc.close()

        mock_session.close.assert_not_called()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants match the design document."""

    def test_gamma_api_base(self):
        assert GAMMA_API_BASE == "https://gamma-api.polymarket.com"

    def test_relevant_tags(self):
        expected = [
            "economics",
            "politics",
            "fed",
            "inflation",
            "recession",
            "interest-rates",
        ]
        assert RELEVANT_TAGS == expected
