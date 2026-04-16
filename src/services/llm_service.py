"""LLMService: manages all interactions with the Anthropic LLM API.

Enforces the daily call limit and logs token usage for cost tracking.
All calls are persisted to the database via StateManager.

Requirements: 23.1-23.7, 21.5
"""

import logging
import os
from datetime import date, datetime
from typing import List, Optional

import anthropic

from src.config import AgentConfig
from src.services.state_manager import StateManager

logger = logging.getLogger(__name__)


class LLMService:
    """Manages LLM API calls with rate limiting and cost tracking."""

    def __init__(self, config: AgentConfig, state_manager: StateManager) -> None:
        self._config = config
        self._state_manager = state_manager
        self._client: Optional[anthropic.AsyncAnthropic] = None
        self._daily_call_count: int = 0
        self._last_reset_date: Optional[date] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the Anthropic async client and load today's call count."""
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        self._daily_call_count = await self._state_manager.get_llm_calls_today()
        self._last_reset_date = date.today()
        logger.info(
            "LLMService initialized — model=%s, calls_today=%d, limit=%d",
            self._config.llm_model,
            self._daily_call_count,
            self._config.max_llm_calls_per_day,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def interpret_sentiment(
        self, polymarket_data: List[dict], news_context: str
    ) -> str:
        """Interpret Polymarket sentiment and news context.

        Called by PolymarketService to produce a structured sentiment
        analysis from prediction-market data and recent news.
        """
        system_prompt = (
            "You are a financial sentiment analyst specializing in prediction markets "
            "and macroeconomic indicators. Analyze the provided Polymarket data and news "
            "context to produce a concise sentiment assessment. Focus on: overall market "
            "sentiment (bullish/bearish/neutral), key risk factors, and potential impact "
            "on US equities. Be specific and quantitative where possible."
        )

        user_message = (
            f"Polymarket Data:\n{_format_polymarket_data(polymarket_data)}\n\n"
            f"News Context:\n{news_context}"
        )

        return await self._call_llm(
            purpose="sentiment_interpretation",
            system_prompt=system_prompt,
            user_message=user_message,
        )

    async def generate_report_content(
        self, portfolio_data: dict, trades_summary: str
    ) -> str:
        """Generate narrative content for the daily email report.

        Called once daily by ReportGenerator to produce a human-readable
        summary of trading activity and portfolio performance.
        """
        system_prompt = (
            "You are a trading report writer for an automated equity trading agent. "
            "Write a concise daily summary covering: portfolio performance, notable "
            "trades, risk metrics, and market conditions. Use a professional but "
            "accessible tone. Include specific numbers and percentages."
        )

        user_message = (
            f"Portfolio Data:\n{_format_dict(portfolio_data)}\n\n"
            f"Trades Summary:\n{trades_summary}"
        )

        return await self._call_llm(
            purpose="report_generation",
            system_prompt=system_prompt,
            user_message=user_message,
        )

    async def interpret_unusual_conditions(self, market_data: dict) -> str:
        """Interpret unusual market conditions that don't match known patterns.

        Called exceptionally when the StrategyEngine encounters market
        behaviour outside its rule-based models.
        """
        system_prompt = (
            "You are a market analyst specializing in unusual trading conditions. "
            "Analyze the provided market data and identify: what is unusual, possible "
            "causes, recommended risk adjustments, and whether trading should be paused. "
            "Be concise and actionable."
        )

        user_message = f"Market Data:\n{_format_dict(market_data)}"

        return await self._call_llm(
            purpose="unusual_conditions",
            system_prompt=system_prompt,
            user_message=user_message,
        )

    @property
    def daily_calls_remaining(self) -> int:
        """Number of LLM calls remaining for today."""
        return max(0, self._config.max_llm_calls_per_day - self._daily_call_count)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call_llm(
        self, purpose: str, system_prompt: str, user_message: str
    ) -> str:
        """Make a rate-limited LLM API call with full logging and persistence.

        Returns the response text, or an empty string if the daily limit
        is reached or an API error occurs.
        """
        self._reset_daily_counter_if_needed()

        if self._daily_call_count >= self._config.max_llm_calls_per_day:
            logger.warning(
                "LLM daily limit reached (%d/%d) — skipping call for purpose=%s",
                self._daily_call_count,
                self._config.max_llm_calls_per_day,
                purpose,
            )
            return ""

        try:
            response = await self._client.messages.create(
                model=self._config.llm_model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract response text
            response_text = response.content[0].text if response.content else ""

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            logger.info(
                "LLM call complete: purpose=%s model=%s input_tokens=%d "
                "output_tokens=%d total_tokens=%d",
                purpose,
                self._config.llm_model,
                input_tokens,
                output_tokens,
                total_tokens,
            )

            await self._state_manager.persist_llm_call(
                purpose=purpose,
                model=self._config.llm_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                success=True,
            )

            self._daily_call_count += 1
            return response_text

        except Exception as exc:
            logger.warning(
                "LLM API error for purpose=%s: %s",
                purpose,
                exc,
            )
            await self._state_manager.persist_llm_call(
                purpose=purpose,
                model=self._config.llm_model,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                success=False,
                error_message=str(exc),
            )
            self._daily_call_count += 1
            return ""

    def _reset_daily_counter_if_needed(self) -> None:
        """Reset the daily call counter if the date has changed."""
        today = date.today()
        if today != self._last_reset_date:
            self._daily_call_count = 0
            self._last_reset_date = today
            logger.info("LLM daily call counter reset for %s", today)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _format_polymarket_data(data: List[dict]) -> str:
    """Format Polymarket data list into a readable string for the LLM."""
    if not data:
        return "No Polymarket data available."

    lines = []
    for i, market in enumerate(data, 1):
        question = market.get("question", "Unknown")
        outcome_prices = market.get("outcomePrices", "N/A")
        volume = market.get("volume", "N/A")
        lines.append(
            f"{i}. {question}\n"
            f"   Outcome Prices: {outcome_prices}\n"
            f"   Volume: {volume}"
        )
    return "\n".join(lines)


def _format_dict(data: dict) -> str:
    """Format a dict into a readable key-value string for the LLM."""
    if not data:
        return "No data available."

    lines = []
    for key, value in data.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)
