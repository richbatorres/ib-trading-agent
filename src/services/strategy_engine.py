"""StrategyEngine: evaluates trading strategies and generates signals.

Evaluates three strategies (momentum, mean reversion, trend following) on
each tick, applies volume confirmation, earnings blackout, and market hours
filters, incorporates Polymarket sentiment weighting, and handles
multi-strategy agreement for confidence boosting.

Requirements: 4.3, 4.4, 5.2, 5.3, 6.2, 6.3, 7.1, 7.2, 8.2, 13.3
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import numpy as np

from src.models.domain import TradeSignal
from src.strategies.indicator_cache import IndicatorCache
from src.strategies.indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
)

logger = logging.getLogger(__name__)


class MarketHours(Protocol):
    """Protocol for market hours checking."""

    def is_market_open(self) -> bool: ...


class StrategyEngine:
    """Evaluates trading strategies and generates trade signals.

    Runs momentum, mean reversion, and trend following strategies on each
    tick.  Applies volume confirmation, earnings blackout, and market hours
    filters before emitting a signal.  Polymarket sentiment adjusts
    confidence but never triggers a trade alone.  When multiple strategies
    agree on direction the confidence is boosted.
    """

    # Minimum data lengths required by each indicator
    _MIN_RSI_PERIODS = 15       # 14-period RSI needs at least 15 prices
    _MIN_MACD_PERIODS = 35      # MACD(12,26,9) needs at least 35 prices
    _MIN_BB_PERIODS = 20        # 20-period Bollinger Bands
    _MIN_EMA_LONG_PERIODS = 21  # 21-period EMA for trend following

    # Volume confirmation multiplier
    _VOLUME_MULTIPLIER = 1.5

    # Base confidence values per strategy
    _BASE_CONFIDENCE = {
        "momentum": 0.7,
        "mean_reversion": 0.6,
        "trend_following": 0.65,
    }

    def __init__(
        self,
        market_hours: MarketHours,
        polymarket_sentiment: float = 0.0,
        earnings_blackout_symbols: Optional[Set[str]] = None,
        indicator_cache: Optional[IndicatorCache] = None,
        market_data_type: str = "1",
    ) -> None:
        self._market_hours = market_hours
        self.polymarket_sentiment = polymarket_sentiment
        self.earnings_blackout_symbols: Set[str] = (
            earnings_blackout_symbols if earnings_blackout_symbols is not None else set()
        )
        self._indicator_cache = indicator_cache
        self._market_data_type = market_data_type  # "1"=real-time, "3"/"4"=delayed, "yahoo"

        # Per-symbol previous indicator values for crossover detection
        self._prev_indicators: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        prices: np.ndarray,
        volumes: np.ndarray,
        avg_daily_volume: float,
    ) -> Optional[TradeSignal]:
        """Main entry point.  Evaluate all strategies, apply filters,
        return the highest-confidence signal or ``None``.

        Parameters
        ----------
        symbol : str
            Ticker symbol.
        price : float
            Current price.
        volume : float
            Current tick volume.
        prices : np.ndarray
            Historical price array (most recent last).
        volumes : np.ndarray
            Historical volume array (most recent last).
        avg_daily_volume : float
            20-day average daily volume for the symbol.

        Returns
        -------
        Optional[TradeSignal]
            A trade signal with the highest confidence among agreeing
            strategies, or ``None`` if no valid signal is produced.
        """
        # --- Filters (applied before strategy evaluation) ---

        # Market hours filter
        if not self._market_hours.is_market_open():
            logger.info(
                "Signal suppressed for %s: market is closed", symbol
            )
            return None

        # Earnings blackout filter
        if symbol in self.earnings_blackout_symbols:
            logger.info(
                "Signal suppressed for %s: symbol is in earnings blackout",
                symbol,
            )
            return None

        # Volume confirmation filter
        # For delayed/frozen data (types 3, 4), volume is cumulative daily
        # volume which is always larger than the average — skip the per-tick
        # volume check. For real-time data (type 1), apply the standard
        # 1.5× average volume threshold.
        if self._market_data_type == "1":
            if avg_daily_volume > 0 and volume < self._VOLUME_MULTIPLIER * avg_daily_volume:
                logger.info(
                    "Signal rejected for %s: current volume %.0f < %.1f × avg volume %.0f (threshold %.0f)",
                    symbol,
                    volume,
                    self._VOLUME_MULTIPLIER,
                    avg_daily_volume,
                    self._VOLUME_MULTIPLIER * avg_daily_volume,
                )
                return None

        # --- Calculate indicators (with cache if available) ---
        indicators = self._calculate_indicators_cached(symbol, prices)
        if indicators is None:
            return None

        rsi, macd_hist, bb_upper, bb_middle, bb_lower, ema_9, ema_21 = indicators

        # --- Retrieve previous indicator values for crossover detection ---
        prev = self._prev_indicators.get(symbol, {})

        # --- Evaluate strategies ---
        raw_signals: List[Tuple[str, str, float, Dict[str, float]]] = []

        momentum = self._evaluate_momentum(rsi, macd_hist, prev)
        if momentum is not None:
            direction, confidence = momentum
            raw_signals.append((
                "momentum",
                direction,
                confidence,
                {"rsi": rsi, "macd_histogram": macd_hist},
            ))

        mean_rev = self._evaluate_mean_reversion(price, bb_upper, bb_lower, prev)
        if mean_rev is not None:
            direction, confidence = mean_rev
            raw_signals.append((
                "mean_reversion",
                direction,
                confidence,
                {"bb_upper": bb_upper, "bb_middle": bb_middle, "bb_lower": bb_lower},
            ))

        trend = self._evaluate_trend_following(ema_9, ema_21, prev)
        if trend is not None:
            direction, confidence = trend
            raw_signals.append((
                "trend_following",
                direction,
                confidence,
                {"ema_9": ema_9, "ema_21": ema_21},
            ))

        # --- Store current indicators for next tick's crossover detection ---
        self._prev_indicators[symbol] = {
            "rsi": rsi,
            "macd_histogram": macd_hist,
            "price": price,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "ema_9": ema_9,
            "ema_21": ema_21,
        }

        if not raw_signals:
            return None

        # --- Multi-strategy agreement ---
        signal = self._resolve_signals(
            raw_signals, symbol, price, volume, rsi, macd_hist,
            bb_upper, bb_middle, bb_lower, ema_9, ema_21,
        )
        return signal

    # ------------------------------------------------------------------
    # Indicator calculation
    # ------------------------------------------------------------------

    def _calculate_indicators(
        self, prices: np.ndarray
    ) -> Optional[Tuple[float, float, float, float, float, float, float]]:
        """Calculate all required indicators.

        Returns ``None`` if there is insufficient data for any indicator.
        Otherwise returns ``(rsi, macd_hist, bb_upper, bb_middle,
        bb_lower, ema_9, ema_21)``.
        """
        n = len(prices)
        if n < self._MIN_MACD_PERIODS:
            return None

        try:
            rsi = calculate_rsi(prices)
            _, _, macd_hist = calculate_macd(prices)
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
            ema_9 = calculate_ema(prices, 9)
            ema_21 = calculate_ema(prices, 21)
        except ValueError:
            # Not enough data for one of the indicators
            return None

        return rsi, macd_hist, bb_upper, bb_middle, bb_lower, ema_9, ema_21

    def _calculate_indicators_cached(
        self, symbol: str, prices: np.ndarray
    ) -> Optional[Tuple[float, float, float, float, float, float, float]]:
        """Calculate indicators with cache support.

        If an IndicatorCache is available, checks for cached values first
        and only recalculates dirty indicators. Falls back to full
        calculation when no cache is configured.
        """
        if self._indicator_cache is None:
            return self._calculate_indicators(prices)

        cache = self._indicator_cache
        n = len(prices)
        if n < self._MIN_MACD_PERIODS:
            return None

        try:
            # Check cache for each indicator; recalculate only if dirty
            rsi = cache.get_indicator(symbol, "rsi")
            if rsi is None:
                rsi = calculate_rsi(prices)
                cache.set_indicator(symbol, "rsi", rsi)

            macd_hist = cache.get_indicator(symbol, "macd_histogram")
            if macd_hist is None:
                _, _, macd_hist = calculate_macd(prices)
                cache.set_indicator(symbol, "macd_histogram", macd_hist)

            bb_upper = cache.get_indicator(symbol, "bb_upper")
            bb_middle = cache.get_indicator(symbol, "bb_middle")
            bb_lower = cache.get_indicator(symbol, "bb_lower")
            if bb_upper is None or bb_middle is None or bb_lower is None:
                bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
                cache.set_indicator(symbol, "bb_upper", bb_upper)
                cache.set_indicator(symbol, "bb_middle", bb_middle)
                cache.set_indicator(symbol, "bb_lower", bb_lower)

            ema_9 = cache.get_indicator(symbol, "ema_9")
            if ema_9 is None:
                ema_9 = calculate_ema(prices, 9)
                cache.set_indicator(symbol, "ema_9", ema_9)

            ema_21 = cache.get_indicator(symbol, "ema_21")
            if ema_21 is None:
                ema_21 = calculate_ema(prices, 21)
                cache.set_indicator(symbol, "ema_21", ema_21)

        except ValueError:
            return None

        return rsi, macd_hist, bb_upper, bb_middle, bb_lower, ema_9, ema_21

    # ------------------------------------------------------------------
    # Strategy evaluators
    # ------------------------------------------------------------------

    def _evaluate_momentum(
        self,
        rsi: float,
        macd_hist: float,
        prev: Dict[str, Any],
    ) -> Optional[Tuple[str, float]]:
        """Momentum strategy.

        BUY:  RSI crosses above 30 from below AND MACD histogram > 0.
        SELL: RSI crosses below 70 from above AND MACD histogram < 0.

        Returns ``(direction, confidence)`` or ``None``.
        """
        prev_rsi = prev.get("rsi")
        prev_macd_hist = prev.get("macd_histogram")

        if prev_rsi is None or prev_macd_hist is None:
            return None

        # BUY: RSI crosses above 30 from below AND MACD histogram turns positive
        if prev_rsi <= 30 and rsi > 30 and prev_macd_hist <= 0 and macd_hist > 0:
            return ("BUY", self._BASE_CONFIDENCE["momentum"])

        # SELL: RSI crosses below 70 from above AND MACD histogram turns negative
        if prev_rsi >= 70 and rsi < 70 and prev_macd_hist >= 0 and macd_hist < 0:
            return ("SELL", self._BASE_CONFIDENCE["momentum"])

        return None

    def _evaluate_mean_reversion(
        self,
        price: float,
        bb_upper: float,
        bb_lower: float,
        prev: Dict[str, Any],
    ) -> Optional[Tuple[str, float]]:
        """Mean reversion strategy.

        BUY:  Price crosses below lower Bollinger Band.
        SELL: Price crosses above upper Bollinger Band.

        Returns ``(direction, confidence)`` or ``None``.
        """
        prev_price = prev.get("price")
        prev_bb_lower = prev.get("bb_lower")
        prev_bb_upper = prev.get("bb_upper")

        if prev_price is None or prev_bb_lower is None or prev_bb_upper is None:
            return None

        # BUY: price crosses below lower BB
        if prev_price >= prev_bb_lower and price < bb_lower:
            return ("BUY", self._BASE_CONFIDENCE["mean_reversion"])

        # SELL: price crosses above upper BB
        if prev_price <= prev_bb_upper and price > bb_upper:
            return ("SELL", self._BASE_CONFIDENCE["mean_reversion"])

        return None

    def _evaluate_trend_following(
        self,
        ema_9: float,
        ema_21: float,
        prev: Dict[str, Any],
    ) -> Optional[Tuple[str, float]]:
        """Trend following strategy.

        BUY:  9-EMA crosses above 21-EMA.
        SELL: 9-EMA crosses below 21-EMA.

        Returns ``(direction, confidence)`` or ``None``.
        """
        prev_ema_9 = prev.get("ema_9")
        prev_ema_21 = prev.get("ema_21")

        if prev_ema_9 is None or prev_ema_21 is None:
            return None

        # BUY: 9-EMA crosses above 21-EMA
        if prev_ema_9 <= prev_ema_21 and ema_9 > ema_21:
            return ("BUY", self._BASE_CONFIDENCE["trend_following"])

        # SELL: 9-EMA crosses below 21-EMA
        if prev_ema_9 >= prev_ema_21 and ema_9 < ema_21:
            return ("SELL", self._BASE_CONFIDENCE["trend_following"])

        return None

    # ------------------------------------------------------------------
    # Signal resolution & sentiment weighting
    # ------------------------------------------------------------------

    def _resolve_signals(
        self,
        raw_signals: List[Tuple[str, str, float, Dict[str, float]]],
        symbol: str,
        price: float,
        volume: float,
        rsi: float,
        macd_hist: float,
        bb_upper: float,
        bb_middle: float,
        bb_lower: float,
        ema_9: float,
        ema_21: float,
    ) -> Optional[TradeSignal]:
        """Resolve multiple strategy signals into a single TradeSignal.

        When strategies agree on direction, confidence is boosted:
        - 2 agree: confidence × 1.3 (capped at 1.0)
        - 3 agree: confidence × 1.5 (capped at 1.0)

        Polymarket sentiment adjusts confidence:
        ``adjusted = base * (1 + sentiment * 0.2)`` clamped to [0, 1].

        Returns the signal with the highest adjusted confidence, or
        ``None`` if no signals remain.
        """
        # Group signals by direction
        buy_signals: List[Tuple[str, float, Dict[str, float]]] = []
        sell_signals: List[Tuple[str, float, Dict[str, float]]] = []

        for strategy, direction, confidence, inds in raw_signals:
            if direction == "BUY":
                buy_signals.append((strategy, confidence, inds))
            else:
                sell_signals.append((strategy, confidence, inds))

        # Pick the direction with the most agreement; on tie, pick highest confidence
        candidates: List[Tuple[str, List[Tuple[str, float, Dict[str, float]]]]] = []
        if buy_signals:
            candidates.append(("BUY", buy_signals))
        if sell_signals:
            candidates.append(("SELL", sell_signals))

        if not candidates:
            return None

        # Select the direction group with the most strategies agreeing
        best_direction, best_group = max(candidates, key=lambda c: len(c[1]))

        # Multi-strategy agreement boost
        agreement_count = len(best_group)
        if agreement_count >= 3:
            agreement_multiplier = 1.5
        elif agreement_count >= 2:
            agreement_multiplier = 1.3
        else:
            agreement_multiplier = 1.0

        # Find the strategy with the highest base confidence in the group
        best_strategy, best_confidence, best_inds = max(
            best_group, key=lambda s: s[1]
        )

        # Apply agreement multiplier
        boosted_confidence = min(best_confidence * agreement_multiplier, 1.0)

        # Apply Polymarket sentiment weighting
        adjusted_confidence = boosted_confidence * (1.0 + self.polymarket_sentiment * 0.2)
        adjusted_confidence = float(np.clip(adjusted_confidence, 0.0, 1.0))

        # Build combined indicators dict
        all_indicators: Dict[str, float] = {
            "rsi": rsi,
            "macd_histogram": macd_hist,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "ema_9": ema_9,
            "ema_21": ema_21,
        }
        all_indicators.update(best_inds)

        return TradeSignal(
            symbol=symbol,
            direction=best_direction,
            strategy=best_strategy,
            confidence=adjusted_confidence,
            price=price,
            volume=volume,
            indicators=all_indicators,
            polymarket_sentiment=self.polymarket_sentiment,
            timestamp=datetime.now(),
        )
