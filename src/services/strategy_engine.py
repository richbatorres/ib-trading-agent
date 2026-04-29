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
    calculate_trend_strength,
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
        "momentum": 0.75,
        "mean_reversion": 0.65,
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

        # Strategy performance tracking for weighted aggregation
        self._strategy_wins: Dict[str, int] = {"momentum": 0, "mean_reversion": 0, "trend_following": 0}
        self._strategy_total: Dict[str, int] = {"momentum": 0, "mean_reversion": 0, "trend_following": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_indicator(value: float, min_val: float, max_val: float) -> float:
        """Normalize an indicator value to [0, 1] range.

        Clamps to [min_val, max_val] then scales linearly.
        Returns 0.5 if min_val == max_val (no range).
        """
        if max_val == min_val:
            return 0.5
        clamped = max(min_val, min(max_val, value))
        return (clamped - min_val) / (max_val - min_val)

    def record_trade_outcome(self, strategy: str, profitable: bool) -> None:
        """Record whether a trade from a strategy was profitable.

        Used by the agent to feed back trade results for weighted aggregation.
        """
        if strategy in self._strategy_total:
            self._strategy_total[strategy] += 1
            if profitable:
                self._strategy_wins[strategy] += 1
            logger.debug(
                "Strategy %s outcome: %s (win_rate=%.1f%%)",
                strategy,
                "WIN" if profitable else "LOSS",
                self._get_strategy_weight(strategy) * 100,
            )

    def _get_strategy_weight(self, strategy: str) -> float:
        """Get performance-based weight for a strategy.

        Returns win rate if enough data (>= 10 trades), otherwise 0.5 (neutral).
        """
        total = self._strategy_total.get(strategy, 0)
        if total < 10:
            return 0.5  # Not enough data — neutral weight
        wins = self._strategy_wins.get(strategy, 0)
        return wins / total

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

        # Market hours filter — check configured sessions
        # If any configured session is active, allow trading.
        # Falls back to the original is_market_open() for backward compatibility.
        configured_sessions = getattr(self, '_configured_sessions', None)
        if configured_sessions:
            any_active = False
            for sess_name in configured_sessions:
                if hasattr(self._market_hours, 'is_session_active') and self._market_hours.is_session_active(sess_name):
                    any_active = True
                    break
            if not any_active and not self._market_hours.is_market_open():
                logger.info(
                    "Signal suppressed for %s: no configured session active", symbol
                )
                return None
        elif not self._market_hours.is_market_open():
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

        # Missing data robustness — skip if price array has NaN or zero-length
        if len(prices) == 0 or np.any(np.isnan(prices)):
            logger.debug("Skipping %s: missing or invalid price data", symbol)
            return None
        if len(volumes) == 0 or np.any(np.isnan(volumes)):
            logger.debug("Skipping %s: missing or invalid volume data", symbol)
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

        rsi, macd_hist, bb_upper, bb_middle, bb_lower, ema_9, ema_21, trend_strength = indicators

        # --- Retrieve previous indicator values for crossover detection ---
        prev = self._prev_indicators.get(symbol, {})

        # --- Evaluate strategies ---
        raw_signals: List[Tuple[str, str, float, Dict[str, float]]] = []

        momentum = self._evaluate_momentum(rsi, macd_hist, prev, trend_strength)
        if momentum is not None:
            direction, confidence = momentum
            raw_signals.append((
                "momentum",
                direction,
                confidence,
                {"rsi": rsi, "macd_histogram": macd_hist},
            ))

        mean_rev = self._evaluate_mean_reversion(price, bb_upper, bb_lower, rsi, prev)
        if mean_rev is not None:
            direction, confidence = mean_rev
            raw_signals.append((
                "mean_reversion",
                direction,
                confidence,
                {"bb_upper": bb_upper, "bb_middle": bb_middle, "bb_lower": bb_lower},
            ))

        trend = self._evaluate_trend_following(ema_9, ema_21, prev, trend_strength)
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
            "trend_strength": trend_strength,
        }

        if not raw_signals:
            return None

        # --- Multi-strategy agreement ---
        signal = self._resolve_signals(
            raw_signals, symbol, price, volume, rsi, macd_hist,
            bb_upper, bb_middle, bb_lower, ema_9, ema_21, trend_strength,
        )
        return signal

    # ------------------------------------------------------------------
    # Indicator calculation
    # ------------------------------------------------------------------

    def _calculate_indicators(
        self, prices: np.ndarray
    ) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
        """Calculate all required indicators.

        Returns ``None`` if there is insufficient data for any indicator.
        Otherwise returns ``(rsi, macd_hist, bb_upper, bb_middle,
        bb_lower, ema_9, ema_21, trend_strength)``.
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
            # Trend strength requires at least 21 prices (period=20 + 1)
            if n >= 21:
                trend_strength = calculate_trend_strength(prices)
            else:
                trend_strength = 50.0  # neutral default
        except ValueError:
            # Not enough data for one of the indicators
            return None

        return rsi, macd_hist, bb_upper, bb_middle, bb_lower, ema_9, ema_21, trend_strength

    def _calculate_indicators_cached(
        self, symbol: str, prices: np.ndarray
    ) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
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

            trend_strength = cache.get_indicator(symbol, "trend_strength")
            if trend_strength is None:
                if n >= 21:
                    trend_strength = calculate_trend_strength(prices)
                else:
                    trend_strength = 50.0  # neutral default
                cache.set_indicator(symbol, "trend_strength", trend_strength)

        except ValueError:
            return None

        return rsi, macd_hist, bb_upper, bb_middle, bb_lower, ema_9, ema_21, trend_strength

    # ------------------------------------------------------------------
    # Strategy evaluators
    # ------------------------------------------------------------------

    def _evaluate_momentum(
        self,
        rsi: float,
        macd_hist: float,
        prev: Dict[str, Any],
        trend_strength: float = 30.0,
    ) -> Optional[Tuple[str, float]]:
        """Momentum strategy.

        BUY:  RSI < 35 AND MACD histogram turns positive (from ≤ 0).
        SELL: RSI > 65 AND MACD histogram turns negative (from ≥ 0).

        Trend filter: weak trend (< 20) reduces confidence by 30%;
        strong trend (> 40) boosts confidence by 10%.

        Returns ``(direction, confidence)`` or ``None``.
        """
        prev_macd_hist = prev.get("macd_histogram")

        if prev_macd_hist is None:
            return None

        confidence = self._BASE_CONFIDENCE["momentum"]

        # BUY: RSI in oversold zone AND MACD histogram turns positive
        if rsi < 35 and prev_macd_hist <= 0 and macd_hist > 0:
            # Apply trend filter
            if trend_strength < 20:
                confidence *= 0.70
            elif trend_strength > 40:
                confidence *= 1.10
            return ("BUY", min(confidence, 1.0))

        # SELL: RSI in overbought zone AND MACD histogram turns negative
        if rsi > 65 and prev_macd_hist >= 0 and macd_hist < 0:
            if trend_strength < 20:
                confidence *= 0.70
            elif trend_strength > 40:
                confidence *= 1.10
            return ("SELL", min(confidence, 1.0))

        return None

    def _evaluate_mean_reversion(
        self,
        price: float,
        bb_upper: float,
        bb_lower: float,
        rsi: float,
        prev: Dict[str, Any],
    ) -> Optional[Tuple[str, float]]:
        """Mean reversion strategy.

        BUY:  Price crosses below lower Bollinger Band AND RSI < 40.
        SELL: Price crosses above upper Bollinger Band AND RSI > 60.

        RSI confirmation reduces noise signals from minor BB touches.

        Returns ``(direction, confidence)`` or ``None``.
        """
        prev_price = prev.get("price")
        prev_bb_lower = prev.get("bb_lower")
        prev_bb_upper = prev.get("bb_upper")

        if prev_price is None or prev_bb_lower is None or prev_bb_upper is None:
            return None

        # BUY: price crosses below lower BB AND RSI confirms oversold
        if prev_price >= prev_bb_lower and price < bb_lower and rsi < 40:
            return ("BUY", self._BASE_CONFIDENCE["mean_reversion"])

        # SELL: price crosses above upper BB AND RSI confirms overbought
        if prev_price <= prev_bb_upper and price > bb_upper and rsi > 60:
            return ("SELL", self._BASE_CONFIDENCE["mean_reversion"])

        return None

    def _evaluate_trend_following(
        self,
        ema_9: float,
        ema_21: float,
        prev: Dict[str, Any],
        trend_strength: float = 30.0,
    ) -> Optional[Tuple[str, float]]:
        """Trend following strategy.

        BUY:  9-EMA crosses above 21-EMA.
        SELL: 9-EMA crosses below 21-EMA.

        Trend filter: weak trend (< 20) reduces confidence by 30%;
        strong trend (> 40) boosts confidence by 10%.

        Returns ``(direction, confidence)`` or ``None``.
        """
        prev_ema_9 = prev.get("ema_9")
        prev_ema_21 = prev.get("ema_21")

        if prev_ema_9 is None or prev_ema_21 is None:
            return None

        confidence = self._BASE_CONFIDENCE["trend_following"]

        # BUY: 9-EMA crosses above 21-EMA
        if prev_ema_9 <= prev_ema_21 and ema_9 > ema_21:
            if trend_strength < 20:
                confidence *= 0.70
            elif trend_strength > 40:
                confidence *= 1.10
            return ("BUY", min(confidence, 1.0))

        # SELL: 9-EMA crosses below 21-EMA
        if prev_ema_9 >= prev_ema_21 and ema_9 < ema_21:
            if trend_strength < 20:
                confidence *= 0.70
            elif trend_strength > 40:
                confidence *= 1.10
            return ("SELL", min(confidence, 1.0))

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
        trend_strength: float = 30.0,
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

        # Apply performance-based strategy weight
        strategy_weight = self._get_strategy_weight(best_strategy)
        # Weight adjusts confidence: 0.5 = neutral, >0.5 = boost, <0.5 = reduce
        weight_factor = 0.5 + strategy_weight  # range [0.5, 1.5]
        boosted_confidence = min(best_confidence * agreement_multiplier * weight_factor, 1.0)

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
            "trend_strength": trend_strength,
        }
        all_indicators.update(best_inds)

        # Normalized features for downstream consumers
        all_indicators["norm_rsi"] = self._normalize_indicator(rsi, 0.0, 100.0)
        all_indicators["norm_trend_strength"] = self._normalize_indicator(trend_strength, 0.0, 100.0)
        all_indicators["norm_polymarket"] = self._normalize_indicator(self.polymarket_sentiment, -1.0, 1.0)

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
