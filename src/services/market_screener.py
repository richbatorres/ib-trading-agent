"""Market Screener: scans S&P 500 daily and selects top trading candidates.

Runs once before market open. Fetches all S&P 500 symbols from Wikipedia,
scores each by volume, volatility, and momentum, then returns the top N
candidates for active trading that day.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default number of candidates to select for active trading
DEFAULT_TOP_N = 30


@dataclass
class CandidateScore:
    """Screening score for a single stock."""
    symbol: str
    avg_volume: float
    volatility: float      # 20-day std dev of returns
    momentum_score: float  # RSI-based: distance from 50 (oversold/overbought)
    total_score: float


class MarketScreener:
    """Scans S&P 500 and selects top trading candidates daily.

    Scoring criteria (higher = better candidate for our strategies):
    1. Volume — high volume = better liquidity, more reliable signals
    2. Volatility — moderate volatility = more trading opportunities
    3. Momentum — stocks near RSI extremes (oversold/overbought) = mean reversion
    """

    def __init__(self, top_n: int = DEFAULT_TOP_N, max_share_price: float = 0.0) -> None:
        self._top_n = top_n
        self._max_share_price = max_share_price  # 0 = no filter
        self._sp500_symbols: List[str] = []
        self._last_candidates: List[str] = []

    def get_sp500_symbols(self) -> List[str]:
        """Fetch current S&P 500 constituent symbols from Wikipedia."""
        try:
            import pandas as pd
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df["Symbol"].tolist()
            # Clean up: replace dots with dashes (BRK.B -> BRK-B for Yahoo)
            symbols = [s.replace(".", "-") for s in symbols]
            self._sp500_symbols = symbols
            logger.info("Loaded %d S&P 500 symbols", len(symbols))
            return symbols
        except Exception as exc:
            logger.warning("Failed to fetch S&P 500 list: %s — using cached", exc)
            if self._sp500_symbols:
                return self._sp500_symbols
            # Fallback: top 50 by market cap
            return self._fallback_symbols()

    def screen(self, symbols: Optional[List[str]] = None) -> List[str]:
        """Screen symbols and return top N candidates for today.

        Fetches 5 days of data for each symbol, scores by volume,
        volatility, and momentum, returns the top N.

        Args:
            symbols: List to screen. If None, fetches S&P 500.

        Returns:
            List of top N symbol strings, sorted by score descending.
        """
        import yfinance as yf

        if symbols is None:
            symbols = self.get_sp500_symbols()

        logger.info("Screening %d symbols for top %d candidates...", len(symbols), self._top_n)

        scores: List[CandidateScore] = []
        batch_size = 50

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_str = " ".join(batch)

            try:
                data = yf.download(
                    batch_str,
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    threads=True,
                )

                for sym in batch:
                    try:
                        if len(batch) == 1:
                            close = data["Close"].dropna().values
                            volume = data["Volume"].dropna().values
                        else:
                            close = data[sym]["Close"].dropna().values
                            volume = data[sym]["Volume"].dropna().values

                        if len(close) < 3 or len(volume) < 3:
                            continue

                        score = self._score_candidate(sym, close, volume)
                        if score is not None:
                            scores.append(score)
                    except Exception:
                        continue

            except Exception as exc:
                logger.debug("Batch download failed for %d symbols: %s", len(batch), exc)
                continue

            if (i + batch_size) % 200 == 0:
                logger.info("Screened %d/%d symbols...", min(i + batch_size, len(symbols)), len(symbols))

        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # Take top N
        top = scores[:self._top_n]
        self._last_candidates = [s.symbol for s in top]

        logger.info(
            "Screening complete: %d scored, top %d selected",
            len(scores), len(top),
        )
        for s in top[:5]:
            logger.info(
                "  Top candidate: %s (score=%.2f, vol=%.0f, volatility=%.4f, momentum=%.2f)",
                s.symbol, s.total_score, s.avg_volume, s.volatility, s.momentum_score,
            )

        return self._last_candidates

    @property
    def last_candidates(self) -> List[str]:
        """Return the most recently screened candidates."""
        return self._last_candidates

    def _score_candidate(
        self, symbol: str, close: np.ndarray, volume: np.ndarray
    ) -> Optional[CandidateScore]:
        """Score a single candidate based on volume, volatility, and momentum."""
        avg_volume = float(np.mean(volume))

        # Filter: minimum average volume of 500k shares
        if avg_volume < 500_000:
            return None

        # Filter: max share price (for small capital accounts)
        current_price = float(close[-1])
        if self._max_share_price > 0 and current_price > self._max_share_price:
            return None

        # Volatility: standard deviation of daily returns
        returns = np.diff(close) / close[:-1]
        volatility = float(np.std(returns)) if len(returns) > 1 else 0.0

        # Filter: skip extremely low volatility (boring stocks)
        if volatility < 0.005:
            return None

        # Momentum score: simple RSI-like measure
        # How far from the middle (50)? Stocks near extremes are better candidates
        gains = returns[returns > 0]
        losses = -returns[returns < 0]
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.001
        rs = avg_gain / avg_loss if avg_loss > 0 else 1.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        momentum_score = abs(rsi - 50.0)  # distance from neutral

        # Total score: weighted combination
        # Normalize each component to roughly 0-1 range
        vol_score = min(avg_volume / 10_000_000, 1.0)  # 10M+ volume = max score
        volat_score = min(volatility / 0.05, 1.0)       # 5%+ daily volatility = max
        mom_score = min(momentum_score / 30.0, 1.0)      # RSI 20 or 80 = max

        total = (
            vol_score * 0.3       # 30% weight on volume
            + volat_score * 0.4   # 40% weight on volatility
            + mom_score * 0.3     # 30% weight on momentum
        )

        return CandidateScore(
            symbol=symbol,
            avg_volume=avg_volume,
            volatility=volatility,
            momentum_score=momentum_score,
            total_score=total,
        )

    @staticmethod
    def _fallback_symbols() -> List[str]:
        """Fallback list: top 50 S&P 500 by market cap."""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
            "ACN", "TMO", "ABT", "DHR", "CRM", "NKE", "TXN", "NEE", "PM",
            "UNP", "RTX", "HON", "LOW", "ORCL", "INTC", "AMD", "QCOM", "IBM",
            "CAT", "GS", "BA", "AMGN", "SBUX",
        ]

    def get_ftse100_symbols(self) -> List[str]:
        """Fetch FTSE 100 constituent symbols for EU trading.
        
        Returns symbols with .L suffix for London Stock Exchange.
        Falls back to a hardcoded top-30 list if fetch fails.
        """
        try:
            import pandas as pd
            url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
            tables = pd.read_html(url)
            # The constituents table typically has a 'Ticker' or 'EPIC' column
            for table in tables:
                for col in ["Ticker", "EPIC", "ticker", "epic"]:
                    if col in table.columns:
                        symbols = [str(s).strip() + ".L" for s in table[col].tolist() if str(s).strip()]
                        if len(symbols) > 20:
                            logger.info("Loaded %d FTSE 100 symbols", len(symbols))
                            return symbols
            raise ValueError("Could not find ticker column in FTSE 100 table")
        except Exception as exc:
            logger.warning("Failed to fetch FTSE 100 list: %s — using fallback", exc)
            return self._fallback_eu_symbols()

    def get_nikkei225_symbols(self) -> List[str]:
        """Get top Nikkei 225 symbols for ASIA trading.
        
        Returns symbols with .T suffix for Tokyo Stock Exchange.
        Uses a curated list (Nikkei 225 Wikipedia page is harder to parse).
        """
        return self._fallback_asia_symbols()

    @staticmethod
    def _fallback_eu_symbols() -> List[str]:
        """Fallback: top 30 FTSE 100 by market cap with .L suffix."""
        return [
            "SHEL.L", "AZN.L", "ULVR.L", "HSBA.L", "BP.L",
            "RIO.L", "GSK.L", "DGE.L", "BATS.L", "REL.L",
            "LSEG.L", "AAL.L", "BHP.L", "VOD.L", "NG.L",
            "PRU.L", "CPG.L", "EXPN.L", "CRH.L", "AHT.L",
            "RKT.L", "BARC.L", "LLOY.L", "GLEN.L", "IMB.L",
            "SSE.L", "ABF.L", "ANTO.L", "TSCO.L", "WPP.L",
        ]

    @staticmethod
    def _fallback_asia_symbols() -> List[str]:
        """Fallback: top 20 Nikkei 225 by market cap with .T suffix."""
        return [
            "7203.T", "6758.T", "9984.T", "8306.T", "6861.T",
            "9432.T", "6501.T", "7267.T", "4502.T", "6902.T",
            "8035.T", "6098.T", "4063.T", "7974.T", "9433.T",
            "3382.T", "8058.T", "2914.T", "4568.T", "6367.T",
        ]

    def screen_for_session(self, session: str) -> List[str]:
        """Screen symbols for a specific trading session.
        
        Args:
            session: "US", "EU", or "ASIA"
        
        Returns:
            Top N candidates for the given session.
        """
        if session == "EU":
            symbols = self.get_ftse100_symbols()
        elif session == "ASIA":
            symbols = self.get_nikkei225_symbols()
        else:
            symbols = self.get_sp500_symbols()
        
        return self.screen(symbols=symbols)
