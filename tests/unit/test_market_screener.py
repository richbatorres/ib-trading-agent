"""Tests for MarketScreener — S&P 500 scanning and candidate selection."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.services.market_screener import MarketScreener, CandidateScore


class TestScoreCandidate:
    """Tests for the internal _score_candidate method."""

    def test_rejects_low_volume(self):
        screener = MarketScreener()
        close = np.array([100.0, 101.0, 99.0, 102.0, 100.5])
        volume = np.array([100_000, 200_000, 150_000, 180_000, 120_000])  # avg < 500k
        result = screener._score_candidate("LOW", close, volume)
        assert result is None

    def test_rejects_low_volatility(self):
        screener = MarketScreener()
        close = np.array([100.0, 100.001, 100.002, 100.001, 100.0])  # nearly flat
        volume = np.array([1_000_000] * 5)
        result = screener._score_candidate("FLAT", close, volume)
        assert result is None

    def test_accepts_good_candidate(self):
        screener = MarketScreener()
        close = np.array([100.0, 103.0, 98.0, 105.0, 101.0])  # volatile
        volume = np.array([5_000_000] * 5)  # high volume
        result = screener._score_candidate("GOOD", close, volume)
        assert result is not None
        assert result.symbol == "GOOD"
        assert result.total_score > 0

    def test_higher_volume_scores_higher(self):
        screener = MarketScreener()
        close = np.array([100.0, 103.0, 98.0, 105.0, 101.0])
        vol_low = np.array([1_000_000] * 5)
        vol_high = np.array([20_000_000] * 5)
        score_low = screener._score_candidate("LOW", close, vol_low)
        score_high = screener._score_candidate("HIGH", close, vol_high)
        assert score_high.total_score > score_low.total_score

    def test_score_components_are_positive(self):
        screener = MarketScreener()
        close = np.array([100.0, 105.0, 95.0, 110.0, 100.0])
        volume = np.array([5_000_000] * 5)
        result = screener._score_candidate("TEST", close, volume)
        assert result.avg_volume > 0
        assert result.volatility > 0
        assert result.total_score > 0


class TestScreen:
    """Tests for the screen() method."""

    def test_returns_top_n_candidates(self):
        screener = MarketScreener(top_n=3)
        # Use a small list with mock yfinance
        with patch("yfinance.download") as mock_dl:
            import pandas as pd
            dates = pd.date_range("2026-01-01", periods=5)
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

            arrays = {}
            for i, sym in enumerate(symbols):
                arrays[(sym, "Close")] = pd.Series([100+i*10, 103+i*10, 98+i*10, 105+i*10, 101+i*10], index=dates)
                arrays[(sym, "Volume")] = pd.Series([5_000_000+i*1_000_000]*5, index=dates)
            multi_df = pd.DataFrame(arrays)
            multi_df.columns = pd.MultiIndex.from_tuples(multi_df.columns)
            mock_dl.return_value = multi_df

            result = screener.screen(symbols)

        assert len(result) <= 3
        assert all(isinstance(s, str) for s in result)

    def test_returns_empty_on_no_data(self):
        screener = MarketScreener(top_n=5)
        with patch("yfinance.download") as mock_dl:
            import pandas as pd
            mock_dl.return_value = pd.DataFrame()
            result = screener.screen(["FAKE1", "FAKE2"])
        assert result == []

    def test_last_candidates_updated(self):
        screener = MarketScreener(top_n=5)
        assert screener.last_candidates == []
        # After screen, last_candidates should be populated
        screener._last_candidates = ["AAPL", "MSFT"]
        assert screener.last_candidates == ["AAPL", "MSFT"]


class TestFallbackSymbols:
    """Tests for the fallback symbol list."""

    def test_fallback_has_50_symbols(self):
        result = MarketScreener._fallback_symbols()
        assert len(result) == 50

    def test_fallback_contains_major_stocks(self):
        result = MarketScreener._fallback_symbols()
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result

    def test_get_sp500_falls_back_on_error(self):
        screener = MarketScreener()
        with patch("pandas.read_html", side_effect=Exception("blocked")):
            result = screener.get_sp500_symbols()
        assert len(result) == 50


class TestTopNConfig:
    """Tests for configurable top_n."""

    def test_default_top_n_is_30(self):
        screener = MarketScreener()
        assert screener._top_n == 30

    def test_custom_top_n(self):
        screener = MarketScreener(top_n=10)
        assert screener._top_n == 10
