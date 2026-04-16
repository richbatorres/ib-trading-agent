"""Unit tests for ReportGenerator."""

import asyncio
import os
import shutil
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AgentConfig
from src.models.domain import PortfolioSnapshot, TradeRecord
from src.services.report_generator import ReportGenerator


@pytest.fixture
def config():
    return AgentConfig(
        ib_account_id="DU12345",
        ib_host="127.0.0.1",
        ib_port=7497,
        environment="paper",
        email_address="trader@example.com",
        email_smtp_host="smtp.example.com",
        email_smtp_port=587,
        email_smtp_user="user@example.com",
        email_smtp_password="secret",
    )


@pytest.fixture
def state_manager():
    return MagicMock()


@pytest.fixture
def generator(config, state_manager):
    return ReportGenerator(config, state_manager)


@pytest.fixture
def portfolio():
    return PortfolioSnapshot(
        total_value=105000.0,
        cash_balance=20000.0,
        positions_value=85000.0,
        daily_pnl=1500.0,
        total_pnl=5000.0,
        total_pnl_pct=5.0,
        num_open_positions=3,
        hard_stop_active=False,
        snapshot_time=datetime(2024, 1, 15, 18, 0, 0),
    )


@pytest.fixture
def sample_trades():
    return [
        TradeRecord(
            symbol="AAPL",
            direction="BUY",
            entry_price=150.0,
            quantity=100,
            stop_loss_price=142.5,
            strategy="momentum",
            signal_confidence=0.8,
            polymarket_sentiment=0.3,
            entry_time=datetime(2024, 1, 15, 10, 0),
            exit_price=155.0,
            exit_time=datetime(2024, 1, 15, 14, 0),
            realized_pnl=500.0,
            status="CLOSED",
        ),
        TradeRecord(
            symbol="MSFT",
            direction="BUY",
            entry_price=380.0,
            quantity=50,
            stop_loss_price=361.0,
            strategy="mean_reversion",
            signal_confidence=0.7,
            polymarket_sentiment=0.2,
            entry_time=datetime(2024, 1, 15, 11, 0),
            exit_price=370.0,
            exit_time=datetime(2024, 1, 15, 15, 0),
            realized_pnl=-500.0,
            status="CLOSED",
        ),
        TradeRecord(
            symbol="GOOGL",
            direction="SELL",
            entry_price=140.0,
            quantity=200,
            stop_loss_price=147.0,
            strategy="trend_following",
            signal_confidence=0.9,
            polymarket_sentiment=-0.1,
            entry_time=datetime(2024, 1, 15, 12, 0),
            status="OPEN",
        ),
    ]


@pytest.fixture
def open_positions():
    return [
        {
            "symbol": "GOOGL",
            "quantity": 200,
            "avg_cost": 140.0,
            "current_price": 138.0,
            "unrealized_pnl": -400.0,
        },
        {
            "symbol": "TSLA",
            "quantity": 50,
            "avg_cost": 240.0,
            "current_price": 250.0,
            "unrealized_pnl": 500.0,
        },
    ]


@pytest.fixture
def cleanup_reports():
    """Clean up data/reports directory after tests."""
    yield
    reports_dir = os.path.join("data", "reports")
    if os.path.exists(reports_dir):
        shutil.rmtree(reports_dir)


class TestGenerateReport:
    """Tests for generate_report method."""

    @pytest.mark.asyncio
    async def test_report_contains_portfolio_summary(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert "Portfolio Summary" in html
        assert "$105,000.00" in html
        assert "$20,000.00" in html
        assert "$85,000.00" in html

    @pytest.mark.asyncio
    async def test_report_contains_daily_pnl(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert "+$1,500.00" in html or "$+1,500.00" in html or "+1,500.00" in html

    @pytest.mark.asyncio
    async def test_report_contains_total_pnl(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert "+$5,000.00" in html or "$+5,000.00" in html or "+5,000.00" in html
        assert "+5.00%" in html

    @pytest.mark.asyncio
    async def test_report_contains_trades(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert "AAPL" in html
        assert "MSFT" in html
        assert "GOOGL" in html
        assert "Trades Today" in html

    @pytest.mark.asyncio
    async def test_report_contains_open_positions(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert "Open Positions" in html
        assert "GOOGL" in html
        assert "TSLA" in html

    @pytest.mark.asyncio
    async def test_report_contains_sentiment(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert "Polymarket Sentiment" in html
        assert "Bullish" in html

    @pytest.mark.asyncio
    async def test_report_no_trades(self, generator, portfolio, open_positions):
        html = await generator.generate_report(portfolio, [], open_positions, 0.0)

        assert "No trades executed today" in html

    @pytest.mark.asyncio
    async def test_report_no_open_positions(self, generator, portfolio, sample_trades):
        html = await generator.generate_report(portfolio, sample_trades, [], 0.0)

        assert "No open positions" in html

    @pytest.mark.asyncio
    async def test_report_is_valid_html(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<body" in html

    @pytest.mark.asyncio
    async def test_report_top_winners_and_losers(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        assert "Top Winners" in html
        assert "Top Losers" in html

    @pytest.mark.asyncio
    async def test_report_color_coded_pnl(self, generator, portfolio, sample_trades, open_positions):
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.5)

        # Green for gains, red for losses
        assert "#2ecc71" in html  # green
        assert "#e74c3c" in html  # red


class TestWarningBanner:
    """Tests for _render_warning_banner method."""

    def test_no_banner_when_loss_under_10(self, generator):
        result = generator._render_warning_banner(-5.0)
        assert result == ""

    def test_no_banner_when_positive(self, generator):
        result = generator._render_warning_banner(5.0)
        assert result == ""

    def test_warning_banner_at_10_percent_loss(self, generator):
        result = generator._render_warning_banner(-10.0)
        assert "WARNING" in result
        assert "10%" in result

    def test_warning_banner_at_15_percent_loss(self, generator):
        result = generator._render_warning_banner(-15.0)
        assert "WARNING" in result
        assert "CRITICAL" not in result

    def test_critical_banner_at_20_percent_loss(self, generator):
        result = generator._render_warning_banner(-20.0)
        assert "CRITICAL" in result
        assert "20%" in result

    def test_critical_banner_at_25_percent_loss(self, generator):
        result = generator._render_warning_banner(-25.0)
        assert "CRITICAL" in result

    @pytest.mark.asyncio
    async def test_warning_banner_in_report(self, generator, sample_trades, open_positions):
        portfolio = PortfolioSnapshot(
            total_value=85000.0,
            cash_balance=10000.0,
            positions_value=75000.0,
            daily_pnl=-2000.0,
            total_pnl=-15000.0,
            total_pnl_pct=-15.0,
            num_open_positions=2,
            hard_stop_active=False,
            snapshot_time=datetime(2024, 1, 15, 18, 0, 0),
        )
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.0)
        assert "WARNING" in html

    @pytest.mark.asyncio
    async def test_critical_banner_in_report(self, generator, sample_trades, open_positions):
        portfolio = PortfolioSnapshot(
            total_value=75000.0,
            cash_balance=5000.0,
            positions_value=70000.0,
            daily_pnl=-5000.0,
            total_pnl=-25000.0,
            total_pnl_pct=-25.0,
            num_open_positions=2,
            hard_stop_active=True,
            snapshot_time=datetime(2024, 1, 15, 18, 0, 0),
        )
        html = await generator.generate_report(portfolio, sample_trades, open_positions, 0.0)
        assert "CRITICAL" in html


class TestSendReport:
    """Tests for send_report method."""

    @pytest.mark.asyncio
    async def test_send_success(self, generator):
        with patch("src.services.report_generator.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            await generator.send_report("<html>test</html>")

            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("user@example.com", "secret")
            mock_server.sendmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_failure_stores_locally(self, generator, cleanup_reports):
        with patch("src.services.report_generator.smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = Exception("Connection refused")

            with patch("src.services.report_generator.asyncio.sleep", new_callable=AsyncMock):
                await generator.send_report("<html>fallback test</html>")

            # Verify file was stored locally
            reports_dir = os.path.join("data", "reports")
            assert os.path.exists(reports_dir)
            files = os.listdir(reports_dir)
            assert len(files) == 1
            assert files[0].startswith("report_")
            assert files[0].endswith(".html")

            # Verify content
            with open(os.path.join(reports_dir, files[0]), "r") as f:
                content = f.read()
            assert "<html>fallback test</html>" in content

    @pytest.mark.asyncio
    async def test_send_retries_once_before_fallback(self, generator, cleanup_reports):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("SMTP error")

        with patch("src.services.report_generator.smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = side_effect

            with patch("src.services.report_generator.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await generator.send_report("<html>retry test</html>")

                # Should have retried once with 60s delay
                mock_sleep.assert_called_once_with(60)

            # Both attempts failed, so 2 SMTP calls
            assert call_count == 2


class TestSentimentRendering:
    """Tests for sentiment display."""

    def test_bullish_sentiment(self, generator):
        html = generator._render_sentiment(0.5)
        assert "Bullish" in html
        assert "#2ecc71" in html

    def test_bearish_sentiment(self, generator):
        html = generator._render_sentiment(-0.5)
        assert "Bearish" in html
        assert "#e74c3c" in html

    def test_neutral_sentiment(self, generator):
        html = generator._render_sentiment(0.0)
        assert "Neutral" in html
