"""Daily HTML email report generator.

Produces professional HTML reports with portfolio summary, trade details,
winners/losers, open positions, Polymarket sentiment, and warning banners.
Sends via SMTP with retry and local fallback.
"""

import asyncio
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

from src.config import AgentConfig
from src.models.domain import PortfolioSnapshot, TradeRecord
from src.services.state_manager import StateManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates and sends daily HTML email reports."""

    def __init__(self, config: AgentConfig, state_manager: StateManager) -> None:
        self._config = config
        self._state = state_manager

    async def generate_report(
        self,
        portfolio: PortfolioSnapshot,
        trades: List[TradeRecord],
        open_positions: List[dict],
        polymarket_sentiment: float,
    ) -> str:
        """Build an HTML email report.

        Args:
            portfolio: Current portfolio snapshot.
            trades: All trades executed today.
            open_positions: Current open positions with unrealized P&L.
            polymarket_sentiment: Current Polymarket sentiment score.

        Returns:
            Complete HTML string for the email report.
        """
        report_date = datetime.now().strftime("%Y-%m-%d")
        daily_pnl_pct = (
            (portfolio.daily_pnl / (portfolio.total_value - portfolio.daily_pnl)) * 100
            if (portfolio.total_value - portfolio.daily_pnl) != 0
            else 0.0
        )

        warning_banner = self._render_warning_banner(portfolio.total_pnl_pct)

        # Top 3 winners and losers by realized P&L
        closed_trades = [t for t in trades if t.realized_pnl is not None]
        sorted_by_pnl = sorted(closed_trades, key=lambda t: t.realized_pnl, reverse=True)
        top_winners = sorted_by_pnl[:3]
        top_losers = sorted_by_pnl[-3:] if len(sorted_by_pnl) >= 3 else sorted_by_pnl
        # Losers should be sorted worst first
        top_losers = sorted(top_losers, key=lambda t: t.realized_pnl)
        # Only include actual losers (negative P&L)
        top_losers = [t for t in top_losers if t.realized_pnl < 0]
        # Only include actual winners (positive P&L)
        top_winners = [t for t in top_winners if t.realized_pnl > 0]

        trades_html = self._render_trades_table(trades)
        winners_html = self._render_top_trades(top_winners, "Top Winners")
        losers_html = self._render_top_trades(top_losers, "Top Losers")
        positions_html = self._render_open_positions(open_positions)
        sentiment_html = self._render_sentiment(polymarket_sentiment)

        pnl_color = "#2ecc71" if portfolio.daily_pnl >= 0 else "#e74c3c"
        total_pnl_color = "#2ecc71" if portfolio.total_pnl >= 0 else "#e74c3c"

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#f4f4f4;font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f4;padding:20px 0;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:8px;overflow:hidden;">

<!-- Header -->
<tr>
<td style="background-color:#2c3e50;padding:24px 30px;color:#ffffff;">
<h1 style="margin:0;font-size:22px;font-weight:bold;">IB Trading Agent — Daily Report</h1>
<p style="margin:6px 0 0 0;font-size:14px;color:#bdc3c7;">{report_date}</p>
</td>
</tr>

{warning_banner}

<!-- Portfolio Summary -->
<tr>
<td style="padding:24px 30px;">
<h2 style="margin:0 0 16px 0;font-size:18px;color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;">Portfolio Summary</h2>
<table width="100%" cellpadding="8" cellspacing="0">
<tr>
<td style="font-size:14px;color:#7f8c8d;">Total Value</td>
<td style="font-size:14px;font-weight:bold;text-align:right;">${portfolio.total_value:,.2f}</td>
</tr>
<tr style="background-color:#f9f9f9;">
<td style="font-size:14px;color:#7f8c8d;">Cash Balance</td>
<td style="font-size:14px;text-align:right;">${portfolio.cash_balance:,.2f}</td>
</tr>
<tr>
<td style="font-size:14px;color:#7f8c8d;">Positions Value</td>
<td style="font-size:14px;text-align:right;">${portfolio.positions_value:,.2f}</td>
</tr>
<tr style="background-color:#f9f9f9;">
<td style="font-size:14px;color:#7f8c8d;">Daily P&amp;L</td>
<td style="font-size:14px;font-weight:bold;text-align:right;color:{pnl_color};">${portfolio.daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)</td>
</tr>
<tr>
<td style="font-size:14px;color:#7f8c8d;">Total P&amp;L</td>
<td style="font-size:14px;font-weight:bold;text-align:right;color:{total_pnl_color};">${portfolio.total_pnl:+,.2f} ({portfolio.total_pnl_pct:+.2f}%)</td>
</tr>
<tr style="background-color:#f9f9f9;">
<td style="font-size:14px;color:#7f8c8d;">Open Positions</td>
<td style="font-size:14px;text-align:right;">{portfolio.num_open_positions}</td>
</tr>
</table>
</td>
</tr>

<!-- Trades Today -->
{trades_html}

<!-- Top Winners -->
{winners_html}

<!-- Top Losers -->
{losers_html}

<!-- Open Positions -->
{positions_html}

<!-- Polymarket Sentiment -->
{sentiment_html}

<!-- Footer -->
<tr>
<td style="background-color:#ecf0f1;padding:16px 30px;text-align:center;">
<p style="margin:0;font-size:12px;color:#95a5a6;">Generated by IB Trading Agent at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</td>
</tr>

</table>
</td></tr>
</table>
</body>
</html>"""

        logger.info("Generated daily report for %s with %d trades", report_date, len(trades))
        return html

    async def send_report(self, html: str) -> None:
        """Send the HTML report via SMTP.

        On failure, retries once after 60 seconds. If still failing,
        stores the HTML locally in data/reports/.

        Args:
            html: The complete HTML report string.
        """
        for attempt in range(2):
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"IB Trading Agent — Daily Report {datetime.now().strftime('%Y-%m-%d')}"
                msg["From"] = self._config.email_smtp_user
                msg["To"] = self._config.email_address
                msg.attach(MIMEText(html, "html"))

                with smtplib.SMTP(self._config.email_smtp_host, self._config.email_smtp_port) as server:
                    server.starttls()
                    server.login(self._config.email_smtp_user, self._config.email_smtp_password)
                    server.sendmail(
                        self._config.email_smtp_user,
                        self._config.email_address,
                        msg.as_string(),
                    )

                logger.info("Daily report email sent to %s", self._config.email_address)
                return
            except Exception as exc:
                if attempt == 0:
                    logger.warning("SMTP send failed (attempt 1/2): %s — retrying in 60s", exc)
                    await asyncio.sleep(60)
                else:
                    logger.error("SMTP send failed (attempt 2/2): %s — storing report locally", exc)

        # Fallback: store HTML locally
        reports_dir = os.path.join("data", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(reports_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        logger.error("Report stored locally at %s", filepath)

    def _render_warning_banner(self, loss_pct: float) -> str:
        """Render a warning or critical banner based on cumulative loss.

        Args:
            loss_pct: The total portfolio P&L percentage (negative = loss).

        Returns:
            HTML string for the banner, or empty string if no warning needed.
        """
        # loss_pct is total_pnl_pct which is negative when losing
        # We check the absolute loss: if loss_pct <= -20 => CRITICAL, <= -10 => WARNING
        if loss_pct <= -20:
            return """<tr>
<td style="background-color:#c0392b;padding:16px 30px;text-align:center;">
<p style="margin:0;font-size:16px;font-weight:bold;color:#ffffff;">⚠️ CRITICAL: Portfolio loss exceeds 20%! Hard stop may be active.</p>
</td>
</tr>"""
        elif loss_pct <= -10:
            return """<tr>
<td style="background-color:#e67e22;padding:16px 30px;text-align:center;">
<p style="margin:0;font-size:16px;font-weight:bold;color:#ffffff;">⚠️ WARNING: Portfolio loss exceeds 10%. Review positions carefully.</p>
</td>
</tr>"""
        return ""

    def _render_trades_table(self, trades: List[TradeRecord]) -> str:
        """Render the trades executed today as an HTML table section."""
        if not trades:
            return """<tr>
<td style="padding:24px 30px;">
<h2 style="margin:0 0 16px 0;font-size:18px;color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;">Trades Today</h2>
<p style="font-size:14px;color:#95a5a6;">No trades executed today.</p>
</td>
</tr>"""

        rows = ""
        for i, trade in enumerate(trades):
            bg = 'background-color:#f9f9f9;' if i % 2 == 1 else ''
            direction_color = "#2ecc71" if trade.direction == "BUY" else "#e74c3c"
            exit_price_str = f"${trade.exit_price:,.2f}" if trade.exit_price is not None else "—"
            pnl_str = ""
            if trade.realized_pnl is not None:
                pnl_color = "#2ecc71" if trade.realized_pnl >= 0 else "#e74c3c"
                pnl_str = f'<span style="color:{pnl_color};font-weight:bold;">${trade.realized_pnl:+,.2f}</span>'
            else:
                pnl_str = "—"

            rows += f"""<tr style="{bg}">
<td style="padding:8px;font-size:13px;font-weight:bold;">{trade.symbol}</td>
<td style="padding:8px;font-size:13px;color:{direction_color};font-weight:bold;">{trade.direction}</td>
<td style="padding:8px;font-size:13px;text-align:right;">${trade.entry_price:,.2f}</td>
<td style="padding:8px;font-size:13px;text-align:right;">{exit_price_str}</td>
<td style="padding:8px;font-size:13px;text-align:right;">{pnl_str}</td>
</tr>"""

        return f"""<tr>
<td style="padding:24px 30px;">
<h2 style="margin:0 0 16px 0;font-size:18px;color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;">Trades Today ({len(trades)})</h2>
<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
<tr style="background-color:#2c3e50;">
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:left;">Symbol</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:left;">Direction</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">Entry</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">Exit</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">P&amp;L</th>
</tr>
{rows}
</table>
</td>
</tr>"""

    def _render_top_trades(self, trades: List[TradeRecord], title: str) -> str:
        """Render a top winners or losers section."""
        if not trades:
            return ""

        rows = ""
        for i, trade in enumerate(trades):
            bg = 'background-color:#f9f9f9;' if i % 2 == 1 else ''
            pnl_color = "#2ecc71" if trade.realized_pnl >= 0 else "#e74c3c"
            rows += f"""<tr style="{bg}">
<td style="padding:8px;font-size:13px;font-weight:bold;">{trade.symbol}</td>
<td style="padding:8px;font-size:13px;">{trade.strategy}</td>
<td style="padding:8px;font-size:13px;text-align:right;color:{pnl_color};font-weight:bold;">${trade.realized_pnl:+,.2f}</td>
</tr>"""

        return f"""<tr>
<td style="padding:24px 30px;">
<h2 style="margin:0 0 16px 0;font-size:18px;color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;">{title}</h2>
<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
<tr style="background-color:#2c3e50;">
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:left;">Symbol</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:left;">Strategy</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">P&amp;L</th>
</tr>
{rows}
</table>
</td>
</tr>"""

    def _render_open_positions(self, open_positions: List[dict]) -> str:
        """Render the open positions section."""
        if not open_positions:
            return """<tr>
<td style="padding:24px 30px;">
<h2 style="margin:0 0 16px 0;font-size:18px;color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;">Open Positions</h2>
<p style="font-size:14px;color:#95a5a6;">No open positions.</p>
</td>
</tr>"""

        rows = ""
        for i, pos in enumerate(open_positions):
            bg = 'background-color:#f9f9f9;' if i % 2 == 1 else ''
            unrealized_pnl = pos.get("unrealized_pnl", 0.0)
            pnl_color = "#2ecc71" if unrealized_pnl >= 0 else "#e74c3c"
            symbol = pos.get("symbol", "N/A")
            quantity = pos.get("quantity", 0)
            avg_cost = pos.get("avg_cost", 0.0)
            current_price = pos.get("current_price", 0.0)

            rows += f"""<tr style="{bg}">
<td style="padding:8px;font-size:13px;font-weight:bold;">{symbol}</td>
<td style="padding:8px;font-size:13px;text-align:right;">{quantity}</td>
<td style="padding:8px;font-size:13px;text-align:right;">${avg_cost:,.2f}</td>
<td style="padding:8px;font-size:13px;text-align:right;">${current_price:,.2f}</td>
<td style="padding:8px;font-size:13px;text-align:right;color:{pnl_color};font-weight:bold;">${unrealized_pnl:+,.2f}</td>
</tr>"""

        return f"""<tr>
<td style="padding:24px 30px;">
<h2 style="margin:0 0 16px 0;font-size:18px;color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;">Open Positions ({len(open_positions)})</h2>
<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
<tr style="background-color:#2c3e50;">
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:left;">Symbol</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">Qty</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">Avg Cost</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">Current</th>
<th style="padding:10px 8px;font-size:13px;color:#ffffff;text-align:right;">Unrealized P&amp;L</th>
</tr>
{rows}
</table>
</td>
</tr>"""

    def _render_sentiment(self, sentiment: float) -> str:
        """Render the Polymarket sentiment summary section."""
        if sentiment > 0.3:
            label = "Bullish"
            color = "#2ecc71"
        elif sentiment < -0.3:
            label = "Bearish"
            color = "#e74c3c"
        else:
            label = "Neutral"
            color = "#f39c12"

        bar_pct = int((sentiment + 1.0) / 2.0 * 100)
        bar_pct = max(0, min(100, bar_pct))

        return f"""<tr>
<td style="padding:24px 30px;">
<h2 style="margin:0 0 16px 0;font-size:18px;color:#2c3e50;border-bottom:2px solid #ecf0f1;padding-bottom:8px;">Polymarket Sentiment</h2>
<table width="100%" cellpadding="8" cellspacing="0">
<tr>
<td style="font-size:14px;color:#7f8c8d;">Sentiment Score</td>
<td style="font-size:14px;font-weight:bold;text-align:right;color:{color};">{sentiment:+.2f} ({label})</td>
</tr>
<tr style="background-color:#f9f9f9;">
<td colspan="2" style="padding:12px 8px;">
<div style="background-color:#ecf0f1;border-radius:4px;height:20px;width:100%;overflow:hidden;">
<div style="background-color:{color};height:100%;width:{bar_pct}%;border-radius:4px;"></div>
</div>
<p style="margin:4px 0 0 0;font-size:11px;color:#95a5a6;text-align:center;">Bearish (-1.0) ← → Bullish (+1.0)</p>
</td>
</tr>
</table>
</td>
</tr>"""
