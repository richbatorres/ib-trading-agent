# Trading Strategy Conventions

inclusion: always

## Instrument Restrictions

- Trade ONLY stocks (equities). No options, futures, forex, or crypto.
- Trade ONLY stocks with average daily volume > 500,000 shares (liquidity filter).
- AVOID trading immediately before or after earnings reports.

## Strategy Selection

The agent combines multiple strategies and selects the most appropriate based on market conditions:

1. **Momentum Trading:** RSI and MACD indicators for short-term momentum detection.
2. **Mean Reversion:** Bollinger Bands to detect excessive deviation from average price.
3. **Volume Analysis:** High volume as signal confirmation. NEVER trade without volume confirmation.
4. **Trend Following:** EMA crossover signals (9/21 EMA).

## Signal Confirmation

- Every trade signal MUST be confirmed by volume analysis.
- Polymarket sentiment is a SECONDARY factor only — never the sole reason for a trade.
- Multiple strategy agreement increases signal confidence.

## Market Hours

- Trade ONLY during regular market hours (NYSE/NASDAQ: 9:30–16:00 ET).
- Outside market hours: monitor, analyze, prepare watchlist — but do NOT execute trades.
- Auto-detect exchange hours; do not hardcode timezone offsets.

## Polymarket Integration

- Polymarket data is refreshed every 15 minutes.
- Polymarket signals are used exclusively as secondary/additional decision factors.
- Track macroeconomic events, political risks, Fed decisions, and similar market-moving events.
