# Performance Analysis — IB Trading Agent
**Date:** April 20, 2025  
**Period Analyzed:** April 15–20, 2025  
**Role:** Experienced Broker Analysis

---

## 1. Current State Diagnosis

### 1.1 Portfolio Overview
| Metric | Value |
|--------|-------|
| Net Liquidation Value | ~$1,015,167 |
| Cash Balance | $771,901 (76% of portfolio) |
| Positions Value | ~$243,266 (24%) |
| Open Positions | 1 (AAPL: 893 shares @ $271.09 avg) |
| Unrealized P&L | +$1,006 |
| Total Trades (all time) | 163 (16 open, 147 closed) |
| Realized P&L | **Unknown** — all 163 trades have `realized_pnl = None` |

### 1.2 Why Is the Portfolio Concentrated in Apple?

The concentration in a single stock (AAPL) is **not a deliberate strategy** — it is the result of cascading bugs:

1. **position_size_0 rejection (81% of all rejections):** The `calculate_position_size()` method in `RiskManager` returns 0 shares for most signals. Root cause: portfolio snapshots record `total_value=0.0` because the snapshot logic reads IB account values with `currency == "USD"` but the IB paper account reports values with `currency == "BASE"`. When `_current_portfolio_value == 0`, the formula `max_position_size_pct / 100 * 0 / price = 0` always yields 0 shares.

2. **SELL signals without positions (18% of rejections):** The `StrategyEngine` generates SELL signals for symbols the agent doesn't hold. The strategy evaluates crossovers purely on indicator math — it has no awareness of current positions. The `RiskManager` correctly rejects these, but they represent wasted computation and log noise.

3. **The AAPL position survived** because it was likely opened during the brief window when portfolio values were correctly loaded at startup (the `initialize()` method has a multi-fallback chain for reading `NetLiquidation`). Subsequent portfolio updates via `_check_portfolio_loss()` and `_take_portfolio_snapshot()` use `currency == "USD"` without the fallback, causing `total_value` to drop to 0.

### 1.3 Trade History Analysis
| Symbol | Trades | Strategy Breakdown |
|--------|--------|--------------------|
| V | 22 | Mostly mean_reversion |
| JNJ | 19 | Mixed |
| NVDA | 18 | Mixed |
| JPM | 17 | Mixed |
| AAPL | 16 | Mixed |
| MSFT | 15 | Mixed |
| GOOGL | 15 | Mixed |
| META | 14 | Mixed |
| TSLA | 14 | Mixed |
| AMZN | 13 | Mixed |

**Strategy distribution:** mean_reversion: 95 (58%), trend_following: 66 (40%), momentum: 2 (1.2%)

### 1.4 Signal Pipeline Efficiency
| Stage | Count | % of Total |
|-------|-------|------------|
| Signals Generated | 2,093 | 100% |
| Signals Rejected | 1,920 | 91.7% |
| Signals Executed | ~173 | 8.3% |

**Rejection breakdown:**
- `position_size_0`: 1,564 (81.5%) — **BUG: portfolio value reads as 0**
- `no_long_position`: 339 (17.7%) — SELL signals for unheld stocks
- `duplicate_position`: ~5 (0.3%) — already holding
- `total_exposure > 90%`: ~15 (0.8%) — exposure limit hit

---

## 2. Identified Weaknesses

### 2.1 Critical Bugs (Must Fix)

| # | Bug | Impact | Root Cause |
|---|-----|--------|------------|
| B1 | Portfolio value reads as 0 in snapshots | 81% of trades rejected | `_take_portfolio_snapshot()` and `_check_portfolio_loss()` filter by `currency == "USD"` but IB paper returns `currency == "BASE"` |
| B2 | Realized P&L never computed | Cannot measure performance | `_handle_fill()` in `OrderExecutor` creates OPEN trades but never closes them with exit price/P&L when SELL fills |
| B3 | SELL signals for unheld stocks | 18% wasted rejections + log noise | `StrategyEngine` has no position awareness |
| B4 | 2.3 GB daily log file | Disk exhaustion risk | `YahooDataProvider.poll()` logs every 10-second poll at INFO level even when no new data |
| B5 | Polymarket sentiment broken | Sentiment always 0.991 or 0.0 | Likely API response parsing issue — `outcomePrices` format may have changed |

### 2.2 Strategy Weaknesses

| # | Weakness | Evidence |
|---|----------|----------|
| W1 | Momentum strategy almost never fires | Only 2 trades out of 163 — requires simultaneous RSI crossover at 30/70 AND MACD histogram sign change, which is extremely rare |
| W2 | Mean reversion generates noise signals | 58% of trades at only 0.60 confidence — BB crossover triggers too easily on normal price fluctuations |
| W3 | No minimum confidence threshold | Agent executes trades at 0.60 confidence — no quality gate |
| W4 | Volume filter too aggressive for delayed data | Volume filter requires 1.5× avg volume per tick, but delayed/frozen data has different volume semantics |
| W5 | No position-aware signal filtering | Strategy generates SELL for stocks not held, wasting cycles |

### 2.3 Operational Weaknesses

| # | Weakness | Impact |
|---|----------|--------|
| O1 | Portfolio value not synced after trades | RiskManager uses stale values between snapshot intervals |
| O2 | No trade closing logic | SELL fills don't update the original BUY trade record |
| O3 | Snapshot P&L always 0 | `daily_pnl`, `total_pnl`, `total_pnl_pct` hardcoded to 0.0 |

---

## 3. Performance Metrics

### 3.1 What We Can Measure
- **Signal generation rate:** ~700 signals/day (2,093 in ~3 days)
- **Execution rate:** ~8% of signals pass all filters
- **Strategy hit rate:** Unknown (no P&L tracking)
- **Win rate:** Unknown (all `realized_pnl = None`)
- **Sharpe ratio:** Cannot compute (no return series)
- **Max drawdown:** Cannot compute (snapshots show `total_value=0`)

### 3.2 What the Data Tells Us
- The agent is **functionally impaired** — it can only trade during the brief startup window when portfolio values are correctly loaded
- After the first snapshot cycle (5 minutes), `_current_portfolio_value` drops to 0 and all subsequent BUY signals are rejected
- The single AAPL position represents the only successful trade window
- **92% rejection rate is not a strategy problem — it's a data pipeline bug**

---

## 4. Summary

The agent's poor performance is **primarily caused by bugs, not strategy deficiencies**. The three critical fixes are:

1. **Fix portfolio value reading** (B1) — use the same multi-fallback chain from `initialize()` everywhere
2. **Implement trade closing with P&L** (B2) — when SELL fills, find the matching BUY trade and compute realized P&L
3. **Add position-aware signal filtering** (B3) — skip SELL signal generation for unheld stocks

Secondary improvements (strategy tuning, confidence thresholds, log reduction) should follow after the critical bugs are fixed.

**Next step:** KORAK 2 — Improvement Strategy
