# Improvement Strategy — IB Trading Agent
**Date:** April 20, 2025  
**Based on:** Performance Analysis (KORAK 1)

---

## 1. Critical Bug Fixes (Priority 1)

### Fix 1.1: Portfolio Value Reading (B1)
**Module:** `src/agent.py` — `_check_portfolio_loss()`, `_take_portfolio_snapshot()`, `_process_tick_async()`

**Problem:** These methods filter `accountValues()` by `currency == "USD"` but IB paper accounts report with `currency == "BASE"`.

**Solution:** Extract a shared helper method `_read_portfolio_from_ib()` that uses the same multi-fallback chain already working in `initialize()`:
1. Try `currency == "BASE"` first
2. Fallback to `currency == "USD"` 
3. Fallback to any currency

This ensures consistent portfolio value reading across all code paths.

**Status:** ✅ Fixed (April 20, 2025)

### Fix 1.6: Available Funds for Cash Reading (B6)
**Module:** `src/agent.py` — `_read_portfolio_from_ib()`, `agent.py` (root) reconnect logic

**Problem:** `_read_portfolio_from_ib()` used `TotalCashBalance` for the cash component, but IB margin/paper accounts report this as a large negative number (e.g., -$268,876) when positions are held on margin. This caused `calculate_position_size()` to always compute `available_cash < 0` → 0 shares → every trade rejected with "position size is 0".

**Solution:** Replace `TotalCashBalance` with `AvailableFunds` (what IB allows for new positions). Fallback chain for cash:
1. `AvailableFunds` (BASE → USD) — actual buying power for new positions
2. `BuyingPower × 0.25` — conservative estimate (RegT margin = 4× AvailableFunds)
3. `NetLiquidation − GrossPositionValue` — computed available cash
4. `NetLiquidation × 0.10` — absolute fallback (10% of portfolio)

Also fixed the reconnect logic in `agent.py` (root) which was passing `NetLiquidation` as both total_value AND cash — now uses `_read_portfolio_from_ib()`.

**Status:** ✅ Fixed (April 22, 2026)

### Fix 1.2: Trade Closing with P&L (B2)
**Modules:** `src/services/order_executor.py`, `src/services/state_manager.py`

**Problem:** When a SELL order fills, the `_handle_fill()` method creates a new OPEN trade record instead of closing the matching BUY trade with exit price and realized P&L.

**Solution:**
1. In `OrderExecutor._handle_fill()`: when direction is SELL, find the matching OPEN BUY trade for that symbol and call `state_manager.close_trade()` with exit price and computed P&L
2. Add `StateManager.close_trade(symbol, exit_price, exit_time)` method that:
   - Finds the OPEN trade for the symbol
   - Computes `realized_pnl = (exit_price - entry_price) * quantity`
   - Updates status to CLOSED, sets exit_price, exit_time, realized_pnl

### Fix 1.3: Position-Aware Signal Filtering (B3)
**Module:** `src/agent.py` — `_process_tick_async()`

**Problem:** StrategyEngine generates SELL signals for stocks the agent doesn't hold, causing 18% of rejections.

**Solution:** Add a pre-filter in `_process_tick_async()` before calling `strategy_engine.process_tick()`:
- If the signal is SELL and the symbol is not in `risk_manager._open_positions`, skip it early
- Actually, the better approach: after getting the signal, if it's SELL and we don't hold the position, log at DEBUG and return early (before calling evaluate_signal). This avoids changing the StrategyEngine's pure signal generation logic.

### Fix 1.4: Log Spam Reduction (B4)
**Module:** `src/services/yahoo_data_provider.py`

**Problem:** `poll()` logs at INFO level every 10 seconds even when no new data arrives.

**Solution:** 
- Change the "no update" path to DEBUG level
- Only log at INFO when data actually changes
- Add a summary log every 5 minutes instead of per-poll

### Fix 1.5: Portfolio Snapshot P&L (O3)
**Module:** `src/agent.py` — `_take_portfolio_snapshot()`

**Problem:** `daily_pnl`, `total_pnl`, `total_pnl_pct` are hardcoded to 0.0.

**Solution:** Compute from initial portfolio value stored in RiskManager:
- `total_pnl = current_value - initial_value`
- `total_pnl_pct = total_pnl / initial_value * 100`
- `daily_pnl` = difference from first snapshot of the day (query DB)

---

## 2. Strategy Improvements (Priority 2)

### Improvement 2.1: Minimum Confidence Threshold
**Module:** `src/agent.py` — `_process_tick_async()`

**Change:** Add a minimum confidence threshold of 0.65. Signals below this are logged at DEBUG and discarded.

**Rationale:** Mean reversion fires at 0.60 base confidence, generating noise. Requiring 0.65 means only multi-strategy agreement or sentiment-boosted signals pass through.

### Improvement 2.2: Relax Momentum Strategy Thresholds
**Module:** `src/services/strategy_engine.py`

**Current:** Momentum requires RSI to cross exactly at 30/70 AND MACD histogram to change sign simultaneously. This is too restrictive — only 2 trades ever.

**Change:** 
- BUY: RSI < 35 (was: crosses above 30 from below) AND MACD histogram > 0
- SELL: RSI > 65 (was: crosses below 70 from above) AND MACD histogram < 0
- Keep the crossover requirement for MACD but relax RSI to a zone check
- Increase base confidence to 0.75 (from 0.70) since relaxed conditions need higher base

### Improvement 2.3: Mean Reversion Tightening
**Module:** `src/services/strategy_engine.py`

**Current:** Triggers on any BB crossover, generating too many signals at 0.60 confidence.

**Change:**
- Require price to be at least 1 standard deviation beyond the band (not just touching)
- Add RSI confirmation: BUY only if RSI < 40, SELL only if RSI > 60
- Increase base confidence to 0.65 (from 0.60)

### Improvement 2.4: Volume Filter for Delayed Data
**Module:** `src/services/strategy_engine.py`

**Current:** Volume filter is already bypassed for delayed data types (3, 4). For Yahoo data, the volume semantics are different (cumulative daily volume).

**Change:** For `market_data_type == "yahoo"`, skip the per-tick volume filter entirely (already handled by the `if self._market_data_type == "1"` check). No code change needed — this is already correct.

---

## 3. Position Sizing Optimization

### Current State
With $1M portfolio and 25% max position size:
- Max per position: $250,000
- Cash buffer (10%): $100,000
- Available for trading: $900,000
- Max positions at 25% each: 3-4 positions

### Recommended Parameters (No Change)
The current parameters are appropriate for a $1M paper account:
- `MAX_POSITION_SIZE_PCT=25` — allows 3-4 diversified positions
- `CASH_BUFFER_PCT=10` — $100K cash reserve
- `STOP_LOSS_PCT=5` — reasonable per-position risk
- `MAX_PORTFOLIO_LOSS_PCT=20` — hard stop at $200K loss

The problem isn't the parameters — it's that `calculate_position_size()` gets `portfolio_value=0` due to bug B1, and `cash=-268876` (negative) due to bug B6 (TotalCashBalance is negative on margin accounts).

---

## 4. Entry/Exit Criteria Summary

### Momentum Strategy (Revised)
- **BUY:** RSI < 35 AND MACD histogram turns positive (from ≤ 0 to > 0) AND volume confirmed
- **SELL:** RSI > 65 AND MACD histogram turns negative (from ≥ 0 to < 0) AND volume confirmed
- **Base confidence:** 0.75

### Mean Reversion Strategy (Revised)
- **BUY:** Price < lower BB AND RSI < 40 AND volume confirmed
- **SELL:** Price > upper BB AND RSI > 60 AND volume confirmed
- **Base confidence:** 0.65

### Trend Following Strategy (Unchanged)
- **BUY:** 9-EMA crosses above 21-EMA AND volume confirmed
- **SELL:** 9-EMA crosses below 21-EMA AND volume confirmed
- **Base confidence:** 0.65

### Global Filters
- **Minimum confidence:** 0.65 (new)
- **Market hours:** Only during regular hours
- **Earnings blackout:** 2 days before, 1 day after
- **Position check:** SELL only if holding the stock

---

## 5. Expected Performance Targets

After implementing all fixes and improvements:

| Metric | Current | Target |
|--------|---------|--------|
| Signal rejection rate | 92% | < 40% |
| Trades per day | ~2-3 (broken) | 5-15 |
| Active positions | 1 | 3-5 |
| Win rate | Unknown | Measurable (P&L tracking) |
| Portfolio utilization | 24% | 50-80% |
| Daily log size | 2.3 GB | < 50 MB |
| Momentum trades | 1.2% | 10-20% |

---

## 6. Implementation Order

1. **Bug fixes first** (B1→B2→B3→B4→B5→O3→B6) — these unlock the agent's basic functionality
2. **Strategy improvements** (2.1→2.2→2.3) — tune signal quality
3. **Tests** — verify all changes
4. **Documentation** — update design.md, requirements.md, tasks.md

**Status:** B1-B5, O3 fixed April 20, 2025. B6 fixed April 22, 2026.

**Next step:** KORAK 3 — Implementation
