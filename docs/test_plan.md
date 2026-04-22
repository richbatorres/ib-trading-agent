# Test Plan — Performance Boost Changes
**Date:** April 20, 2025

---

## 1. Test Levels

### Unit Tests (tests/unit/)
- Portfolio value reading helper (`_read_portfolio_from_ib`)
- Available funds cash reading (AvailableFunds fallback chain)
- Trade closing with P&L (`StateManager.close_trade`)
- Updated momentum strategy thresholds
- Updated mean reversion with RSI confirmation
- Minimum confidence threshold filtering
- Position-aware SELL signal filtering
- Log level changes in Yahoo data provider

### Integration Tests (not needed for these changes)
- The changes are internal logic fixes, not new service integrations
- Existing integration tests in test/test_agent.py cover the pipeline

### Performance Tests (not needed)
- No performance-critical changes — same NumPy vectorization

### Backtest Smoke Tests (not needed)
- Strategy parameter changes don't affect backtest infrastructure

## 2. Edge Cases

| Edge Case | Test Level | Description |
|-----------|-----------|-------------|
| No IB account values | Unit | `_read_portfolio_from_ib` returns (0, 0) |
| BASE currency only | Unit | Reads correctly from BASE |
| USD currency only | Unit | Falls back to USD |
| Mixed currencies | Unit | Prefers BASE over USD |
| Negative TotalCashBalance | Unit | Uses AvailableFunds instead |
| No AvailableFunds tag | Unit | Falls back to BuyingPower×0.25 |
| No BuyingPower tag | Unit | Falls back to NLV−GrossPositionValue |
| All cash tags missing | Unit | Falls back to NLV×0.10 |
| No OPEN trade for SELL | Unit | `close_trade` returns None |
| Multiple OPEN trades | Unit | Closes oldest first |
| Confidence exactly at threshold | Unit | 0.65 passes, 0.64 doesn't |
| SELL with no position | Unit | Filtered before risk manager |
| SELL with position | Unit | Passes through to risk manager |
| RSI at boundary (35/65) | Unit | Momentum doesn't fire at exactly 35 |
| Mean reversion without RSI confirm | Unit | Signal suppressed |

## 3. Negative Tests (Agent MUST NOT trade)

- MUST NOT execute trade with confidence < 0.65
- MUST NOT generate SELL signal for unheld stock (at agent level)
- MUST NOT use stale portfolio value of 0
- MUST NOT use negative TotalCashBalance for position sizing (use AvailableFunds)
- Mean reversion MUST NOT fire if RSI is in neutral zone (40-60)

## 4. Positive Tests (Agent MUST trade)

- MUST execute trade with confidence ≥ 0.65
- MUST close trade with correct P&L on SELL fill
- MUST read portfolio value correctly from BASE currency
- Momentum MUST fire when RSI < 35 and MACD crosses positive
