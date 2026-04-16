# Trading Risk Management Rules

inclusion: always

## Capital Protection Priority

Capital protection is ALWAYS the #1 priority. Profit maximization is secondary.

## Hard Limits

These limits are non-negotiable and must be enforced at all times:

- **Max Portfolio Loss (Hard Stop):** 20% of total portfolio value. When reached, the agent MUST automatically close ALL positions and halt trading until manually restarted.
- **Stop-Loss Per Position:** Max 5% loss per individual stock. Enforce via stop-loss orders placed immediately upon entry.
- **Trailing Stop-Loss:** Implement trailing stop-loss to protect unrealized gains on winning positions.
- **Max Exposure Per Stock:** Max 25% of total portfolio in any single stock.
- **Cash Buffer:** Always maintain minimum 10% of portfolio value in cash. Never deploy 100% of capital.

## Risk Check Order

Before every trade, validate in this exact order:
1. Is the hard stop triggered? (portfolio loss >= 20%) → HALT
2. Would this trade violate the cash buffer? (remaining cash < 10%) → REJECT
3. Would this trade exceed max position size? (position > 25% of portfolio) → REJECT
4. Is a stop-loss order configured for this position? → REQUIRED

## Configuration

All risk parameters must be configurable via .env file with sensible defaults:
- MAX_PORTFOLIO_LOSS_PCT=20
- MAX_POSITION_SIZE_PCT=25
- STOP_LOSS_PCT=5
- CASH_BUFFER_PCT=10
