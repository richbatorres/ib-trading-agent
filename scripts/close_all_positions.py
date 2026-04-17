"""Close ALL open positions — reset to cash only."""
from ib_insync import IB, MarketOrder, Stock

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=98)
ib.sleep(2)

portfolio = ib.portfolio()
if not portfolio:
    print("No open positions — nothing to close.")
    ib.disconnect()
    exit()

print(f"Closing {len(portfolio)} positions...")
for item in portfolio:
    sym = item.contract.symbol
    qty = int(item.position)
    if qty == 0:
        continue

    # Create a proper contract with SMART exchange
    contract = Stock(sym, "SMART", "USD")
    ib.qualifyContracts(contract)

    # Reverse the position: sell if long, buy if short
    if qty > 0:
        action = "SELL"
        close_qty = qty
    else:
        action = "BUY"
        close_qty = abs(qty)

    order = MarketOrder(action, close_qty)
    trade = ib.placeOrder(contract, order)
    print(f"  {action} {close_qty} {sym} (closing {'LONG' if qty > 0 else 'SHORT'})")

# Wait for fills
print("Waiting for fills...")
ib.sleep(10)

# Verify
remaining = ib.portfolio()
open_count = sum(1 for p in remaining if int(p.position) != 0)
print(f"Remaining open positions: {open_count}")

# Show final cash
for av in ib.accountValues():
    if av.tag == "TotalCashBalance" and av.currency == "BASE":
        print(f"Cash balance: ${float(av.value):,.2f}")

ib.disconnect()
print("DONE")
