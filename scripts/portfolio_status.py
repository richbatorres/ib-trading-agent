"""Print current IB portfolio status."""
from ib_insync import IB

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=98)
ib.sleep(2)

net_liq = 0.0
cash = 0.0
unrealized = 0.0
realized = 0.0
for av in ib.accountValues():
    if av.currency == "BASE":
        if av.tag == "NetLiquidation":
            net_liq = float(av.value)
        if av.tag == "TotalCashBalance":
            cash = float(av.value)
        if av.tag == "UnrealizedPnL":
            unrealized = float(av.value)
        if av.tag == "RealizedPnL":
            realized = float(av.value)

print("=" * 65)
print("  IB TRADING AGENT — PORTFOLIO STATUS")
print("=" * 65)
print(f"  Net Liquidation:  ${net_liq:>12,.2f}")
print(f"  Cash Balance:     ${cash:>12,.2f}")
print(f"  Unrealized P&L:   ${unrealized:>+12,.2f}")
print(f"  Realized P&L:     ${realized:>+12,.2f}")
print(f"  Total P&L:        ${unrealized + realized:>+12,.2f}")
print()

portfolio = ib.portfolio()
if portfolio:
    print(f"  {'Symbol':<8} {'Qty':>6} {'AvgCost':>9} {'MktPrc':>9} {'Value':>11} {'P&L':>9}  Type")
    print("  " + "-" * 62)
    total_pos = 0.0
    for item in portfolio:
        sym = item.contract.symbol
        qty = int(item.position)
        avg = item.averageCost
        mkt = item.marketPrice
        val = item.marketValue
        pnl = item.unrealizedPNL
        total_pos += abs(val)
        typ = "LONG" if qty > 0 else "SHORT"
        print(f"  {sym:<8} {qty:>6} {avg:>9.2f} {mkt:>9.2f} {val:>+11,.0f} {pnl:>+9,.0f}  {typ}")
    print("  " + "-" * 62)
    print(f"  Total Position Value: ${total_pos:>12,.0f}")
    if net_liq > 0:
        print(f"  Exposure Ratio:       {total_pos / net_liq * 100:.1f}%")
else:
    print("  No open positions")

print()
fills = ib.fills()
print(f"  Fills today: {len(fills)}")
if fills:
    buys = sum(1 for f in fills if f.execution.side == "BOT")
    sells = sum(1 for f in fills if f.execution.side == "SLD")
    print(f"  Buys: {buys}, Sells: {sells}")
print("=" * 65)

ib.disconnect()
