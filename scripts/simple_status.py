"""Simple portfolio status for non-traders."""
from ib_insync import IB

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=98)
ib.sleep(2)

cash = 0.0
upnl = 0.0
rpnl = 0.0
for av in ib.accountValues():
    if av.currency == "BASE":
        if av.tag == "TotalCashBalance":
            cash = float(av.value)
        if av.tag == "UnrealizedPnL":
            upnl = float(av.value)
        if av.tag == "RealizedPnL":
            rpnl = float(av.value)

start = 1_000_000.0
total_pnl = upnl + rpnl
current_total = start + total_pnl
pct = (total_pnl / start) * 100

D = "$"  # avoid PowerShell eating dollar signs

print()
print("=" * 55)
print("  JEDNOSTAVAN PREGLED AGENTA")
print("=" * 55)
print()
print(f"  Pocetni kapital:       {D}{start:>12,.0f}")
print(f"  Trenutna gotovina:     {D}{cash:>12,.0f}")
print(f"  Nerealizirani P/L:     {D}{upnl:>+12,.0f}")
print(f"  Realizirani P/L:       {D}{rpnl:>+12,.0f}")
print(f"  ----------------------------------------")
print(f"  UKUPNI PROFIT/GUBITAK: {D}{total_pnl:>+12,.0f}  ({pct:+.2f}%)")
print(f"  UKUPNA VRIJEDNOST:     {D}{current_total:>12,.0f}")
print()

portfolio = ib.portfolio()
long_value = sum(item.marketValue for item in portfolio if item.position > 0)
short_value = sum(abs(item.marketValue) for item in portfolio if item.position < 0)

print(f"  Ulozeno u LONG pozicije:  {D}{long_value:>12,.0f}")
print(f"  Ulozeno u SHORT pozicije: {D}{short_value:>12,.0f}")
print(f"  Ukupno ulozeno:           {D}{long_value + short_value:>12,.0f}")
print()
print(f"  Otvorenih pozicija: {len(portfolio)}")
print(f"  Tradeova danas:    {len(ib.fills())}")
print()

if total_pnl < 0:
    print(f"  >>> Agent je u GUBITKU od {D}{abs(total_pnl):,.0f} ({abs(pct):.2f}%)")
else:
    print(f"  >>> Agent je u DOBITKU od {D}{total_pnl:,.0f} ({pct:.2f}%)")

print("=" * 55)
ib.disconnect()
