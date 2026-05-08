"""Check market data availability on LIVE account.

Connects to live IB Gateway (port 4001 or 7496) and checks if
US real-time market data is available for free.

SAFE: This script only reads market data — it does NOT place any orders.

Usage:
    python scripts/check_live_market_data.py [port]
    
    Default port: 7496 (TWS live)
    Alternative: 4001 (Gateway live)
"""
import asyncio
import sys
from ib_insync import IB, Stock, util

util.patchAsyncio()

# Use port from command line or default to 7496 (TWS live)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7496


async def check():
    ib = IB()
    
    print(f"Connecting to LIVE IB on port {PORT}...")
    print("(This is READ-ONLY — no orders will be placed)")
    print()
    
    try:
        await ib.connectAsync("127.0.0.1", PORT, clientId=998, readonly=True)
    except Exception as e:
        print(f"Connection failed: {e}")
        print()
        print("Make sure:")
        print(f"  - TWS or IB Gateway is running on port {PORT}")
        print("  - API connections are enabled")
        print("  - You are logged into your LIVE account")
        return
    
    accounts = ib.managedAccounts()
    print(f"Connected. Accounts: {accounts}")
    print()
    
    # Test US real-time for AAPL
    contract = Stock("AAPL", "SMART", "USD")
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print("Failed to qualify AAPL")
        ib.disconnect()
        return
    
    contract = qualified[0]
    
    print("Testing AAPL with Market Data Type 1 (Real-Time)...")
    ib.reqMarketDataType(1)
    ticker = ib.reqMktData(contract, genericTickList="233,165")
    
    for i in range(10):
        ib.sleep(0.5)
        if ticker.last is not None and ticker.last == ticker.last:
            break
    
    mdt_names = {1: "REAL-TIME", 2: "FROZEN", 3: "DELAYED", 4: "FROZEN-DELAYED"}
    actual = mdt_names.get(ticker.marketDataType, f"UNKNOWN({ticker.marketDataType})")
    
    print(f"  Last: {ticker.last}")
    print(f"  Bid: {ticker.bid}")
    print(f"  Ask: {ticker.ask}")
    print(f"  Volume: {ticker.volume}")
    print(f"  Data type received: {actual}")
    print()
    
    if ticker.marketDataType == 1:
        print("✅ US REAL-TIME WORKS! Free US Stock Bundle is active.")
    elif ticker.marketDataType == 3:
        print("⚠️  Got DELAYED data — US real-time subscription NOT active.")
        print("   Go to Client Portal → Settings → Market Data Subscriptions")
        print("   and activate 'US Securities Snapshot and Futures Value Bundle'")
    else:
        print(f"❓ Got {actual} — check your subscriptions.")
    
    ib.cancelMktData(contract)
    ib.disconnect()
    print("\nDone. No orders were placed.")


asyncio.run(check())
