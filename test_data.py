"""
test_data.py
============
Run this FIRST to diagnose bar fetching issues.
Usage: python test_data.py

Checks:
  1. API connection
  2. Snapshot data
  3. Daily bar history
  4. 5-minute bar history
  5. Full indicator computation
"""
import os
import sys

# ── Load keys from config ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import Config
config = Config()

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

api = tradeapi.REST(
    config.ALPACA_API_KEY,
    config.ALPACA_SECRET_KEY,
    base_url    = config.ALPACA_BASE_URL,
    api_version = "v2"
)

TICKER = "AAPL"

print("=" * 60)
print(f"  Claude Trading Bot — Data Diagnostics")
print(f"  Testing ticker: {TICKER}")
print(f"  Feed: {config.ALPACA_DATA_FEED}")
print(f"  URL:  {config.ALPACA_BASE_URL}")
print("=" * 60)

# ── Test 1: Account ──────────────────────────────────────────────
print("\n[1] Account connection...")
try:
    acct = api.get_account()
    print(f"  ✅ Connected | Equity=${float(acct.equity):,.2f} | Cash=${float(acct.cash):,.2f}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print("  → Check your API keys in config.py")
    sys.exit(1)

# ── Test 2: Snapshot ─────────────────────────────────────────────
print(f"\n[2] Snapshot for {TICKER}...")
try:
    snaps = api.get_snapshots([TICKER])
    snap  = snaps.get(TICKER)
    if snap:
        price = snap.latest_trade.price if snap.latest_trade else "N/A"
        vol   = snap.daily_bar.v if snap.daily_bar else "N/A"
        print(f"  ✅ Price=${price}  Volume={vol}")
    else:
        print(f"  ❌ No snapshot returned for {TICKER}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# ── Test 3: Daily bars ───────────────────────────────────────────
print(f"\n[3] Daily bars for {TICKER} (limit=60)...")
try:
    df = api.get_bars(
        TICKER,
        TimeFrame.Day,
        limit      = 60,
        adjustment = "raw",
        feed       = config.ALPACA_DATA_FEED,
    ).df
    if df is not None and not df.empty:
        print(f"  ✅ Got {len(df)} daily bars")
        print(f"  Latest bar: {df.index[-1]}  close={df['close'].iloc[-1]:.2f}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print(f"  ❌ Empty dataframe returned")
        print("  → Try changing ALPACA_DATA_FEED to 'iex' in config.py")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    print("  → Trying fallback method...")
    try:
        bars = list(api.get_bars_iter(TICKER, TimeFrame.Day, limit=60))
        print(f"  ✅ Fallback got {len(bars)} bars")
    except Exception as e2:
        print(f"  ❌ Fallback also failed: {e2}")

# ── Test 4: 5-minute bars ────────────────────────────────────────
print(f"\n[4] 5-minute bars for {TICKER} (limit=75)...")
try:
    df5 = api.get_bars(
        TICKER,
        TimeFrame.Minute,
        limit      = 75,
        adjustment = "raw",
        feed       = config.ALPACA_DATA_FEED,
    ).df
    if df5 is not None and not df5.empty:
        print(f"  ✅ Got {len(df5)} minute bars")
    else:
        print(f"  ⚠️  Empty — 5m RSI will be unavailable (non-critical)")
except Exception as e:
    print(f"  ⚠️  Failed: {e} (non-critical)")

# ── Test 5: Full indicator computation ───────────────────────────
print(f"\n[5] Full indicator test for {TICKER}...")
try:
    from data_feed import MarketDataFeed
    feed = MarketDataFeed(config)
    data = feed.get_quotes([TICKER])
    if TICKER in data:
        d = data[TICKER]
        print(f"  ✅ Indicators computed successfully!")
        print(f"  Price      : ${d.get('price', 'N/A'):.2f}")
        print(f"  RSI (daily): {d.get('rsi_14', 'N/A')}")
        print(f"  RSI (5m)   : {d.get('rsi_5m', 'N/A')}")
        print(f"  MACD       : {d.get('macd', 'N/A')}")
        bb = d.get('bb')
        if bb:
            print(f"  BB         : lower={bb['lower']}  mid={bb['mid']}  upper={bb['upper']}")
        else:
            print(f"  BB         : N/A")
        print(f"  ATR(14)    : {d.get('atr_14', 'N/A')}")
        print(f"  Regime     : {d.get('market_regime', 'N/A')}")
        print(f"  Vol ratio  : {d.get('vol_ratio', 'N/A')}")
        sig = d.get('signals', {})
        print(f"  Signal bias: {sig.get('bias', 'N/A')}")
        print(f"  Buy sigs   : {sig.get('buy_signals', [])}")
        print(f"  Sell sigs  : {sig.get('sell_signals', [])}")
    else:
        print(f"  ❌ No data returned for {TICKER}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DIAGNOSIS COMPLETE")
print("  If test [3] failed, do this:")
print("  1. In config.py change: ALPACA_DATA_FEED = 'iex'")
print("  2. Make sure your Alpaca subscription is active")
print("  3. Run this test again")
print("=" * 60)
