import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
from backtest import (
    download_ticker,
    calc_rsi,
    calc_macd,
    calc_atr,
    calc_ema,
    calc_market_regime,
    calc_volume_ratio,
)

ticker = "NVDA"
df = download_ticker(ticker, "2024-01-01", "2025-12-31")

closes = df["Close"]
highs = df["High"]
lows = df["Low"]
volumes = df["Volume"]

rsi = calc_rsi(closes, 14)
macd, sig, hist = calc_macd(closes)
atr = calc_atr(highs, lows, closes)
regime = calc_market_regime(closes)
vol_r = calc_volume_ratio(volumes)
ema21 = calc_ema(closes, 21)

rsi_low = rsi < 48
regime_bull = regime >= 1
above_ema21 = closes > ema21
rsi_bounce = (rsi > rsi.shift(1)).fillna(False) & (rsi > rsi.shift(2)).fillna(False)
price_bounce = (closes > closes.shift(1)).fillna(False)
hist_improve = (hist > hist.shift(1)).fillna(False)
vol_ok = vol_r > 1.0

print(f"\nCondition breakdown for {ticker}:")
print(f"  RSI < 48          : {rsi_low.sum()} days")
print(f"  Regime >= BULL    : {regime_bull.sum()} days")
print(f"  Price > EMA21     : {above_ema21.sum()} days")
print(f"  RSI bouncing up   : {rsi_bounce.sum()} days")
print(f"  Price bouncing up : {price_bounce.sum()} days")
print(f"  MACD hist improve : {hist_improve.sum()} days")
print(f"  Volume OK         : {vol_ok.sum()} days")
print(f"\nCombinations:")
print(f"  RSI low + Bull    : {(rsi_low & regime_bull).sum()} days")
print(f"  + above EMA21     : {(rsi_low & regime_bull & above_ema21).sum()} days")
print(
    f"  + RSI bouncing    : {(rsi_low & regime_bull & above_ema21 & rsi_bounce).sum()} days"
)
print(
    f"  + price bouncing  : {(rsi_low & regime_bull & above_ema21 & rsi_bounce & price_bounce).sum()} days"
)
print(
    f"  + hist improving  : {(rsi_low & regime_bull & above_ema21 & rsi_bounce & price_bounce & hist_improve).sum()} days"
)
print(
    f"  ALL conditions    : {(rsi_low & regime_bull & above_ema21 & rsi_bounce & price_bounce & hist_improve & vol_ok).sum()} days"
)
