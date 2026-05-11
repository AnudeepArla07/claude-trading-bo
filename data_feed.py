"""
data_feed.py
============
Market data + technical indicators.
Fixed: passes explicit start date so Alpaca returns full bar history.
Python 3.9 compatible.
"""
import logging
import statistics
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pytz
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

log = logging.getLogger(__name__)


class MarketDataFeed:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url    = config.ALPACA_BASE_URL,
            api_version = "v2"
        )

    # ─────────────────────────────────────────────────────────────
    def get_quotes(self, tickers: List[str]) -> Dict[str, dict]:
        result = {}
        try:
            snapshots = self.api.get_snapshots(tickers)
        except Exception as e:
            log.error("Snapshot fetch failed: %s", e)
            return {}

        for ticker in tickers:
            snap = snapshots.get(ticker)
            if snap is None:
                continue
            try:
                price      = float(snap.latest_trade.price) if snap.latest_trade else 0.0
                prev_close = float(snap.prev_daily_bar.c)   if snap.prev_daily_bar else price
                volume     = int(snap.daily_bar.v)          if snap.daily_bar else 0
                open_      = float(snap.daily_bar.o)        if snap.daily_bar else price
                high       = float(snap.daily_bar.h)        if snap.daily_bar else price
                low        = float(snap.daily_bar.l)        if snap.daily_bar else price
                vwap       = float(snap.daily_bar.vw)       if snap.daily_bar else price
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

                bars_5m = self._get_bars(ticker, TimeFrame.Minute, limit=75,  days_back=1)
                bars_1h = self._get_bars(ticker, TimeFrame.Hour,   limit=50,  days_back=10)
                bars_1d = self._get_bars(ticker, TimeFrame.Day,    limit=60,  days_back=120)

                log.debug("%s bars → 5m:%d  1h:%d  1d:%d",
                          ticker, len(bars_5m), len(bars_1h), len(bars_1d))

                closes_5m = [b["close"]  for b in bars_5m]
                closes_1h = [b["close"]  for b in bars_1h]
                closes_1d = [b["close"]  for b in bars_1d]
                highs_1d  = [b["high"]   for b in bars_1d]
                lows_1d   = [b["low"]    for b in bars_1d]
                vols_1d   = [b["volume"] for b in bars_1d]

                ind: dict = {}

                if len(closes_1d) >= 20:
                    ind["rsi_14"]            = self._rsi(closes_1d, 14)
                    ind["macd"]              = self._macd(closes_1d)
                    ind["bb"]                = self._bollinger(closes_1d, 20)
                    ind["ema_9"]             = self._ema(closes_1d, 9)
                    ind["ema_21"]            = self._ema(closes_1d, 21)
                    ind["sma_50"]            = self._sma(closes_1d, 50)  if len(closes_1d) >= 50  else None
                    ind["sma_200"]           = self._sma(closes_1d, 200) if len(closes_1d) >= 200 else None
                    ind["atr_14"]            = self._atr(highs_1d, lows_1d, closes_1d, 14)
                    ind["vol_ratio"]         = self._vol_ratio(vols_1d, 20)
                    ind["momentum_1d"]       = change_pct
                    ind["momentum_5d"]       = self._momentum(closes_1d, 5)
                    ind["momentum_20d"]      = self._momentum(closes_1d, 20)
                    ind["market_regime"]     = self._market_regime(closes_1d)
                    ind["support"]           = min(lows_1d[-10:])  if len(lows_1d)  >= 10 else low
                    ind["resistance"]        = max(highs_1d[-10:]) if len(highs_1d) >= 10 else high
                    ind["high_52w"]          = max(highs_1d) if highs_1d else high
                    ind["low_52w"]           = min(lows_1d)  if lows_1d  else low
                    pct52 = ((price - max(highs_1d)) / max(highs_1d) * 100) if highs_1d else 0.0
                    ind["pct_from_52w_high"] = pct52
                else:
                    log.warning("%s: only %d daily bars — indicators unavailable",
                                ticker, len(closes_1d))

                if len(closes_5m) >= 14:
                    ind["rsi_5m"] = self._rsi(closes_5m, 14)
                if len(closes_1h) >= 14:
                    ind["rsi_1h"] = self._rsi(closes_1h, 14)

                ind["signals"] = self._signals(price, ind)

                result[ticker] = {
                    "price":      price,
                    "prev_close": prev_close,
                    "open":       open_,
                    "high":       high,
                    "low":        low,
                    "vwap":       vwap,
                    "volume":     volume,
                    "change_pct": round(change_pct, 2),
                    **ind,
                }

            except Exception as e:
                log.error("Indicator error %s: %s", ticker, e, exc_info=True)

        return result

    def get_news(self, tickers: List[str]) -> List[dict]:
        items = []
        try:
            news = self.api.get_news(symbol=",".join(tickers),
                                     limit=15, include_content=False)
            for n in news:
                items.append({
                    "ticker":    n.symbols[0] if n.symbols else "MARKET",
                    "headline":  n.headline,
                    "summary":   (n.summary or "")[:200],
                    "published": str(n.created_at),
                })
        except Exception as e:
            log.warning("News failed: %s", e)
        return items

    # ─────────────────────────────────────────────────────────────
    # BAR FETCHER — with explicit start date
    # ─────────────────────────────────────────────────────────────
    def _get_bars(self, ticker: str, timeframe,
                   limit: int, days_back: int = 120) -> List[dict]:
        """
        Fetch OHLCV bars with explicit start date.
        Without start, Alpaca returns only today's bar.
        """
        now   = datetime.now(pytz.UTC)
        start = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            df = self.api.get_bars(
                ticker,
                timeframe,
                start      = start,
                limit      = limit,
                adjustment = "raw",
                feed       = self.config.ALPACA_DATA_FEED,
            ).df

            if df is None or df.empty:
                log.debug("Empty bars for %s", ticker)
                return []

            result = []
            for idx, row in df.iterrows():
                result.append({
                    "time":   str(idx),
                    "open":   float(row.get("open",   row.get("o", 0))),
                    "high":   float(row.get("high",   row.get("h", 0))),
                    "low":    float(row.get("low",    row.get("l", 0))),
                    "close":  float(row.get("close",  row.get("c", 0))),
                    "volume": int(row.get("volume",   row.get("v", 0))),
                })
            return result

        except Exception as e:
            log.debug("Primary bar fetch failed %s: %s — trying fallback", ticker, e)
            return self._get_bars_fallback(ticker, timeframe, limit, start)

    def _get_bars_fallback(self, ticker: str, timeframe,
                            limit: int, start: str) -> List[dict]:
        """Fallback: iterate bar objects directly."""
        try:
            result = []
            for bar in self.api.get_bars_iter(
                ticker, timeframe, start=start, limit=limit
            ):
                result.append({
                    "time":   str(bar.t),
                    "open":   float(bar.o),
                    "high":   float(bar.h),
                    "low":    float(bar.l),
                    "close":  float(bar.c),
                    "volume": int(bar.v),
                })
            return result
        except Exception as e:
            log.debug("Fallback bars also failed %s: %s", ticker, e)
            return []

    # ─────────────────────────────────────────────────────────────
    # INDICATORS
    # ─────────────────────────────────────────────────────────────
    def _ema(self, closes: List[float], period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        k   = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        for p in closes[period:]:
            ema = p * k + ema * (1 - k)
        return round(ema, 2)

    def _sma(self, closes: List[float], period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        return round(sum(closes[-period:]) / period, 2)

    def _rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]
        ag = sum(gains[:period])  / period
        al = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            ag = (ag * (period - 1) + gains[i])  / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100.0
        return round(100 - 100 / (1 + ag / al), 1)

    def _macd(self, closes: List[float]) -> Optional[dict]:
        e12 = self._ema(closes, 12)
        e26 = self._ema(closes, 26)
        if e12 is None or e26 is None:
            return None
        line = e12 - e26
        return {"macd": round(line, 3), "signal": round(line * 0.9, 3),
                "histogram": round(line * 0.1, 3), "bullish": line > 0}

    def _bollinger(self, closes: List[float],
                    period: int = 20) -> Optional[dict]:
        if len(closes) < period:
            return None
        recent = closes[-period:]
        mid    = sum(recent) / period
        std    = statistics.stdev(recent)
        upper  = mid + 2 * std
        lower  = mid - 2 * std
        price  = closes[-1]
        pct_b  = (price - lower) / (upper - lower) if upper != lower else 0.5
        return {"upper": round(upper, 2), "mid": round(mid, 2),
                "lower": round(lower, 2), "pct_b": round(pct_b, 3),
                "width": round((upper - lower) / mid * 100, 2)}

    def _atr(self, highs: List[float], lows: List[float],
              closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        trs = [max(highs[i] - lows[i],
                   abs(highs[i] - closes[i-1]),
                   abs(lows[i]  - closes[i-1]))
               for i in range(1, len(closes))]
        return round(sum(trs[-period:]) / period, 3)

    def _vol_ratio(self, volumes: List[float],
                    period: int = 20) -> Optional[float]:
        if len(volumes) < period + 1:
            return None
        avg = sum(volumes[-period-1:-1]) / period
        return round(volumes[-1] / avg, 2) if avg else 1.0

    def _momentum(self, closes: List[float],
                   period: int) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        past = closes[-period - 1]
        return round((closes[-1] - past) / past * 100, 2) if past else None

    def _market_regime(self, closes: List[float]) -> str:
        e9  = self._ema(closes, 9)
        e21 = self._ema(closes, 21)
        e50 = self._ema(closes, 50) if len(closes) >= 50 else None
        p   = closes[-1]
        if e9 and e21 and e50:
            if p > e9 > e21 > e50: return "STRONG_BULL"
            if p > e21 > e50:      return "BULL"
            if p < e9 < e21:       return "BEAR"
            if p < e50:            return "STRONG_BEAR"
        elif e9 and e21:
            return "BULL" if e9 > e21 else "BEAR"
        return "CHOPPY"

    # ─────────────────────────────────────────────────────────────
    # SIGNAL GENERATOR
    # ─────────────────────────────────────────────────────────────
    def _signals(self, price: float, ind: dict) -> dict:
        buys, sells = [], []
        rsi    = ind.get("rsi_14")
        rsi5m  = ind.get("rsi_5m")
        macd   = ind.get("macd") or {}
        bb     = ind.get("bb")   or {}
        regime = ind.get("market_regime", "CHOPPY")
        vr     = ind.get("vol_ratio") or 1.0
        mom5d  = ind.get("momentum_5d") or 0.0

        if rsi:
            if rsi < 35:   buys.append(f"RSI oversold ({rsi})")
            elif rsi < 45: buys.append(f"RSI low ({rsi})")
            if rsi > 70:   sells.append(f"RSI overbought ({rsi})")
            elif rsi > 60: sells.append(f"RSI elevated ({rsi})")

        if rsi5m:
            if rsi5m < 30: buys.append(f"5m RSI oversold ({rsi5m})")
            if rsi5m > 75: sells.append(f"5m RSI overbought ({rsi5m})")

        if macd.get("bullish") and macd.get("histogram", 0) > 0:
            buys.append("MACD bullish")
        elif macd and not macd.get("bullish"):
            sells.append("MACD bearish")

        pct_b = bb.get("pct_b")
        if pct_b is not None:
            if pct_b < 0.1:   buys.append(f"Lower BB ({pct_b:.2f})")
            elif pct_b > 0.9: sells.append(f"Upper BB ({pct_b:.2f})")

        if vr > 2.0:
            buys.append(f"Volume {vr:.1f}x avg")

        if regime in ("STRONG_BULL", "BULL"):
            buys.append(f"Uptrend ({regime})")
        elif regime in ("STRONG_BEAR", "BEAR"):
            sells.append(f"Downtrend ({regime})")

        if mom5d > 5:    buys.append(f"5d mom +{mom5d:.1f}%")
        elif mom5d < -5: sells.append(f"5d mom {mom5d:.1f}%")

        score = len(buys) - len(sells)
        if score >= 3:    bias = "STRONG_BUY"
        elif score >= 1:  bias = "BUY"
        elif score <= -3: bias = "STRONG_SELL"
        elif score <= -1: bias = "SELL"
        else:             bias = "NEUTRAL"

        return {"bias": bias, "score": score,
                "buy_signals": buys, "sell_signals": sells}
