"""
scanner.py
==========
Dynamic watchlist generator. Scans market every morning.
Python 3.9 compatible.
"""
import logging
from typing import List

import alpaca_trade_api as tradeapi

log = logging.getLogger(__name__)


class MarketScanner:
    def __init__(self, config):
        self.config = config
        try:
            self.api = tradeapi.REST(
                config.ALPACA_API_KEY,
                config.ALPACA_SECRET_KEY,
                base_url    = config.ALPACA_BASE_URL,
                api_version = "v2"
            )
        except Exception as e:
            log.error("Scanner init failed: %s", e)
            self.api = None

    def get_dynamic_watchlist(self) -> List[str]:
        if not self.api:
            return self.config.WATCHLIST
        candidates = set()
        candidates.update(self.scan_volume_leaders())
        candidates.update(self.scan_gap_ups())
        candidates.update(self.scan_momentum())
        candidates.update(self.scan_sector_rotation())
        candidates.update(self.scan_earnings_movers())
        log.info("🔍 Total candidates: %d", len(candidates))
        ranked = self.rank_candidates(list(candidates))
        return ranked[:12] if ranked else self.config.WATCHLIST

    def scan_volume_leaders(self) -> List[str]:
        try:
            symbols = [
                "AAPL","MSFT","NVDA","AMD","TSLA","META","AMZN","GOOGL",
                "SPY","QQQ","SOFI","PLTR","RIVN","NIO","BABA","UBER","LYFT"
            ]
            snaps   = self.api.get_snapshots(symbols)
            return [t for t, s in snaps.items()
                    if s and s.daily_bar and s.daily_bar.v > 5_000_000]
        except Exception as e:
            log.error("Volume scan: %s", e); return []

    def scan_gap_ups(self) -> List[str]:
        try:
            symbols = [
                "AAPL","MSFT","GOOGL","NVDA","TSLA","META","AMZN",
                "AMD","NFLX","CRM","ORCL","ADBE","SHOP","UBER","PLTR",
                "SOFI","RIVN","NIO","BABA","JD","PDD","MSTR","COIN"
            ]
            snaps = self.api.get_snapshots(symbols)
            gaps  = []
            for t, s in snaps.items():
                if s and s.daily_bar and s.prev_daily_bar:
                    prev = s.prev_daily_bar.c
                    if prev > 0:
                        gap = (s.daily_bar.o - prev) / prev
                        if 0.02 <= gap <= 0.10:
                            gaps.append(t)
            return gaps
        except Exception as e:
            log.error("Gap scan: %s", e); return []

    def scan_momentum(self) -> List[str]:
        try:
            symbols = [
                "NVDA","MSFT","AAPL","META","GOOGL","AMZN","TSLA",
                "AVGO","LLY","JPM","V","MA","UNH","XOM","JNJ"
            ]
            snaps = self.api.get_snapshots(symbols)
            movers = []
            for t, s in snaps.items():
                if s and s.daily_bar and s.prev_daily_bar:
                    prev = s.prev_daily_bar.c
                    if prev > 0 and (s.daily_bar.c - prev) / prev > 0.015:
                        movers.append(t)
            return movers
        except Exception as e:
            log.error("Momentum scan: %s", e); return []

    def scan_sector_rotation(self) -> List[str]:
        sector_map = {
            "XLK": ["AAPL","MSFT","NVDA","AVGO","AMD"],
            "XLF": ["JPM","BAC","WFC","GS","MS"],
            "XLE": ["XOM","CVX","COP","EOG","SLB"],
            "XLV": ["UNH","JNJ","LLY","ABBV","MRK"],
            "XLY": ["AMZN","TSLA","HD","MCD","SBUX"],
            "XBI": ["MRNA","BNTX","REGN","VRTX","GILD"],
        }
        try:
            snaps = self.api.get_snapshots(list(sector_map.keys()))
            best_etf, best_chg = None, -999.0
            for etf, s in snaps.items():
                if s and s.daily_bar and s.prev_daily_bar:
                    prev = s.prev_daily_bar.c
                    if prev > 0:
                        chg = (s.daily_bar.c - prev) / prev
                        if chg > best_chg:
                            best_chg = chg; best_etf = etf
            if best_etf and best_chg > 0:
                log.info("🔥 Hottest sector: %s (+%.2f%%)", best_etf, best_chg*100)
                return sector_map[best_etf]
            return []
        except Exception as e:
            log.error("Sector scan: %s", e); return []

    def scan_earnings_movers(self) -> List[str]:
        try:
            news = self.api.get_news(symbol=None, limit=50)
            kws  = ["earnings beat","revenue beat","raised guidance",
                    "record revenue","topped estimates"]
            tickers = []
            for item in news:
                if any(k in item.headline.lower() for k in kws):
                    tickers.extend(item.symbols or [])
            return list(set(tickers))[:5]
        except Exception as e:
            log.warning("Earnings scan: %s", e); return []

    def rank_candidates(self, tickers: List[str]) -> List[str]:
        if not tickers:
            return []
        try:
            snaps  = self.api.get_snapshots(tickers)
            scored = []
            for t, s in snaps.items():
                if not s or not s.daily_bar or not s.prev_daily_bar:
                    continue
                price  = s.daily_bar.c
                vol    = s.daily_bar.v
                prev   = s.prev_daily_bar.c
                chg    = (price - prev) / prev if prev > 0 else 0
                score  = 0
                if vol > 10_000_000:  score += 30
                elif vol > 5_000_000: score += 20
                elif vol > 1_000_000: score += 10
                if chg > 0.03:   score += 25
                elif chg > 0.01: score += 15
                elif chg < 0:    score -= 20
                if 10 < price < 500: score += 15
                elif price > 500:    score -= 10
                elif price < 5:      score -= 20
                scored.append((t, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            ranked = [t for t, _ in scored]
            log.info("🏆 Top watchlist: %s", ranked[:10])
            return ranked
        except Exception as e:
            log.error("Ranking failed: %s", e); return tickers
