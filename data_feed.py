"""
data_feed.py
============
Market data feed + technical indicators for live trading.

All indicator calculations delegated to indicators.py —
the same module used by backtest.py — so live signals are
100% identical to backtest signals, eliminating the #1 cause
of live vs backtest divergence.

Flow per ticker:
  1. Fetch snapshot (latest price, volume, OHLC)
  2. Fetch 1d / 1h / 5m bars from Alpaca
  3. Compute indicators via indicators.py (shared with backtest)
  4. Score signals via indicators.score_signals() — exact match
  5. Return unified dict consumed by claude_brain + risk_manager
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytz
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

import indicators as ind

log = logging.getLogger(__name__)


class MarketDataFeed:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL,
            api_version="v2",
        )
        # Ticker type cache — set by classify_watchlist() at startup
        self.ticker_types: Dict[str, str] = {}

    def set_ticker_types(self, types: Dict[str, str]):
        """Called by bot.py after auto-classification."""
        self.ticker_types = types
        log.info("📐 Ticker types loaded: %s", types)

    # ─────────────────────────────────────────────────────────────
    # PRIMARY ENTRY POINT
    # ─────────────────────────────────────────────────────────────

    def get_quotes(self, tickers: List[str]) -> Dict[str, dict]:
        """
        Fetch market data and compute indicators for all tickers.
        Returns dict keyed by ticker symbol.
        """
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
                data = self._process_ticker(ticker, snap)
                if data:
                    result[ticker] = data
            except Exception as e:
                log.error("Indicator error %s: %s", ticker, e, exc_info=True)

        return result

    def get_news(self, tickers: List[str]) -> List[dict]:
        items = []
        try:
            news = self.api.get_news(
                symbol=",".join(tickers),
                limit=15,
                include_content=False,
            )
            for n in news:
                items.append(
                    {
                        "ticker": n.symbols[0] if n.symbols else "MARKET",
                        "headline": n.headline,
                        "summary": (n.summary or "")[:200],
                        "published": str(n.created_at),
                    }
                )
        except Exception as e:
            log.warning("News failed: %s", e)
        return items

    # ─────────────────────────────────────────────────────────────
    # PER-TICKER PROCESSING
    # ─────────────────────────────────────────────────────────────

    def _process_ticker(self, ticker: str, snap) -> Optional[dict]:
        """
        Full indicator pipeline for one ticker.
        Uses indicators.py for all calculations — same as backtest.
        """
        # ── Extract snapshot values ───────────────────────────
        price = float(snap.latest_trade.price) if snap.latest_trade else 0.0
        prev_close = float(snap.prev_daily_bar.c) if snap.prev_daily_bar else price
        volume = int(snap.daily_bar.v) if snap.daily_bar else 0
        open_ = float(snap.daily_bar.o) if snap.daily_bar else price
        high = float(snap.daily_bar.h) if snap.daily_bar else price
        low_ = float(snap.daily_bar.l) if snap.daily_bar else price
        vwap = float(snap.daily_bar.vw) if snap.daily_bar else price
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

        # ── Fetch bars ────────────────────────────────────────
        bars_1d = self._get_bars(ticker, TimeFrame.Day, limit=60, days_back=120)
        bars_1h = self._get_bars(ticker, TimeFrame.Hour, limit=50, days_back=10)
        bars_5m = self._get_bars(ticker, TimeFrame.Minute, limit=75, days_back=1)

        log.debug(
            "%s bars → 1d:%d 1h:%d 5m:%d",
            ticker,
            len(bars_1d),
            len(bars_1h),
            len(bars_5m),
        )

        if len(bars_1d) < 20:
            log.warning("%s: only %d daily bars — skipping", ticker, len(bars_1d))
            return None

        # ── Build price arrays ────────────────────────────────
        closes_1d = [b["close"] for b in bars_1d]
        highs_1d = [b["high"] for b in bars_1d]
        lows_1d = [b["low"] for b in bars_1d]
        vols_1d = [b["volume"] for b in bars_1d]
        closes_1h = [b["close"] for b in bars_1h]
        closes_5m = [b["close"] for b in bars_5m]

        # ── Ticker type ───────────────────────────────────────
        ticker_type = self.ticker_types.get(ticker, "quality")

        # ── Score signals via indicators.py ───────────────────
        # This matches backtest.py generate_signals() EXACTLY
        scored = ind.score_signals(
            closes=closes_1d,
            highs=highs_1d,
            lows=lows_1d,
            volumes=vols_1d,
            ticker_type=ticker_type,
        )

        # ── Multi-timeframe RSI ───────────────────────────────
        rsi_5m = ind.calc_rsi_value(closes_5m) if len(closes_5m) >= 14 else None
        rsi_1h = ind.calc_rsi_value(closes_1h) if len(closes_1h) >= 14 else None

        # ── Additional indicators ─────────────────────────────
        sma_50 = ind.calc_sma(closes_1d, 50)
        sma_200 = ind.calc_sma(closes_1d, 200) if len(closes_1d) >= 200 else None
        bb = ind.calc_bollinger(closes_1d, 20)
        high_52w = max(highs_1d) if highs_1d else high
        low_52w = min(lows_1d) if lows_1d else low_
        pct_52w = ((price - high_52w) / high_52w * 100) if high_52w else 0.0

        # ── Build unified result ──────────────────────────────
        return {
            # Price / volume
            "price": price,
            "prev_close": prev_close,
            "open": open_,
            "high": high,
            "low": low_,
            "vwap": vwap,
            "volume": volume,
            "change_pct": round(change_pct, 2),
            # Classification
            "ticker_type": ticker_type,
            # Backtest-matched indicators (from score_signals)
            "rsi_14": scored["rsi"],
            "atr_14": scored["atr"],
            "vol_ratio": scored["vol_ratio"],
            "momentum_5d": scored["mom5d"],
            "momentum_20d": scored["mom20d"],
            "ema_9": scored["ema9"],
            "ema_21": scored["ema21"],
            "ema_50": scored["ema50"],
            "pct_b": scored["pct_b"],
            "market_regime": scored["regime_str"],
            "market_regime_num": scored["regime"],
            # Signal scoring (matches backtest thresholds)
            "signal_count": scored["signal_count"],
            "entry_signal": scored["entry"],
            "exit_signal": scored["exit"],
            # MACD
            "macd": scored["macd"]["macd"],
            "macd_signal": scored["macd"]["signal"],
            "macd_hist": scored["macd"]["histogram"],
            "macd_bullish": scored["macd"]["bullish"],
            # Multi-timeframe RSI
            "rsi_5m": rsi_5m,
            "rsi_1h": rsi_1h,
            # Additional indicators
            "sma_50": sma_50,
            "sma_200": sma_200,
            "bb": bb,
            "momentum_1d": round(change_pct, 2),
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pct_from_52w_high": round(pct_52w, 2),
            # Legacy signals dict (for claude_brain.py prompt)
            "signals": scored["signals"],
        }

    # ─────────────────────────────────────────────────────────────
    # BAR FETCHER
    # ─────────────────────────────────────────────────────────────

    def _get_bars(
        self, ticker: str, timeframe, limit: int, days_back: int = 120
    ) -> List[dict]:
        """
        Fetch OHLCV bars with explicit start date.
        Without start, Alpaca returns only the current bar.
        """
        now = datetime.now(pytz.UTC)
        start = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            df = self.api.get_bars(
                ticker,
                timeframe,
                start=start,
                limit=limit,
                adjustment="raw",
                feed=self.config.ALPACA_DATA_FEED,
            ).df
            if df is None or df.empty:
                return []
            result = []
            for _, row in df.iterrows():
                result.append(
                    {
                        "open": float(row.get("open", row.get("o", 0))),
                        "high": float(row.get("high", row.get("h", 0))),
                        "low": float(row.get("low", row.get("l", 0))),
                        "close": float(row.get("close", row.get("c", 0))),
                        "volume": int(row.get("volume", row.get("v", 0))),
                    }
                )
            return result
        except Exception as e:
            log.debug("Bar fetch failed %s: %s — trying fallback", ticker, e)
            return self._get_bars_fallback(ticker, timeframe, limit, start)

    def _get_bars_fallback(
        self, ticker: str, timeframe, limit: int, start: str
    ) -> List[dict]:
        try:
            result = []
            for bar in self.api.get_bars_iter(
                ticker, timeframe, start=start, limit=limit
            ):
                result.append(
                    {
                        "open": float(bar.o),
                        "high": float(bar.h),
                        "low": float(bar.l),
                        "close": float(bar.c),
                        "volume": int(bar.v),
                    }
                )
            return result
        except Exception as e:
            log.debug("Fallback bars failed %s: %s", ticker, e)
            return []

    # ─────────────────────────────────────────────────────────────
    # AUTO-CLASSIFY WATCHLIST
    # ─────────────────────────────────────────────────────────────

    def classify_watchlist(self, tickers: List[str]) -> Dict[str, str]:
        """
        Classify each ticker as momentum / quality / index.
        Call once at bot startup before first cycle.
        Uses same algorithm as backtest.py classify_ticker().

        Returns and caches: {"NVDA": "momentum", "AAPL": "quality", ...}
        """
        log.info("🔍 Auto-classifying watchlist (%d tickers)...", len(tickers))

        raw_data: Dict[str, dict] = {}
        for ticker in tickers:
            bars = self._get_bars(ticker, TimeFrame.Day, limit=300, days_back=400)
            if len(bars) >= 60:
                raw_data[ticker] = {
                    "closes": [b["close"] for b in bars],
                    "highs": [b["high"] for b in bars],
                    "lows": [b["low"] for b in bars],
                    "volumes": [b["volume"] for b in bars],
                }

        # SPY closes for beta calculation
        spy_closes = None
        if "SPY" in raw_data:
            spy_closes = raw_data["SPY"]["closes"]
        else:
            spy_bars = self._get_bars("SPY", TimeFrame.Day, limit=300, days_back=400)
            if spy_bars:
                spy_closes = [b["close"] for b in spy_bars]

        # Classify each ticker
        icons = {"momentum": "🚀", "quality": "📈", "index": "📊"}
        classifications = {}

        for ticker, d in raw_data.items():
            t_type = ind.classify_ticker_live(
                closes=d["closes"],
                highs=d["highs"],
                lows=d["lows"],
                volumes=d["volumes"],
                spy_closes=spy_closes,
            )
            classifications[ticker] = t_type
            log.info("  %s %-6s → %s", icons.get(t_type, "•"), ticker, t_type)

        self.set_ticker_types(classifications)
        return classifications
