"""
live_feed.py
============
Real-time WebSocket price feed with live stop management.

Features:
  - Streams live 1-minute bars from Alpaca
  - Checks stops on every price update (not every 10 minutes)
  - Moves stop to breakeven when position up 1%
  - Locks in 1% profit when position up 2%
  - Trails stop continuously using ATR
  - Time stop: exits flat positions after 2 hours
  - Detects significant moves and triggers Claude re-analysis
  - Auto-reconnects on disconnect
  - Volume spike detection
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pytz

from position_manager import PositionManager

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


class LiveFeed:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        watchlist: List[str],
        position_manager: PositionManager,
        on_update: Callable,  # callback(symbol, price, volume, change_pct)
        on_exit: Callable,  # callback(symbol, reason, price)
        on_trigger: Callable,  # callback(symbol, trigger_name, price, pl_pct)
        data_feed: str = "iex",
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.watchlist = [t.upper() for t in watchlist]
        self.position_manager = position_manager
        self.on_update = on_update
        self.on_exit = on_exit
        self.on_trigger = on_trigger
        self.data_feed = data_feed

        # State
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._running: bool = False
        self._prev_prices: Dict[str, float] = {}
        self._last_prices: Dict[str, float] = {}
        self._bar_counts: Dict[str, int] = {}

        log.info(
            "📡 LiveFeed initialized | feed=%s | tickers=%s", data_feed, self.watchlist
        )

    # ─────────────────────────────────────────────────────────────
    # POSITION REGISTRATION (delegated to PositionManager)
    # ─────────────────────────────────────────────────────────────

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        qty: int,
        side: str = "long",
    ):
        """Register a new position for real-time monitoring."""
        self.position_manager.register(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            qty=qty,
            side=side,
        )

    def remove_position(self, symbol: str):
        """Remove position from monitoring after it's closed."""
        self.position_manager.remove(symbol)

    def update_stop(self, symbol: str, new_stop: float):
        """Update stop level for an existing position."""
        self.position_manager.update_stop(symbol, new_stop)

    # ─────────────────────────────────────────────────────────────
    # REAL-TIME STOP MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def _check_position(self, symbol: str, price: float):
        """
        Full position check on every price update.
        Delegates to PositionManager for state management.
        """
        # Check for exit conditions (stop, target, time)
        exit_reason, pl_pct = self.position_manager.check_position(symbol, price)
        if exit_reason:
            log.info(
                "🔔 EXIT TRIGGERED: %s  price=$%.2f  %s",
                symbol,
                price,
                exit_reason,
            )
            self.on_exit(symbol, exit_reason, price)
            self.remove_position(symbol)
            return

        # Check for breakeven stop trigger
        trigger = self.position_manager.update_breakeven_stop(symbol, price)
        if trigger:
            self.on_trigger(symbol, trigger, price, pl_pct)

        # Check for locked profit trigger
        trigger = self.position_manager.update_locked_profit(symbol, price)
        if trigger:
            self.on_trigger(symbol, trigger, price, pl_pct)

        # Update trailing stop
        if self.position_manager.update_trailing_stop(symbol, price):
            # Trailing stop was updated, could log but keep quiet for noise reduction
            pass

        # Check for significant move triggers
        self._check_triggers(symbol, price, pl_pct)

    def _check_triggers(self, symbol: str, price: float, pl_pct: float):
        """
        Fire callbacks when position hits key levels.
        Triggers Claude re-analysis at important moments.
        """
        # Get position to check trigger flags
        pos = self.position_manager.get(symbol)
        if not pos:
            return

        # Up 1% for first time
        if pl_pct >= 1.0 and not pos["triggered_1up"]:
            pos["triggered_1up"] = True
            log.info("⚡ TRIGGER +1%%: %s at $%.2f", symbol, price)
            self.on_trigger(symbol, "up_1pct", price, pl_pct)

        # Up 2% for first time
        if pl_pct >= 2.0 and not pos["triggered_2up"]:
            pos["triggered_2up"] = True
            log.info("⚡ TRIGGER +2%%: %s at $%.2f", symbol, price)
            self.on_trigger(symbol, "up_2pct", price, pl_pct)

        # Down 1% for first time
        if pl_pct <= -1.0 and not pos["triggered_1dn"]:
            pos["triggered_1dn"] = True
            log.info("⚡ TRIGGER -1%%: %s at $%.2f", symbol, price)
            self.on_trigger(symbol, "down_1pct", price, pl_pct)

        # Recovered to breakeven after being negative
        if pos["was_negative"] and pl_pct >= 0 and not pos.get("triggered_recovery"):
            pos["triggered_recovery"] = True
            log.info("⚡ TRIGGER RECOVERY: %s back to breakeven", symbol)
            self.on_trigger(symbol, "breakeven_recovery", price, pl_pct)

    # ─────────────────────────────────────────────────────────────
    # WEBSOCKET STREAM
    # ─────────────────────────────────────────────────────────────

    def start(self):
        """Start WebSocket in background daemon thread."""
        if self._running:
            log.warning("LiveFeed already running.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_with_reconnect,
            name="LiveFeed",
            daemon=True,
        )
        self._thread.start()
        log.info("✅ LiveFeed thread started.")

    def stop(self):
        """Gracefully stop the WebSocket."""
        log.info("🛑 Stopping LiveFeed...")
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("LiveFeed stopped.")

    def _run_with_reconnect(self):
        """Auto-reconnect with exponential backoff."""
        backoff = 10
        attempt = 0
        while self._running:
            attempt += 1
            if attempt > 1:
                log.info("🔄 Reconnect #%d in %ds...", attempt, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            try:
                log.info("🔌 Connecting to stream (attempt #%d)...", attempt)
                self._run_stream()
                backoff = 10
            except Exception as e:
                if self._running:
                    log.error("❌ Stream error: %s", e)
                else:
                    break

    def _run_stream(self):
        """Create and run the Alpaca WebSocket stream."""
        try:
            from alpaca_trade_api.stream import Stream
        except ImportError:
            log.error("alpaca-trade-api not installed.")
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        stream = Stream(
            self.api_key,
            self.secret_key,
            base_url="https://stream.data.alpaca.markets",
            data_feed=self.data_feed,
        )
        self._stream = stream

        async def on_bar(bar):
            try:
                symbol = bar.symbol
                price = float(bar.close)
                volume = int(bar.volume)
                prev = self._prev_prices.get(symbol, price)
                chg = ((price - prev) / prev * 100) if prev else 0.0

                self._prev_prices[symbol] = price
                self._last_prices[symbol] = price
                self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1

                # Volume spike detection
                self._check_volume_spike(symbol, volume)

                # Real-time position management
                self._check_position(symbol, price)

                # General update callback
                self.on_update(symbol, price, volume, chg)

            except Exception as e:
                log.error("Bar handler error: %s", e)

        for ticker in self.watchlist:
            stream.subscribe_bars(on_bar, ticker)

        log.info("📡 Subscribed to bars: %s", self.watchlist)
        stream.run()

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _check_volume_spike(self, symbol: str, volume: int):
        """Detect unusual volume — signals institutional activity."""
        key = f"_vol_{symbol}"
        history = getattr(self, key, [])
        history.append(volume)
        if len(history) > 5:
            history = history[-5:]
        setattr(self, key, history)

        if len(history) >= 3:
            avg = sum(history[:-1]) / len(history[:-1])
            if avg > 0 and volume >= avg * 3:
                log.info(
                    "⚡ VOL SPIKE: %s  current=%s  avg=%s  ratio=%.1fx",
                    symbol,
                    f"{volume:,}",
                    f"{int(avg):,}",
                    volume / avg,
                )

    def get_price(self, symbol: str) -> Optional[float]:
        return self._last_prices.get(symbol.upper())

    def get_all_prices(self) -> Dict[str, float]:
        return dict(self._last_prices)

    def get_position_status(self, symbol: str) -> Optional[dict]:
        return self.position_manager.get(symbol)

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "tickers": self.watchlist,
            "bar_counts": dict(self._bar_counts),
            "prices": dict(self._last_prices),
            "positions": self.position_manager.get_symbols(),
        }
