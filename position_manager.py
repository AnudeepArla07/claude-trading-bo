"""
position_manager.py
===================
Centralized position state management for stocks and options.
Handles reconciliation, stop updates, and trigger detection.
Python 3.9 compatible.
"""

import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytz

log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

# Thread-safe lock for position updates
_position_lock = threading.Lock()


class PositionManager:
    """
    Manages all open positions with thread-safe state tracking.
    Coordinates between live_feed (real-time) and bot cycle (periodic checks).
    """

    def __init__(self):
        # Open positions for real-time management
        # Format: {
        #   "NVDA": {
        #     "entry_price": 450.0,
        #     "stop_loss":   441.0,
        #     "take_profit": 478.0,
        #     "atr":         4.5,
        #     "peak_price":  450.0,
        #     "entry_time":  datetime,
        #     "qty":         10,
        #     "side":        "long",
        #     "breakeven_moved": False,
        #     "locked_1pct":     False,
        #     "triggered_1up":   False,
        #     "triggered_2up":   False,
        #     "triggered_1dn":   False,
        #     "was_negative":    False,
        #   }
        # }
        self._positions: Dict[str, dict] = {}

    # ─────────────────────────────────────────────────────────────
    # POSITION REGISTRATION & REMOVAL
    # ─────────────────────────────────────────────────────────────

    def register(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        atr: float,
        qty: int,
        side: str = "long",
    ) -> None:
        """Register a new position for tracking."""
        with _position_lock:
            self._positions[symbol] = {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": atr,
                "peak_price": entry_price,
                "entry_time": datetime.now(ET),
                "qty": qty,
                "side": side,
                "breakeven_moved": False,
                "locked_1pct": False,
                "triggered_1up": False,
                "triggered_2up": False,
                "triggered_1dn": False,
                "was_negative": False,
            }
            log.info(
                "📌 Position registered: %s  entry=$%.2f  stop=$%.2f  target=$%.2f  atr=$%.2f",
                symbol,
                entry_price,
                stop_loss,
                take_profit,
                atr,
            )

    def remove(self, symbol: str) -> None:
        """Remove position from tracking."""
        with _position_lock:
            if symbol in self._positions:
                del self._positions[symbol]
                log.info("🗑️  Position removed: %s", symbol)

    def get(self, symbol: str) -> Optional[dict]:
        """Get position state (thread-safe copy)."""
        with _position_lock:
            return self._positions.get(symbol)

    def get_all(self) -> Dict[str, dict]:
        """Get all positions as thread-safe copy."""
        with _position_lock:
            return dict(self._positions)

    def get_symbols(self) -> List[str]:
        """Get list of all tracked symbols."""
        with _position_lock:
            return list(self._positions.keys())

    def count(self) -> int:
        """Get number of open positions."""
        with _position_lock:
            return len(self._positions)

    # ─────────────────────────────────────────────────────────────
    # STOP & TARGET MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def update_stop(self, symbol: str, new_stop: float) -> bool:
        """Update stop loss for a position. Returns True if updated."""
        with _position_lock:
            if symbol not in self._positions:
                log.warning("Update stop: %s not found", symbol)
                return False
            old_stop = self._positions[symbol]["stop_loss"]
            self._positions[symbol]["stop_loss"] = new_stop
            log.info(
                "🔄 Stop updated: %s  $%.2f → $%.2f",
                symbol,
                old_stop,
                new_stop,
            )
            return True

    def update_target(self, symbol: str, new_target: float) -> bool:
        """Update take profit target. Returns True if updated."""
        with _position_lock:
            if symbol not in self._positions:
                log.warning("Update target: %s not found", symbol)
                return False
            old_target = self._positions[symbol]["take_profit"]
            self._positions[symbol]["take_profit"] = new_target
            log.info(
                "🎯 Target updated: %s  $%.2f → $%.2f",
                symbol,
                old_target,
                new_target,
            )
            return True

    # ─────────────────────────────────────────────────────────────
    # REAL-TIME POSITION CHECKS
    # ─────────────────────────────────────────────────────────────

    def check_position(self, symbol: str, price: float) -> Tuple[Optional[str], float]:
        """
        Check position state against current price.
        Returns: (exit_reason, pl_pct) or (None, pl_pct) if no exit
        
        Exit reasons:
        - "stop_loss": hard stop hit
        - "profit_target": take profit hit
        - "time_stop": held > 2hrs with pl < 0.5%
        """
        with _position_lock:
            pos = self._positions.get(symbol)
            if not pos:
                return None, 0.0

            entry = pos["entry_price"]
            stop = pos["stop_loss"]
            target = pos["take_profit"]
            atr = pos["atr"]
            side = pos["side"]
            entry_tm = pos["entry_time"]

            # Calculate P&L
            if side == "long":
                pl_pct = (price - entry) / entry * 100
            else:
                pl_pct = (entry - price) / entry * 100

            # Track if ever negative
            if pl_pct < 0:
                pos["was_negative"] = True

            # Check stop
            if side == "long" and price <= stop:
                return f"stop_loss (P&L: {pl_pct:.1f}%)", pl_pct

            # Check target
            if side == "long" and price >= target:
                return f"profit_target (+{pl_pct:.1f}%)", pl_pct

            # Check time stop (2 hours, flat/losing)
            now = datetime.now(ET)
            held_hr = (now - entry_tm).seconds / 3600
            if held_hr > 2.0 and pl_pct < 0.5:
                return f"time_stop (held {held_hr:.1f}hr at {pl_pct:.1f}%)", pl_pct

            return None, pl_pct

    def update_breakeven_stop(self, symbol: str, price: float) -> Optional[str]:
        """
        Move stop to breakeven when position up 1%.
        Returns trigger name if triggered, None otherwise.
        """
        with _position_lock:
            pos = self._positions.get(symbol)
            if not pos or pos["breakeven_moved"]:
                return None

            entry = pos["entry_price"]
            pl_pct = (price - entry) / entry * 100

            if pl_pct >= 1.0:
                breakeven = entry * 1.001  # just above entry
                if breakeven > pos["stop_loss"]:
                    pos["stop_loss"] = breakeven
                    pos["breakeven_moved"] = True
                    log.info(
                        "📈 BREAKEVEN: %s up %.1f%% — stop moved to $%.2f",
                        symbol,
                        pl_pct,
                        breakeven,
                    )
                    return "breakeven_stop"
            return None

    def update_locked_profit(self, symbol: str, price: float) -> Optional[str]:
        """
        Lock in 1% profit when position up 2%.
        Returns trigger name if triggered, None otherwise.
        """
        with _position_lock:
            pos = self._positions.get(symbol)
            if not pos or pos["locked_1pct"]:
                return None

            entry = pos["entry_price"]
            pl_pct = (price - entry) / entry * 100

            if pl_pct >= 2.0:
                lock_stop = entry * 1.01  # lock in 1%
                if lock_stop > pos["stop_loss"]:
                    pos["stop_loss"] = lock_stop
                    pos["locked_1pct"] = True
                    log.info(
                        "🔒 LOCK 1%%: %s up %.1f%% — stop locked at $%.2f",
                        symbol,
                        pl_pct,
                        lock_stop,
                    )
                    return "locked_1pct"
            return None

    def update_trailing_stop(self, symbol: str, price: float) -> bool:
        """
        Trail stop by 1.5x ATR from peak. Returns True if updated.
        """
        with _position_lock:
            pos = self._positions.get(symbol)
            if not pos:
                return False

            if price > pos["peak_price"]:
                pos["peak_price"] = price
                atr = pos["atr"]
                new_trail = price - atr * 1.5
                if new_trail > pos["stop_loss"]:
                    pos["stop_loss"] = new_trail
                    log.info(
                        "📈 TRAIL: %s peak=$%.2f  stop=$%.2f",
                        symbol,
                        price,
                        new_trail,
                    )
                    return True
            return False

    # ─────────────────────────────────────────────────────────────
    # RECONCILIATION
    # ─────────────────────────────────────────────────────────────

    def reconcile_with_broker(self, broker_positions: List[dict]) -> Tuple[List[str], List[str]]:
        """
        Reconcile internal positions with broker positions.
        Returns (missing_symbols, extra_symbols)
        
        missing: positions we have but broker doesn't (stale)
        extra: positions broker has but we don't (need to register)
        """
        broker_symbols = {p["symbol"] for p in broker_positions}
        tracked_symbols = set(self.get_symbols())

        missing = list(tracked_symbols - broker_symbols)
        extra = list(broker_symbols - tracked_symbols)

        if missing:
            log.warning("⚠️  Stale positions removed: %s", missing)
            for sym in missing:
                self.remove(sym)

        if extra:
            log.warning("⚠️  Positions not tracked: %s (register them)", extra)

        return missing, extra

    # ─────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Clear all positions (for shutdown)."""
        with _position_lock:
            self._positions.clear()
            log.info("🗑️  All positions cleared")

    def summary(self) -> str:
        """Get summary of all positions."""
        with _position_lock:
            if not self._positions:
                return "No open positions"

            lines = [f"\n📊 Open Positions ({len(self._positions)})"]
            for symbol, pos in self._positions.items():
                entry = pos["entry_price"]
                current_stop = pos["stop_loss"]
                target = pos["take_profit"]
                qty = pos["qty"]
                lines.append(
                    f"  {symbol}: {qty} shares | "
                    f"entry=${entry:.2f} | stop=${current_stop:.2f} | target=${target:.2f}"
                )
            return "\n".join(lines)
