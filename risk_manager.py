"""
risk_manager.py
===============
Hard risk rules that override Claude.
Includes trend alignment check and ATR-based position sizing.
Python 3.9 compatible.
"""

import logging
from datetime import date
from typing import Tuple, Optional, Dict, List

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config):
        self.config = config
        self._daily_trades: Dict[date, int] = {}
        self._daily_start_equity: Optional[float] = None
        self._peak_equity: Optional[float] = None
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        self._peak_prices: Dict[str, float] = {}
        self._trailing_stops: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────
    # PRIMARY APPROVAL
    # ─────────────────────────────────────────────────────────────

    def approve(self, decision: dict, portfolio: dict) -> Tuple[bool, str]:
        if decision.get("action") == "hold":
            return True, "Hold."

        checks = [
            self._check_market_open_time(),
            self._check_confidence(decision),
            self._check_risk_reward(decision),
            self._check_trend_alignment(decision),
            self._check_daily_loss(portfolio),
            self._check_drawdown(portfolio),
            self._check_daily_count(),
            self._check_consecutive_losses(),
            self._check_quantity(decision),
            self._check_duplicate(decision, portfolio),
        ]

        for ok, reason in checks:
            if not ok:
                return False, reason

        self._inc()
        return True, "All checks passed ✅"

    # ─────────────────────────────────────────────────────────────
    # INDIVIDUAL CHECKS
    # ─────────────────────────────────────────────────────────────

    def _check_market_open_time(self) -> Tuple[bool, str]:
        from datetime import datetime
        import pytz

        et = pytz.timezone("America/New_York")
        now = datetime.now(et)
        if now.hour == 9 and now.minute < 35:
            return False, "Waiting for 9:35 AM ET open"
        return True, ""

    def _check_confidence(self, decision: dict) -> Tuple[bool, str]:
        try:
            conf = decision.get("confidence")
            if conf is None:
                return False, "Confidence not provided in decision"
            conf = float(conf)
            if conf < 0 or conf > 1:
                return False, f"Invalid confidence {conf} (must be 0-1)"
        except (TypeError, ValueError) as e:
            return False, f"Confidence parsing error: {e}"
        
        minimum = self.config.MIN_CONFIDENCE
        if self._consecutive_losses >= 2:
            minimum = min(minimum + 0.05 * self._consecutive_losses, 0.90)
        if conf < minimum:
            return False, (
                f"Confidence {conf:.0%} < required {minimum:.0%} "
                f"(loss streak: {self._consecutive_losses})"
            )
        return True, ""

    def _check_risk_reward(self, decision: dict) -> Tuple[bool, str]:
        try:
            rr = decision.get("risk_reward")
            entry = decision.get("entry_price") or decision.get("price")
            stop = decision.get("stop_loss")
            target = decision.get("take_profit")
            
            if rr is not None:
                rr = float(rr)
                if rr < self.config.MIN_RISK_REWARD:
                    return False, f"R:R {rr:.1f} < min {self.config.MIN_RISK_REWARD}"
            
            if entry and stop and target and decision.get("action") == "buy":
                entry = float(entry)
                stop = float(stop)
                target = float(target)
                risk = abs(entry - stop)
                reward = abs(target - entry)
                if risk > 0:
                    actual_rr = reward / risk
                    if actual_rr < self.config.MIN_RISK_REWARD:
                        return False, (
                            f"R:R {actual_rr:.1f} < min {self.config.MIN_RISK_REWARD}"
                        )
        except (TypeError, ValueError) as e:
            return False, f"Risk/reward parsing error: {e}"
        
        return True, ""

    def _check_trend_alignment(self, decision: dict) -> Tuple[bool, str]:
        """
        Prevent options strategies conflicting with market trend.
        This is the last line of defense after options_brain filters.
        """
        strategy = decision.get("strategy", "")
        regime = decision.get("regime", "")
        bias = decision.get("bias", "")

        if not strategy:
            return True, ""

        # Bearish options on bullish stock
        if strategy in ("LONG_PUT", "BEAR_PUT_SPREAD"):
            if regime in ("STRONG_BULL", "BULL"):
                return False, (
                    f"Trend conflict: {strategy} blocked — " f"stock in {regime} regime"
                )
            if bias in ("STRONG_BUY", "BUY"):
                return False, (
                    f"Trend conflict: {strategy} blocked — " f"bias is {bias}"
                )

        # Bullish options on bearish stock
        if strategy in ("LONG_CALL", "BULL_CALL_SPREAD"):
            if regime in ("STRONG_BEAR", "BEAR"):
                return False, (
                    f"Trend conflict: {strategy} blocked — " f"stock in {regime} regime"
                )
            if bias in ("STRONG_SELL", "SELL"):
                return False, (
                    f"Trend conflict: {strategy} blocked — " f"bias is {bias}"
                )

        # No options in choppy market
        if strategy and regime == "CHOPPY":
            return False, (f"Trend conflict: {strategy} blocked — CHOPPY regime")

        return True, ""

    def _check_daily_loss(self, portfolio: dict) -> Tuple[bool, str]:
        equity = portfolio.get("equity", 0)
        daily_pl = portfolio.get("daily_pl", 0)
        if self._daily_start_equity is None:
            self._daily_start_equity = equity - daily_pl
        if self._daily_start_equity and self._daily_start_equity > 0:
            pct = daily_pl / self._daily_start_equity
            if pct < -self.config.DAILY_LOSS_LIMIT:
                return False, (
                    f"Daily loss {pct:.2%} > limit "
                    f"{self.config.DAILY_LOSS_LIMIT:.2%} — halted"
                )
        return True, ""

    def _check_drawdown(self, portfolio: dict) -> Tuple[bool, str]:
        equity = portfolio.get("equity", 0)
        if self._peak_equity is None:
            self._peak_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            if dd > self.config.MAX_DRAWDOWN:
                return False, (
                    f"Drawdown {dd:.2%} > max "
                    f"{self.config.MAX_DRAWDOWN:.2%} — suspended"
                )
        return True, ""

    def _check_daily_count(self) -> Tuple[bool, str]:
        today = date.today()
        if self._daily_trades.get(today, 0) >= self.config.MAX_TRADES_PER_DAY:
            return False, (f"Daily limit ({self.config.MAX_TRADES_PER_DAY}) reached")
        return True, ""

    def _check_consecutive_losses(self) -> Tuple[bool, str]:
        if self._consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            return False, (
                f"{self._consecutive_losses} consecutive losses — " f"cooling down"
            )
        return True, ""

    def _check_quantity(self, decision: dict) -> Tuple[bool, str]:
        qty = decision.get("quantity", 0)
        if not isinstance(qty, (int, float)) or qty <= 0:
            return False, f"Invalid quantity: {qty}"
        return True, ""

    def _check_duplicate(self, decision: dict, portfolio: dict) -> Tuple[bool, str]:
        ticker = (decision.get("ticker") or "").upper()
        if decision.get("action") == "buy":
            for pos in portfolio.get("positions", []):
                if pos["symbol"] == ticker and pos["qty"] > 0:
                    return False, f"Already long {ticker}"
        return True, ""

    # ─────────────────────────────────────────────────────────────
    # TRAILING STOPS
    # ─────────────────────────────────────────────────────────────

    def update_trailing_stops(
        self, prices: Dict[str, float], atr_map: Dict[str, float]
    ) -> List[str]:
        """
        Fallback trailing stop check for positions not tracked
        by live_feed. Runs every cycle as a safety net.
        """
        stopped = []
        for ticker, price in prices.items():
            atr = atr_map.get(ticker, 0)
            if atr <= 0:
                continue
            if ticker not in self._peak_prices:
                self._peak_prices[ticker] = price
            self._peak_prices[ticker] = max(self._peak_prices[ticker], price)
            trail = self._peak_prices[ticker] - atr * 2.0
            self._trailing_stops[ticker] = trail
            if price <= trail:
                log.info(
                    "🛑 Cycle trailing stop: %s $%.2f <= $%.2f",
                    ticker,
                    price,
                    trail,
                )
                stopped.append(ticker)
        return stopped

    # ─────────────────────────────────────────────────────────────
    # POSITION SIZING
    # ─────────────────────────────────────────────────────────────

    def max_shares(
        self, portfolio: dict, price: float, atr: Optional[float] = None
    ) -> int:
        equity = portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        risk_pct = self.config.RISK_PER_TRADE
        if self._consecutive_losses >= 2:
            risk_pct *= 0.5
        if atr and atr > 0 and price > 0:
            qty = int(equity * risk_pct / (atr * 1.5))
        else:
            qty = (
                int(min(equity * self.config.MAX_POSITION_PCT, cash) / price)
                if price > 0
                else 0
            )
        cap = int(cash * 0.20 / price) if price > 0 else 0
        return max(1, min(qty, cap))

    # ─────────────────────────────────────────────────────────────
    # RESULT TRACKING
    # ─────────────────────────────────────────────────────────────

    def record_trade_result(self, profitable: bool):
        if profitable:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    def reset_daily(self):
        self._daily_start_equity = None

    def status(self) -> dict:
        today = date.today()
        return {
            "trades_today": self._daily_trades.get(today, 0),
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "peak_equity": self._peak_equity,
        }

    def _inc(self):
        today = date.today()
        self._daily_trades[today] = self._daily_trades.get(today, 0) + 1
