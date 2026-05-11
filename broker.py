"""
broker.py
=========
Alpaca stock order execution. Python 3.9 compatible.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import alpaca_trade_api as tradeapi

log = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL,
            api_version="v2"
        )
        mode = "PAPER" if "paper" in config.ALPACA_BASE_URL else "🔴 LIVE"
        log.info("📡 Alpaca broker connected | %s", mode)

    def is_market_open(self) -> bool:
        try:
            return self.api.get_clock().is_open
        except Exception as e:
            log.error("Clock check failed: %s", e)
            return False

    def get_portfolio(self) -> dict:
        try:
            acct = self.api.get_account()
            positions = self.api.list_positions()
            pos_list = []
            for p in positions:
                pos_list.append({
                    "symbol":           p.symbol,
                    "qty":              float(p.qty),
                    "avg_entry":        float(p.avg_entry_price),
                    "current_price":    float(p.current_price),
                    "market_value":     float(p.market_value),
                    "unrealized_pl":    float(p.unrealized_pl),
                    "unrealized_plpc":  float(p.unrealized_plpc),
                })
            return {
                "cash":          float(acct.cash),
                "equity":        float(acct.equity),
                "daily_pl":      float(acct.equity) - float(acct.last_equity),
                "buying_power":  float(acct.buying_power),
                "positions":     pos_list,
            }
        except Exception as e:
            log.error("Portfolio fetch failed: %s", e)
            return {"cash": 0, "equity": 0, "daily_pl": 0,
                    "buying_power": 0, "positions": []}

    def execute(self, decision: dict) -> Optional[dict]:
        action  = decision.get("action")
        ticker  = decision.get("ticker")
        qty     = int(decision.get("quantity", 0))
        stop    = decision.get("stop_loss")
        target  = decision.get("take_profit")
        if action == "hold" or not ticker or qty <= 0:
            return None

        if self.config.DRY_RUN:
            log.info(
                "[DRY RUN] Simulated stock order: %s %s x%d",
                action.upper(),
                ticker,
                qty,
            )
            return {
                "order_id":  f"DRYRUN-{ticker}-{action}",
                "symbol":    ticker,
                "side":      action,
                "qty":       float(qty),
                "status":    "dry_run",
                "submitted": str(datetime.utcnow()),
            }

        try:
            if action == "buy" and stop and target:
                order = self.api.submit_order(
                    symbol        = ticker,
                    qty           = qty,
                    side          = "buy",
                    type          = "market",
                    time_in_force = "day",
                    order_class   = "bracket",
                    stop_loss     = {"stop_price":   round(stop,   2)},
                    take_profit   = {"limit_price":  round(target, 2)},
                )
            else:
                order = self.api.submit_order(
                    symbol        = ticker,
                    qty           = qty,
                    side          = action,
                    type          = "market",
                    time_in_force = "day",
                )
            return {
                "order_id":  order.id,
                "symbol":    order.symbol,
                "side":      order.side,
                "qty":       float(order.qty),
                "status":    order.status,
                "submitted": str(order.submitted_at),
            }
        except Exception as e:
            log.error("Order failed [%s %s x%d]: %s", action, ticker, qty, e)
            return None

    def close_position(self, ticker: str):
        if self.config.DRY_RUN:
            log.info("[DRY RUN] Simulated stock close: %s", ticker)
            return
        try:
            self.api.close_position(ticker)
            log.info("🔒 Closed stock position: %s", ticker)
        except Exception as e:
            log.error("Close position failed %s: %s", ticker, e)

    def cancel_all(self):
        if self.config.DRY_RUN:
            log.info("[DRY RUN] Simulated cancel all orders.")
            return
        try:
            self.api.cancel_all_orders()
            log.info("🚫 All open orders cancelled.")
        except Exception as e:
            log.error("Cancel all failed: %s", e)
