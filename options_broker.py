"""
options_broker.py
=================
Options order execution, position tracking, auto-exit management.
Python 3.9 compatible.
"""

import re
import logging
import requests
from typing import Optional, Dict, List

import alpaca_trade_api as tradeapi

log = logging.getLogger(__name__)


class OptionsBroker:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url=config.ALPACA_BASE_URL,
            api_version="v2",
        )
        self.headers = {
            "APCA-API-KEY-ID": config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
            "accept": "application/json",
        }
        self._tracked: Dict[str, dict] = {}
        log.info("📊 Options broker ready")

    # ─────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────

    def execute_options_trade(self, decision: dict) -> Optional[dict]:
        strategy = decision.get("strategy", "")
        action = decision.get("action", "hold")
        if action == "hold":
            return None
        if strategy in ("LONG_CALL", "LONG_PUT"):
            return self._buy_single(decision)
        if strategy in ("BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"):
            return self._buy_spread(decision)
        log.error("Unknown strategy: %s", strategy)
        return None

    # ─────────────────────────────────────────────────────────────
    # SINGLE LEG ORDER
    # ─────────────────────────────────────────────────────────────

    def _buy_single(self, decision: dict) -> Optional[dict]:
        symbol = decision.get("option_symbol")
        contracts = int(decision.get("contracts", 1))
        limit = decision.get("limit_price")

        if not symbol:
            log.error("No option_symbol in decision")
            return None

        if self.config.DRY_RUN:
            log.info(
                "[DRY RUN] Simulated options single order: BUY %d x %s @ %s",
                contracts,
                symbol,
                f"${limit:.2f}" if limit else "market",
            )
            return {
                "order_id": f"DRYRUN-{symbol}",
                "symbol": symbol,
                "strategy": decision.get("strategy"),
                "side": "buy",
                "contracts": contracts,
                "limit": limit,
                "status": "dry_run",
                "max_loss": decision.get("max_loss", 0),
            }

        # Validate symbol exists on Alpaca
        symbol = self._validate_symbol(symbol)
        if not symbol:
            return None

        # Fit contracts to available buying power
        contracts = self._fit_to_buying_power(contracts, limit)
        if contracts == 0:
            log.warning("Skipping — insufficient buying power")
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=contracts,
                side="buy",
                type="limit" if limit else "market",
                time_in_force="day",
                limit_price=round(float(limit), 2) if limit else None,
            )
            result = {
                "order_id": order.id,
                "symbol": symbol,
                "strategy": decision.get("strategy"),
                "side": "buy",
                "contracts": contracts,
                "limit": limit,
                "status": order.status,
                "max_loss": decision.get("max_loss", 0),
            }
            self._tracked[symbol] = {
                **result,
                "entry_price": limit or 0,
                "stop_loss_pct": float(decision.get("stop_loss_pct", 50)),
                "ticker": decision.get("ticker"),
            }
            log.info(
                "✅ Options order: BUY %d x %s @ $%.2f",
                contracts,
                symbol,
                limit or 0,
            )
            return result
        except Exception as e:
            log.error("Order failed [%s x%d]: %s", symbol, contracts, e)
            return None

    # ─────────────────────────────────────────────────────────────
    # SPREAD ORDER
    # ─────────────────────────────────────────────────────────────

    def _buy_spread(self, decision: dict) -> Optional[dict]:
        long_sym = decision.get("long_symbol")
        short_sym = decision.get("short_symbol")
        contracts = int(decision.get("contracts", 1))
        net_debit = float(decision.get("net_debit", 0))
        long_mid = float(decision.get("long_mid", net_debit * 1.5))
        short_mid = float(decision.get("short_mid", net_debit * 0.5))

        if not long_sym or not short_sym:
            log.error("Spread missing leg symbols")
            return None

        if self.config.DRY_RUN:
            log.info(
                "[DRY RUN] Simulated spread order: %s | %s/%s | %d contracts | debit $%.2f",
                decision.get("strategy"),
                long_sym,
                short_sym,
                contracts,
                net_debit,
            )
            return {
                "order_id": f"DRYRUN-{long_sym}-{short_sym}",
                "strategy": decision.get("strategy"),
                "long_symbol": long_sym,
                "short_symbol": short_sym,
                "contracts": contracts,
                "net_debit": net_debit,
                "max_loss": decision.get("max_loss", 0),
                "max_profit": decision.get("max_profit", 0),
                "status": "dry_run",
            }

        try:
            long_order = self.api.submit_order(
                symbol=long_sym,
                qty=contracts,
                side="buy",
                type="limit",
                time_in_force="day",
                limit_price=round(long_mid, 2),
            )
            self.api.submit_order(
                symbol=short_sym,
                qty=contracts,
                side="sell",
                type="limit",
                time_in_force="day",
                limit_price=round(short_mid, 2),
            )
            result = {
                "order_id": long_order.id,
                "strategy": decision.get("strategy"),
                "long_symbol": long_sym,
                "short_symbol": short_sym,
                "contracts": contracts,
                "net_debit": net_debit,
                "max_loss": decision.get("max_loss", 0),
                "max_profit": decision.get("max_profit", 0),
                "status": "submitted",
            }
            log.info(
                "✅ Spread: %s | %s/%s | %d contracts | debit $%.2f",
                decision.get("strategy"),
                long_sym,
                short_sym,
                contracts,
                net_debit,
            )
            return result
        except Exception as e:
            log.error("Spread order failed: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────
    # SYMBOL VALIDATION
    # ─────────────────────────────────────────────────────────────

    def _validate_symbol(self, symbol: str) -> Optional[str]:
        """Verify symbol exists on Alpaca. Find nearest if not."""
        try:
            url = f"{self.config.ALPACA_BASE_URL}/v2/options/contracts/{symbol}"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                log.info("   ✅ Symbol validated: %s", symbol)
                return symbol
            log.warning("   ⚠️  Symbol not found: %s — searching nearest...", symbol)
            return self._find_nearest_contract(symbol)
        except Exception as e:
            log.error("Symbol validation failed: %s", e)
            return None

    def _find_nearest_contract(self, symbol: str) -> Optional[str]:
        """Parse symbol and find the nearest real Alpaca contract."""
        try:
            match = re.match(r"([A-Z]+)(\d{6})([CP])(\d{8})", symbol)
            if not match:
                log.error("Cannot parse symbol: %s", symbol)
                return None

            ticker = match.group(1)
            exp_str = match.group(2)
            opt_type = "call" if match.group(3) == "C" else "put"
            strike = int(match.group(4)) / 1000
            expiry = f"20{exp_str[:2]}-{exp_str[2:4]}-{exp_str[4:6]}"

            log.info(
                "   Searching: %s %s %s strike=$%.0f",
                ticker,
                expiry,
                opt_type,
                strike,
            )

            url = f"{self.config.ALPACA_BASE_URL}/v2/options/contracts"
            params = {
                "underlying_symbols": ticker,
                "expiration_date": expiry,
                "type": opt_type,
                "strike_price_gte": strike * 0.95,
                "strike_price_lte": strike * 1.05,
                "limit": 10,
            }
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)

            if resp.status_code != 200:
                log.error("Contract search failed: %s", resp.text[:200])
                return None

            contracts = resp.json().get("option_contracts", [])
            if not contracts:
                log.warning(
                    "   No contracts near $%.0f for %s %s",
                    strike,
                    ticker,
                    expiry,
                )
                return None

            best = min(
                contracts,
                key=lambda c: abs(float(c["strike_price"]) - strike),
            )
            real_symbol = best["symbol"]
            log.info(
                "   ✅ Nearest found: %s (strike=$%s)",
                real_symbol,
                best["strike_price"],
            )
            return real_symbol

        except Exception as e:
            log.error("Nearest contract search failed: %s", e)
            return None

    # ─────────────────────────────────────────────────────────────
    # BUYING POWER CHECK
    # ─────────────────────────────────────────────────────────────

    def _fit_to_buying_power(self, contracts: int, limit_price: Optional[float]) -> int:
        """Scale down contracts to fit available buying power."""
        try:
            acct = self.api.get_account()
            bp = float(
                getattr(acct, "options_buying_power", None)
                or getattr(acct, "regt_buying_power", None)
                or acct.buying_power
            )
            if not limit_price or limit_price <= 0:
                return contracts
            max_c = int(bp / (limit_price * 100))
            max_c = max(0, min(max_c, contracts))
            if max_c < contracts:
                log.info(
                    "⚠️  Contracts reduced %d→%d (BP=$%.0f)",
                    contracts,
                    max_c,
                    bp,
                )
            return max_c
        except Exception as e:
            log.warning("BP check failed: %s — defaulting to 1", e)
            return 1

    # ─────────────────────────────────────────────────────────────
    # POSITION MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def get_options_positions(self) -> List[dict]:
        try:
            positions = self.api.list_positions()
            opts = []
            for p in positions:
                if len(p.symbol) > 10:
                    opts.append(
                        {
                            "symbol": p.symbol,
                            "qty": float(p.qty),
                            "avg_entry": float(p.avg_entry_price),
                            "current_price": float(p.current_price),
                            "market_value": float(p.market_value),
                            "unrealized_pl": float(p.unrealized_pl),
                            "pl_pct": float(p.unrealized_plpc) * 100,
                        }
                    )
            return opts
        except Exception as e:
            log.error("Options positions fetch failed: %s", e)
            return []

    def check_exits(self) -> List[dict]:
        """Return positions that hit stop-loss or profit target."""
        to_close = []
        for pos in self.get_options_positions():
            sym = pos["symbol"]
            pl_pct = pos["pl_pct"]
            tracked = self._tracked.get(sym, {})
            stop_pct = tracked.get("stop_loss_pct", 50)
            if pl_pct <= -stop_pct:
                to_close.append({"symbol": sym, "reason": f"stop_loss ({pl_pct:.1f}%)"})
            elif pl_pct >= 80:
                to_close.append(
                    {"symbol": sym, "reason": f"profit_target (+{pl_pct:.1f}%)"}
                )
        return to_close

    def close_option(self, symbol: str) -> bool:
        if self.config.DRY_RUN:
            log.info("[DRY RUN] Simulated option close: %s", symbol)
            self._tracked.pop(symbol, None)
            return True
        try:
            pos = self.api.get_position(symbol)
            self.api.submit_order(
                symbol=symbol,
                qty=abs(float(pos.qty)),
                side="sell",
                type="market",
                time_in_force="day",
            )
            self._tracked.pop(symbol, None)
            log.info("✅ Closed option: %s", symbol)
            return True
        except Exception as e:
            log.error("Close option %s failed: %s", symbol, e)
            return False

    def close_all_options(self):
        if self.config.DRY_RUN:
            log.info("[DRY RUN] Simulated close all options.")
            self._tracked.clear()
            return
        for pos in self.get_options_positions():
            self.close_option(pos["symbol"])
        log.info("✅ All options closed.")

    def options_summary(self) -> dict:
        positions = self.get_options_positions()
        return {
            "open_positions": len(positions),
            "total_value": round(sum(p["market_value"] for p in positions), 2),
            "total_pl": round(sum(p["unrealized_pl"] for p in positions), 2),
            "positions": positions,
        }
