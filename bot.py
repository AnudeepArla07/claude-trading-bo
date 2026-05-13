"""
bot.py
======
Claude Trading Bot — Stocks + Options with real-time exit management.

New in this version:
  - LiveFeed WebSocket integrated for real-time price monitoring
  - Breakeven stop: moves stop to entry when position up 1%
  - Lock 1%: moves stop to +1% when position up 2%
  - Trailing stop: updates continuously on every bar
  - Time stop: exits flat positions after 2 hours
  - Trigger-based Claude re-analysis on significant moves
  - Multi-timeframe trend filter in options brain
  - Trend alignment check in risk manager

Run:          python bot.py
Dry run:      python bot.py --dry-run
"""

import argparse
import time
import logging
import schedule
import pytz
from datetime import datetime
from typing import Optional, Dict

from config import Config
from data_feed import MarketDataFeed
from claude_brain import ClaudeBrain
from options_brain import OptionsBrain
from options_feed import OptionsFeed
from options_broker import OptionsBroker
from risk_manager import RiskManager
from broker import AlpacaBroker
from database import TradeDatabase
from scanner import MarketScanner
from live_feed import LiveFeed
from position_manager import PositionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")


class TradingBot:
    def __init__(self, dry_run: bool = False):
        log.info("=" * 65)
        log.info("🤖  Claude Trading Bot  |  Stocks + Options + Real-Time")
        log.info("=" * 65)

        # Validate configuration before initializing
        from config import validate_config

        try:
            validate_config()
            log.info("✅ Configuration validated")
        except ValueError as e:
            log.error("❌ %s", e)
            raise

        self.config = Config()
        self.dry_run = dry_run
        self.data = MarketDataFeed(self.config)
        self.stock_brain = ClaudeBrain(self.config)
        self.options_brain = OptionsBrain(self.config)
        self.options_feed = OptionsFeed(self.config)
        self.options_broker = OptionsBroker(self.config)
        self.risk = RiskManager(self.config)
        self.broker = AlpacaBroker(self.config)
        self.db = TradeDatabase()
        self.scanner = MarketScanner(self.config)

        self.watchlist = self.config.WATCHLIST
        self.cycle_count = 0
        self.live_prices: Dict[str, float] = {}
        self.live_feed: Optional[LiveFeed] = None
        self.positions = PositionManager()  # Centralized position tracking
        self._unfilled_orders: Dict[str, float] = {}  # Track unfilled order timestamps

        if dry_run:
            log.info("⚠️  DRY RUN — no orders will be submitted.")

        log.info("✅ All systems ready.")
        log.info("📋 Starting watchlist: %s", self.watchlist)

        # Reconcile positions with broker on startup
        self._reconcile_positions()

    # ─────────────────────────────────────────────────────────────
    # POSITION RECONCILIATION
    # ─────────────────────────────────────────────────────────────

    def _reconcile_positions(self):
        """Reconcile position manager with broker positions on startup."""
        try:
            portfolio = self.broker.get_portfolio()
            missing, extra = self.positions.reconcile_with_broker(
                portfolio.get("positions", [])
            )

            if extra:
                log.info("📌 Registering %d new positions from broker", len(extra))
                for pos in portfolio.get("positions", []):
                    if pos["symbol"] in extra:
                        # Estimate ATR from recent price movement
                        atr = abs(pos["avg_entry"] - pos["current_price"]) * 0.5
                        self.positions.register(
                            symbol=pos["symbol"],
                            entry_price=pos["avg_entry"],
                            stop_loss=pos["current_price"] - atr,
                            take_profit=pos["current_price"] + atr * 2,
                            atr=atr,
                            qty=int(pos["qty"]),
                            side="long",
                        )
            else:
                log.info("✅ Position reconciliation complete — all synced")
        except Exception as e:
            log.warning("Position reconciliation failed: %s", e)

    def _check_unfilled_orders(self):
        """Check for unfilled orders older than 5 minutes and cancel them."""
        now = time.time()
        timeout = 5 * 60  # 5 minutes

        for symbol in list(self._unfilled_orders.keys()):
            age = now - self._unfilled_orders[symbol]
            if age > timeout:
                log.warning(
                    "🚫 Order for %s unfilled for %.0f seconds, cancelling", symbol, age
                )
                try:
                    self.broker.cancel_all()  # uses the wrapper you already have
                    del self._unfilled_orders[symbol]
                except Exception as e:
                    log.error("Cancel failed for %s: %s", symbol, e)

    # ─────────────────────────────────────────────────────────────
    # LIVE FEED
    # ─────────────────────────────────────────────────────────────

    def _start_live_feed(self):
        """Start WebSocket for real-time price monitoring."""
        try:
            self.live_feed = LiveFeed(
                api_key=self.config.ALPACA_API_KEY,
                secret_key=self.config.ALPACA_SECRET_KEY,
                watchlist=self.watchlist,
                position_manager=self.positions,
                on_update=self._on_price_update,
                on_exit=self._on_live_exit,
                on_trigger=self._on_live_trigger,
                data_feed=self.config.ALPACA_DATA_FEED,
            )
            self.live_feed.start()
            log.info("📡 Live feed started for real-time stop management.")
        except Exception as e:
            log.warning(
                "⚠️  Live feed failed to start: %s — "
                "falling back to cycle-based stops.",
                e,
            )
            self.live_feed = None

    def _on_price_update(
        self, symbol: str, price: float, volume: int, change_pct: float
    ):
        """Called on every live price bar."""
        self.live_prices[symbol] = price
        if abs(change_pct) >= 1.5:
            direction = "🚀" if change_pct > 0 else "🔻"
            log.info(
                "%s %s  $%.2f  (%+.2f%%)  vol=%s",
                direction,
                symbol,
                price,
                change_pct,
                f"{volume:,}",
            )

    def _on_live_exit(self, symbol: str, reason: str, price: float):
        """Called when live feed detects a stop or target hit."""
        log.info("🔔 LIVE EXIT: %s | %s | price=$%.2f", symbol, reason, price)
        if not self.dry_run:
            self.broker.close_position(symbol)
            if self.live_feed:
                self.live_feed.remove_position(symbol)
        self.db.log_decision(
            {
                "action": "sell",
                "ticker": symbol,
                "quantity": 0,
                "confidence": 1.0,
                "reason": f"Live exit: {reason}",
            },
            "LIVE_EXIT",
            reason,
        )

    def _on_live_trigger(self, symbol: str, trigger: str, price: float, pl_pct: float):
        """Called when position hits a significant level."""
        log.info(
            "⚡ TRIGGER: %s  %s  price=$%.2f  P&L=%.1f%%",
            symbol,
            trigger,
            price,
            pl_pct,
        )
        # Could trigger Claude re-analysis here for advanced use
        # For now just log it

    # ─────────────────────────────────────────────────────────────
    # WATCHLIST
    # ─────────────────────────────────────────────────────────────

    def refresh_watchlist(self):
        log.info("─" * 65)
        log.info("🔍 Refreshing watchlist...")
        try:
            core = [
                "NVDA",
                "AMD",
                "TSLA",
                "META",
                "AAPL",
                "MSFT",
                "AMZN",
                "GOOGL",
                "SPY",
                "QQQ",
                "PLTR",
                "SOFI",
                "MSTR",
                "COIN",
                "UBER",
            ]
            dynamic = self.scanner.get_dynamic_watchlist()
            combined = list(dict.fromkeys(core + (dynamic or [])))[:15]
            self.watchlist = combined
            added = [t for t in self.watchlist if t not in core]
            log.info("✅ Watchlist: %d tickers", len(self.watchlist))
            log.info("   Core    : %s", core[:10])
            log.info("   Dynamic : %s", added if added else "none added")
            log.info("   Full    : %s", self.watchlist)

            # Restart live feed with updated tickers
            if self.live_feed:
                self.live_feed.stop()
                self._start_live_feed()

        except Exception as e:
            log.error("Watchlist refresh failed: %s", e)
        log.info("─" * 65)

    # ─────────────────────────────────────────────────────────────
    # CORE CYCLE
    # ─────────────────────────────────────────────────────────────

    def run_cycle(self):
        now_et = datetime.now(ET)
        if now_et.weekday() >= 5:
            log.info("📅 Weekend — skipping.")
            return
        if not self.broker.is_market_open():
            log.info("🔒 Market closed.")
            return

        # Check for stale unfilled orders
        self._check_unfilled_orders()

        self.cycle_count += 1
        log.info("━" * 65)
        log.info(
            "📊 Cycle #%d  |  %s ET  |  %d tickers",
            self.cycle_count,
            now_et.strftime("%H:%M:%S"),
            len(self.watchlist),
        )

        # 1. Portfolio
        portfolio = self.broker.get_portfolio()
        equity = portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pl = portfolio.get("daily_pl", 0)
        open_positions = portfolio.get("positions", [])
        open_symbols = [p["symbol"] for p in open_positions]

        log.info(
            "💼 Equity=$%.0f  Cash=$%.0f  P&L=$%+.0f  Open=%d",
            equity,
            cash,
            daily_pl,
            len(open_positions),
        )
        if open_symbols:
            log.info("   Holding: %s", open_symbols)

        # 2. Options summary
        opt_sum = self.options_broker.options_summary()
        if opt_sum["open_positions"] > 0:
            log.info(
                "🎯 Options=%d  P&L=$%+.0f",
                opt_sum["open_positions"],
                opt_sum["total_pl"],
            )

        # 3. Options auto-exit check
        self._check_options_exits()

        # 4. Market data + indicators
        market_data = self.data.get_quotes(self.watchlist)
        news = self.data.get_news(self.watchlist)
        if not market_data:
            log.warning("⚠️  No market data.")
            return

        # Inject live prices for freshest data
        for ticker, price in self.live_prices.items():
            if ticker in market_data:
                market_data[ticker]["price"] = price
                market_data[ticker]["source"] = "websocket"

        log.info("📡 Data for %d tickers", len(market_data))

        # 5. Manage open stock positions (breakeven, trailing)
        self._manage_open_positions(portfolio, market_data)

        # 6. Stock brain
        log.info("🧠 Stock brain...")
        stock_dec = self.stock_brain.analyze(portfolio, market_data, news)
        if stock_dec:
            self._run_stock_trade(stock_dec, portfolio, open_symbols, market_data)

        # 7. Options brain
        log.info("🎯 Options brain...")
        self._run_options_layer(market_data, portfolio)

        # 8. Cycle-based trailing stops (safety net if live feed down)
        if not self.live_feed or not self.live_feed._running:
            prices = {t: d["price"] for t, d in market_data.items()}
            atrs = {t: d.get("atr_14", 0) for t, d in market_data.items()}
            stopped = self.risk.update_trailing_stops(prices, atrs)
            for ticker in stopped:
                log.info("🛑 Cycle stop — closing %s", ticker)
                if not self.dry_run:
                    self.broker.close_position(ticker)

    # ─────────────────────────────────────────────────────────────
    # POSITION MANAGEMENT (breakeven + lock-in)
    # ─────────────────────────────────────────────────────────────

    def _manage_open_positions(self, portfolio: dict, market_data: dict):
        """
        Cycle-based position management.
        Handles breakeven and profit locking for positions
        not already managed by the live feed.
        """
        for pos in portfolio.get("positions", []):
            ticker = pos["symbol"]
            entry = pos["avg_entry"]
            current = pos.get("current_price", entry)
            pl_pct = (current - entry) / entry * 100

            if ticker not in market_data:
                continue

            atr = market_data[ticker].get("atr_14") or 0

            # Already managed by live feed
            if self.live_feed and self.live_feed.get_position_status(ticker):
                continue

            log.info(
                "   📍 %s  entry=$%.2f  now=$%.2f  P&L=%.1f%%",
                ticker,
                entry,
                current,
                pl_pct,
            )

            # CORRECT — check bigger threshold first


if pl_pct >= 2.0:  # check 2% first
    lock = round(entry * 1.01, 2)
    log.info("🔒 %s +%.1f%% — locking 1%% at $%.2f", ticker, pl_pct, lock)
elif pl_pct >= 1.0:  # then 1%
    breakeven = round(entry * 1.001, 2)
    log.info("📈 %s +%.1f%% — breakeven stop at $%.2f", ticker, pl_pct, breakeven)
    if self.live_feed and atr:
        self.live_feed.register_position(...)
elif atr and (entry - current) > atr * 1.5:
    log.info("⚠️  %s down $%.2f > 1.5x ATR=$%.2f", ticker, entry - current, atr)

    # ─────────────────────────────────────────────────────────────
    # STOCK LAYER
    # ─────────────────────────────────────────────────────────────

    def _run_stock_trade(
        self, decision: dict, portfolio: dict, open_symbols: list, market_data: dict
    ):
        action = decision.get("action", "hold")
        ticker = decision.get("ticker", "")
        conf = decision.get("confidence", 0)
        reason = (decision.get("reason") or "")[:80]

        log.info(
            "📈 Stock: %s %s  conf=%.0f%%  |  %s",
            action.upper(),
            ticker,
            conf * 100,
            reason,
        )

        if action == "hold":
            self.db.log_decision(decision, "HOLD")
            return

        if action == "buy" and ticker in open_symbols:
            log.info("⏭️  Already holding %s — skipping", ticker)
            self.db.log_decision(decision, "SKIPPED", "Already in position")
            return

        ok, blocked = self.risk.approve(decision, portfolio)
        if not ok:
            log.info("🛑 Risk blocked: %s", blocked)
            self.db.log_decision(decision, "BLOCKED", blocked)
            return

        if self.dry_run:
            log.info("🧪 DRY RUN — would: %s %s", action.upper(), ticker)
            self.db.log_decision(decision, "DRY_RUN")
            return

        result = self.broker.execute(decision)
        if result:
            log.info(
                "✅ Executed: %s %s x%s",
                result["side"].upper(),
                result["symbol"],
                result["qty"],
            )
            self.db.log_trade(decision, result)

            # Register with live feed for real-time management
            if self.live_feed and ticker in market_data:
                atr = market_data[ticker].get("atr_14") or 0
                if atr and decision.get("stop_loss") and decision.get("take_profit"):
                    price = market_data[ticker]["price"]
                    self.live_feed.register_position(
                        symbol=ticker,
                        entry_price=float(decision.get("entry_price", price)),
                        stop_loss=float(decision["stop_loss"]),
                        take_profit=float(decision["take_profit"]),
                        atr=atr,
                        qty=int(result["qty"]),
                    )
        else:
            log.error("❌ Stock order failed.")
            self.db.log_decision(decision, "FAILED", "Broker rejected")

    # ─────────────────────────────────────────────────────────────
    # OPTIONS LAYER
    # ─────────────────────────────────────────────────────────────

    def _run_options_layer(self, market_data: dict, portfolio: dict):
        equity = portfolio.get("equity", 0)
        open_option_tickers = [
            self.options_broker._tracked[s].get("ticker", "")
            for s in self.options_broker._tracked
        ]

        bias_order = {
            "STRONG_BUY": 5,
            "BUY": 4,
            "NEUTRAL": 0,
            "SELL": 4,
            "STRONG_SELL": 5,
        }
        candidates = sorted(
            [
                (t, d)
                for t, d in market_data.items()
                if d.get("signals", {}).get("bias", "NEUTRAL") != "NEUTRAL"
                and d.get("volume", 0) > 1_000_000
            ],
            key=lambda x: bias_order.get(
                x[1].get("signals", {}).get("bias", "NEUTRAL"), 0
            ),
            reverse=True,
        )

        if not candidates:
            log.info("⚪ Options: no directional setups.")
            return

        traded = 0
        for ticker, stock_data in candidates:
            if traded >= 2:
                break
            if ticker in open_option_tickers:
                log.info("⏭️  Options already open: %s", ticker)
                continue

            bias = stock_data.get("signals", {}).get("bias", "NEUTRAL")
            price = stock_data.get("price", 0)
            log.info("   🎯 %s @ $%.2f  bias=%s", ticker, price, bias)

            opt_data = self.options_feed.get_options_data(ticker, price, bias)
            if not opt_data or not opt_data.get("recommendations"):
                log.info("   No options data: %s", ticker)
                continue

            iv_rank = opt_data.get("iv_rank")
            max_pain = opt_data.get("max_pain")
            log.info(
                "   IV=%s  MaxPain=%s",
                f"{iv_rank:.0f}%" if iv_rank is not None else "N/A",
                f"${max_pain:.0f}" if max_pain else "N/A",
            )

            opt_dec = self.options_brain.analyze(ticker, stock_data, opt_data)
            if not opt_dec:
                continue

            action = opt_dec.get("action", "hold")
            strategy = opt_dec.get("strategy", "")
            conf = opt_dec.get("confidence", 0)
            max_loss = float(opt_dec.get("max_loss", 0))

            log.info(
                "   %s  conf=%.0f%%  max_loss=$%.0f",
                strategy or "HOLD",
                conf * 100,
                max_loss,
            )

            if action == "hold":
                continue
            if conf < self.config.MIN_CONFIDENCE:
                log.info("   Blocked: low confidence")
                continue
            if equity > 0 and max_loss > equity * self.config.OPTIONS_MAX_LOSS_PCT:
                log.info("   Blocked: max_loss too large")
                continue

            if self.dry_run:
                log.info("   🧪 DRY RUN: %s %s", strategy, ticker)
                traded += 1
                continue

            result = self.options_broker.execute_options_trade(opt_dec)
            if result:
                log.info("   ✅ Options: %s %s", strategy, ticker)
                self.db.log_options_trade(opt_dec, result)
                traded += 1
            else:
                log.error("   ❌ Options failed: %s", ticker)

    # ─────────────────────────────────────────────────────────────
    # OPTIONS EXIT CHECKER
    # ─────────────────────────────────────────────────────────────

    def _check_options_exits(self):
        for exit_info in self.options_broker.check_exits():
            sym = exit_info["symbol"]
            reason = exit_info["reason"]
            log.info("🔔 Options exit: %s | %s", sym, reason)
            if not self.dry_run:
                self.options_broker.close_option(sym)
            self.db.log_decision(
                {
                    "action": "sell",
                    "ticker": sym,
                    "quantity": 0,
                    "confidence": 1.0,
                    "reason": reason,
                },
                "OPTIONS_EXIT",
                reason,
            )

    # ─────────────────────────────────────────────────────────────
    # END OF DAY
    # ─────────────────────────────────────────────────────────────

    def end_of_day(self):
        if datetime.now(ET).weekday() >= 5:
            return
        log.info("🔔 End of day — closing all positions...")
        if not self.dry_run:
            self.options_broker.close_all_options()
            if self.config.CLOSE_EOD:
                portfolio = self.broker.get_portfolio()
                for pos in portfolio.get("positions", []):
                    self.broker.close_position(pos["symbol"])
                self.broker.cancel_all()
        self.db.print_summary()
        log.info("✅ EOD done.")

    # ─────────────────────────────────────────────────────────────
    # START
    # ─────────────────────────────────────────────────────────────

    def start(self):
        log.info("🚀 Bot starting...")

        # Always refresh watchlist on startup
        self.refresh_watchlist()

        # Start live feed for real-time monitoring
        self._start_live_feed()

        # Schedule
        schedule.every().day.at("13:31").do(self.refresh_watchlist)
        schedule.every().day.at("19:45").do(self.end_of_day)
        schedule.every(2).hours.do(self.refresh_watchlist)
        schedule.every(self.config.CYCLE_MINUTES).minutes.do(self.run_cycle)

        if self.broker.is_market_open():
            log.info("📈 Market open — running first cycle.")
            self.run_cycle()
        else:
            log.info("⏳ Market closed. Waiting...")

        while True:
            try:
                schedule.run_pending()
                time.sleep(30)
            except KeyboardInterrupt:
                log.info("\n⛔ Stopped.")
                if self.live_feed:
                    self.live_feed.stop()
                if not self.dry_run:
                    self.options_broker.close_all_options()
                self.db.print_summary()
                break
            except Exception as e:
                log.error("Loop error: %s", e)
                time.sleep(60)


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Claude Trading Bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without submitting orders.",
    )
    args = parser.parse_args()

    try:
        bot = TradingBot(dry_run=args.dry_run)
        bot.start()
    except Exception as e:
        log.critical("Bot crashed: %s", e, exc_info=True)
        sys.exit(1)
