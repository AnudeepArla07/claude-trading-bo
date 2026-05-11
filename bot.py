"""
bot.py
======
Claude Trading Bot — Stocks + Options dual-layer system.

Fixes in this version:
  - Position check: never buys the same stock twice
  - Options check: never opens a second options position on same ticker
  - Clean cycle logging

Flow each cycle:
  1. Dynamic watchlist scan (daily at 9:31 AM)
  2. Fetch quotes + compute 15 technical indicators per ticker
  3. Stock brain (Claude) → best stock trade
  4. Options brain (Claude) → best options trade on same signals
  5. Risk checks on both decisions
  6. Execute via Alpaca (paper or live)
  7. Auto-exit options hitting stop/target
  8. Trailing stops on stock positions
  9. Log everything to SQLite

Run: python bot.py
"""

import argparse
import sys
import time
import logging
import schedule
from datetime import datetime
from typing import Optional

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

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


class TradingBot:
    def __init__(self, dry_run: bool = False):
        log.info("=" * 65)
        log.info("🤖  Claude Trading Bot  |  Stocks + Options")
        log.info("=" * 65)

        self.config = Config()
        if dry_run:
            self.config.DRY_RUN = True
            log.info("⚠️  DRY RUN enabled — no live orders will be submitted.")
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

        log.info("✅ All systems ready.")
        log.info("📋 Watchlist: %s", self.watchlist)
        log.info("🔁 Cycle every %d min", self.config.CYCLE_MINUTES)

    # ─────────────────────────────────────────────────────────────
    # WATCHLIST
    # ─────────────────────────────────────────────────────────────

    def refresh_watchlist(self):
        log.info("🔍 Refreshing watchlist...")

        try:
            # Always keep these core tickers regardless of scanner
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

            # Scanner finds today's best movers
            dynamic = self.scanner.get_dynamic_watchlist()

            # Combine — core always included, dynamic adds top movers
            combined = list(dict.fromkeys(core + (dynamic or [])))[:15]
            self.watchlist = combined

            log.info("✅ Watchlist: %d tickers", len(self.watchlist))
            log.info("   Core    : %s", core)
            log.info("   Dynamic : %s", [t for t in self.watchlist if t not in core])

        except Exception as e:
            log.error("Watchlist refresh failed: %s — keeping previous", e)

    # ─────────────────────────────────────────────────────────────
    # CORE CYCLE
    # ─────────────────────────────────────────────────────────────

    def run_cycle(self):
        from datetime import datetime
        import pytz

        # Skip weekends
        et = pytz.timezone("America/New_York")
        now = datetime.now(et)
        if now.weekday() >= 5:
            log.info("📅 Weekend — market closed.")
            return

        if not self.broker.is_market_open():
            log.info("🔒 Market closed.")
            return

        self.cycle_count += 1
        log.info("━" * 65)
        log.info(
            "📊 Cycle #%d  |  %s",
            self.cycle_count,
            datetime.now().strftime("%H:%M:%S ET"),
        )

        # 1. Portfolio snapshot
        portfolio = self.broker.get_portfolio()
        equity = portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pl = portfolio.get("daily_pl", 0)
        open_positions = portfolio.get("positions", [])
        open_symbols = [p["symbol"] for p in open_positions]

        log.info(
            "💼 Equity=$%.0f  Cash=$%.0f  Day P&L=$%+.0f  Open=%d",
            equity,
            cash,
            daily_pl,
            len(open_positions),
        )

        if open_symbols:
            log.info("   Holding: %s", open_symbols)

        # 2. Options position summary
        opt_sum = self.options_broker.options_summary()
        if opt_sum["open_positions"] > 0:
            log.info(
                "🎯 Options open=%d  P&L=$%+.0f",
                opt_sum["open_positions"],
                opt_sum["total_pl"],
            )

        # 3. Auto-exit options hitting stop or target
        self._check_options_exits()

        # 4. Fetch market data + indicators
        market_data = self.data.get_quotes(self.watchlist)
        news = self.data.get_news(self.watchlist)
        if not market_data:
            log.warning("⚠️  No market data — skipping cycle.")
            return
        log.info("📡 Data for %d tickers", len(market_data))

        # 5. Stock brain
        log.info("🧠 Stock brain analyzing...")
        stock_dec = self.stock_brain.analyze(portfolio, market_data, news)
        if stock_dec:
            self._run_stock_trade(stock_dec, portfolio, open_symbols)

        # 6. Options brain
        log.info("🎯 Options brain analyzing...")
        self._run_options_layer(market_data, portfolio)

        # 7. Trailing stops on stock positions
        prices = {t: d["price"] for t, d in market_data.items()}
        atrs = {t: d.get("atr_14", 0) for t, d in market_data.items()}
        stopped = self.risk.update_trailing_stops(prices, atrs)
        for ticker in stopped:
            log.info("🛑 Trailing stop — closing %s", ticker)
            self.broker.close_position(ticker)

    # ─────────────────────────────────────────────────────────────
    # STOCK LAYER
    # ─────────────────────────────────────────────────────────────

    def _run_stock_trade(self, decision: dict, portfolio: dict, open_symbols: list):
        action = decision.get("action", "hold")
        ticker = decision.get("ticker", "")
        conf = decision.get("confidence", 0)
        reason = (decision.get("reason") or "")[:80]

        log.info(
            "📈 Stock decision: %s %s  conf=%.0f%%  |  %s",
            action.upper(),
            ticker,
            conf * 100,
            reason,
        )

        # Hold — nothing to do
        if action == "hold":
            self.db.log_decision(decision, "HOLD")
            return

        # ── Already holding this ticker — skip ────────────────
        if action == "buy" and ticker in open_symbols:
            log.info("⏭️  Already holding %s — skipping to avoid repeat buy", ticker)
            self.db.log_decision(decision, "SKIPPED", "Already in position")
            return

        # ── Risk check ─────────────────────────────────────────
        ok, reason_blocked = self.risk.approve(decision, portfolio)
        if not ok:
            log.info("🛑 Risk blocked: %s", reason_blocked)
            self.db.log_decision(decision, "BLOCKED", reason_blocked)
            return

        # ── Execute ────────────────────────────────────────────
        result = self.broker.execute(decision)
        if result:
            log.info(
                "✅ Stock executed: %s %s x%s",
                result["side"].upper(),
                result["symbol"],
                result["qty"],
            )
            self.db.log_trade(decision, result)
        else:
            log.error("❌ Stock order failed.")
            self.db.log_decision(decision, "FAILED", "Broker rejected")

    # ─────────────────────────────────────────────────────────────
    # OPTIONS LAYER
    # ─────────────────────────────────────────────────────────────

    def _run_options_layer(self, market_data: dict, portfolio: dict):
        equity = portfolio.get("equity", 0)

        # Tickers already tracked in open options positions
        open_option_tickers = [
            self.options_broker._tracked[s].get("ticker", "")
            for s in self.options_broker._tracked
        ]

        # Sort candidates by signal strength
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
            log.info("⚪ Options: no directional setups this cycle.")
            return

        traded = 0
        for ticker, stock_data in candidates:
            # Max 2 options trades per cycle
            if traded >= 2:
                break

            # ── Already have options on this ticker — skip ─────
            if ticker in open_option_tickers:
                log.info("⏭️  Already have options on %s — skipping", ticker)
                continue

            bias = stock_data.get("signals", {}).get("bias", "NEUTRAL")
            price = stock_data.get("price", 0)
            log.info("   🎯 %s @ $%.2f  bias=%s", ticker, price, bias)

            # Fetch options chain
            opt_data = self.options_feed.get_options_data(ticker, price, bias)
            if not opt_data or not opt_data.get("recommendations"):
                log.info("   No options data for %s", ticker)
                continue

            iv_rank = opt_data.get("iv_rank")
            max_pain = opt_data.get("max_pain")
            log.info(
                "   IV Rank=%s  MaxPain=$%s",
                f"{iv_rank:.0f}%" if iv_rank is not None else "N/A",
                f"{max_pain:.0f}" if max_pain else "N/A",
            )

            # Options brain decision
            opt_dec = self.options_brain.analyze(ticker, stock_data, opt_data)
            if not opt_dec:
                continue

            action = opt_dec.get("action", "hold")
            strategy = opt_dec.get("strategy", "")
            conf = opt_dec.get("confidence", 0)
            max_loss = float(opt_dec.get("max_loss", 0))

            log.info(
                "   Decision: %s  conf=%.0f%%  max_loss=$%.0f",
                strategy or "HOLD",
                conf * 100,
                max_loss,
            )

            if action == "hold":
                continue

            # Risk gates
            if conf < self.config.MIN_CONFIDENCE:
                log.info(
                    "   Blocked: conf %.0f%% < min %.0f%%",
                    conf * 100,
                    self.config.MIN_CONFIDENCE * 100,
                )
                continue

            if equity > 0 and max_loss > equity * self.config.OPTIONS_MAX_LOSS_PCT:
                log.info(
                    "   Blocked: max_loss $%.0f > %.0f%% of portfolio",
                    max_loss,
                    self.config.OPTIONS_MAX_LOSS_PCT * 100,
                )
                continue

            # Execute
            result = self.options_broker.execute_options_trade(opt_dec)
            if result:
                log.info("   ✅ Options placed: %s %s", strategy, ticker)
                self.db.log_options_trade(opt_dec, result)
                traded += 1
            else:
                log.error("   ❌ Options order failed: %s", ticker)

    # ─────────────────────────────────────────────────────────────
    # OPTIONS EXIT CHECKER
    # ─────────────────────────────────────────────────────────────

    def _check_options_exits(self):
        for exit_info in self.options_broker.check_exits():
            sym = exit_info["symbol"]
            reason = exit_info["reason"]
            log.info("🔔 Auto-exit: %s | %s", sym, reason)
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
        log.info("🔔 3:45 PM — end of day cleanup...")
        self.options_broker.close_all_options()
        if self.config.CLOSE_EOD:
            portfolio = self.broker.get_portfolio()
            for pos in portfolio.get("positions", []):
                self.broker.close_position(pos["symbol"])
            self.broker.cancel_all()
        self.db.print_summary()
        log.info("✅ End-of-day done.")

    # ─────────────────────────────────────────────────────────────
    # START
    # ─────────────────────────────────────────────────────────────

    def start(self):
        log.info("🚀 Bot running. Press Ctrl+C to stop.")

        # Times in UTC (9:31 AM ET = 13:31 UTC, 3:45 PM ET = 19:45 UTC)
        schedule.every().day.at("13:31").do(self.refresh_watchlist)  # 9:31 AM ET
        schedule.every().day.at("19:45").do(self.end_of_day)         # 3:45 PM ET
        schedule.every(self.config.CYCLE_MINUTES).minutes.do(self.run_cycle)

        if self.broker.is_market_open():
            self.refresh_watchlist()
            self.run_cycle()
        else:
            log.info("⏳ Market closed. Bot standing by...")

        while True:
            try:
                schedule.run_pending()
                time.sleep(30)
            except KeyboardInterrupt:
                log.info("\n⛔ Stopped by user.")
                self.options_broker.close_all_options()
                self.db.print_summary()
                break
            except Exception as e:
                log.error("Main loop error: %s", e)
                time.sleep(60)


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Start the Claude trading bot.")
    parser.add_argument("--dry-run", action="store_true", help="Run without submitting any live orders.")
    args = parser.parse_args()

    try:
        bot = TradingBot(dry_run=args.dry_run)
        bot.start()
    except Exception as e:
        log.critical("Bot crashed: %s", e, exc_info=True)
        sys.exit(1)
