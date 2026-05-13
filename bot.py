"""
bot.py
======
Claude Trading Bot — final unified version.

Architecture:
  1. Signal pre-filter (indicators.py via data_feed.py)
     Only tickers with 4+ aligned signals reach Claude
  2. Claude decision (claude_brain.py)
     Makes the final BUY/HOLD decision on pre-filtered setups
  3. Risk check (risk_manager.py)
     Hard rules that override Claude if needed
  4. Execution (broker.py)
     ATR-based bracket orders for immediate stop protection

Paper/Live toggle: edit ALPACA_BASE_URL in config.py

Run:      python bot.py
Dry run:  python bot.py --dry-run
"""

import argparse
import time
import logging
import schedule
import pytz
from datetime import datetime
from typing import Optional, Dict

from config import Config, validate_config
from data_feed import MarketDataFeed
from claude_brain import ClaudeBrain
from options_brain import OptionsBrain
from options_feed import OptionsFeed
from options_broker import OptionsBroker
from risk_manager import RiskManager
from broker import AlpacaBroker
from database import TradeDatabase
from scanner import MarketScanner

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
        log.info("🤖  Claude Trading Bot  |  Stocks + Options")
        log.info("=" * 65)

        # ── Step 1: Config first — everything depends on it ───
        self.config = Config()
        self.dry_run = dry_run

        # ── Step 2: Validate API keys ─────────────────────────
        try:
            validate_config()
            log.info("✅ Configuration validated")
        except ValueError as e:
            log.error("❌ %s", e)
            raise SystemExit(1)

        mode = (
            "🧪 DRY RUN"
            if dry_run
            else ("📄 PAPER" if "paper" in self.config.ALPACA_BASE_URL else "💰 LIVE")
        )
        log.info("💡 Mode: %s", mode)

        # ── Step 3: Initialize components ─────────────────────
        self.data = MarketDataFeed(self.config)
        self.stock_brain = ClaudeBrain(self.config)
        self.options_brain = OptionsBrain(self.config)
        self.options_feed = OptionsFeed(self.config)
        self.options_broker = OptionsBroker(self.config)
        self.risk = RiskManager(self.config)
        self.broker = AlpacaBroker(self.config)
        self.db = TradeDatabase()
        self.scanner = MarketScanner(self.config)

        self.watchlist: list = list(self.config.WATCHLIST)
        self.cycle_count: int = 0
        self.ticker_classifications: Dict[str, str] = {}
        self.live_prices: Dict[str, float] = {}

        log.info("✅ All components ready.")
        log.info("📋 Watchlist: %s", self.watchlist)

        # ── Step 4: Classify tickers ──────────────────────────
        try:
            log.info("🔍 Classifying watchlist...")
            self.ticker_classifications = self.data.classify_watchlist(
                self.config.WATCHLIST
            )
            log.info("✅ Classification: %s", self.ticker_classifications)
        except Exception as e:
            log.warning("⚠️  Classification failed: %s — defaulting to quality", e)
            self.ticker_classifications = {t: "quality" for t in self.config.WATCHLIST}

        # ── Step 5: Reconcile open positions with broker ──────
        self._reconcile_positions()

    # ─────────────────────────────────────────────────────────────
    # LIVE FEED — disabled (Python 3.14 + websockets conflict)
    # Use Alpaca bracket orders for instant stop protection instead
    # ─────────────────────────────────────────────────────────────

    def _start_live_feed(self):
        """
        Live feed disabled — websockets incompatible with Python 3.14.

        Protection layers without live feed:
          1. Alpaca bracket orders (instant exchange-level stops)
          2. _manage_open_positions() every cycle (breakeven, trailing)
          3. EOD close if CLOSE_EOD=True

        To re-enable: install Python 3.11 + websockets==10.4
        and replace this method with the live_feed.py integration.
        """
        log.info("📡 Live feed disabled (Python 3.14/websockets conflict).")
        log.info("   Protection: Alpaca bracket orders + 10-min cycle stops.")

    # ─────────────────────────────────────────────────────────────
    # RECONCILE
    # ─────────────────────────────────────────────────────────────

    def _reconcile_positions(self):
        """Sync internal state with actual broker positions on startup."""
        try:
            portfolio = self.broker.get_portfolio()
            positions = portfolio.get("positions", [])
            if positions:
                log.info("📋 Existing positions: %s", [p["symbol"] for p in positions])
            else:
                log.info("📋 No open positions.")
        except Exception as e:
            log.warning("Reconcile failed: %s", e)

    # ─────────────────────────────────────────────────────────────
    # WATCHLIST
    # ─────────────────────────────────────────────────────────────

    def refresh_watchlist(self):
        log.info("─" * 65)
        log.info("🔍 Refreshing watchlist...")
        try:
            core = list(self.config.WATCHLIST)
            dynamic = self.scanner.get_dynamic_watchlist()
            combined = [
                t
                for t in dict.fromkeys(core + (dynamic or []))
                if t not in self.config.NO_TRADE_TICKERS
            ][:15]
            self.watchlist = combined
            added = [t for t in self.watchlist if t not in core]
            log.info("✅ Watchlist: %d tickers", len(self.watchlist))
            log.info("   Core    : %s", core)
            log.info("   Dynamic : %s", added if added else "none")
            log.info("   Full    : %s", self.watchlist)
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

        self.cycle_count += 1
        log.info("━" * 65)
        log.info(
            "📊 Cycle #%d  |  %s ET  |  %d tickers",
            self.cycle_count,
            now_et.strftime("%H:%M:%S"),
            len(self.watchlist),
        )

        # 1. Portfolio snapshot
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

        # 3. Options auto-exit
        self._check_options_exits()

        # 4. Market data + indicators
        # Always include REGIME_TICKER (SPY) for market regime even if not in watchlist
        fetch_list = list(self.watchlist)
        if self.config.REGIME_TICKER not in fetch_list:
            fetch_list.append(self.config.REGIME_TICKER)
        market_data = self.data.get_quotes(fetch_list)
        news = self.data.get_news(self.watchlist)
        if not market_data:
            log.warning("⚠️  No market data — skipping cycle.")
            return

        # ── LEVER 5: Market regime filter ─────────────────────
        # Only trade when SPY is BULL or STRONG_BULL
        spy_data = market_data.get(self.config.REGIME_TICKER, {})
        spy_regime = spy_data.get("market_regime_num", 0)
        spy_str = spy_data.get("market_regime", "UNKNOWN")

        if spy_regime < self.config.SPY_BULL_MIN:
            log.info(
                "🔒 SPY regime=%s (%.0f) — sitting out this cycle", spy_str, spy_regime
            )
            # Still manage existing positions even in bear market
            self._manage_open_positions(portfolio, market_data)
            return

        # Tradeable data — strip regime-only and no-trade tickers before Claude/options
        tradeable_data = {
            t: d
            for t, d in market_data.items()
            if t not in self.config.NO_TRADE_TICKERS
        }
        log.info(
            "📡 SPY=%s  Data for %d tickers (%d tradeable)",
            spy_str,
            len(market_data),
            len(tradeable_data),
        )

        # 5. Manage open positions (breakeven, lock-in, trailing)
        self._manage_open_positions(portfolio, market_data)

        # 6. Stock brain (Claude)
        log.info(
            "🧠 Stock brain analyzing %d pre-filtered setups...",
            sum(
                1
                for d in tradeable_data.values()
                if d.get("signal_count", 0) >= self.config.MIN_SIGNALS_PRE_FILTER
            ),
        )

        stock_dec = self.stock_brain.analyze(portfolio, tradeable_data, news)
        if stock_dec:
            self._run_stock_trade(stock_dec, portfolio, open_symbols, market_data)

        # 7. Options brain
        log.info("🎯 Options brain analyzing...")
        self._run_options_layer(market_data, portfolio)

        # 8. Cycle-based trailing stops (fallback without live feed)
        prices = {t: d["price"] for t, d in market_data.items()}
        atrs = {t: d.get("atr_14", 0) for t, d in market_data.items()}
        stopped = self.risk.update_trailing_stops(prices, atrs)
        for ticker in stopped:
            log.info("🛑 Trailing stop hit — closing %s", ticker)
            if not self.dry_run:
                self.broker.close_position(ticker)

    # ─────────────────────────────────────────────────────────────
    # POSITION MANAGEMENT (breakeven + lock-in)
    # ─────────────────────────────────────────────────────────────

    def _manage_open_positions(self, portfolio: dict, market_data: dict):
        """
        Cycle-based stop management.
        Checks breakeven and profit locking for open positions.
        Aligned with backtest partial profit thresholds.
        """
        for pos in portfolio.get("positions", []):
            ticker = pos["symbol"]
            entry = pos.get("avg_entry", 0)
            current = pos.get("current_price", entry)

            if entry <= 0:
                continue

            pl_pct = (current - entry) / entry * 100
            atr = (market_data.get(ticker) or {}).get("atr_14") or 0
            t_type = self.ticker_classifications.get(ticker, "quality")
            params = self.config.ATR_PARAMS.get(t_type, {"stop": 3.5, "target": 10.0})
            partial = (
                self.config.PARTIAL_PROFIT_MOM
                if t_type == "momentum"
                else self.config.PARTIAL_PROFIT_STD
            )

            log.info(
                "   📍 %s [%s]  entry=$%.2f  now=$%.2f  P&L=%.1f%%",
                ticker,
                t_type,
                entry,
                current,
                pl_pct,
            )

            # ── Check 2% FIRST, then 1% (order matters!) ──────
            # Partial profit threshold (type-specific)
            if pl_pct >= partial * 100:
                log.info(
                    "   💰 %s +%.1f%% — partial profit threshold hit", ticker, pl_pct
                )

            # Up 2% — lock in 1% profit
            elif pl_pct >= 2.0:
                lock = round(entry * 1.01, 2)
                log.info(
                    "   🔒 %s +%.1f%% — locking 1%% at $%.2f", ticker, pl_pct, lock
                )

            # Up 1% — move to breakeven
            elif pl_pct >= 1.0:
                breakeven = round(entry * 1.001, 2)
                log.info(
                    "   📈 %s +%.1f%% — breakeven stop at $%.2f",
                    ticker,
                    pl_pct,
                    breakeven,
                )

            # Down more than 1.5x ATR — warn
            elif atr and (entry - current) > atr * 1.5:
                log.info(
                    "   ⚠️  %s down $%.2f > 1.5x ATR=$%.2f — watch",
                    ticker,
                    entry - current,
                    atr,
                )

    # ─────────────────────────────────────────────────────────────
    # STOCK LAYER
    # ─────────────────────────────────────────────────────────────

    def _run_stock_trade(
        self, decision: dict, portfolio: dict, open_symbols: list, market_data: dict
    ):
        action = decision.get("action", "hold")
        ticker = decision.get("ticker", "")
        conf = decision.get("confidence", 0)
        reason = (decision.get("reason") or "")[:100]
        t_type = decision.get(
            "ticker_type", self.ticker_classifications.get(ticker, "quality")
        )

        log.info(
            "📈 Claude: %s %s  conf=%.0f%%  type=%s  |  %s",
            action.upper(),
            ticker,
            conf * 100,
            t_type,
            reason,
        )

        if action == "hold":
            self.db.log_decision(decision, "HOLD")
            return

        if action == "buy" and ticker in open_symbols:
            log.info("⏭️  Already holding %s — skipping", ticker)
            self.db.log_decision(decision, "SKIPPED", "Already in position")
            return

        # Inject ATR-based stop/target if Claude didn't set them
        if ticker in market_data:
            atr = market_data[ticker].get("atr_14") or 0
            price = market_data[ticker].get("price", 0)
            params = self.config.ATR_PARAMS.get(t_type, {"stop": 3.5, "target": 10.0})
            if atr > 0 and price > 0:
                if not decision.get("stop_loss"):
                    decision["stop_loss"] = round(price - atr * params["stop"], 2)
                if not decision.get("take_profit"):
                    decision["take_profit"] = round(price + atr * params["target"], 2)
                if decision.get("stop_loss") and decision.get("take_profit"):
                    risk = price - decision["stop_loss"]
                    reward = decision["take_profit"] - price
                    decision["risk_reward"] = round(reward / risk, 2) if risk > 0 else 0

        ok, blocked = self.risk.approve(decision, portfolio)
        if not ok:
            log.info("🛑 Risk blocked: %s", blocked)
            self.db.log_decision(decision, "BLOCKED", blocked)
            return

        if self.dry_run:
            log.info(
                "🧪 DRY RUN — would %s %s  stop=$%s  target=$%s",
                action.upper(),
                ticker,
                decision.get("stop_loss", "?"),
                decision.get("take_profit", "?"),
            )
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

        # Only consider tickers with directional signal AND signal filter passed
        # Exclude regime-only and no-trade tickers (SPY, QQQ)
        candidates = sorted(
            [
                (t, d)
                for t, d in market_data.items()
                if t not in self.config.NO_TRADE_TICKERS
                and d.get("signals", {}).get("bias", "NEUTRAL") != "NEUTRAL"
                and d.get("volume", 0) > 1_000_000
                and d.get("signal_count", 0) >= self.config.MIN_SIGNALS_PRE_FILTER
            ],
            key=lambda x: x[1].get("signal_count", 0),
            reverse=True,
        )

        if not candidates:
            log.info("⚪ Options: no qualifying setups.")
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
            log.info(
                "   🎯 %s @ $%.2f  bias=%s  sigs=%d/7",
                ticker,
                price,
                bias,
                stock_data.get("signal_count", 0),
            )

            opt_data = self.options_feed.get_options_data(ticker, price, bias)
            if not opt_data or not opt_data.get("recommendations"):
                log.info("   No options data: %s", ticker)
                continue

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
    # OPTIONS EXIT
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
        log.info("🔔 End of day cleanup...")
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

        # Refresh watchlist immediately
        self.refresh_watchlist()

        # Live feed disabled (Python 3.14 compat — see _start_live_feed)
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
                log.info("\n⛔ Stopped by user.")
                if not self.dry_run:
                    self.options_broker.close_all_options()
                self.db.print_summary()
                break
            except Exception as e:
                log.error("Main loop error: %s", e)
                time.sleep(60)


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Claude Trading Bot")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without submitting orders"
    )
    args = parser.parse_args()

    try:
        bot = TradingBot(dry_run=args.dry_run)
        bot.start()
    except SystemExit:
        sys.exit(1)
    except Exception as e:
        log.critical("Bot crashed: %s", e, exc_info=True)
        sys.exit(1)
