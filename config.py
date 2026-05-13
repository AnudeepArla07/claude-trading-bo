"""
config.py
=========
Single source of truth for ALL bot and backtest settings.

HOW TO SWITCH BETWEEN PAPER AND LIVE TRADING:
  1. Scroll to the TRADING MODE section below
  2. Comment out PAPER lines, uncomment LIVE lines
  3. Set ALPACA_PAPER_KEY / ALPACA_LIVE_KEY in your .env file
  4. Never commit real keys to GitHub
"""

import os

# ══════════════════════════════════════════════════════════════════
# ENV LOADER — auto-reads .env file without needing `source .env`
# ══════════════════════════════════════════════════════════════════


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("export ", "")
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()


# ══════════════════════════════════════════════════════════════════
# VALIDATION — runs at bot startup before any API call
# ══════════════════════════════════════════════════════════════════


def validate_config() -> None:
    required = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY", ""),
        "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }
    missing = [k for k, v in required.items() if not v or "YOUR_" in v]
    if missing:
        raise ValueError(
            f"Missing or placeholder API keys: {missing}\n"
            f"Set them in your .env file and restart."
        )


class Config:

    # ── API Keys ──────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")

    # ══════════════════════════════════════════════════════════════
    # TRADING MODE — comment/uncomment to switch
    # ══════════════════════════════════════════════════════════════

    # ── PAPER TRADING (safe — use this until strategy is proven) ──
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    ALPACA_DATA_FEED: str = "iex"  # free 15-min delayed

    # ── LIVE TRADING (real money — uncomment when ready) ──────────
    # ALPACA_BASE_URL:  str = "https://api.alpaca.markets"
    # ALPACA_DATA_FEED: str = "sip"   # $9/mo real-time feed

    DRY_RUN: bool = False  # overridden at runtime by --dry-run flag

    # ══════════════════════════════════════════════════════════════
    # WATCHLIST
    # SOFI removed — consistent underperformer in backtesting
    # META, AMZN removed — underperformed vs momentum tickers
    # SPY/QQQ removed from trading — index ETFs not worth the risk/reward
    # SPY still fetched every cycle as regime reference (REGIME_TICKER)
    # ══════════════════════════════════════════════════════════════

    WATCHLIST: list = [
        "NVDA",
        "AMD",
        "TSLA",
        "AAPL",
        "GOOGL",
        "MSFT",
        "PLTR",
        "MSTR",
        "COIN",
        "UBER",
    ]

    REGIME_TICKER: str = "SPY"  # fetched every cycle for regime, never traded
    NO_TRADE_TICKERS: set = {"SPY", "QQQ"}  # never enter stock or options positions

    # ══════════════════════════════════════════════════════════════
    # CLAUDE MODEL
    # ══════════════════════════════════════════════════════════════

    MODEL: str = "claude-sonnet-4-6"
    MAX_TOKENS: int = 1200

    # ══════════════════════════════════════════════════════════════
    # STRATEGY PARAMETERS
    # These match backtest_v10.py exactly — DO NOT change without
    # re-running the backtest to verify impact.
    # ══════════════════════════════════════════════════════════════

    # ── Claude decision thresholds ────────────────────────────────
    MIN_CONFIDENCE: float = 0.65  # min confidence to trade
    # 5/7 signals = 0.71 ≥ 0.65 ✅
    # 4/7 signals = 0.57 < 0.65 ❌
    MIN_RISK_REWARD: float = 2.0  # min R:R ratio

    # ── Position sizing (Lever 3 from backtest) ───────────────────
    RISK_PER_TRADE: float = 0.02  # 2% of equity per trade
    MAX_POSITION_PCT: float = 0.20  # max 20% per position
    MAX_POSITIONS: int = 3  # max concurrent positions

    # ── ATR-based stop/target (must match backtest ATR_PARAMS) ────
    # Used by broker.py to compute bracket order levels
    ATR_PARAMS: dict = {
        "momentum": {"stop": 2.5, "target": 12.0},  # 4.8:1 R:R
        "quality": {"stop": 3.5, "target": 10.0},  # 2.9:1 R:R
        "index": {"stop": 4.0, "target": 8.0},  # 2.0:1 R:R
    }

    # ── Market regime filter (Lever 5) ────────────────────────────
    # SPY must be at least BULL (1) before any trades are allowed
    # Regime values: 2=STRONG_BULL 1=BULL 0=CHOPPY -1=BEAR -2=STRONG_BEAR
    SPY_BULL_MIN: float = 1.0  # sit out in CHOPPY/BEAR markets

    # ── Signal pre-filter ─────────────────────────────────────────
    # Only call Claude when this many signals are already aligned
    # Reduces API calls, ensures Claude sees quality setups only
    MIN_SIGNALS_PRE_FILTER: int = 4  # 4/7 minimum before calling Claude

    # ── Partial profit thresholds (match backtest) ────────────────
    PARTIAL_PROFIT_MOM: float = 0.06  # momentum: take half at +6%
    PARTIAL_PROFIT_STD: float = 0.04  # quality/index: take half at +4%

    # ── Risk guards ───────────────────────────────────────────────
    DAILY_LOSS_LIMIT: float = 0.03  # halt if down 3% in a day
    MAX_DRAWDOWN: float = 0.08  # halt if down 8% from peak
    MAX_TRADES_PER_DAY: int = 15
    MAX_CONSECUTIVE_LOSSES: int = 3

    # ══════════════════════════════════════════════════════════════
    # OPTIONS SETTINGS
    # ══════════════════════════════════════════════════════════════

    OPTIONS_MAX_LOSS_PCT: float = 0.02
    OPTIONS_PROFIT_TARGET: float = 0.80
    OPTIONS_STOP_LOSS: float = 0.50
    OPTIONS_MIN_VOLUME: int = 10
    OPTIONS_MAX_SPREAD_PCT: float = 0.15
    OPTIONS_MIN_DTE: int = 7
    OPTIONS_MAX_DTE: int = 45

    # ══════════════════════════════════════════════════════════════
    # BOT BEHAVIOR
    # ══════════════════════════════════════════════════════════════

    CYCLE_MINUTES: int = 10
    CLOSE_EOD: bool = False  # False = let winners run overnight
    # True  = safer for paper trading
    # PAPER: set True | LIVE: set False
