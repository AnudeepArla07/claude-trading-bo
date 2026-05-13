"""
config.py
=========
All settings in one place. Edit this file before running the bot.
"""

import os


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))


def _mask_key(key: str) -> str:
    """Mask API key for safe logging (show first and last 4 chars)."""
    if not key or len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def validate_config() -> None:
    """Validate all required API keys are set."""
    required_keys = {
        "ANTHROPIC_API_KEY": "Anthropic API key from https://console.anthropic.com",
        "ALPACA_API_KEY": "Alpaca API key from https://alpaca.markets",
        "ALPACA_SECRET_KEY": "Alpaca secret key from https://alpaca.markets",
    }

    for key, desc in required_keys.items():
        value = os.getenv(key, "")
        if not value or value.startswith("YOUR_"):
            raise ValueError(
                f"❌ {key} not configured!\n"
                f"   {desc}\n"
                f"   Set in .env file or environment variable."
            )


class Config:
    # ── API Keys ──────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_KEY")
    ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_KEY")
    ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET")

    # Paper trading (safe). Change to "https://api.alpaca.markets" for live.
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    # "iex" = free 15-min delayed. "sip" = $9/mo real-time.
    ALPACA_DATA_FEED = "iex"

    # Run the bot without placing any orders. Useful for testing and validation.
    DRY_RUN: bool = os.getenv("DRY_RUN", "0").lower() in ("1", "true", "yes")

    # ── Watchlist ─────────────────────────────────────────────────────────────
    # High-volume momentum tickers best suited for this strategy + options.
    WATCHLIST: list = [
        "NVDA",
        "AMD",
        "TSLA",
        "META",
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "PLTR",
        "SOFI",
        "MSTR",
        "COIN",
        "UBER",
    ]

    # ── Claude Model ──────────────────────────────────────────────────────────
    MODEL: str = "claude-sonnet-4-6"
    MAX_TOKENS: int = 1200

    # ── Risk Settings ─────────────────────────────────────────────────────────
    MIN_CONFIDENCE: float = 0.72  # min Claude confidence to trade
    MIN_RISK_REWARD: float = 2.0  # min R:R ratio
    RISK_PER_TRADE: float = 0.01  # risk 1% of equity per stock trade
    MAX_POSITION_PCT: float = 0.15  # max 15% of portfolio per position
    DAILY_LOSS_LIMIT: float = 0.03  # halt if down 3% in a day
    MAX_DRAWDOWN: float = 0.08  # halt if down 8% from peak ever
    MAX_TRADES_PER_DAY: int = 15  # max total trades per day
    MAX_CONSECUTIVE_LOSSES: int = 3  # pause after 3 losses in a row

    # ── Options Risk ──────────────────────────────────────────────────────────
    OPTIONS_MAX_LOSS_PCT: float = 0.02  # max 2% of portfolio per options trade
    OPTIONS_PROFIT_TARGET: float = 0.80  # close at 80% gain on premium
    OPTIONS_STOP_LOSS: float = 0.50  # close at 50% loss on premium
    OPTIONS_MIN_VOLUME: int = 10  # minimum contract volume
    OPTIONS_MAX_SPREAD_PCT: float = 0.15  # max 15% bid/ask spread
    OPTIONS_MIN_DTE: int = 7  # min days to expiry
    OPTIONS_MAX_DTE: int = 45  # max days to expiry

    # ── Bot Behavior ──────────────────────────────────────────────────────────
    CYCLE_MINUTES: int = 10  # analyze every 10 minutes
    CLOSE_EOD: bool = True  # close all positions at 3:45 PM ET
