"""
backtest.py  (v9)
=================
Backtests the Claude Trading Bot with automatic ticker classification.

NEW: Auto-Classifier
  Analyzes each ticker's price behavior over the training window
  and automatically assigns it to one of three strategy types:

    MOMENTUM  — high volatility, RSI stays overbought, strong trends
                Strategy: buy breakouts, ride the trend
                Tickers: NVDA, TSLA, MSTR, COIN, PLTR usually land here

    QUALITY   — stable growers, RSI mean-reverts, regular pullbacks
                Strategy: buy RSI dips in uptrend
                Tickers: AAPL, MSFT, GOOGL usually land here

    INDEX     — low volatility, slow moves, range-bound RSI
                Strategy: buy dips with wider stops, longer holds
                Tickers: SPY, QQQ usually land here

  Classification runs automatically before the backtest using
  rolling 252-bar (1 year) windows so it adapts as market regimes change.

Usage:
    python backtest.py                     full 6-year backtest
    python backtest.py --ticker NVDA       single ticker
    python backtest.py --quick             2024 only
    python backtest.py --quiet             no trade logs
    python backtest.py --quiet --regimes   add regime breakdown
    python backtest.py --classify          show classification only
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from typing import Optional, Dict

# Fix yfinance SQLite cache (prevents OperationalError in regime tests)
try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DEFAULT_WATCHLIST = [
    "NVDA",
    "AMD",
    "TSLA",
    "AAPL",
    "GOOGL",
    "MSFT",
    "QQQ",
    "PLTR",
    "MSTR",
    "COIN",
    "UBER",
    "META",
    "AMZN",
    "SPY",
]

INIT_CASH = 100_000
COMMISSION = 0.001

# Per-type stop and target multipliers
# Momentum: wider targets (trends run far), moderate stops
# Quality:  moderate targets, wider stops (more volatile per ATR)
# Index:    widest stops (slow, grindy), moderate targets
ATR_PARAMS = {
    "momentum": {"stop": 2.5, "target": 7.5, "rr": 3.0},
    "quality": {"stop": 3.5, "target": 8.75, "rr": 2.5},
    "index": {"stop": 4.0, "target": 8.0, "rr": 2.0},
}

RSI_PERIOD = 14
BB_PERIOD = 20
RISK_PER_TRADE = 0.01  # 1% of equity per trade
MIN_HOLD_DAYS = 3  # min days before stop fires
MAX_POSITIONS = 4  # max concurrent positions
MAX_POSITION_PCT = 0.20  # max 20% equity per position
PARTIAL_PROFIT = 0.03  # take half at +3%

# Auto-classification thresholds
MOMENTUM_BETA_MIN = 1.3  # beta above this → momentum
MOMENTUM_RSI_OB_MIN = 0.35  # fraction of time RSI > 65 → momentum
QUALITY_BETA_MAX = 1.3  # beta below this → quality or index
INDEX_VOL_MAX = 0.18  # annualized vol below this → index


# ══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════


def download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download OHLCV. Handles all yfinance versions."""
    try:
        try:
            yf.set_tz_cache_location("/tmp/yf_cache")
        except Exception:
            pass

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        col_map = {}
        for c in raw.columns:
            n = str(c).strip().lower()
            if "open" in n:
                col_map[c] = "Open"
            elif "high" in n:
                col_map[c] = "High"
            elif "low" in n:
                col_map[c] = "Low"
            elif "close" in n:
                col_map[c] = "Close"
            elif "vol" in n:
                col_map[c] = "Volume"
        raw = raw.rename(columns=col_map)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in raw.columns for c in required):
            return None

        df = raw[required].copy()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        return df if len(df) >= 60 else None

    except Exception as e:
        print(f"  ❌ {ticker}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════


def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calc_macd(closes: pd.Series) -> tuple:
    e12 = closes.ewm(span=12, adjust=False).mean()
    e26 = closes.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def calc_ema(closes: pd.Series, period: int) -> pd.Series:
    return closes.ewm(span=period, adjust=False).mean()


def calc_regime(closes: pd.Series) -> pd.Series:
    """2=STRONG_BULL 1=BULL 0=CHOPPY -1=BEAR -2=STRONG_BEAR"""
    e9 = calc_ema(closes, 9)
    e21 = calc_ema(closes, 21)
    e50 = calc_ema(closes, 50)
    p = closes
    r = pd.Series(0.0, index=closes.index)
    r[(p > e9) & (e9 > e21) & (e21 > e50)] = 2.0
    r[(p > e21) & (e21 > e50) & (r != 2)] = 1.0
    r[(p < e9) & (e9 < e21)] = -1.0
    r[(p < e50) & (r == -1)] = -2.0
    return r


def calc_vol_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg = volume.rolling(period).mean()
    return (volume / avg.replace(0, np.nan)).fillna(1.0)


def calc_bollinger_pctb(closes: pd.Series, period: int = 20) -> pd.Series:
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    lower = mid - 2 * std
    upper = mid + 2 * std
    denom = (upper - lower).replace(0, np.nan)
    return ((closes - lower) / denom).fillna(0.5)


# ══════════════════════════════════════════════════════════════════
# AUTO-CLASSIFIER
# ══════════════════════════════════════════════════════════════════


def classify_ticker(
    df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None, window: int = 252
) -> str:
    """
    Automatically classify a ticker as momentum, quality, or index
    by analyzing its actual price behavior over the last 'window' bars.

    Scoring model — each characteristic adds points:

    MOMENTUM signals:
      +3  Beta vs SPY > MOMENTUM_BETA_MIN (high beta = amplified moves)
      +2  RSI overbought (>65) > 35% of time (stays extended = trend)
      +2  Average 20-day momentum > 3% (moves fast)
      +1  Annualized volatility > 40%
      +1  Regime = STRONG_BULL > 40% of time (sustained strong trends)

    INDEX signals:
      +3  Annualized volatility < INDEX_VOL_MAX (very stable)
      +2  Beta < 0.8 (moves less than market)
      +1  RSI range (max-min) < 45 (stays range-bound)

    QUALITY signals:
      +2  Beta between 0.8 and MOMENTUM_BETA_MIN
      +2  RSI mean-reverts (crosses 50 frequently)
      +1  Volatility between index and momentum thresholds

    Returns: "momentum" | "quality" | "index"
    """
    closes = df["Close"].tail(window)
    highs = df["High"].tail(window)
    lows = df["Low"].tail(window)

    # ── Compute characteristics ────────────────────────────────
    rsi = calc_rsi(closes)
    regime = calc_regime(closes)
    ret = closes.pct_change().dropna()

    # Annualized volatility
    ann_vol = ret.std() * np.sqrt(252)

    # Beta vs SPY
    beta = 1.0
    if spy_df is not None and len(spy_df) >= window:
        spy_ret = spy_df["Close"].tail(window).pct_change().dropna()
        common = ret.index.intersection(spy_ret.index)
        if len(common) > 50:
            r = ret.loc[common]
            sr = spy_ret.loc[common]
            cov = np.cov(r, sr)[0][1]
            spy_var = sr.var()
            beta = cov / spy_var if spy_var > 0 else 1.0

    # RSI characteristics
    rsi_ob_frac = (rsi > 65).mean()  # fraction overbought
    rsi_os_frac = (rsi < 35).mean()  # fraction oversold
    rsi_range = rsi.max() - rsi.min()
    rsi_crosses = ((rsi > 50) != (rsi > 50).shift(1)).sum()  # crosses 50

    # Momentum characteristics
    mom20d = closes.pct_change(20).dropna()
    avg_mom20 = mom20d.mean() * 100

    # Regime
    strong_bull_frac = (regime == 2).mean()

    # ── Score each type ────────────────────────────────────────
    momentum_score = 0
    quality_score = 0
    index_score = 0

    # Momentum scoring
    if beta > MOMENTUM_BETA_MIN:
        momentum_score += 3
    elif beta > 1.1:
        momentum_score += 1
    if rsi_ob_frac > MOMENTUM_RSI_OB_MIN:
        momentum_score += 2
    elif rsi_ob_frac > 0.20:
        momentum_score += 1
    if avg_mom20 > 3:
        momentum_score += 2
    elif avg_mom20 > 1.5:
        momentum_score += 1
    if ann_vol > 0.45:
        momentum_score += 2
    elif ann_vol > 0.30:
        momentum_score += 1
    if strong_bull_frac > 0.40:
        momentum_score += 1

    # Index scoring
    if ann_vol < INDEX_VOL_MAX:
        index_score += 3
    elif ann_vol < 0.22:
        index_score += 1
    if beta < 0.8:
        index_score += 3
    elif beta < 1.0:
        index_score += 1
    if rsi_range < 45:
        index_score += 2

    # Quality scoring
    if 0.8 <= beta <= MOMENTUM_BETA_MIN:
        quality_score += 2
    if rsi_crosses > 30:
        quality_score += 2
    if 0.20 < ann_vol < 0.40:
        quality_score += 1
    if 0.15 < rsi_ob_frac < 0.30:
        quality_score += 1

    # ── Decision ──────────────────────────────────────────────
    scores = {
        "momentum": momentum_score,
        "quality": quality_score,
        "index": index_score,
    }
    return max(scores, key=scores.get)


def classify_all_tickers(
    all_data: Dict[str, pd.DataFrame],
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Classify all tickers using their downloaded data.
    Uses SPY as the market benchmark for beta calculation.
    """
    spy_df = all_data.get("SPY")

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  🔍 AUTO-CLASSIFICATION")
        print(f"  Analyzing price behavior to assign strategy type")
        print(f"{'═'*65}")
        print(
            f"  {'Ticker':<8} {'Type':<12} {'Beta':>6} "
            f"{'Vol%':>6} {'RSI_OB%':>8} {'Score': >6}"
        )
        print(f"  {'─'*55}")

    classifications = {}

    for ticker, df in all_data.items():
        closes = df["Close"].tail(252)
        rsi = calc_rsi(closes)
        ret = closes.pct_change().dropna()
        ann_vol = ret.std() * np.sqrt(252)
        rsi_ob_frac = (rsi > 65).mean()

        # Beta calculation
        beta = 1.0
        if spy_df is not None:
            spy_ret = spy_df["Close"].tail(252).pct_change().dropna()
            common = ret.index.intersection(spy_ret.index)
            if len(common) > 50:
                cov = np.cov(ret.loc[common], spy_ret.loc[common])[0][1]
                spy_var = spy_ret.loc[common].var()
                beta = cov / spy_var if spy_var > 0 else 1.0

        t_type = classify_ticker(df, spy_df)
        classifications[ticker] = t_type

        if verbose:
            icon = {"momentum": "🚀", "quality": "📈", "index": "📊"}[t_type]
            print(
                f"  {icon}{ticker:<7} {t_type:<12} "
                f"{beta:>6.2f} "
                f"{ann_vol*100:>5.0f}%  "
                f"{rsi_ob_frac*100:>6.0f}%"
            )

    if verbose:
        types = pd.Series(classifications).value_counts()
        print(f"\n  Summary:")
        for t, n in types.items():
            tickers = [k for k, v in classifications.items() if v == t]
            icon = {"momentum": "🚀", "quality": "📈", "index": "📊"}[t]
            print(f"  {icon} {t:<12}: {n} tickers — {tickers}")
        print(f"{'═'*65}\n")

    return classifications


# ══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION — strategy per ticker type
# ══════════════════════════════════════════════════════════════════


def generate_signals(df: pd.DataFrame, ticker_type: str = "quality") -> pd.DataFrame:
    """
    Generate entry/exit signals based on ticker type.

    MOMENTUM: buy breakouts in STRONG_BULL regime
    QUALITY:  buy RSI dips in BULL regime
    INDEX:    buy RSI dips conservatively in BULL regime
    """
    df = df.copy()
    closes = df["Close"]
    highs = df["High"]
    lows = df["Low"]
    volumes = df["Volume"]

    rsi = calc_rsi(closes, RSI_PERIOD)
    macd, sig, h = calc_macd(closes)
    atr = calc_atr(highs, lows, closes)
    regime = calc_regime(closes)
    vol_ratio = calc_vol_ratio(volumes)
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    ema50 = calc_ema(closes, 50)
    pct_b = calc_bollinger_pctb(closes, BB_PERIOD)
    mom5d = closes.pct_change(5).fillna(0) * 100
    mom20d = closes.pct_change(20).fillna(0) * 100

    df["rsi"] = rsi
    df["macd"] = macd
    df["macd_sig"] = sig
    df["macd_hist"] = h
    df["atr"] = atr
    df["regime"] = regime
    df["vol_ratio"] = vol_ratio
    df["ema9"] = ema9
    df["ema21"] = ema21
    df["mom5d"] = mom5d

    if ticker_type == "momentum":
        # ── MOMENTUM STRATEGY ─────────────────────────────────
        # Buy breakouts in strong uptrends — NOT RSI dips.
        # RSI > 60 in uptrend means momentum continuation.
        # Requires STRONG_BULL (both daily EMAs aligned perfectly).
        near_high_20d = closes >= closes.rolling(20).max().shift(1) * 0.97
        strong_bull = regime == 2
        macd_bull = ((macd > sig) & (h > 0)).fillna(False)
        vol_surge = vol_ratio > 1.5
        fast_mom = mom5d > 2
        above_ema9 = closes > ema9
        macd_growing = (h > h.shift(1)).fillna(False)

        df["signal_count"] = (
            near_high_20d.astype(int)
            + strong_bull.astype(int)
            + macd_bull.astype(int)
            + vol_surge.astype(int)
            + fast_mom.astype(int)
            + above_ema9.astype(int)
            + macd_growing.astype(int)
        )

        # Entry: strong bull trend + 4+ signals
        df["entry"] = (strong_bull & (df["signal_count"] >= 4) & atr.notna()).fillna(
            False
        )

        # Exit: trend weakens significantly or momentum collapses
        df["exit"] = (
            (regime <= 0)  # no longer in BULL
            | (rsi < 35)  # momentum completely collapsed
        ).fillna(False)

    elif ticker_type == "index":
        # ── INDEX STRATEGY ────────────────────────────────────
        # Indices are slow and steady — buy RSI dips conservatively.
        # Use tighter RSI threshold (< 42) since indices don't
        # pull back as deeply.
        s1 = (regime >= 1).astype(int)
        s2 = (rsi < 42).astype(int)  # tighter RSI for indices
        s3 = (closes > ema21).astype(int)
        s4 = (macd > sig).astype(int)
        s5 = (vol_ratio > 1.0).astype(int)
        s6 = (mom5d > -1).astype(int)  # not falling hard
        s7 = (closes > ema50).astype(int)  # above 50-day MA

        df["signal_count"] = s1 + s2 + s3 + s4 + s5 + s6 + s7

        # Entry: BULL + 5 signals (strict quality filter)
        df["entry"] = ((regime >= 1) & (df["signal_count"] >= 5) & atr.notna()).fillna(
            False
        )

        # Exit: overbought for index OR trend breaks
        df["exit"] = ((rsi > 70) | (regime <= -1)).fillna(  # indices rarely go above 70
            False
        )

    else:
        # ── QUALITY STRATEGY (default) ────────────────────────
        # Buy RSI pullbacks in uptrend — works well for stable growers.
        s1 = (regime >= 1).astype(int)
        s2 = (rsi < 50).astype(int)
        s3 = (closes > ema21).astype(int)
        s4 = (macd > sig).astype(int)
        s5 = (vol_ratio > 1.2).astype(int)
        s6 = (mom5d > 0).astype(int)
        s7 = (pct_b < 0.35).astype(int)

        df["signal_count"] = s1 + s2 + s3 + s4 + s5 + s6 + s7

        # Entry: BULL + 5 signals
        df["entry"] = ((regime >= 1) & (df["signal_count"] >= 5) & atr.notna()).fillna(
            False
        )

        # Exit: overbought OR trend breaks
        df["exit"] = ((rsi > 72) | (regime <= -1)).fillna(False)

    return df


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO
# ══════════════════════════════════════════════════════════════════


class Portfolio:
    """
    Portfolio with per-ticker-type stop and target sizing.
    ATR multipliers are set per classification:
      momentum: tighter stop, bigger target (trends run far)
      quality:  wider stop (more noise), moderate target
      index:    widest stop (slow moving), moderate target
    """

    def __init__(self, init_cash: float):
        self.cash = init_cash
        self.equity = init_cash
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def update_equity(self, prices: dict):
        pos_val = sum(
            self.positions[t]["qty"] * prices.get(t, 0) for t in self.positions
        )
        self.equity = self.cash + pos_val
        self.equity_curve.append(self.equity)

    def can_buy(self, price: float, atr: float, stop_mult: float) -> int:
        if atr <= 0 or price <= 0:
            return 0
        qty = int(self.equity * RISK_PER_TRADE / (atr * stop_mult))
        max_qty = int(self.equity * MAX_POSITION_PCT / (price * (1 + COMMISSION)))
        cash_qty = int(self.cash * 0.95 / (price * (1 + COMMISSION)))
        return max(0, min(qty, max_qty, cash_qty))

    def buy(
        self, ticker: str, price: float, atr: float, date, ticker_type: str = "quality"
    ) -> bool:
        if ticker in self.positions:
            return False

        params = ATR_PARAMS[ticker_type]
        stop_mult = params["stop"]
        tgt_mult = params["target"]

        qty = self.can_buy(price, atr, stop_mult)
        if qty <= 0:
            return False
        cost = qty * price * (1 + COMMISSION)
        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions[ticker] = {
            "qty": qty,
            "entry": price,
            "stop": price - atr * stop_mult,
            "target": price + atr * tgt_mult,
            "peak": price,
            "atr": atr,
            "date": date,
            "stop_mult": stop_mult,
            "ticker_type": ticker_type,
            "partial_taken": False,
        }
        return True

    def sell(self, ticker: str, price: float, date, reason: str) -> Optional[dict]:
        if ticker not in self.positions:
            return None
        pos = self.positions.pop(ticker)
        qty = pos["qty"]
        proceeds = qty * price * (1 - COMMISSION)
        cost = qty * pos["entry"] * (1 + COMMISSION)
        pl = proceeds - cost
        pl_pct = pl / cost * 100
        self.cash += proceeds
        trade = {
            "ticker": ticker,
            "ticker_type": pos.get("ticker_type", "quality"),
            "entry": pos["entry"],
            "exit": price,
            "qty": qty,
            "pl": pl,
            "pl_pct": pl_pct,
            "entry_dt": pos["date"],
            "exit_dt": date,
            "hold_days": (date - pos["date"]).days,
            "reason": reason,
        }
        self.trades.append(trade)
        return trade

    def check_stops(self, prices: dict, date) -> list:
        to_sell = []

        for ticker, pos in list(self.positions.items()):
            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            hold_days = (date - pos["date"]).days
            if hold_days < MIN_HOLD_DAYS:
                continue

            entry = pos["entry"]
            stop_mult = pos.get("stop_mult", 3.0)
            ttype = pos.get("ticker_type", "quality")

            # Partial profit (different thresholds by type)
            partial_pct = 0.05 if ttype == "momentum" else PARTIAL_PROFIT
            if (
                not pos["partial_taken"]
                and price >= entry * (1 + partial_pct)
                and pos["qty"] >= 2
            ):
                half = pos["qty"] // 2
                pos["qty"] -= half
                self.cash += half * price * (1 - COMMISSION)
                pos["stop"] = entry * 1.001
                pos["partial_taken"] = True
                self.trades.append(
                    {
                        "ticker": ticker,
                        "ticker_type": ttype,
                        "entry": entry,
                        "exit": price,
                        "qty": half,
                        "pl": half * (price - entry),
                        "pl_pct": (price - entry) / entry * 100,
                        "entry_dt": pos["date"],
                        "exit_dt": date,
                        "hold_days": hold_days,
                        "reason": "partial_profit",
                    }
                )

            # Trailing stop
            if price > pos["peak"]:
                pos["peak"] = price
                new_trail = price - pos["atr"] * stop_mult
                if new_trail > pos["stop"]:
                    pos["stop"] = new_trail

            # Breakeven after +1%
            if price >= entry * 1.01:
                be = entry * 1.001
                if be > pos["stop"]:
                    pos["stop"] = be

            if price <= pos["stop"]:
                to_sell.append((ticker, price, "stop_loss"))
            elif price >= pos["target"]:
                to_sell.append((ticker, price, "profit_target"))

        return to_sell


# ══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════


def run_backtest(
    tickers: list,
    start_date: str,
    end_date: str,
    init_cash: float = INIT_CASH,
    verbose: bool = True,
    show_classify: bool = True,
) -> dict:
    print(f"\n{'═'*65}")
    print(f"  📊 BACKTEST ENGINE  v9")
    print(f"  Period  : {start_date} → {end_date}")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Capital : ${init_cash:,.0f}")
    print(f"{'═'*65}")

    # Download all data
    print("\n📥 Downloading data...")
    all_data = {}
    for ticker in tickers:
        df = download_ticker(ticker, start_date, end_date)
        if df is None:
            print(f"  ⚠️  {ticker}: skipped")
            continue
        all_data[ticker] = df
        print(f"  ✅ {ticker}: {len(df)} bars")

    if not all_data:
        print("\n❌ No data.")
        return {}

    # Auto-classify all tickers
    classifications = classify_all_tickers(all_data, verbose=show_classify)

    # Generate signals using per-type strategy
    print("📐 Generating signals per ticker type...")
    for ticker, df in all_data.items():
        t_type = classifications.get(ticker, "quality")
        all_data[ticker] = generate_signals(df, t_type)
        entries = int(all_data[ticker]["entry"].sum())
        exits = int(all_data[ticker]["exit"].sum())
        print(
            f"  {ticker:<6} [{t_type:<9}]: " f"entries={entries:>3}  exits={exits:>3}"
        )

    total_entries = sum(int(df["entry"].sum()) for df in all_data.values())
    if total_entries == 0:
        print("\n⚠️  Zero entry signals.")
        return {}

    all_dates = sorted(set.union(*[set(df.index) for df in all_data.values()]))

    print(
        f"\n🔄 Simulating {len(all_dates)} trading days "
        f"({total_entries} total entry signals)..."
    )

    port = Portfolio(init_cash)

    for date in all_dates:
        prices = {
            t: float(df.loc[date, "Close"])
            for t, df in all_data.items()
            if date in df.index
        }

        # 1. Stop and target checks
        for ticker, price, reason in port.check_stops(prices, date):
            trade = port.sell(ticker, price, date, reason)
            if trade and verbose:
                icon = "🎯" if reason == "profit_target" else "🛑"
                print(
                    f"  {icon} {str(date.date())} {ticker:<6} "
                    f"[{trade['ticker_type']:<9}] "
                    f"{reason:<14} "
                    f"P&L: ${trade['pl']:>+8,.0f} "
                    f"({trade['pl_pct']:>+5.1f}%)"
                )

        # 2. Signal exits
        for ticker in list(port.positions.keys()):
            if date not in all_data[ticker].index:
                continue
            if bool(all_data[ticker].loc[date, "exit"]):
                price = prices.get(ticker, 0)
                trade = port.sell(ticker, price, date, "signal_exit")
                if trade and verbose:
                    print(
                        f"  📤 {str(date.date())} {ticker:<6} "
                        f"[{trade['ticker_type']:<9}] "
                        f"{'signal_exit':<14} "
                        f"P&L: ${trade['pl']:>+8,.0f} "
                        f"({trade['pl_pct']:>+5.1f}%)"
                    )

        # 3. Entries — ranked by signal count
        if len(port.positions) < MAX_POSITIONS:
            opps = []
            for ticker, df in all_data.items():
                if ticker in port.positions:
                    continue
                if date not in df.index:
                    continue
                row = df.loc[date]
                if bool(row["entry"]) and row["atr"] > 0:
                    opps.append(
                        (
                            ticker,
                            float(row["signal_count"]),
                            float(prices.get(ticker, 0)),
                            float(row["atr"]),
                            classifications.get(ticker, "quality"),
                        )
                    )

            opps.sort(key=lambda x: x[1], reverse=True)
            for ticker, sigs, price, atr, t_type in opps[:3]:
                if len(port.positions) >= MAX_POSITIONS:
                    break
                if port.buy(ticker, price, atr, date, t_type):
                    pos = port.positions[ticker]
                    if verbose:
                        icon = {"momentum": "🚀", "quality": "📈", "index": "📊"}[
                            t_type
                        ]
                        print(
                            f"  🟢 {str(date.date())} {ticker:<6} "
                            f"{icon}[{t_type:<9}] "
                            f"BUY x{pos['qty']:<5} "
                            f"@ ${price:.2f}  "
                            f"stop=${pos['stop']:.2f}  "
                            f"target=${pos['target']:.2f}"
                        )

        port.update_equity(prices)

    # Close remaining
    last_date = all_dates[-1]
    for ticker in list(port.positions.keys()):
        price = (
            float(all_data[ticker].loc[last_date, "Close"])
            if last_date in all_data[ticker].index
            else 0
        )
        port.sell(ticker, price, last_date, "end_of_test")

    return {
        "trades": port.trades,
        "equity_curve": port.equity_curve,
        "final_equity": port.equity,
        "init_cash": init_cash,
        "start_date": start_date,
        "end_date": end_date,
        "all_dates": all_dates,
        "all_data": all_data,
        "classifications": classifications,
    }


# ══════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════


def analyze_results(results: dict) -> dict:
    trades = results["trades"]
    equity = results["equity_curve"]
    init_cash = results["init_cash"]
    final_eq = results["final_equity"]

    if not trades:
        print("⚠️  No trades executed.")
        return {}

    df_t = pd.DataFrame(trades)

    total_return = (final_eq - init_cash) / init_cash * 100
    n_years = len(equity) / 252
    annual_return = (
        ((final_eq / init_cash) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    )

    winners = df_t[df_t["pl"] > 0]
    losers = df_t[df_t["pl"] < 0]
    win_rate = len(winners) / len(df_t) * 100
    avg_win = winners["pl_pct"].mean() if len(winners) else 0
    avg_loss = losers["pl_pct"].mean() if len(losers) else 0
    pf = (
        winners["pl"].sum() / abs(losers["pl"].sum())
        if len(losers) and losers["pl"].sum() != 0
        else float("inf")
    )

    eq_s = pd.Series(equity)
    daily = eq_s.pct_change().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()

    ticker_stats = (
        df_t.groupby("ticker")
        .agg(
            type=("ticker_type", "first"),
            trades=("pl", "count"),
            total_pl=("pl", "sum"),
            win_rate=("pl", lambda x: (x > 0).mean() * 100),
            avg_hold=("hold_days", "mean"),
        )
        .sort_values("total_pl", ascending=False)
    )

    # Per-type breakdown
    type_stats = (
        (
            df_t.groupby("ticker_type")
            .agg(
                trades=("pl", "count"),
                total_pl=("pl", "sum"),
                win_rate=("pl", lambda x: (x > 0).mean() * 100),
                avg_pl=("pl_pct", "mean"),
            )
            .sort_values("total_pl", ascending=False)
        )
        if "ticker_type" in df_t.columns
        else None
    )

    reason_stats = df_t.groupby("reason").agg(
        count=("pl", "count"),
        avg_pl=("pl_pct", "mean"),
        total=("pl", "sum"),
    )

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_trades": len(df_t),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": pf,
        "winners": len(winners),
        "losers": len(losers),
        "ticker_stats": ticker_stats,
        "type_stats": type_stats,
        "reason_stats": reason_stats,
        "df_trades": df_t,
    }


# ══════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════


def print_report(results: dict, metrics: dict):
    m = metrics
    tr = m["total_return"]
    ar = m["annual_return"]
    sh = m["sharpe"]
    dd = m["max_drawdown"]
    wr = m["win_rate"]
    pf = m["profit_factor"]
    rr = abs(m["avg_win"] / m["avg_loss"]) if m["avg_loss"] != 0 else 0

    def grade(s):
        if s > 2.0:
            return "🏆 EXCELLENT"
        if s > 1.5:
            return "✅ GOOD"
        if s > 1.0:
            return "⚠️  ACCEPTABLE"
        if s > 0.5:
            return "❌ POOR"
        return "🚨 VERY POOR"

    print(f"\n{'═'*65}")
    print(f"  📊 BACKTEST RESULTS  v9")
    print(f"  Period : {results['start_date']} → {results['end_date']}")
    print(f"{'═'*65}")

    print(f"\n  OVERALL PERFORMANCE")
    print(f"  {'─'*50}")
    print(f"  Total Return     : {tr:>+8.1f}%")
    print(f"  Annual Return    : {ar:>+8.1f}%")
    print(f"  Sharpe Ratio     : {sh:>8.2f}  {grade(sh)}")
    print(
        f"  Max Drawdown     : {dd:>8.1f}%  "
        f"{'✅' if dd > -20 else '⚠️' if dd > -30 else '❌'}"
    )
    print(f"  Starting Capital : ${results['init_cash']:>12,.0f}")
    print(f"  Final Capital    : ${results['final_equity']:>12,.0f}")
    print(
        f"  Net Profit       : "
        f"${results['final_equity']-results['init_cash']:>+12,.0f}"
    )

    print(f"\n  TRADE STATISTICS")
    print(f"  {'─'*50}")
    print(f"  Total Trades     : {m['total_trades']:>8}")
    print(
        f"  Winners          : {m['winners']:>8}  ({wr:.1f}%)  "
        f"{'✅' if wr > 55 else '⚠️' if wr > 45 else '❌'}"
    )
    print(f"  Losers           : {m['losers']:>8}  ({100-wr:.1f}%)")
    print(f"  Avg Win          : {m['avg_win']:>+8.2f}%")
    print(f"  Avg Loss         : {m['avg_loss']:>+8.2f}%")
    print(f"  Win / Loss R:R   : {rr:>8.1f}:1  " f"{'✅' if rr > 1.5 else '❌'}")
    print(
        f"  Profit Factor    : {pf:>8.2f}  "
        f"{'✅' if pf > 1.5 else '⚠️' if pf > 1.0 else '❌'}"
    )

    # Per-type breakdown
    if m.get("type_stats") is not None:
        print(f"\n  PERFORMANCE BY STRATEGY TYPE")
        print(f"  {'─'*58}")
        print(
            f"  {'Type':<12} {'Trades':>6} {'Win%':>6} "
            f"{'Avg P&L%':>9} {'Total P&L':>12}"
        )
        print(f"  {'─'*58}")
        icons = {"momentum": "🚀", "quality": "📈", "index": "📊"}
        for t_type, row in m["type_stats"].iterrows():
            icon = icons.get(t_type, "•")
            pl_icon = "✅" if row["total_pl"] > 0 else "❌"
            print(
                f"  {pl_icon}{icon}{t_type:<10} "
                f"{row['trades']:>6.0f} "
                f"{row['win_rate']:>5.1f}% "
                f"{row['avg_pl']:>+8.2f}% "
                f"${row['total_pl']:>10,.0f}"
            )

    print(f"\n  PERFORMANCE BY TICKER")
    print(f"  {'─'*65}")
    print(
        f"  {'Ticker':<8} {'Type':<11} {'Trades':>6} "
        f"{'Win%':>6} {'Total P&L':>12} {'Avg Hold':>9}"
    )
    print(f"  {'─'*65}")
    icons = {"momentum": "🚀", "quality": "📈", "index": "📊"}
    for ticker, row in m["ticker_stats"].iterrows():
        pl_icon = "✅" if row["total_pl"] > 0 else "❌"
        t_icon = icons.get(row.get("type", "quality"), "•")
        print(
            f"  {pl_icon}{t_icon}{ticker:<6} "
            f"{row.get('type',''):<11} "
            f"{row['trades']:>6.0f} "
            f"{row['win_rate']:>5.1f}% "
            f"${row['total_pl']:>10,.0f} "
            f"{row['avg_hold']:>7.1f}d"
        )

    print(f"\n  EXIT REASONS")
    print(f"  {'─'*58}")
    print(f"  {'Reason':<18} {'Count':>6} {'Avg P&L%':>9} {'Total':>12}")
    print(f"  {'─'*58}")
    for reason, row in m["reason_stats"].iterrows():
        icon = "✅" if row["avg_pl"] > 0 else "❌"
        print(
            f"  {icon}{reason:<17} {row['count']:>6.0f} "
            f"{row['avg_pl']:>+8.1f}% "
            f"${row['total']:>10,.0f}"
        )

    print(f"\n{'═'*65}")
    print(f"  VERDICT")
    print(f"{'═'*65}")

    issues, positives = [], []
    if sh > 1.5:
        positives.append(f"✅ Strong Sharpe ({sh:.2f})")
    elif sh > 1.0:
        positives.append(f"⚠️  Acceptable Sharpe ({sh:.2f})")
    else:
        issues.append(f"❌ Weak Sharpe ({sh:.2f})")
    if dd > -20:
        positives.append(f"✅ Drawdown manageable ({dd:.1f}%)")
    elif dd > -30:
        issues.append(f"⚠️  High drawdown ({dd:.1f}%)")
    else:
        issues.append(f"❌ Drawdown too high ({dd:.1f}%)")
    if wr > 55:
        positives.append(f"✅ Good win rate ({wr:.1f}%)")
    elif wr > 45:
        positives.append(f"⚠️  Acceptable win rate ({wr:.1f}%)")
    else:
        issues.append(f"❌ Low win rate ({wr:.1f}%)")
    if pf > 1.5:
        positives.append(f"✅ Strong profit factor ({pf:.2f})")
    elif pf > 1.0:
        issues.append(f"⚠️  Weak profit factor ({pf:.2f})")
    else:
        issues.append(f"❌ Losing overall (PF={pf:.2f})")
    if ar > 20:
        positives.append(f"✅ Strong annual return ({ar:.1f}%)")
    elif ar > 10:
        positives.append(f"⚠️  Moderate return ({ar:.1f}%)")
    else:
        issues.append(f"❌ Low return ({ar:.1f}%)")

    for p in positives:
        print(f"  {p}")
    for i in issues:
        print(f"  {i}")

    if not issues:
        print("\n  🚀 READY — start paper trading.")
    elif len(issues) <= 2:
        print("\n  ⚠️  CLOSE — paper trade while optimizing.")
    else:
        print("\n  🛑 NOT READY — fix before real money.")

    print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════


def save_charts(results: dict, metrics: dict) -> str:
    print("📈 Generating charts...")

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    G, R, GR, TXT = "#00ff88", "#ff4444", "#888888", "#ffffff"
    TYPE_COLORS = {"momentum": "#ff9900", "quality": "#00aaff", "index": "#aa88ff"}

    def style(ax, title):
        ax.set_facecolor("#16213e")
        ax.set_title(title, color=TXT, pad=8, fontsize=9)
        ax.tick_params(colors=TXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333366")
        ax.yaxis.label.set_color(TXT)
        ax.xaxis.label.set_color(TXT)

    eq = pd.Series(results["equity_curve"])
    dates = results["all_dates"][: len(eq)]
    init = results["init_cash"]

    # 1. Equity curve (spans all 3 columns)
    ax1 = fig.add_subplot(gs[0, :])
    color = G if eq.iloc[-1] >= init else R
    ax1.plot(dates, eq.values, color=color, linewidth=1.5)
    ax1.axhline(init, color=GR, linestyle="--", linewidth=0.8, alpha=0.7, label="Start")
    ax1.fill_between(dates, eq.values, init, alpha=0.1, color=color)
    ax1.set_ylabel("Portfolio ($)")
    ax1.legend(facecolor="#1a1a2e", labelcolor=TXT, fontsize=8)
    style(
        ax1,
        f"Equity Curve  |  Final ${results['final_equity']:,.0f}  "
        f"({metrics['total_return']:+.1f}%)  |  "
        f"Sharpe {metrics['sharpe']:.2f}  |  "
        f"Annual {metrics['annual_return']:+.1f}%",
    )

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax2.fill_between(range(len(dd)), dd.values, 0, color=R, alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    style(ax2, f"Drawdown  |  Max {metrics['max_drawdown']:.1f}%")

    # 3. Monthly P&L
    ax3 = fig.add_subplot(gs[1, 1])
    if results["trades"]:
        df_t = pd.DataFrame(results["trades"])
        df_t["exit_dt"] = pd.to_datetime(df_t["exit_dt"])
        monthly = df_t.groupby(df_t["exit_dt"].dt.to_period("M"))["pl"].sum()
        ax3.bar(
            range(len(monthly)),
            monthly.values,
            color=[G if v >= 0 else R for v in monthly.values],
        )
        ax3.axhline(0, color=GR, linewidth=0.8)
        step = max(1, len(monthly) // 6)
        ax3.set_xticks(range(0, len(monthly), step))
        ax3.set_xticklabels(
            [str(p) for p in monthly.index[::step]],
            rotation=45,
            ha="right",
            fontsize=6,
        )
        ax3.set_ylabel("P&L ($)")
    style(ax3, "Monthly P&L")

    # 4. P&L by strategy type (NEW)
    ax4 = fig.add_subplot(gs[1, 2])
    if metrics.get("type_stats") is not None:
        ts = metrics["type_stats"]
        clrs = [TYPE_COLORS.get(t, GR) for t in ts.index]
        ax4.bar(ts.index, ts["total_pl"].values, color=clrs)
        ax4.axhline(0, color=GR, linewidth=0.8)
        ax4.set_ylabel("Total P&L ($)")
        ax4.tick_params(axis="x", labelsize=8)
    style(ax4, "P&L by Strategy Type")

    # 5. Trade distribution
    ax5 = fig.add_subplot(gs[2, 0])
    if results["trades"]:
        df_t = pd.DataFrame(results["trades"])
        wins = df_t[df_t["pl_pct"] > 0]["pl_pct"]
        loss = df_t[df_t["pl_pct"] < 0]["pl_pct"]
        if len(wins):
            ax5.hist(wins, bins=20, color=G, alpha=0.7, label=f"Winners ({len(wins)})")
        if len(loss):
            ax5.hist(loss, bins=20, color=R, alpha=0.7, label=f"Losers ({len(loss)})")
        ax5.axvline(0, color=GR, linewidth=1)
        ax5.set_xlabel("Return %")
        ax5.legend(facecolor="#1a1a2e", labelcolor=TXT, fontsize=7)
    style(ax5, "Trade Distribution")

    # 6. Per-ticker P&L colored by type
    ax6 = fig.add_subplot(gs[2, 1:])
    ts = metrics["ticker_stats"].sort_values("total_pl")
    type_col = [
        TYPE_COLORS.get(ts.loc[t, "type"], GR) if ts.loc[t, "total_pl"] > 0 else R
        for t in ts.index
    ]
    ax6.barh(
        [f"{t}" for t in ts.index],
        ts["total_pl"].values,
        color=type_col,
    )
    ax6.axvline(0, color=GR, linewidth=0.8)
    ax6.set_xlabel("Total P&L ($)")
    ax6.tick_params(axis="y", labelsize=7)
    # Legend for type colors
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=TYPE_COLORS["momentum"], label="🚀 Momentum"),
        Patch(facecolor=TYPE_COLORS["quality"], label="📈 Quality"),
        Patch(facecolor=TYPE_COLORS["index"], label="📊 Index"),
    ]
    ax6.legend(
        handles=legend_elements,
        facecolor="#1a1a2e",
        labelcolor=TXT,
        fontsize=7,
        loc="lower right",
    )
    style(ax6, "P&L by Ticker  (color = strategy type)")

    plt.suptitle(
        f"Backtest v9 {results['start_date']} → {results['end_date']}  |  "
        f"Auto-Classified Strategy",
        color=TXT,
        fontsize=11,
        y=1.01,
    )

    fname = (
        f"backtest_v9_" f"{results['start_date'][:4]}_" f"{results['end_date'][:4]}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Chart saved: {fname}")
    return fname


# ══════════════════════════════════════════════════════════════════
# REGIME BREAKDOWN
# ══════════════════════════════════════════════════════════════════


def run_regime_breakdown(results: dict):
    periods = {
        "COVID Crash+Recovery (2020)": ("2020-01-01", "2020-12-31"),
        "Bull Market (2021)": ("2021-01-01", "2021-12-31"),
        "Bear Market (2022)": ("2022-01-01", "2022-12-31"),
        "AI Boom (2023)": ("2023-01-01", "2023-12-31"),
        "Election+Cuts (2024)": ("2024-01-01", "2024-12-31"),
    }
    tickers = list(results["all_data"].keys())

    print(f"\n{'═'*65}")
    print(f"  📅 PERFORMANCE BY MARKET REGIME")
    print(f"{'═'*65}")
    print(
        f"  {'Period':<30} {'Return':>8} {'Sharpe':>7} " f"{'MaxDD':>7} {'Trades':>7}"
    )
    print(f"  {'─'*63}")

    for name, (start, end) in periods.items():
        try:
            time.sleep(3)
            r = run_backtest(tickers, start, end, verbose=False, show_classify=False)
            if not r or not r["trades"]:
                print(f"  ⚪ {name:<30} no trades")
                continue
            m = analyze_results(r)
            if not m:
                continue
            icon = "✅" if m["annual_return"] > 0 else "❌"
            print(
                f"  {icon}{name:<29} "
                f"{m['annual_return']:>+7.1f}% "
                f"{m['sharpe']:>6.2f}  "
                f"{m['max_drawdown']:>5.1f}%  "
                f"{m['total_trades']:>6}"
            )
        except Exception as e:
            print(f"  ❌ {name:<30} error: {e}")

    print(f"{'═'*65}\n")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Claude Trading Bot — Backtester v9 (Auto-Classifier)"
    )
    parser.add_argument("--ticker", type=str, help="Single ticker e.g. NVDA")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument(
        "--end", type=str, default=datetime.today().strftime("%Y-%m-%d")
    )
    parser.add_argument("--quick", action="store_true", help="2024 only")
    parser.add_argument("--quiet", action="store_true", help="No trade logs")
    parser.add_argument("--regimes", action="store_true", help="Regime breakdown")
    parser.add_argument(
        "--classify", action="store_true", help="Show classification only, no backtest"
    )
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else DEFAULT_WATCHLIST
    start_date = "2024-01-01" if args.quick else args.start
    end_date = args.end
    verbose = not args.quiet

    print(f"\n🤖 Claude Trading Bot — Backtester v9")
    print(f"   Auto-classifies each ticker into:")
    print(f"   🚀 Momentum → buy breakouts (NVDA, TSLA type)")
    print(f"   📈 Quality  → buy RSI dips  (AAPL, MSFT type)")
    print(f"   📊 Index    → buy dips, wider stops (SPY, QQQ type)")

    # Classify-only mode
    if args.classify:
        print(f"\n📥 Downloading data for classification...")
        all_data = {}
        for ticker in tickers:
            df = download_ticker(ticker, start_date, end_date)
            if df:
                all_data[ticker] = df
                print(f"  ✅ {ticker}")
        if all_data:
            classify_all_tickers(all_data, verbose=True)
        return

    results = run_backtest(
        tickers,
        start_date,
        end_date,
        verbose=verbose,
        show_classify=True,
    )
    if not results:
        return

    metrics = analyze_results(results)
    if not metrics:
        return

    print_report(results, metrics)
    save_charts(results, metrics)

    if args.regimes:
        run_regime_breakdown(results)

    print("✅ Done.")


if __name__ == "__main__":
    main()
