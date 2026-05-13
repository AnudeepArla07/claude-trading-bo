"""
backtest.py  (v10 — Claude-aligned)
====================================
Backtest with Claude as decision maker in both live and simulation.

HOW THIS MIRRORS THE LIVE BOT:
  Live bot flow:
    data_feed (signal pre-filter, 4+ signals)
    → Claude BUY/HOLD decision (confidence ≥ 0.65)
    → Risk check
    → Execute with ATR-based bracket orders

  Backtest flow (this file):
    generate_signals (same indicators as data_feed via indicators.py)
    → _claude_proxy (confidence = signal_count/7, same threshold 0.65)
    → Portfolio (same ATR stops, same sizing)

  Both use MIN_CONFIDENCE = 0.65 as the gate.
  Both use ATR_PARAMS from config.py for stops/targets.
  Both use the same ticker classification logic.

Claude proxy mapping:
  7/7 signals → confidence 1.00 → STRONG_BUY → enters
  6/7 signals → confidence 0.86 → BUY        → enters
  5/7 signals → confidence 0.71 → BUY        → enters
  4/7 signals → confidence 0.57 → HOLD       → filtered out
  3/7 signals → confidence 0.43 → HOLD       → filtered out

Usage:
  python backtest.py                   full 6-year backtest
  python backtest.py --ticker NVDA     single ticker
  python backtest.py --quick           2024 only
  python backtest.py --quiet           no trade logs
  python backtest.py --quiet --regimes regime breakdown
  python backtest.py --classify        show classification only
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

# ── Shared config (same as live bot) ─────────────────────────────
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from config import Config

    _cfg = Config()
    MIN_CONFIDENCE = _cfg.MIN_CONFIDENCE
    ATR_PARAMS = _cfg.ATR_PARAMS
    RISK_PER_TRADE = _cfg.RISK_PER_TRADE
    MAX_POSITION_PCT = _cfg.MAX_POSITION_PCT
    MAX_POSITIONS = _cfg.MAX_POSITIONS
    SPY_BULL_MIN = _cfg.SPY_BULL_MIN
    PARTIAL_PROFIT_MOM = _cfg.PARTIAL_PROFIT_MOM
    PARTIAL_PROFIT_STD = _cfg.PARTIAL_PROFIT_STD
    DEFAULT_WATCHLIST = list(_cfg.WATCHLIST)
except Exception:
    # Fallback if config.py not found
    MIN_CONFIDENCE = 0.65
    ATR_PARAMS = {
        "momentum": {"stop": 2.5, "target": 12.0},
        "quality": {"stop": 3.5, "target": 10.0},
        "index": {"stop": 4.0, "target": 8.0},
    }
    RISK_PER_TRADE = 0.02
    MAX_POSITION_PCT = 0.20
    MAX_POSITIONS = 3
    SPY_BULL_MIN = 1.0
    PARTIAL_PROFIT_MOM = 0.06
    PARTIAL_PROFIT_STD = 0.04
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
        "SPY",
    ]

COMMISSION = 0.001
MIN_HOLD_DAYS = 3
MIN_EXIT_HOLD = {"momentum": 999, "quality": 8, "index": 8}

# Classifier thresholds
MOMENTUM_BETA_MIN = 1.3
MOMENTUM_RSI_OB_MIN = 0.25
INDEX_VOL_MAX = 0.18

RSI_PERIOD = 14
BB_PERIOD = 20

try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════
# CLAUDE PROXY — mirrors claude_brain.py decision logic
# ══════════════════════════════════════════════════════════════════


def claude_proxy(
    signal_count: int, ticker_type: str, regime: float, bias: str
) -> tuple:
    """
    Simulates Claude's BUY/HOLD decision without calling the API.

    Uses same threshold as live claude_brain.py:
      confidence = signal_count / 7
      entry if confidence >= MIN_CONFIDENCE and regime is BULL+

    Returns: (action, confidence, reason)
    """
    confidence = signal_count / 7.0

    # Regime gate — mirrors live bot's SPY regime filter
    if regime < SPY_BULL_MIN:
        return "hold", confidence, f"Regime {regime:.0f} < SPY_BULL_MIN"

    # Momentum: needs STRONG_BULL (regime == 2)
    if ticker_type == "momentum" and regime < 2.0:
        return "hold", confidence, "Momentum requires STRONG_BULL regime"

    # Confidence gate — same threshold as live MIN_CONFIDENCE
    if confidence < MIN_CONFIDENCE:
        return "hold", confidence, f"Confidence {confidence:.2f} < {MIN_CONFIDENCE}"

    # Bias must be bullish
    if bias not in ("BUY", "STRONG_BUY"):
        return "hold", confidence, f"Bias {bias} not bullish"

    action = "buy"
    if confidence >= 0.85:
        reason = f"STRONG_BUY: {signal_count}/7 signals, conf={confidence:.0%}"
    else:
        reason = f"BUY: {signal_count}/7 signals, conf={confidence:.0%}"

    return action, confidence, reason


# ══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════


def download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        try:
            yf.set_tz_cache_location("/tmp/yf_cache")
        except Exception:
            pass
        raw = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
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
# INDICATORS (same implementations as indicators.py)
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


def calc_atr(high, low, close, period: int = 14) -> pd.Series:
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
# SIGNAL GENERATION — same logic as indicators.score_signals()
# ══════════════════════════════════════════════════════════════════


def generate_signals(df: pd.DataFrame, ticker_type: str = "quality") -> pd.DataFrame:
    """
    Generate signals IDENTICAL to indicators.py score_signals().
    Ensures backtest and live use the exact same entry logic.
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

    df["rsi"] = rsi
    df["macd"] = macd
    df["hist"] = h
    df["atr"] = atr
    df["regime"] = regime
    df["vol_r"] = vol_ratio
    df["ema9"] = ema9
    df["ema21"] = ema21
    df["pct_b"] = pct_b
    df["mom5d"] = mom5d

    if ticker_type == "momentum":
        # Same as indicators.score_signals() momentum branch
        strong_bull = regime == 2
        bull_3d = (
            strong_bull
            & strong_bull.shift(1).fillna(False)
            & strong_bull.shift(2).fillna(False)
        )
        near_high = (closes >= closes.rolling(20).max().shift(1) * 0.99).fillna(False)
        macd_bull = ((macd > sig) & (h > 0)).fillna(False)
        macd_accel = (h > h.shift(1)).fillna(False) & (h.shift(1) > h.shift(2)).fillna(
            False
        )
        vol_surge = vol_ratio > 2.0
        fast_mom = mom5d > 5
        above_ema9 = closes > ema9

        df["signal_count"] = (
            near_high.astype(int)
            + bull_3d.astype(int)
            + macd_bull.astype(int)
            + macd_accel.astype(int)
            + vol_surge.astype(int)
            + fast_mom.astype(int)
            + above_ema9.astype(int)
        )
        df["bias"] = "NEUTRAL"
        df.loc[bull_3d & (df["signal_count"] >= 4), "bias"] = "STRONG_BUY"
        df.loc[
            bull_3d & (df["signal_count"] >= 3) & (df["bias"] == "NEUTRAL"), "bias"
        ] = "BUY"

        # Pre-filter for claude_proxy: signal_count >= 4
        df["pre_filter"] = bull_3d & (df["signal_count"] >= 4) & atr.notna()

        # Exit: BEAR or sustained below EMA21
        below_ema21_2d = (closes < ema21) & (closes.shift(1) < ema21.shift(1)).fillna(
            False
        )
        df["exit"] = ((regime <= -1) | below_ema21_2d.fillna(False)).fillna(False)

    elif ticker_type == "index":
        s1 = (regime >= 1).astype(int)
        s2 = (rsi < 42).astype(int)
        s3 = (closes > ema21).astype(int)
        s4 = (macd > sig).astype(int)
        s5 = (vol_ratio > 1.0).astype(int)
        s6 = (mom5d > -1).astype(int)
        s7 = (closes > ema50).astype(int)
        df["signal_count"] = s1 + s2 + s3 + s4 + s5 + s6 + s7
        df["bias"] = "NEUTRAL"
        df.loc[(regime >= 1) & (df["signal_count"] >= 5), "bias"] = "BUY"
        df.loc[(regime >= 2) & (df["signal_count"] >= 6), "bias"] = "STRONG_BUY"
        df["pre_filter"] = (regime >= 1) & (df["signal_count"] >= 4) & atr.notna()
        df["exit"] = ((rsi > 70) | (regime <= -1)).fillna(False)

    else:  # quality
        s1 = (regime >= 1).astype(int)
        s2 = (rsi < 50).astype(int)
        s3 = (closes > ema21).astype(int)
        s4 = (macd > sig).astype(int)
        s5 = (vol_ratio > 1.2).astype(int)
        s6 = (mom5d > 0).astype(int)
        s7 = (pct_b < 0.35).astype(int)
        df["signal_count"] = s1 + s2 + s3 + s4 + s5 + s6 + s7
        df["bias"] = "NEUTRAL"
        df.loc[(regime >= 1) & (df["signal_count"] >= 5), "bias"] = "BUY"
        df.loc[(regime >= 2) & (df["signal_count"] >= 6), "bias"] = "STRONG_BUY"
        df["pre_filter"] = (regime >= 1) & (df["signal_count"] >= 4) & atr.notna()
        df["exit"] = ((rsi > 72) | (regime <= -1)).fillna(False)

    return df


# ══════════════════════════════════════════════════════════════════
# AUTO-CLASSIFIER (same as indicators.classify_ticker_live)
# ══════════════════════════════════════════════════════════════════


def _calc_beta(df, spy_df, window=252):
    if spy_df is None or len(spy_df) < 60:
        return 1.0
    ret = df["Close"].tail(window).pct_change().dropna()
    spy_ret = spy_df["Close"].tail(window).pct_change().dropna()
    common = ret.index.intersection(spy_ret.index)
    if len(common) < 50:
        return 1.0
    r, sr = ret.loc[common], spy_ret.loc[common]
    cov = np.cov(r, sr)[0][1]
    spy_var = sr.var()
    return cov / spy_var if spy_var > 0 else 1.0


def classify_ticker(df, spy_df=None, window=252):
    closes = df["Close"].tail(window)
    rsi_s = calc_rsi(closes)
    regime_s = calc_regime(closes)
    ret = closes.pct_change().dropna()
    ann_vol = float(ret.std() * np.sqrt(252))
    beta = _calc_beta(df, spy_df, window)
    rsi_ob_frac = float((rsi_s > 65).mean())
    rsi_range = float(rsi_s.max() - rsi_s.min())
    rsi_crosses = int(((rsi_s > 50) != (rsi_s > 50).shift(1)).sum())
    avg_mom20 = float(closes.pct_change(20).dropna().mean() * 100)
    sb_frac = float((regime_s == 2).mean())

    mom = qty = idx = 0
    if beta > MOMENTUM_BETA_MIN:
        mom += 3
    elif beta > 1.1:
        mom += 1
    if rsi_ob_frac > MOMENTUM_RSI_OB_MIN:
        mom += 2
    elif rsi_ob_frac > 0.15:
        mom += 1
    if avg_mom20 > 3:
        mom += 2
    elif avg_mom20 > 1.5:
        mom += 1
    if ann_vol > 0.45:
        mom += 2
    elif ann_vol > 0.30:
        mom += 1
    if sb_frac > 0.40:
        mom += 1

    if ann_vol < INDEX_VOL_MAX:
        idx += 3
    elif ann_vol < 0.22:
        idx += 1
    if beta < 0.8:
        idx += 3
    elif beta < 1.0:
        idx += 1
    if rsi_range < 45:
        idx += 2

    if 0.8 <= beta <= MOMENTUM_BETA_MIN:
        qty += 2
    if rsi_crosses > 30:
        qty += 2
    if 0.20 < ann_vol < 0.40:
        qty += 1
    if 0.15 < rsi_ob_frac < 0.30:
        qty += 1

    scores = {"momentum": mom, "quality": qty, "index": idx}
    return max(scores, key=scores.get)


def classify_all(all_data, verbose=True):
    # Fix: explicit None check (avoids pandas truth ambiguity)
    spy_df = all_data.get("SPY")
    if spy_df is None:
        spy_df = all_data.get("__SPY_REF__")

    if verbose:
        print(f"\n{'═'*65}")
        print(f"  🔍 AUTO-CLASSIFICATION (same as live bot)")
        print(f"{'═'*65}")
        print(
            f"  {'Ticker':<8} {'Type':<12} {'Beta':>6} " f"{'Vol%':>6} {'RSI_OB%':>8}"
        )
        print(f"  {'─'*50}")

    out = {}
    for ticker, df in all_data.items():
        if ticker.startswith("__"):
            continue
        t_type = classify_ticker(df, spy_df)
        out[ticker] = t_type
        if verbose:
            closes = df["Close"].tail(252)
            ret = closes.pct_change().dropna()
            ann_vol = float(ret.std() * np.sqrt(252))
            beta = _calc_beta(df, spy_df)
            rsi_ob = float((calc_rsi(closes) > 65).mean())
            icon = {"momentum": "🚀", "quality": "📈", "index": "📊"}[t_type]
            print(
                f"  {icon}{ticker:<7} {t_type:<12} "
                f"{beta:>6.2f} {ann_vol*100:>5.0f}%  {rsi_ob*100:>6.0f}%"
            )

    if verbose:
        from collections import Counter

        cnt = Counter(out.values())
        icons = {"momentum": "🚀", "quality": "📈", "index": "📊"}
        print(f"\n  Summary:")
        for t in ["momentum", "quality", "index"]:
            if cnt.get(t, 0) > 0:
                tks = [k for k, v in out.items() if v == t]
                print(f"  {icons[t]} {t:<12}: {cnt[t]} — {tks}")
        print(f"{'═'*65}\n")

    return out


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO — same sizing and stop logic as live bot
# ══════════════════════════════════════════════════════════════════


class Portfolio:

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
        self,
        ticker: str,
        price: float,
        atr: float,
        date,
        ticker_type: str = "quality",
        confidence: float = 0.7,
    ) -> bool:
        if ticker in self.positions:
            return False
        params = ATR_PARAMS.get(ticker_type, ATR_PARAMS["quality"])
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
            "confidence": confidence,
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
            "confidence": pos.get("confidence", 0),
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
            stop_mult = pos.get("stop_mult", 3.5)
            ttype = pos.get("ticker_type", "quality")

            # Partial profit (matches live bot thresholds)
            partial_pct = (
                PARTIAL_PROFIT_MOM if ttype == "momentum" else PARTIAL_PROFIT_STD
            )
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
                        "confidence": pos.get("confidence", 0),
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
    tickers, start_date, end_date, init_cash=100_000, verbose=True, show_classify=True
):
    print(f"\n{'═'*65}")
    print(f"  📊 BACKTEST ENGINE  (Claude-aligned)")
    print(f"  Period   : {start_date} → {end_date}")
    print(f"  Tickers  : {len(tickers)}")
    print(f"  Capital  : ${init_cash:,.0f}")
    print(f"  Claude proxy: confidence = signal_count/7 ≥ {MIN_CONFIDENCE}")
    print(f"{'═'*65}")

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

    # SPY reference for beta + regime filter
    spy_trading = all_data.get("SPY")
    if "SPY" not in all_data:
        spy_ref = download_ticker("SPY", start_date, end_date)
        if spy_ref is not None:
            all_data["__SPY_REF__"] = spy_ref
            spy_trading = spy_ref
            print(f"  ✅ SPY (reference): {len(spy_ref)} bars")

    # SPY regime series for Lever 5
    spy_regime_df = None
    if spy_trading is not None:
        spy_regime_df = spy_trading.copy()
        spy_regime_df["regime"] = calc_regime(spy_trading["Close"])

    # Auto-classify
    classifications = classify_all(all_data, verbose=show_classify)
    all_data.pop("__SPY_REF__", None)

    # Generate signals per type
    print("📐 Generating signals (Claude pre-filter logic)...")
    for ticker, df in all_data.items():
        t_type = classifications.get(ticker, "quality")
        all_data[ticker] = generate_signals(df, t_type)
        pf_count = int(all_data[ticker]["pre_filter"].sum())
        print(
            f"  {'🚀' if t_type=='momentum' else '📈' if t_type=='quality' else '📊'} "
            f"{ticker:<6} [{t_type:<9}]: pre-filter signals={pf_count}"
        )

    total_pf = sum(int(df["pre_filter"].sum()) for df in all_data.values())
    if total_pf == 0:
        print("\n⚠️  Zero pre-filter signals.")
        return {}

    all_dates = sorted(set.union(*[set(df.index) for df in all_data.values()]))
    print(
        f"\n🔄 Simulating {len(all_dates)} days " f"({total_pf} pre-filtered setups)..."
    )

    port = Portfolio(init_cash)
    bear_days = 0
    claude_buys = 0
    claude_hold = 0

    for date in all_dates:
        prices = {
            t: float(df.loc[date, "Close"])
            for t, df in all_data.items()
            if date in df.index
        }

        # Lever 5: SPY regime filter
        spy_regime = 0.0
        if spy_regime_df is not None and date in spy_regime_df.index:
            spy_regime = float(spy_regime_df.loc[date, "regime"])
        if spy_regime < SPY_BULL_MIN:
            bear_days += 1
            for ticker, price, reason in port.check_stops(prices, date):
                trade = port.sell(ticker, price, date, reason)
                if trade and verbose:
                    icon = "🎯" if reason == "profit_target" else "🛑"
                    print(
                        f"  {icon} {str(date.date())} {ticker:<6} "
                        f"{reason:<14} P&L: ${trade['pl']:>+8,.0f} "
                        f"({trade['pl_pct']:>+5.1f}%)"
                    )
            port.update_equity(prices)
            continue

        # Stops + targets
        for ticker, price, reason in port.check_stops(prices, date):
            trade = port.sell(ticker, price, date, reason)
            if trade and verbose:
                icon = "🎯" if reason == "profit_target" else "🛑"
                print(
                    f"  {icon} {str(date.date())} {ticker:<6} "
                    f"[{trade['ticker_type']:<9}] {reason:<14} "
                    f"P&L: ${trade['pl']:>+8,.0f} ({trade['pl_pct']:>+5.1f}%)"
                )

        # Signal exits
        for ticker in list(port.positions.keys()):
            if date not in all_data[ticker].index:
                continue
            pos = port.positions[ticker]
            hold = (date - pos["date"]).days
            ttype = pos.get("ticker_type", "quality")
            min_ex = MIN_EXIT_HOLD.get(ttype, 8)
            if hold < min_ex:
                continue
            if bool(all_data[ticker].loc[date, "exit"]):
                price = prices.get(ticker, 0)
                trade = port.sell(ticker, price, date, "signal_exit")
                if trade and verbose:
                    print(
                        f"  📤 {str(date.date())} {ticker:<6} "
                        f"[{trade['ticker_type']:<9}] signal_exit     "
                        f"P&L: ${trade['pl']:>+8,.0f} ({trade['pl_pct']:>+5.1f}%)"
                    )

        # Entries via Claude proxy
        if len(port.positions) < MAX_POSITIONS:
            opps = []
            for ticker, df in all_data.items():
                if ticker in port.positions:
                    continue
                if date not in df.index:
                    continue
                row = df.loc[date]
                if not bool(row["pre_filter"]):
                    continue
                if row["atr"] <= 0 or pd.isna(row["atr"]):
                    continue
                t_type = classifications.get(ticker, "quality")
                regime = float(row["regime"])
                bias = str(row.get("bias", "NEUTRAL"))
                sigs = int(row["signal_count"])

                # ── Claude proxy decision ──────────────────────
                action, conf, reason = claude_proxy(sigs, t_type, regime, bias)
                if action == "buy":
                    opps.append(
                        (
                            ticker,
                            conf,
                            float(prices.get(ticker, 0)),
                            float(row["atr"]),
                            t_type,
                            sigs,
                        )
                    )

            # Take top candidates by confidence
            opps.sort(key=lambda x: x[1], reverse=True)
            for ticker, conf, price, atr, t_type, sigs in opps[:2]:
                if len(port.positions) >= MAX_POSITIONS:
                    break
                if port.buy(ticker, price, atr, date, t_type, conf):
                    claude_buys += 1
                    pos = port.positions[ticker]
                    icon = {"momentum": "🚀", "quality": "📈", "index": "📊"}.get(
                        t_type, "•"
                    )
                    if verbose:
                        print(
                            f"  🟢 {str(date.date())} {ticker:<6} "
                            f"{icon}[{t_type:<9}] conf={conf:.0%} "
                            f"sigs={sigs}/7 "
                            f"BUY x{pos['qty']:<5} @ ${price:.2f}  "
                            f"stop=${pos['stop']:.2f}  "
                            f"target=${pos['target']:.2f}"
                        )
                else:
                    claude_hold += 1

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

    n_days = len(all_dates)
    bull_d = n_days - bear_days
    print(
        f"\n  📊 Regime: bull={bull_d}d ({bull_d/n_days*100:.0f}%)  "
        f"bear/choppy={bear_days}d  [Lever 5 filter]"
    )
    print(f"  🧠 Claude proxy: {claude_buys} buys  {claude_hold} holds filtered")

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
        "bear_days": bear_days,
        "claude_buys": claude_buys,
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
        print("⚠️  No trades.")
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


def print_report(results: dict, m: dict):
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
    print(f"  📊 BACKTEST RESULTS  (Claude-aligned)")
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
        f"  Net Profit       : ${results['final_equity']-results['init_cash']:>+12,.0f}"
    )

    n_days = len(results["all_dates"])
    bull_d = n_days - results.get("bear_days", 0)
    print(f"\n  LEVER SUMMARY")
    print(f"  {'─'*50}")
    print(
        f"  Bull days traded : {bull_d:>8}  ({bull_d/n_days*100:.0f}%)  [L5 regime filter]"
    )
    print(f"  Bear days skipped: {results.get('bear_days',0):>8}")
    print(f"  Claude buys      : {results.get('claude_buys',0):>8}  [proxy decisions]")

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

    if m.get("type_stats") is not None:
        icons = {"momentum": "🚀", "quality": "📈", "index": "📊"}
        print(f"\n  BY STRATEGY TYPE")
        print(f"  {'─'*58}")
        print(
            f"  {'Type':<14} {'Trades':>6} {'Win%':>6} "
            f"{'Avg P&L%':>9} {'Total P&L':>12}"
        )
        print(f"  {'─'*58}")
        for t_type, row in m["type_stats"].iterrows():
            icon = icons.get(t_type, "•")
            pl_icon = "✅" if row["total_pl"] > 0 else "❌"
            print(
                f"  {pl_icon}{icon}{t_type:<12} {row['trades']:>6.0f} "
                f"{row['win_rate']:>5.1f}% {row['avg_pl']:>+8.2f}% "
                f"${row['total_pl']:>10,.0f}"
            )

    print(f"\n  BY TICKER")
    print(f"  {'─'*65}")
    icons = {"momentum": "🚀", "quality": "📈", "index": "📊"}
    for ticker, row in m["ticker_stats"].iterrows():
        pl_icon = "✅" if row["total_pl"] > 0 else "❌"
        t_icon = icons.get(row.get("type", ""), "•")
        print(
            f"  {pl_icon}{t_icon}{ticker:<6} {row.get('type',''):<11} "
            f"{row['trades']:>6.0f} {row['win_rate']:>5.1f}% "
            f"${row['total_pl']:>10,.0f} {row['avg_hold']:>7.1f}d"
        )

    print(f"\n  EXIT REASONS")
    print(f"  {'─'*58}")
    for reason, row in m["reason_stats"].iterrows():
        icon = "✅" if row["avg_pl"] > 0 else "❌"
        print(
            f"  {icon}{reason:<17} {row['count']:>6.0f} "
            f"{row['avg_pl']:>+8.1f}% ${row['total']:>10,.0f}"
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
        issues.append(f"❌ Losing overall")
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
        print("\n  ⚠️  CLOSE — paper trade while tuning.")
    else:
        print("\n  🛑 NOT READY — fix issues first.")
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
    TC = {"momentum": "#ff9900", "quality": "#00aaff", "index": "#aa88ff"}

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

    ax1 = fig.add_subplot(gs[0, :])
    c = G if eq.iloc[-1] >= init else R
    ax1.plot(dates, eq.values, color=c, linewidth=1.5)
    ax1.axhline(init, color=GR, linestyle="--", linewidth=0.8, alpha=0.7, label="Start")
    ax1.fill_between(dates, eq.values, init, alpha=0.1, color=c)
    ax1.set_ylabel("Portfolio ($)")
    ax1.legend(facecolor="#1a1a2e", labelcolor=TXT, fontsize=8)
    style(
        ax1,
        f"Equity Curve  |  Final ${results['final_equity']:,.0f}  "
        f"({metrics['total_return']:+.1f}%)  |  Sharpe {metrics['sharpe']:.2f}  "
        f"|  Annual {metrics['annual_return']:+.1f}%",
    )

    ax2 = fig.add_subplot(gs[1, 0])
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax2.fill_between(range(len(dd)), dd.values, 0, color=R, alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    style(ax2, f"Drawdown  |  Max {metrics['max_drawdown']:.1f}%")

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
            [str(p) for p in monthly.index[::step]], rotation=45, ha="right", fontsize=6
        )
        ax3.set_ylabel("P&L ($)")
    style(ax3, "Monthly P&L")

    ax4 = fig.add_subplot(gs[1, 2])
    if metrics.get("type_stats") is not None:
        ts = metrics["type_stats"]
        ax4.bar(
            ts.index, ts["total_pl"].values, color=[TC.get(t, GR) for t in ts.index]
        )
        ax4.axhline(0, color=GR, linewidth=0.8)
        ax4.set_ylabel("Total P&L ($)")
    style(ax4, "P&L by Strategy Type")

    ax5 = fig.add_subplot(gs[2, 0])
    if results["trades"]:
        df_t = pd.DataFrame(results["trades"])
        wins = df_t[df_t["pl_pct"] > 0]["pl_pct"]
        loss = df_t[df_t["pl_pct"] < 0]["pl_pct"]
        if len(wins):
            ax5.hist(wins, bins=20, color=G, alpha=0.7, label=f"W ({len(wins)})")
        if len(loss):
            ax5.hist(loss, bins=20, color=R, alpha=0.7, label=f"L ({len(loss)})")
        ax5.axvline(0, color=GR, linewidth=1)
        ax5.set_xlabel("Return %")
        ax5.legend(facecolor="#1a1a2e", labelcolor=TXT, fontsize=7)
    style(ax5, "Trade Distribution")

    ax6 = fig.add_subplot(gs[2, 1:])
    ts = metrics["ticker_stats"].sort_values("total_pl")
    clr = [
        TC.get(ts.loc[t, "type"], GR) if ts.loc[t, "total_pl"] > 0 else R
        for t in ts.index
    ]
    ax6.barh([str(t) for t in ts.index], ts["total_pl"].values, color=clr)
    ax6.axvline(0, color=GR, linewidth=0.8)
    ax6.set_xlabel("Total P&L ($)")
    ax6.tick_params(axis="y", labelsize=7)
    from matplotlib.patches import Patch

    ax6.legend(
        handles=[
            Patch(facecolor=TC["momentum"], label="🚀 Momentum"),
            Patch(facecolor=TC["quality"], label="📈 Quality"),
            Patch(facecolor=TC["index"], label="📊 Index"),
            Patch(facecolor=R, label="❌ Loss"),
        ],
        facecolor="#1a1a2e",
        labelcolor=TXT,
        fontsize=7,
        loc="lower right",
    )
    style(ax6, "P&L by Ticker")

    plt.suptitle(
        f"Backtest (Claude-aligned)  {results['start_date']} → {results['end_date']}  |  "
        f"Sharpe {metrics['sharpe']:.2f}  |  Annual {metrics['annual_return']:+.1f}%",
        color=TXT,
        fontsize=11,
        y=1.01,
    )
    fname = f"backtest_{results['start_date'][:4]}_{results['end_date'][:4]}.png"
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
    tickers = [t for t in results["all_data"] if not t.startswith("__")]
    print(f"\n{'═'*65}")
    print(f"  📅 PERFORMANCE BY MARKET REGIME")
    print(f"{'═'*65}")
    print(f"  {'Period':<30} {'Return':>8} {'Sharpe':>7} {'MaxDD':>7} {'Trades':>7}")
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
                f"  {icon}{name:<29} {m['annual_return']:>+7.1f}% "
                f"{m['sharpe']:>6.2f}  {m['max_drawdown']:>5.1f}%  "
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
        description="Claude Trading Bot — Backtest (Claude-aligned)"
    )
    parser.add_argument("--ticker", type=str)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument(
        "--end", type=str, default=datetime.today().strftime("%Y-%m-%d")
    )
    parser.add_argument("--quick", action="store_true", help="2024 only")
    parser.add_argument("--quiet", action="store_true", help="No trade logs")
    parser.add_argument("--regimes", action="store_true")
    parser.add_argument("--classify", action="store_true")
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else DEFAULT_WATCHLIST
    start_date = "2024-01-01" if args.quick else args.start
    end_date = args.end
    verbose = not args.quiet

    print(f"\n🤖 Claude Trading Bot — Backtester (Claude-aligned)")
    print(f"   Decision: claude_proxy (confidence = signal_count/7 ≥ {MIN_CONFIDENCE})")
    print(f"   Stop/target: ATR-based from config.ATR_PARAMS")
    print(f"   Regime filter: SPY ≥ BULL ({SPY_BULL_MIN})")
    print(
        f"   Sizing: {RISK_PER_TRADE*100:.0f}% risk per trade  |  Max {MAX_POSITIONS} positions"
    )

    if args.classify:
        print("\n📥 Downloading for classification...")
        all_data = {}
        for t in tickers:
            df = download_ticker(t, start_date, end_date)
            if df:
                all_data[t] = df
                print(f"  ✅ {t}")
        if "SPY" not in all_data:
            spy = download_ticker("SPY", start_date, end_date)
            if spy:
                all_data["__SPY_REF__"] = spy
        if all_data:
            classify_all(all_data, verbose=True)
        return

    results = run_backtest(tickers, start_date, end_date, verbose=verbose)
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
