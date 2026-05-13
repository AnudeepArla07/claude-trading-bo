"""
indicators.py
=============
Single source of truth for all technical indicators.

Used by BOTH data_feed.py (live trading) and backtest.py.
Both files import from here so indicators are 100% identical —
eliminating the #1 cause of live vs backtest divergence.

Supports two input modes:
  pandas Series  → used by backtest.py (vectorized, full history)
  Python list    → used by data_feed.py (streaming, recent bars only)

All functions accept both and return the same result.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Optional, List

# Type alias — accepts either pandas Series or plain list
Prices = Union[pd.Series, List[float]]


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _to_series(x: Prices) -> pd.Series:
    """Convert list or Series to a clean pandas Series."""
    if isinstance(x, pd.Series):
        return x.reset_index(drop=True).astype(float)
    return pd.Series(x, dtype=float)


# ══════════════════════════════════════════════════════════════════
# CORE INDICATORS
# ══════════════════════════════════════════════════════════════════


def calc_rsi(closes: Prices, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's EMA smoothing.
    Returns full Series aligned to input length.
    First `period` values will be NaN; fillna(50) applied.
    """
    s = _to_series(closes)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calc_rsi_value(closes: Prices, period: int = 14) -> float:
    """Returns the latest RSI value as a float."""
    return float(calc_rsi(closes, period).iloc[-1])


def calc_macd(closes: Prices) -> tuple:
    """
    MACD (12, 26, 9).
    Returns (macd_series, signal_series, histogram_series).
    """
    s = _to_series(closes)
    e12 = s.ewm(span=12, adjust=False).mean()
    e26 = s.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def calc_macd_values(closes: Prices) -> dict:
    """Returns latest MACD values as a dict for live trading."""
    macd, signal, hist = calc_macd(closes)
    m = float(macd.iloc[-1])
    s = float(signal.iloc[-1])
    h = float(hist.iloc[-1])
    return {
        "macd": round(m, 4),
        "signal": round(s, 4),
        "histogram": round(h, 4),
        "bullish": m > s,
    }


def calc_atr(high: Prices, low: Prices, close: Prices, period: int = 14) -> pd.Series:
    """ATR using simple rolling mean of true range."""
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)
    tr = pd.concat(
        [
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def calc_atr_value(high: Prices, low: Prices, close: Prices, period: int = 14) -> float:
    """Returns latest ATR value as a float."""
    result = calc_atr(high, low, close, period).iloc[-1]
    return float(result) if not pd.isna(result) else 0.0


def calc_ema(closes: Prices, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return _to_series(closes).ewm(span=period, adjust=False).mean()


def calc_ema_value(closes: Prices, period: int) -> float:
    """Returns latest EMA value as a float."""
    return float(calc_ema(closes, period).iloc[-1])


def calc_sma(closes: Prices, period: int) -> Optional[float]:
    """Simple Moving Average — returns latest value or None."""
    s = _to_series(closes)
    if len(s) < period:
        return None
    return float(s.iloc[-period:].mean())


def calc_bollinger(closes: Prices, period: int = 20) -> dict:
    """
    Bollinger Bands.
    Returns dict with upper, mid, lower, pct_b, width.
    pct_b: 0 = at lower band, 1 = at upper band.
    """
    s = _to_series(closes)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    price = s.iloc[-1]
    u = float(upper.iloc[-1])
    m = float(mid.iloc[-1])
    lo = float(lower.iloc[-1])
    denom = u - lo
    pct_b = (price - lo) / denom if denom > 0 else 0.5
    return {
        "upper": round(u, 2),
        "mid": round(m, 2),
        "lower": round(lo, 2),
        "pct_b": round(float(pct_b), 4),
        "width": round((u - lo) / m * 100, 2) if m > 0 else 0.0,
    }


def calc_bollinger_pctb(closes: Prices, period: int = 20) -> pd.Series:
    """
    Full pct_b Series for backtest vectorized use.
    0 = at lower band, 1 = at upper band.
    """
    s = _to_series(closes)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    lower = mid - 2 * std
    upper = mid + 2 * std
    denom = (upper - lower).replace(0, np.nan)
    return ((s - lower) / denom).fillna(0.5)


def calc_vol_ratio(volume: Prices, period: int = 20) -> pd.Series:
    """Volume relative to N-bar average. Returns full Series."""
    v = _to_series(volume)
    avg = v.rolling(period).mean()
    return (v / avg.replace(0, np.nan)).fillna(1.0)


def calc_vol_ratio_value(volume: Prices, period: int = 20) -> float:
    """Latest volume ratio as a float."""
    return float(calc_vol_ratio(volume, period).iloc[-1])


def calc_momentum(closes: Prices, period: int) -> Optional[float]:
    """
    Price momentum over N bars as percentage.
    Returns None if insufficient data.
    """
    s = _to_series(closes)
    if len(s) < period + 1:
        return None
    past = float(s.iloc[-(period + 1)])
    if past == 0:
        return None
    return round((float(s.iloc[-1]) - past) / past * 100, 2)


def calc_regime(closes: Prices) -> pd.Series:
    """
     Market regime series for backtest vectorized use.
     2 = STRONG_BULL   price > ema9 > ema21 > ema50
     1 = BULL          price > ema21 > ema50
     0 = CHOPPY
    -1 = BEAR          price < ema9 < ema21
    -2 = STRONG_BEAR   price < ema50 and bear
    """
    s = _to_series(closes)
    e9 = calc_ema(s, 9)
    e21 = calc_ema(s, 21)
    e50 = calc_ema(s, 50)
    r = pd.Series(0.0, index=s.index)
    r[(s > e9) & (e9 > e21) & (e21 > e50)] = 2.0
    r[(s > e21) & (e21 > e50) & (r != 2)] = 1.0
    r[(s < e9) & (e9 < e21)] = -1.0
    r[(s < e50) & (r == -1)] = -2.0
    return r


def calc_regime_value(closes: Prices) -> float:
    """
    Current market regime as a single float.
    Used by data_feed.py for live regime detection.
    """
    s = _to_series(closes)
    if len(s) < 50:
        return 0.0
    e9 = calc_ema_value(s, 9)
    e21 = calc_ema_value(s, 21)
    e50 = calc_ema_value(s, 50)
    p = float(s.iloc[-1])

    if p > e9 > e21 > e50:
        return 2.0  # STRONG_BULL
    if p > e21 > e50:
        return 1.0  # BULL
    if p < e9 < e21:
        return -2.0 if p < e50 else -1.0  # STRONG_BEAR / BEAR
    return 0.0  # CHOPPY


def regime_str(value: float) -> str:
    """Convert numeric regime to string label."""
    mapping = {
        2.0: "STRONG_BULL",
        1.0: "BULL",
        0.0: "CHOPPY",
        -1.0: "BEAR",
        -2.0: "STRONG_BEAR",
    }
    return mapping.get(value, "CHOPPY")


# ══════════════════════════════════════════════════════════════════
# SIGNAL SCORING
# Matches backtest.py generate_signals() EXACTLY.
# This is the critical function that keeps live and backtest aligned.
# ══════════════════════════════════════════════════════════════════


def score_signals(
    closes: Prices,
    highs: Prices,
    lows: Prices,
    volumes: Prices,
    ticker_type: str = "quality",
) -> dict:
    """
    Compute signal score for a ticker — identical logic to backtest.py.

    Returns dict with:
      signal_count  int     0-7, higher = stronger setup
      entry         bool    True if entry conditions met
      exit          bool    True if exit conditions met
      regime        float   numeric regime value
      regime_str    str     BULL / BEAR / CHOPPY etc.
      rsi           float
      macd          dict    {macd, signal, histogram, bullish}
      atr           float
      vol_ratio     float
      mom5d         float   5-day momentum %
      ema21         float
      pct_b         float   Bollinger pct_b
      signals       dict    legacy format for claude_brain.py
    """
    s = _to_series(closes)
    h = _to_series(highs)
    lo = _to_series(lows)
    v = _to_series(volumes)

    # ── Compute all indicators ─────────────────────────────────
    rsi_series = calc_rsi(s)
    macd_s, sig_s, hist_s = calc_macd(s)
    atr_series = calc_atr(h, lo, s)
    regime_series = calc_regime(s)
    vol_r_series = calc_vol_ratio(v)
    ema9_series = calc_ema(s, 9)
    ema21_series = calc_ema(s, 21)
    ema50_series = calc_ema(s, 50)
    pct_b_series = calc_bollinger_pctb(s)
    mom5d_val = calc_momentum(s, 5) or 0.0
    mom20d_val = calc_momentum(s, 20) or 0.0

    # Latest values
    rsi_val = float(rsi_series.iloc[-1])
    macd_val = float(macd_s.iloc[-1])
    sig_val = float(sig_s.iloc[-1])
    hist_val = float(hist_s.iloc[-1])
    hist_prev = float(hist_s.iloc[-2]) if len(hist_s) >= 2 else 0.0
    atr_val = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 0.0
    regime_val = float(regime_series.iloc[-1])
    vol_r_val = float(vol_r_series.iloc[-1])
    ema9_val = float(ema9_series.iloc[-1])
    ema21_val = float(ema21_series.iloc[-1])
    ema50_val = float(ema50_series.iloc[-1])
    pct_b_val = float(pct_b_series.iloc[-1])
    price = float(s.iloc[-1])

    # ── Signal scoring — matches backtest generate_signals() ───
    if ticker_type == "momentum":
        # STRONG_BULL sustained for 3 consecutive bars
        strong_bull_now = regime_val == 2.0
        strong_bull_prev1 = (
            float(regime_series.iloc[-2]) == 2.0 if len(regime_series) >= 2 else False
        )
        strong_bull_prev2 = (
            float(regime_series.iloc[-3]) == 2.0 if len(regime_series) >= 3 else False
        )
        bull_3d = strong_bull_now and strong_bull_prev1 and strong_bull_prev2

        high_20d_prev = float(h.iloc[-21:-1].max()) if len(h) >= 21 else float(h.max())
        near_high = price >= high_20d_prev * 0.99

        s1 = int(near_high)
        s2 = int(bull_3d)
        s3 = int(macd_val > sig_val and hist_val > 0)
        s4 = int(hist_val > hist_prev)
        s5 = int(vol_r_val > 2.0)
        s6 = int(mom5d_val > 5)
        s7 = int(price > ema9_val)

        signal_count = s1 + s2 + s3 + s4 + s5 + s6 + s7
        entry_min = 5
        entry = bull_3d and signal_count >= entry_min and atr_val > 0

        # Exit: confirmed BEAR or 2-bar close below EMA21
        ema21_prev = (
            float(ema21_series.iloc[-2]) if len(ema21_series) >= 2 else ema21_val
        )
        price_prev = float(s.iloc[-2]) if len(s) >= 2 else price
        below_ema21_2d = (price < ema21_val) and (price_prev < ema21_prev)
        exit_ = (regime_val <= -1) or below_ema21_2d

    elif ticker_type == "index":
        s1 = int(regime_val >= 1)
        s2 = int(rsi_val < 42)
        s3 = int(price > ema21_val)
        s4 = int(macd_val > sig_val)
        s5 = int(vol_r_val > 1.0)
        s6 = int(mom5d_val > -1)
        s7 = int(price > ema50_val)

        signal_count = s1 + s2 + s3 + s4 + s5 + s6 + s7
        entry_min = 5
        entry = regime_val >= 1 and signal_count >= entry_min and atr_val > 0
        exit_ = (rsi_val > 70) or (regime_val <= -1)

    else:
        # Quality (default)
        s1 = int(regime_val >= 1)
        s2 = int(rsi_val < 50)
        s3 = int(price > ema21_val)
        s4 = int(macd_val > sig_val)
        s5 = int(vol_r_val > 1.2)
        s6 = int(mom5d_val > 0)
        s7 = int(pct_b_val < 0.35)

        signal_count = s1 + s2 + s3 + s4 + s5 + s6 + s7
        entry_min = 5
        entry = regime_val >= 1 and signal_count >= entry_min and atr_val > 0
        exit_ = (rsi_val > 72) or (regime_val <= -1)

    # ── Legacy signals dict (for claude_brain.py prompt) ───────
    buys, sells = [], []
    r_str = regime_str(regime_val)

    if rsi_val < 35:
        buys.append(f"RSI oversold ({rsi_val:.0f})")
    elif rsi_val < 50:
        buys.append(f"RSI low ({rsi_val:.0f})")
    if rsi_val > 72:
        sells.append(f"RSI overbought ({rsi_val:.0f})")
    elif rsi_val > 60:
        sells.append(f"RSI elevated ({rsi_val:.0f})")

    if macd_val > sig_val and hist_val > 0:
        buys.append("MACD bullish")
    elif macd_val < sig_val:
        sells.append("MACD bearish")

    if pct_b_val < 0.1:
        buys.append(f"Lower BB ({pct_b_val:.2f})")
    elif pct_b_val > 0.9:
        sells.append(f"Upper BB ({pct_b_val:.2f})")

    if vol_r_val > 2.0:
        buys.append(f"Volume {vol_r_val:.1f}x avg")
    if r_str in ("STRONG_BULL", "BULL"):
        buys.append(f"Uptrend ({r_str})")
    elif r_str in ("STRONG_BEAR", "BEAR"):
        sells.append(f"Downtrend ({r_str})")
    if mom5d_val > 5:
        buys.append(f"5d mom +{mom5d_val:.1f}%")
    elif mom5d_val < -5:
        sells.append(f"5d mom {mom5d_val:.1f}%")

    score = len(buys) - len(sells)
    if score >= 3:
        bias = "STRONG_BUY"
    elif score >= 1:
        bias = "BUY"
    elif score <= -3:
        bias = "STRONG_SELL"
    elif score <= -1:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    return {
        # Backtest-aligned
        "signal_count": signal_count,
        "entry": entry,
        "exit": exit_,
        "regime": regime_val,
        "regime_str": r_str,
        # Raw indicators
        "rsi": round(rsi_val, 1),
        "atr": round(atr_val, 3),
        "vol_ratio": round(vol_r_val, 2),
        "mom5d": round(mom5d_val, 2),
        "mom20d": round(mom20d_val, 2),
        "ema9": round(ema9_val, 2),
        "ema21": round(ema21_val, 2),
        "ema50": round(ema50_val, 2),
        "pct_b": round(pct_b_val, 4),
        "price": round(price, 2),
        "macd": {
            "macd": round(macd_val, 4),
            "signal": round(sig_val, 4),
            "histogram": round(hist_val, 4),
            "bullish": macd_val > sig_val,
        },
        # Legacy format for claude_brain.py
        "signals": {
            "bias": bias,
            "score": score,
            "buy_signals": buys,
            "sell_signals": sells,
        },
    }


# ══════════════════════════════════════════════════════════════════
# CLASSIFIER  (mirrors backtest.py classify_ticker)
# ══════════════════════════════════════════════════════════════════


def classify_ticker_live(
    closes: Prices,
    highs: Prices,
    lows: Prices,
    volumes: Prices,
    spy_closes: Optional[Prices] = None,
    window: int = 252,
) -> str:
    """
    Auto-classify a ticker as momentum / quality / index.
    Same logic as backtest.py classify_ticker().
    Used by bot.py at startup to set strategy type per ticker.
    """
    s = _to_series(closes).tail(window)
    ret = s.pct_change().dropna()
    ann_vol = float(ret.std() * np.sqrt(252))

    # Beta
    beta = 1.0
    if spy_closes is not None:
        spy = _to_series(spy_closes).tail(window)
        spy_ret = spy.pct_change().dropna()
        common = min(len(ret), len(spy_ret))
        if common >= 50:
            r = ret.iloc[-common:].values
            sr = spy_ret.iloc[-common:].values
            cov = float(np.cov(r, sr)[0][1])
            spy_var = float(np.var(sr))
            beta = cov / spy_var if spy_var > 0 else 1.0

    rsi_vals = calc_rsi(s)
    rsi_ob_fr = float((rsi_vals > 65).mean())
    rsi_range = float(rsi_vals.max() - rsi_vals.min())
    rsi_cross = int(((rsi_vals > 50) != (rsi_vals > 50).shift(1)).sum())
    avg_mom20 = float(s.pct_change(20).dropna().mean() * 100)
    regime_s = calc_regime(s)
    sb_frac = float((regime_s == 2).mean())

    mom_score = qty_score = idx_score = 0

    if beta > 1.3:
        mom_score += 3
    elif beta > 1.1:
        mom_score += 1
    if rsi_ob_fr > 0.25:
        mom_score += 2
    elif rsi_ob_fr > 0.15:
        mom_score += 1
    if avg_mom20 > 3:
        mom_score += 2
    elif avg_mom20 > 1.5:
        mom_score += 1
    if ann_vol > 0.45:
        mom_score += 2
    elif ann_vol > 0.30:
        mom_score += 1
    if sb_frac > 0.40:
        mom_score += 1

    if ann_vol < 0.18:
        idx_score += 3
    elif ann_vol < 0.22:
        idx_score += 1
    if beta < 0.8:
        idx_score += 3
    elif beta < 1.0:
        idx_score += 1
    if rsi_range < 45:
        idx_score += 2

    if 0.8 <= beta <= 1.3:
        qty_score += 2
    if rsi_cross > 30:
        qty_score += 2
    if 0.20 < ann_vol < 0.40:
        qty_score += 1
    if 0.15 < rsi_ob_fr < 0.30:
        qty_score += 1

    scores = {"momentum": mom_score, "quality": qty_score, "index": idx_score}
    return max(scores, key=scores.get)
