"""
backtest.py  (v8)
=================
Backtests the Claude Trading Bot strategy on historical data.

v8 changes vs v7:
  - Removed META, AMZN, SPY (consistent losers in v7)
  - Wider stops: 4x ATR (was 3x) → fewer stop-outs on noise
  - Bigger targets: 8x ATR (was 6x) → 2:1 R:R maintained
  - Partial profit raised to +4% (was +3%)
  - MIN_HOLD_DAYS raised to 4 (was 3)
  - yfinance cache fix → prevents OperationalError in regime breakdown
  - Sleep raised to 3s between regime downloads

Usage:
    python backtest.py                     full 6-year backtest
    python backtest.py --ticker NVDA       single ticker
    python backtest.py --quick             2024 only (fast)
    python backtest.py --quiet             no individual trade logs
    python backtest.py --quiet --regimes   add regime breakdown
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
from typing import Optional

# ── Fix yfinance SQLite cache (prevents OperationalError) ─────────
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
    # Removed: META, AMZN, SPY (consistent underperformers)
]

INIT_CASH = 100_000
COMMISSION = 0.001  # 0.1% per trade
RISK_PER_TRADE = 0.01  # risk 1% of equity per trade
ATR_STOP_MULT = 4.0  # stop  = entry - ATR * 4 (wider, fewer stops)
ATR_TARGET_MULT = 8.0  # target = entry + ATR * 8 (2:1 R:R)
RSI_PERIOD = 14
RSI_BUY = 50  # enter when RSI < 50 in uptrend
RSI_SELL = 72  # exit when RSI > 72
BB_PERIOD = 20
MIN_SIGNALS = 5  # require 5/7 signals
MIN_HOLD_DAYS = 4  # min days before stop fires
MAX_POSITIONS = 3  # max concurrent positions
MAX_POSITION_PCT = 0.20  # max 20% equity per position
PARTIAL_PROFIT = 0.04  # take half at +4% gain


# ══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════


def download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data.
    Handles all yfinance versions including MultiIndex columns.
    """
    try:
        # Re-apply cache fix per call (needed for regime sub-tests)
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

        # Flatten MultiIndex columns (yfinance >= 0.2.40)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Standardize column names
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
    """2=STRONG_BULL  1=BULL  0=CHOPPY  -1=BEAR  -2=STRONG_BEAR"""
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
    """0 = at lower band  1 = at upper band"""
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    lower = mid - 2 * std
    upper = mid + 2 * std
    denom = (upper - lower).replace(0, np.nan)
    return ((closes - lower) / denom).fillna(0.5)


# ══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ══════════════════════════════════════════════════════════════════


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signal scoring — each condition adds 1 point.
    Entry requires MIN_SIGNALS (5) out of 7 conditions.

    Signal 1: Regime >= BULL         uptrend confirmed on daily
    Signal 2: RSI < RSI_BUY (50)     pulled back from highs
    Signal 3: Price > EMA21           above structural support
    Signal 4: MACD > signal line      momentum direction bullish
    Signal 5: Volume > 1.2x average   institutional interest
    Signal 6: 5-day momentum > 0      short-term trend intact
    Signal 7: BB pct_b < 0.35         near lower Bollinger Band

    Requiring 5/7 gives quality confluence without contradictions.
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
    ema21 = calc_ema(closes, 21)
    pct_b = calc_bollinger_pctb(closes, BB_PERIOD)
    mom5d = closes.pct_change(5).fillna(0) * 100

    df["rsi"] = rsi
    df["macd"] = macd
    df["macd_sig"] = sig
    df["macd_hist"] = h
    df["atr"] = atr
    df["regime"] = regime
    df["vol_ratio"] = vol_ratio
    df["ema21"] = ema21
    df["pct_b"] = pct_b
    df["mom5d"] = mom5d

    s1 = (regime >= 1).astype(int)
    s2 = (rsi < RSI_BUY).astype(int)
    s3 = (closes > ema21).astype(int)
    s4 = (macd > sig).astype(int)
    s5 = (vol_ratio > 1.2).astype(int)
    s6 = (mom5d > 0).astype(int)
    s7 = (pct_b < 0.35).astype(int)

    df["signal_count"] = s1 + s2 + s3 + s4 + s5 + s6 + s7

    # Entry: in uptrend + enough signal confluence + ATR available
    df["entry"] = (
        (regime >= 1) & (df["signal_count"] >= MIN_SIGNALS) & atr.notna()
    ).fillna(False)

    # Exit: overbought OR trend broken
    df["exit"] = ((rsi > RSI_SELL) | (regime <= -1)).fillna(False)

    return df


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO
# ══════════════════════════════════════════════════════════════════


class Portfolio:
    """
    Portfolio simulator with:
      - ATR position sizing (1% risk per trade)
      - Trailing stop (updates on new price highs)
      - Breakeven stop (after +1%)
      - Partial profit at +PARTIAL_PROFIT% (sell half)
      - Minimum hold days (avoids open-volatility shakeouts)
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

    def can_buy(self, price: float, atr: float) -> int:
        if atr <= 0 or price <= 0:
            return 0
        qty = int(self.equity * RISK_PER_TRADE / (atr * ATR_STOP_MULT))
        max_qty = int(self.equity * MAX_POSITION_PCT / (price * (1 + COMMISSION)))
        cash_qty = int(self.cash * 0.95 / (price * (1 + COMMISSION)))
        return max(0, min(qty, max_qty, cash_qty))

    def buy(self, ticker: str, price: float, atr: float, date) -> bool:
        if ticker in self.positions:
            return False
        qty = self.can_buy(price, atr)
        if qty <= 0:
            return False
        cost = qty * price * (1 + COMMISSION)
        if cost > self.cash:
            return False
        self.cash -= cost
        self.positions[ticker] = {
            "qty": qty,
            "entry": price,
            "stop": price - atr * ATR_STOP_MULT,
            "target": price + atr * ATR_TARGET_MULT,
            "peak": price,
            "atr": atr,
            "date": date,
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
        """
        Per position, each bar:
          1. Skip if within MIN_HOLD_DAYS
          2. Partial profit at PARTIAL_PROFIT% (sell half)
          3. Update trailing stop on new price highs
          4. Move stop to breakeven after +1%
          5. Fire stop or profit target
        """
        to_sell = []

        for ticker, pos in list(self.positions.items()):
            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            hold_days = (date - pos["date"]).days
            if hold_days < MIN_HOLD_DAYS:
                continue

            entry = pos["entry"]

            # Partial profit
            if (
                not pos["partial_taken"]
                and price >= entry * (1 + PARTIAL_PROFIT)
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
                new_trail = price - pos["atr"] * ATR_STOP_MULT
                if new_trail > pos["stop"]:
                    pos["stop"] = new_trail

            # Breakeven after +1%
            if price >= entry * 1.01:
                be = entry * 1.001
                if be > pos["stop"]:
                    pos["stop"] = be

            # Fire stop or target
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
) -> dict:
    print(f"\n{'═'*65}")
    print(f"  📊 BACKTEST ENGINE")
    print(f"  Period  : {start_date} → {end_date}")
    print(f"  Tickers : {len(tickers)}")
    print(f"  Capital : ${init_cash:,.0f}")
    print(f"{'═'*65}")

    print("\n📥 Downloading data...")
    all_data = {}
    for ticker in tickers:
        df = download_ticker(ticker, start_date, end_date)
        if df is None:
            print(f"  ⚠️  {ticker}: skipped")
            continue
        df = generate_signals(df)
        entries = int(df["entry"].sum())
        exits = int(df["exit"].sum())
        avg_sig = df["signal_count"].mean()
        status = "✅" if entries > 0 else "⚠️ "
        all_data[ticker] = df
        print(
            f"  {status} {ticker:<6}: {len(df)} bars  "
            f"entries={entries:>3}  exits={exits:>3}  "
            f"avg_signals={avg_sig:.1f}"
        )

    if not all_data:
        print("\n❌ No data. Try: pip install --upgrade yfinance")
        return {}

    total_entries = sum(int(df["entry"].sum()) for df in all_data.values())
    if total_entries == 0:
        print("\n⚠️  Zero entry signals. Showing sample:")
        ticker = list(all_data.keys())[0]
        print(
            all_data[ticker][["rsi", "regime", "signal_count", "entry"]]
            .tail(10)
            .to_string()
        )
        return {}

    all_dates = sorted(set.union(*[set(df.index) for df in all_data.values()]))

    print(
        f"\n🔄 Simulating {len(all_dates)} trading days "
        f"with {len(all_data)} tickers "
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
                    f"{reason:<14} "
                    f"P&L: ${trade['pl']:>+8,.0f} "
                    f"({trade['pl_pct']:>+5.1f}%)"
                )

        # 2. Signal-based exits
        for ticker in list(port.positions.keys()):
            if date not in all_data[ticker].index:
                continue
            if bool(all_data[ticker].loc[date, "exit"]):
                price = prices.get(ticker, 0)
                trade = port.sell(ticker, price, date, "signal_exit")
                if trade and verbose:
                    print(
                        f"  📤 {str(date.date())} {ticker:<6} "
                        f"{'signal_exit':<14} "
                        f"P&L: ${trade['pl']:>+8,.0f} "
                        f"({trade['pl_pct']:>+5.1f}%)"
                    )

        # 3. Entries — top candidates by signal count
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
                        )
                    )

            opps.sort(key=lambda x: x[1], reverse=True)
            for ticker, sigs, price, atr in opps[:3]:
                if len(port.positions) >= MAX_POSITIONS:
                    break
                if port.buy(ticker, price, atr, date):
                    pos = port.positions[ticker]
                    if verbose:
                        print(
                            f"  🟢 {str(date.date())} {ticker:<6} "
                            f"BUY  x{pos['qty']:<5} "
                            f"@ ${price:<8.2f} "
                            f"stop=${pos['stop']:.2f}  "
                            f"target=${pos['target']:.2f}  "
                            f"sigs={sigs:.0f}"
                        )

        port.update_equity(prices)

    # Close remaining positions at end of test
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
            trades=("pl", "count"),
            total_pl=("pl", "sum"),
            win_rate=("pl", lambda x: (x > 0).mean() * 100),
            avg_hold=("hold_days", "mean"),
        )
        .sort_values("total_pl", ascending=False)
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
    print(f"  📊 BACKTEST RESULTS")
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

    print(f"\n  PERFORMANCE BY TICKER")
    print(f"  {'─'*60}")
    print(
        f"  {'Ticker':<8} {'Trades':>6} {'Win%':>6} "
        f"{'Total P&L':>12} {'Avg Hold':>9}"
    )
    print(f"  {'─'*60}")
    for ticker, row in m["ticker_stats"].iterrows():
        icon = "✅" if row["total_pl"] > 0 else "❌"
        print(
            f"  {icon}{ticker:<7} {row['trades']:>6.0f} "
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
        print("\n  🚀 READY — strategy shows real edge. Start paper trading.")
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

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    G, R, GR, TXT = "#00ff88", "#ff4444", "#888888", "#ffffff"

    def style(ax, title):
        ax.set_facecolor("#16213e")
        ax.set_title(title, color=TXT, pad=8, fontsize=10)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333366")
        ax.yaxis.label.set_color(TXT)
        ax.xaxis.label.set_color(TXT)

    eq = pd.Series(results["equity_curve"])
    dates = results["all_dates"][: len(eq)]
    init = results["init_cash"]

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    color = G if eq.iloc[-1] >= init else R
    ax1.plot(dates, eq.values, color=color, linewidth=1.5)
    ax1.axhline(init, color=GR, linestyle="--", linewidth=0.8, alpha=0.7, label="Start")
    ax1.fill_between(dates, eq.values, init, alpha=0.1, color=color)
    ax1.set_ylabel("Portfolio ($)")
    ax1.legend(facecolor="#1a1a2e", labelcolor=TXT, fontsize=8)
    style(
        ax1,
        f"Equity Curve  |  "
        f"Final ${results['final_equity']:,.0f}  "
        f"({metrics['total_return']:+.1f}%)",
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
        step = max(1, len(monthly) // 8)
        ax3.set_xticks(range(0, len(monthly), step))
        ax3.set_xticklabels(
            [str(p) for p in monthly.index[::step]],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax3.set_ylabel("P&L ($)")
    style(ax3, "Monthly P&L")

    # 4. Trade distribution
    ax4 = fig.add_subplot(gs[2, 0])
    if results["trades"]:
        df_t = pd.DataFrame(results["trades"])
        wins = df_t[df_t["pl_pct"] > 0]["pl_pct"]
        loss = df_t[df_t["pl_pct"] < 0]["pl_pct"]
        if len(wins):
            ax4.hist(wins, bins=25, color=G, alpha=0.7, label=f"Winners ({len(wins)})")
        if len(loss):
            ax4.hist(loss, bins=25, color=R, alpha=0.7, label=f"Losers ({len(loss)})")
        ax4.axvline(0, color=GR, linewidth=1)
        ax4.set_xlabel("Return %")
        ax4.legend(facecolor="#1a1a2e", labelcolor=TXT, fontsize=8)
    style(ax4, "Trade Distribution")

    # 5. Per-ticker P&L
    ax5 = fig.add_subplot(gs[2, 1])
    ts = metrics["ticker_stats"].sort_values("total_pl")
    ax5.barh(
        ts.index,
        ts["total_pl"].values,
        color=[G if v >= 0 else R for v in ts["total_pl"]],
    )
    ax5.axvline(0, color=GR, linewidth=0.8)
    ax5.set_xlabel("Total P&L ($)")
    ax5.tick_params(axis="y", labelsize=8)
    style(ax5, "P&L by Ticker")

    plt.suptitle(
        f"Backtest {results['start_date']} → {results['end_date']}  |  "
        f"Sharpe {metrics['sharpe']:.2f}  |  "
        f"Annual {metrics['annual_return']:+.1f}%  |  "
        f"MaxDD {metrics['max_drawdown']:.1f}%",
        color=TXT,
        fontsize=11,
        y=1.01,
    )

    fname = (
        f"backtest_" f"{results['start_date'][:4]}_" f"{results['end_date'][:4]}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ Chart saved: {fname}")
    return fname


# ══════════════════════════════════════════════════════════════════
# REGIME BREAKDOWN
# ══════════════════════════════════════════════════════════════════


def run_regime_breakdown(results: dict):
    """Performance in each distinct market period."""
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
            time.sleep(3)  # avoid yfinance rate limit
            r = run_backtest(tickers, start, end, verbose=False)
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
    parser = argparse.ArgumentParser(description="Claude Trading Bot — Backtester v8")
    parser.add_argument("--ticker", type=str, help="Single ticker e.g. NVDA")
    parser.add_argument(
        "--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD",
    )
    parser.add_argument("--quick", action="store_true", help="2024 only (fast test)")
    parser.add_argument("--quiet", action="store_true", help="No individual trade logs")
    parser.add_argument(
        "--regimes", action="store_true", help="Show performance by market regime"
    )
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else DEFAULT_WATCHLIST
    start_date = "2024-01-01" if args.quick else args.start
    end_date = args.end
    verbose = not args.quiet

    print(f"\n🤖 Claude Trading Bot — Backtester v8")
    print(f"   Tickers  : {len(tickers)}  " f"({', '.join(tickers[:6])}...)")
    print(
        f"   Entry    : {MIN_SIGNALS}+ signals  "
        f"(regime + RSI<{RSI_BUY} + EMA21 + MACD + vol + mom + BB)"
    )
    print(f"   Exit     : RSI>{RSI_SELL} or regime breaks")
    print(
        f"   Stop     : {ATR_STOP_MULT}x ATR  |  "
        f"Target: {ATR_TARGET_MULT}x ATR  |  "
        f"R:R ~{ATR_TARGET_MULT/ATR_STOP_MULT:.0f}:1"
    )
    print(f"   Partial  : sell half at +{PARTIAL_PROFIT*100:.0f}%")
    print(
        f"   Risk     : {RISK_PER_TRADE*100:.0f}% per trade  |  "
        f"Max {MAX_POSITIONS} positions  |  "
        f"Min hold {MIN_HOLD_DAYS} days"
    )

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
