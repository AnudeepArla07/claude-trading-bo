"""
options_feed.py
===============
Options chain data, Greeks, IV rank, max pain, contract selection.
Python 3.9 compatible.
"""
import logging
import statistics
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

import requests

log = logging.getLogger(__name__)


class OptionsFeed:
    def __init__(self, config):
        self.config  = config
        self.headers = {
            "APCA-API-KEY-ID":     config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
            "accept":              "application/json",
        }
        self._iv_history: Dict[str, List[float]] = {}

    def get_options_data(self, ticker: str, stock_price: float,
                          signal_bias: str = "NEUTRAL") -> Optional[dict]:
        try:
            expiries = self._get_expiry_dates()
            chains: Dict[str, dict] = {}
            for exp in expiries[:2]:
                chain = self._fetch_chain(ticker, exp, stock_price)
                if chain:
                    chains[exp] = chain

            if not chains:
                return None

            recommendations = self._select_contracts(
                ticker, stock_price, chains, signal_bias
            )
            iv_rank  = self._compute_iv_rank(ticker, chains)
            max_pain = self._compute_max_pain(chains)

            return {
                "ticker":          ticker,
                "stock_price":     stock_price,
                "signal_bias":     signal_bias,
                "iv_rank":         iv_rank,
                "max_pain":        max_pain,
                "recommendations": recommendations,
                "chains":          self._summarize_chains(chains),
            }
        except Exception as e:
            log.error("Options data error %s: %s", ticker, e)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    def _fetch_chain(self, ticker: str, expiry: str,
                      stock_price: float) -> Optional[dict]:
        try:
            url    = "https://paper-api.alpaca.markets/v2/options/contracts"
            params = {
                "underlying_symbols": ticker,
                "expiration_date":    expiry,
                "strike_price_gte":   round(stock_price * 0.85, 0),
                "strike_price_lte":   round(stock_price * 1.15, 0),
                "limit":              100,
            }
            resp = requests.get(url, headers=self.headers,
                                params=params, timeout=10)
            if resp.status_code != 200:
                return None

            contracts = resp.json().get("option_contracts", [])
            if not contracts:
                return None

            symbols   = [c["symbol"] for c in contracts]
            snapshots = self._fetch_snapshots(symbols)

            chain: Dict[str, List[dict]] = {"calls": [], "puts": []}
            for c in contracts:
                sym  = c["symbol"]
                snap = snapshots.get(sym, {})
                bid  = snap.get("bid", 0)
                ask  = snap.get("ask", 0)
                mid  = round((bid + ask) / 2, 2) if bid and ask else 0
                rec  = {
                    "symbol":        sym,
                    "strike":        float(c.get("strike_price", 0)),
                    "expiry":        c.get("expiration_date"),
                    "type":          c.get("type", "").lower(),
                    "bid":           bid,
                    "ask":           ask,
                    "mid":           mid,
                    "last":          snap.get("last", 0),
                    "volume":        snap.get("volume", 0),
                    "open_interest": snap.get("open_interest", 0),
                    "iv":            snap.get("implied_volatility", 0),
                    "delta":         snap.get("delta", 0),
                    "gamma":         snap.get("gamma", 0),
                    "theta":         snap.get("theta", 0),
                    "vega":          snap.get("vega", 0),
                    "dte":           self._dte(expiry),
                    "spread_pct":    self._spread_pct(bid, ask),
                    "moneyness":     self._moneyness(
                        float(c.get("strike_price", 0)),
                        stock_price, c.get("type", "").lower()
                    ),
                }
                if rec["type"] == "call":
                    chain["calls"].append(rec)
                elif rec["type"] == "put":
                    chain["puts"].append(rec)

            chain["calls"].sort(key=lambda x: x["strike"])
            chain["puts"].sort(key=lambda x: x["strike"], reverse=True)
            return chain

        except Exception as e:
            log.error("Chain fetch %s %s: %s", ticker, expiry, e)
            return None

    def _fetch_snapshots(self, symbols: List[str]) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        if not symbols:
            return result
        try:
            url    = "https://data.alpaca.markets/v1beta1/options/snapshots"
            params = {"symbols": ",".join(symbols[:50])}
            headers = dict(self.headers)
            resp   = requests.get(url, headers=headers,
                                  params=params, timeout=10)
            if resp.status_code == 200:
                for sym, snap in resp.json().get("snapshots", {}).items():
                    greeks = snap.get("greeks", {})
                    quote  = snap.get("latestQuote", {})
                    trade  = snap.get("latestTrade", {})
                    bid    = float(quote.get("bp", 0))
                    ask    = float(quote.get("ap", 0))
                    result[sym] = {
                        "bid":                bid,
                        "ask":                ask,
                        "mid":                round((bid+ask)/2, 2) if bid and ask else 0,
                        "last":               float(trade.get("p", 0)),
                        "volume":             int(snap.get("dailyBar", {}).get("v", 0)),
                        "open_interest":      int(snap.get("openInterest", 0)),
                        "implied_volatility": float(snap.get("impliedVolatility", 0)),
                        "delta":              float(greeks.get("delta", 0)),
                        "gamma":              float(greeks.get("gamma", 0)),
                        "theta":              float(greeks.get("theta", 0)),
                        "vega":               float(greeks.get("vega", 0)),
                    }
        except Exception as e:
            log.warning("Snapshots failed: %s", e)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    def _select_contracts(self, ticker: str, price: float,
                           chains: Dict[str, dict],
                           bias: str) -> List[dict]:
        recs = []
        for expiry, chain in chains.items():
            dte = self._dte(expiry)
            if bias in ("STRONG_BUY", "BUY"):
                call = self._best_contract(chain["calls"], 0.40)
                if call:
                    qty = self._size_contracts(call)
                    recs.append({
                        "strategy":  "LONG_CALL",
                        "direction": "bullish",
                        "contract":  call,
                        "contracts": qty,
                        "max_loss":  round(call["mid"] * 100 * qty, 2),
                        "max_profit":"unlimited",
                        "breakeven": round(call["strike"] + call["mid"], 2),
                        "expiry":    expiry,
                        "dte":       dte,
                        "reason":    (
                            f"Buy {qty} x {call['strike']}C exp {expiry}. "
                            f"Delta={call['delta']:.2f}. "
                            f"Premium=${call['mid']:.2f}. "
                            f"Breakeven=${call['strike']+call['mid']:.2f}. "
                            f"Max loss=${call['mid']*100*qty:.0f}."
                        ),
                    })
                spread = self._find_spread(chain["calls"], price, "bull_call")
                if spread:
                    debit = round(spread["long"]["mid"] - spread["short"]["mid"], 2)
                    qty   = max(1, min(int(1500 / (debit * 100 + 1)), 10))
                    mp    = round((spread["short"]["strike"] -
                                   spread["long"]["strike"] - debit) * 100 * qty, 2)
                    recs.append({
                        "strategy":   "BULL_CALL_SPREAD",
                        "direction":  "bullish",
                        "long_leg":   spread["long"],
                        "short_leg":  spread["short"],
                        "contracts":  qty,
                        "net_debit":  debit,
                        "max_loss":   round(debit * 100 * qty, 2),
                        "max_profit": mp,
                        "breakeven":  round(spread["long"]["strike"] + debit, 2),
                        "expiry":     expiry,
                        "dte":        dte,
                        "reason":     (
                            f"Buy {spread['long']['strike']}C / "
                            f"Sell {spread['short']['strike']}C. "
                            f"Debit=${debit:.2f}. Max profit=${mp:.0f}. "
                            f"Defined risk=${debit*100*qty:.0f}."
                        ),
                    })
            elif bias in ("STRONG_SELL", "SELL"):
                put = self._best_contract(chain["puts"], -0.40)
                if put:
                    qty = self._size_contracts(put)
                    recs.append({
                        "strategy":  "LONG_PUT",
                        "direction": "bearish",
                        "contract":  put,
                        "contracts": qty,
                        "max_loss":  round(put["mid"] * 100 * qty, 2),
                        "max_profit":round((put["strike"] - put["mid"]) * 100 * qty, 2),
                        "breakeven": round(put["strike"] - put["mid"], 2),
                        "expiry":    expiry,
                        "dte":       dte,
                        "reason":    (
                            f"Buy {qty} x {put['strike']}P exp {expiry}. "
                            f"Delta={put['delta']:.2f}. "
                            f"Premium=${put['mid']:.2f}. "
                            f"Breakeven=${put['strike']-put['mid']:.2f}. "
                            f"Max loss=${put['mid']*100*qty:.0f}."
                        ),
                    })
        return recs

    def _best_contract(self, contracts: List[dict],
                        target_delta: float) -> Optional[dict]:
        valid = [
            c for c in contracts
            if c["spread_pct"] < 0.15
            and c["volume"] > 5
            and c["mid"] > 0
            and self.config.OPTIONS_MIN_DTE <= c["dte"] <= self.config.OPTIONS_MAX_DTE
        ]
        if not valid:
            valid = [c for c in contracts if c["mid"] > 0]
        if not valid:
            return None
        return min(valid, key=lambda c: abs(abs(c["delta"]) - abs(target_delta)))

    def _find_spread(self, contracts: List[dict], price: float,
                      spread_type: str) -> Optional[dict]:
        valid = [c for c in contracts if c["mid"] > 0 and c["volume"] > 0]
        if len(valid) < 2:
            return None
        if spread_type == "bull_call":
            atm = min(valid, key=lambda c: abs(c["strike"] - price))
            otm = min(valid, key=lambda c: abs(c["strike"] - price * 1.05))
            if atm["symbol"] != otm["symbol"] and atm["strike"] < otm["strike"]:
                return {"long": atm, "short": otm}
        return None

    def _size_contracts(self, contract: dict,
                         win_rate: float = 0.55) -> int:
        mid = contract["mid"]
        if mid <= 0:
            return 1
        avg_win  = mid * 2.0
        avg_loss = mid
        kelly    = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        kelly    = max(0.05, min(kelly, 0.25))
        risk_budget = 2000
        cost = mid * 100
        return max(1, min(int(risk_budget / cost), 10))

    def _compute_iv_rank(self, ticker: str,
                          chains: Dict[str, dict]) -> Optional[float]:
        ivs = [
            c["iv"]
            for chain in chains.values()
            for c in chain.get("calls", []) + chain.get("puts", [])
            if c.get("iv", 0) > 0
        ]
        if not ivs:
            return None
        current_iv = sum(ivs) / len(ivs)
        history    = self._iv_history.get(ticker, [])
        history.append(current_iv)
        history    = history[-252:]
        self._iv_history[ticker] = history
        if len(history) < 10:
            return None
        iv_min = min(history)
        iv_max = max(history)
        if iv_max == iv_min:
            return 50.0
        return round((current_iv - iv_min) / (iv_max - iv_min) * 100, 1)

    def _compute_max_pain(self, chains: Dict[str, dict]) -> Optional[float]:
        try:
            call_oi: Dict[float, int] = {}
            put_oi:  Dict[float, int] = {}
            all_strikes = set()
            for chain in chains.values():
                for c in chain.get("calls", []):
                    s = c["strike"]; all_strikes.add(s)
                    call_oi[s] = call_oi.get(s, 0) + c.get("open_interest", 0)
                for p in chain.get("puts", []):
                    s = p["strike"]; all_strikes.add(s)
                    put_oi[s] = put_oi.get(s, 0) + p.get("open_interest", 0)
            if not all_strikes:
                return None
            best_s = None
            best_p = float("inf")
            for test_s in sorted(all_strikes):
                pain = sum((test_s - s) * oi for s, oi in call_oi.items() if test_s > s)
                pain+= sum((s - test_s) * oi for s, oi in put_oi.items()  if test_s < s)
                if pain < best_p:
                    best_p = pain; best_s = test_s
            return best_s
        except Exception:
            return None

    def _summarize_chains(self, chains: Dict[str, dict]) -> dict:
        summary = {}
        for expiry, chain in chains.items():
            calls = sorted(chain["calls"],
                           key=lambda c: abs(c["delta"] - 0.40))[:5]
            puts  = sorted(chain["puts"],
                           key=lambda c: abs(abs(c["delta"]) - 0.40))[:5]
            summary[expiry] = {
                "calls": [{"strike": c["strike"], "mid": c["mid"],
                           "delta": c["delta"], "iv": round(c["iv"]*100,1),
                           "theta": c["theta"], "volume": c["volume"],
                           "moneyness": c["moneyness"]} for c in calls],
                "puts":  [{"strike": p["strike"], "mid": p["mid"],
                           "delta": p["delta"], "iv": round(p["iv"]*100,1),
                           "theta": p["theta"], "volume": p["volume"],
                           "moneyness": p["moneyness"]} for p in puts],
            }
        return summary

    def _get_expiry_dates(self) -> List[str]:
        exps = []
        d    = date.today()
        while len(exps) < 3:
            d += timedelta(days=1)
            if d.weekday() == 4:
                exps.append(d.strftime("%Y-%m-%d"))
        return exps

    def _dte(self, expiry_str: str) -> int:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return max(0, (exp - date.today()).days)

    def _moneyness(self, strike: float, price: float, opt_type: str) -> str:
        pct = (strike - price) / price * 100
        if opt_type == "call":
            return "ITM" if pct < -3 else "OTM" if pct > 3 else "ATM"
        return "ITM" if pct > 3 else "OTM" if pct < -3 else "ATM"

    def _spread_pct(self, bid: float, ask: float) -> float:
        mid = (bid + ask) / 2
        return round((ask - bid) / mid, 3) if mid > 0 else 1.0
