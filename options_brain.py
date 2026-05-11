"""
options_brain.py
================
Claude AI decision engine specialized for options trading.
Python 3.9 compatible.
"""
import json
import logging
import re
from typing import Optional, List

import anthropic

log = logging.getLogger(__name__)

OPTIONS_SYSTEM_PROMPT = """
You are an elite options trader with 20 years of experience.
You trade options for MAXIMUM LEVERAGE on high-conviction directional moves.

STRATEGY SELECTION:
  LONG_CALL        → Strong bull + IV rank < 40% (cheap options)
  BULL_CALL_SPREAD → Moderate bull + IV rank > 50% (sell expensive premium)
  LONG_PUT         → Strong bear + IV rank < 40%
  BEAR_PUT_SPREAD  → Moderate bear + IV rank > 50%

OPTION SELECTION RULES:
  - Target delta: 0.35–0.45 (best balance of leverage and probability)
  - DTE: 14–30 days (avoid theta decay < 7 DTE, avoid slow moves > 45 DTE)
  - Bid/ask spread: < 15% of mid (liquidity requirement)
  - Volume: > 10 contracts (confirms liquidity)
  - Max loss per trade: $2,000

EXIT RULES:
  - Take profit at 80% gain on premium
  - Cut loss at 50% loss on premium
  - Exit all positions before last 5 DTE

DO NOT TRADE WHEN:
  - No clear directional bias
  - Bid/ask spread > 15%
  - Volume < 10 contracts
  - Max loss would exceed $2,000

RESPOND WITH STRICT JSON ONLY — no prose, no markdown.

For single-leg (LONG_CALL or LONG_PUT):
{
  "action":        "buy",
  "strategy":      "LONG_CALL",
  "ticker":        "NVDA",
  "option_symbol": "NVDA250117C00450000",
  "contracts":     3,
  "limit_price":   4.50,
  "strike":        450.00,
  "expiry":        "2025-01-17",
  "delta":         0.42,
  "dte":           14,
  "max_loss":      1350.00,
  "profit_target": 2430.00,
  "stop_loss_pct": 50,
  "confidence":    0.85,
  "iv_rank":       28.0,
  "reason":        "Under 80 words explaining the trade."
}

For spreads:
{
  "action":       "buy",
  "strategy":     "BULL_CALL_SPREAD",
  "ticker":       "NVDA",
  "long_symbol":  "NVDA250117C00450000",
  "short_symbol": "NVDA250117C00475000",
  "long_mid":     3.20,
  "short_mid":    1.10,
  "contracts":    5,
  "net_debit":    2.10,
  "max_loss":     1050.00,
  "max_profit":   2200.00,
  "breakeven":    452.10,
  "confidence":   0.82,
  "iv_rank":      62.0,
  "reason":       "Under 80 words explaining the trade."
}

For no trade:
{
  "action":     "hold",
  "reason":     "No qualifying setup.",
  "confidence": 0.0
}
"""


class OptionsBrain:
    def __init__(self, config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def analyze(self, ticker: str, stock_data: dict,
                 options_data: dict) -> Optional[dict]:
        if not options_data or not options_data.get("recommendations"):
            return None
        prompt = self._build_prompt(ticker, stock_data, options_data)
        for attempt in range(2):
            try:
                resp = self.client.messages.create(
                    model      = self.config.MODEL,
                    max_tokens = 1000,
                    system     = OPTIONS_SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": prompt}]
                )
                raw = re.sub(r"```json|```", "", resp.content[0].text).strip()
                d   = json.loads(raw)
                self._validate(d)
                return d
            except json.JSONDecodeError:
                if attempt == 0:
                    prompt += "\n\nReturn ONLY the JSON object."
                else:
                    log.error("Options brain JSON failed.")
                    return None
            except Exception as e:
                log.error("Options brain error: %s", e)
                return None

    def _build_prompt(self, ticker: str, stock_data: dict,
                       options_data: dict) -> str:
        price    = stock_data.get("price", 0)
        regime   = stock_data.get("market_regime", "UNKNOWN")
        signals  = stock_data.get("signals", {})
        bias     = signals.get("bias", "NEUTRAL")
        rsi      = stock_data.get("rsi_14", "N/A")
        rsi5m    = stock_data.get("rsi_5m", "N/A")
        macd     = stock_data.get("macd") or {}
        atr      = stock_data.get("atr_14", "N/A")
        vr       = stock_data.get("vol_ratio") or 1.0
        mom5d    = stock_data.get("momentum_5d") or 0.0
        iv_rank  = options_data.get("iv_rank")
        max_pain = options_data.get("max_pain")
        recs     = options_data.get("recommendations", [])
        chains   = options_data.get("chains", {})
        buys     = signals.get("buy_signals", [])
        sells    = signals.get("sell_signals", [])

        iv_advice = ""
        if iv_rank is not None:
            if iv_rank < 30:
                iv_advice = "→ BUY OPTIONS outright (cheap premium)"
            elif iv_rank > 50:
                iv_advice = "→ USE SPREADS (sell expensive premium)"
            else:
                iv_advice = "→ Either strategy viable"

        lines = [
            f"═══ OPTIONS ANALYSIS: {ticker} ═══",
            f"Stock: ${price:.2f}  Regime={regime}  Bias={bias}",
            f"RSI daily={rsi}  5m={rsi5m}  MACD={'bull✅' if macd.get('bullish') else 'bear❌'}",
            f"Volume={vr:.1f}x  5d mom={mom5d:+.1f}%  ATR=${atr}",
            f"Buy signals: {', '.join(buys) if buys else 'none'}",
            f"Sell signals: {', '.join(sells) if sells else 'none'}",
            "",
            f"IV Rank: {f'{iv_rank:.0f}%' if iv_rank is not None else 'N/A'}  {iv_advice}",
            f"Max Pain: ${max_pain:.0f}" if max_pain else "Max Pain: N/A",
            "",
            "═══ AVAILABLE CHAIN ═══",
        ]

        for expiry, cdata in chains.items():
            lines.append(f"\nExpiry {expiry}:")
            lines.append("  CALLS:")
            for c in cdata.get("calls", []):
                lines.append(
                    f"    ${c['strike']:.0f}C  mid=${c['mid']:.2f}  "
                    f"delta={c['delta']:.2f}  IV={c['iv']:.0f}%  "
                    f"vol={c['volume']}  {c['moneyness']}"
                )
            lines.append("  PUTS:")
            for p in cdata.get("puts", []):
                lines.append(
                    f"    ${p['strike']:.0f}P  mid=${p['mid']:.2f}  "
                    f"delta={p['delta']:.2f}  IV={p['iv']:.0f}%  "
                    f"vol={p['volume']}  {p['moneyness']}"
                )

        lines.append("\n═══ PRE-ANALYZED RECOMMENDATIONS ═══")
        for i, rec in enumerate(recs, 1):
            lines.append(f"\n{i}. {rec['strategy']}  (exp {rec.get('expiry','?')}  DTE={rec.get('dte','?')})")
            lines.append(f"   {rec['reason']}")
            mp = rec.get('max_profit')
            mp_str = mp if isinstance(mp, str) else f"${mp:.0f}"
            lines.append(f"   MaxLoss=${rec['max_loss']:.0f}  MaxProfit={mp_str}  Breakeven=${rec.get('breakeven',0):.2f}")

        lines.append(
            "\n═══ TASK ═══\n"
            "Choose the BEST options trade or hold.\n"
            "Must meet ALL:\n"
            "  ✅ Clear bias (BUY or SELL)\n"
            "  ✅ MaxProfit:MaxLoss >= 2:1\n"
            "  ✅ Volume > 10  ✅ DTE 7-45  ✅ Spread < 15%\n"
            "  ✅ Max loss < $2,000\n"
            "JSON only."
        )
        return "\n".join(lines)

    def _validate(self, d: dict):
        if "action" not in d:
            raise ValueError("Missing action")
        if d["action"] not in ("buy", "sell", "hold"):
            raise ValueError(f"Bad action: {d['action']}")
        if d["action"] != "hold":
            for f in ("strategy", "ticker", "contracts", "confidence"):
                if f not in d:
                    raise ValueError(f"Missing: {f}")
            d["contracts"]  = int(d["contracts"])
            d["confidence"] = float(d["confidence"])
