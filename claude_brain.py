"""
claude_brain.py
===============
Aggressive Claude AI stock trading decision engine.
Feeds full technical indicator suite across 3 timeframes.
Python 3.9 compatible.
"""

import json
import logging
import re
from typing import Optional, List

import anthropic

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an elite quantitative day trader with 20 years of experience.
Your mandate: generate maximum risk-adjusted returns on a $100,000 account.
You are AGGRESSIVE but DISCIPLINED — only trade the highest-conviction setups.

TRADING RULES:
1. MOMENTUM IS KING — trade in the direction of the trend, never fight it.
2. REQUIRE 3+ SIGNALS — RSI + MACD + BB + Volume + Regime must align.
3. MINIMUM 2.5:1 R:R — every trade needs defined stop and target.
4. ATR-BASED STOPS — stop = entry - 1.5x ATR. Target = entry + 3.75x ATR.
5. NEVER trade CHOPPY regime unless there is a breakout with 2x+ volume.
6. HOLD when uncertain — protect capital above all else.

AGGRESSIVE ENTRY TRIGGERS:
- STRONG_BUY bias + RSI < 45 + volume > 1.5x avg → BUY immediately
- BUY bias + MACD bullish + price above VWAP → BUY
- STRONG_SELL + RSI > 65 → SELL (exit longs only, no shorting)
- CHOPPY or no clear signal → HOLD

RESPOND WITH STRICT JSON ONLY — no prose, no markdown:
{
  "action":       "buy" | "sell" | "hold",
  "ticker":       "NVDA",
  "quantity":     15,
  "entry_price":  450.00,
  "stop_loss":    441.00,
  "take_profit":  478.00,
  "risk_reward":  3.0,
  "confidence":   0.88,
  "signal_count": 4,
  "trade_type":   "momentum",
  "reason":       "Explain in under 100 words why this trade."
}
"""


class ClaudeBrain:
    def __init__(self, config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._history: List[dict] = []

    def analyze(
        self, portfolio: dict, market_data: dict, news: List[dict]
    ) -> Optional[dict]:
        prompt = self._build_prompt(portfolio, market_data, news)
        for attempt in range(2):
            try:
                resp = self.client.messages.create(
                    model=self.config.MODEL,
                    max_tokens=self.config.MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = re.sub(r"```json|```", "", resp.content[0].text).strip()
                d = json.loads(raw)
                self._validate(d)
                self._history.append(d)
                if len(self._history) > 20:
                    self._history = self._history[-20:]
                return d
            except json.JSONDecodeError:
                if attempt == 0:
                    prompt += "\n\nReturn ONLY the JSON object."
                else:
                    log.error("Stock brain JSON parse failed.")
                    return None
            except Exception as e:
                log.error("Stock brain error: %s", e)
                return None

    def _build_prompt(
        self, portfolio: dict, market_data: dict, news: List[dict]
    ) -> str:
        equity = portfolio.get("equity", 0)
        cash = portfolio.get("cash", 0)
        daily_pl = portfolio.get("daily_pl", 0)
        pct = (daily_pl / equity * 100) if equity else 0

        lines = [
            "═══ PORTFOLIO ═══",
            f"Equity: ${equity:,.0f}  Cash: ${cash:,.0f}  "
            f"Day P&L: ${daily_pl:+,.0f} ({pct:+.2f}%)",
            f"Open positions: {len(portfolio.get('positions',[]))}",
        ]
        for p in portfolio.get("positions", []):
            lines.append(
                f"  {p['symbol']}: {p['qty']:.0f} shares | "
                f"entry ${p['avg_entry']:.2f} | now ${p['current_price']:.2f} | "
                f"P&L ${p['unrealized_pl']:+.2f}"
            )

        # Sort by signal strength
        order = {"STRONG_BUY": 0, "BUY": 1, "NEUTRAL": 2, "SELL": 3, "STRONG_SELL": 4}
        sorted_data = sorted(
            market_data.items(),
            key=lambda x: order.get(x[1].get("signals", {}).get("bias", "NEUTRAL"), 2),
        )

        lines.append("\n═══ MARKET DATA ═══")
        for ticker, d in sorted_data:
            sig = d.get("signals", {})
            bias = sig.get("bias", "NEUTRAL")
            icon = {
                "STRONG_BUY": "🟢🟢",
                "BUY": "🟢",
                "NEUTRAL": "⚪",
                "SELL": "🔴",
                "STRONG_SELL": "🔴🔴",
            }.get(bias, "⚪")
            rsi = d.get("rsi_14")
            rsi5m = d.get("rsi_5m")
            macd = d.get("macd", {}) or {}
            bb = d.get("bb", {}) or {}
            atr = d.get("atr_14")
            vr = d.get("vol_ratio") or 1.0
            regime = d.get("market_regime", "?")
            mom5d = d.get("momentum_5d") or 0.0
            price = d.get("price", 0)
            chg = d.get("change_pct", 0)

            # Suggested levels
            stop_sug = round(price - (atr or 0) * 1.5, 2) if atr else "N/A"
            tp_sug = round(price + (atr or 0) * 3.75, 2) if atr else "N/A"

            lines.append(
                f"\n{icon} {ticker}  ${price:.2f} ({chg:+.2f}%)  BIAS={bias}  "
                f"REGIME={regime}\n"
                f"  RSI: daily={rsi or 'N/A'}  5m={rsi5m or 'N/A'} | "
                f"MACD: {'✅bull' if macd.get('bullish') else '❌bear'} | "
                f"BB pct_b={bb.get('pct_b', 'N/A')} | Vol {vr:.1f}x\n"
                f"  5d mom: {mom5d:+.1f}% | ATR=${atr or 'N/A'} | "
                f"Suggested stop=${stop_sug}  target=${tp_sug}\n"
                f"  BUY signals:  {', '.join(sig.get('buy_signals',[])) or 'none'}\n"
                f"  SELL signals: {', '.join(sig.get('sell_signals',[])) or 'none'}"
            )

        if news:
            lines.append("\n═══ NEWS ═══")
            for n in news[:6]:
                lines.append(f"  [{n['ticker']}] {n['headline']}")

        if self._history:
            lines.append("\n═══ RECENT DECISIONS ═══")
            for h in self._history[-4:]:
                lines.append(
                    f"  {h.get('action','?').upper()} {h.get('ticker','?')} "
                    f"conf={h.get('confidence',0):.0%}"
                )

        lines.append(
            "\n═══ TASK ═══\n"
            "Find the SINGLE best trade. Requirements:\n"
            "  ✅ 2+ aligned signals  ✅ Vol ratio > 1.2x\n"
            "  ✅ Clear stop (ATR*1.5) ✅ R:R >= 2.5:1\n"
            "  ✅ Not CHOPPY regime\n"
            "If nothing qualifies → hold.\nJSON only."
        )
        return "\n".join(lines)

    def _validate(self, d: dict):
        for f in ("action", "ticker", "quantity", "confidence"):
            if f not in d:
                raise ValueError(f"Missing field: {f}")
        if d["action"] not in ("buy", "sell", "hold"):
            raise ValueError(f"Bad action: {d['action']}")
        d["quantity"] = int(d["quantity"])
        d["confidence"] = float(d["confidence"])
