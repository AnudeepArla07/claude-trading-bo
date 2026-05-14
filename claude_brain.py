"""
claude_brain.py
===============
Claude AI stock trading decision engine.

Claude acts as the FINAL decision maker on pre-filtered setups.
The data_feed pre-filters with signal scoring (4+ signals required)
so Claude only sees high-quality setups — not every ticker every cycle.

This makes live and backtest aligned:
  Live:     signals pre-filter → Claude decides → risk check → execute
  Backtest: signals pre-filter → confidence proxy → risk check → execute

Confidence proxy in backtest:  confidence = signal_count / 7
  6/7 = 0.86 ≥ MIN_CONFIDENCE (0.65) → BUY
  5/7 = 0.71 ≥ MIN_CONFIDENCE (0.65) → BUY
  4/7 = 0.57 < MIN_CONFIDENCE (0.65) → HOLD (filtered before Claude)

System prompt updated to:
  - Respect ticker type (momentum/quality/index)
  - Use ATR-based stops matching backtest ATR_PARAMS
  - Market regime as hard filter
"""

import json
import logging
import re
from typing import Optional, List

import anthropic

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an elite quantitative trader managing a $100,000 portfolio.
You are the FINAL decision maker on pre-screened setups.
Every ticker you see has already passed a 4+ signal pre-filter.

═══════════════════════════════════════════════════════
DECISION FRAMEWORK
═══════════════════════════════════════════════════════

TICKER TYPES (shown in data):
  🚀 MOMENTUM  — buy breakouts near 20-day highs in STRONG_BULL
                  Do NOT buy RSI dips on momentum tickers
                  RSI 70+ in a momentum ticker = strength, not sell
  📈 QUALITY   — buy RSI pullbacks (< 50) in BULL regime
                  Classic mean-reversion in uptrend
  📊 INDEX     — buy RSI pullbacks (< 42) in BULL, wider stops

REGIME RULES (non-negotiable):
  STRONG_BULL or BULL  → may trade in direction of trend
  CHOPPY               → HOLD, do not trade
  BEAR or STRONG_BEAR  → HOLD, do not trade, protect capital

ATR-BASED STOPS (use these levels, not arbitrary ones):
  Momentum: stop = entry - ATR * 2.5   target = entry + ATR * 12.0
  Quality:  stop = entry - ATR * 3.5   target = entry + ATR * 10.0
  Index:    stop = entry - ATR * 4.0   target = entry + ATR * 8.0

CONFIDENCE SCORING:
  0.85 - 1.00 → STRONG_BUY (all signals perfectly aligned)
  0.70 - 0.84 → BUY (strong setup, proceed)
  0.65 - 0.69 → marginal, only trade if top-tier ticker type
  below 0.65  → HOLD (insufficient conviction)

═══════════════════════════════════════════════════════
ENTRY REQUIREMENTS
═══════════════════════════════════════════════════════

Momentum entry (all required):
  ✅ STRONG_BULL regime (not just BULL)
  ✅ Price near 20-day high (within 1-2%)
  ✅ MACD bullish and accelerating
  ✅ Volume > 2x average
  ✅ 5-day momentum > 5%

Quality entry (all required):
  ✅ BULL or STRONG_BULL regime
  ✅ RSI pulled back below 50
  ✅ Price above EMA21
  ✅ MACD bullish
  ✅ Volume > 1.2x average

═══════════════════════════════════════════════════════
OUTPUT — STRICT JSON ONLY
═══════════════════════════════════════════════════════

{
  "action":       "buy" | "sell" | "hold",
  "ticker":       "NVDA",
  "quantity":     10,
  "entry_price":  450.00,
  "stop_loss":    438.75,
  "take_profit":  570.00,
  "risk_reward":  3.0,
  "confidence":   0.87,
  "signal_count": 6,
  "ticker_type":  "momentum",
  "reason":       "Under 80 words explaining why."
}

For hold:
{
  "action":     "hold",
  "ticker":     "MARKET",
  "quantity":   0,
  "confidence": 0.0,
  "reason":     "Why no trade this cycle."
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
        """
        Analyze pre-filtered market data and decide the best trade.
        Only tickers with entry_signal=True are shown to Claude.
        """
        # Pre-filter: only show Claude tickers that pass signal threshold
        # Guard against non-dict values that can arrive from bad API snapshots
        filtered_data = {}
        for t, d in market_data.items():
            if not isinstance(d, dict):
                log.warning(
                    "⚠️  Skipping %s — expected dict, got %s", t, type(d).__name__
                )
                continue
            if d.get("signal_count", 0) >= self.config.MIN_SIGNALS_PRE_FILTER or d.get(
                "entry_signal", False
            ):
                filtered_data[t] = d

        if not filtered_data:
            log.info("🧠 Claude: no pre-filtered setups this cycle → HOLD")
            return {
                "action": "hold",
                "ticker": "MARKET",
                "quantity": 0,
                "confidence": 0.0,
                "reason": "No qualifying setups passed signal pre-filter.",
            }

        prompt = self._build_prompt(portfolio, filtered_data, news)

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
        ]
        for p in portfolio.get("positions", []):
            lines.append(
                f"  {p['symbol']}: {p['qty']:.0f} shares | "
                f"entry ${p['avg_entry']:.2f} | "
                f"now ${p['current_price']:.2f} | "
                f"P&L ${p['unrealized_pl']:+.2f}"
            )

        lines.append("\n═══ PRE-FILTERED SETUPS (4+ signals) ═══")

        # Sort by signal count descending — skip any non-dict entries defensively
        sorted_data = sorted(
            [(t, d) for t, d in market_data.items() if isinstance(d, dict)],
            key=lambda x: x[1].get("signal_count", 0),
            reverse=True,
        )

        for ticker, d in sorted_data:
            sig = d.get("signals") if isinstance(d.get("signals"), dict) else {}
            bias = sig.get("bias", "NEUTRAL")
            t_type = d.get("ticker_type", "quality")
            regime = d.get("market_regime", "?")
            rsi = d.get("rsi_14")
            atr = d.get("atr_14")
            vr = d.get("vol_ratio") or 1.0
            mom5d = d.get("momentum_5d") or 0.0
            price = d.get("price", 0)
            chg = d.get("change_pct", 0)
            sigs = d.get("signal_count", 0)
            entry = d.get("entry_signal", False)
            macd_b = d.get("macd_bullish", False)
            conf_sim = sigs / 7.0  # simulated confidence

            # ATR-based suggested levels
            params = self.config.ATR_PARAMS.get(t_type, {"stop": 3.5, "target": 10.0})
            stop_sug = round(price - (atr or 0) * params["stop"], 2) if atr else "N/A"
            tp_sug = round(price + (atr or 0) * params["target"], 2) if atr else "N/A"

            type_icon = {"momentum": "🚀", "quality": "📈", "index": "📊"}.get(
                t_type, "•"
            )
            entry_flag = "✅ENTRY" if entry else "⚠️ watch"

            lines.append(
                f"\n{type_icon} {ticker} [{t_type}]  ${price:.2f} "
                f"({chg:+.2f}%)  sigs={sigs}/7  conf={conf_sim:.0%}  {entry_flag}\n"
                f"  Regime={regime}  Bias={bias}  RSI={rsi or 'N/A'}  "
                f"Vol={vr:.1f}x  5d={mom5d:+.1f}%  MACD={'✅' if macd_b else '❌'}\n"
                f"  ATR=${atr or 'N/A'}  Suggested: stop=${stop_sug}  target=${tp_sug}\n"
                f"  Buy signals:  {', '.join(sig.get('buy_signals', [])) or 'none'}\n"
                f"  Sell signals: {', '.join(sig.get('sell_signals', [])) or 'none'}"
            )

        if news:
            lines.append("\n═══ NEWS ═══")
            for n in news[:5]:
                lines.append(f"  [{n['ticker']}] {n['headline']}")

        if self._history:
            lines.append("\n═══ RECENT DECISIONS ═══")
            for h in self._history[-3:]:
                lines.append(
                    f"  {h.get('action','?').upper()} {h.get('ticker','?')} "
                    f"conf={h.get('confidence',0):.0%}"
                )

        lines.append(
            "\n═══ TASK ═══\n"
            "Pick the SINGLE best trade from the setups above.\n"
            "Requirements:\n"
            "  ✅ entry_signal=True preferred\n"
            "  ✅ Regime is BULL or STRONG_BULL\n"
            "  ✅ Strategy matches ticker type\n"
            "  ✅ Confidence >= 0.65\n"
            "  ✅ Use ATR-based stop/target levels shown above\n"
            "If nothing qualifies → hold.\n"
            "JSON only."
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
