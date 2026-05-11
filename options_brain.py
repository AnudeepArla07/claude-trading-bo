"""
options_brain.py
================
Claude AI options decision engine.
Fixed: multi-timeframe trend scoring prevents puts on bullish stocks
       and handles intraday volatility correctly.
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

═══════════════════════════════════════════════════════
STRATEGY SELECTION
═══════════════════════════════════════════════════════

  LONG_CALL        → Strong bull + IV rank < 40%
  BULL_CALL_SPREAD → Moderate bull + IV rank > 50%
  LONG_PUT         → Strong bear + IV rank < 40%
  BEAR_PUT_SPREAD  → Moderate bear + IV rank > 50%

═══════════════════════════════════════════════════════
CRITICAL TREND ALIGNMENT — NEVER VIOLATE
═══════════════════════════════════════════════════════

  BULL trend  → ONLY CALLS or BULL SPREADS
  BEAR trend  → ONLY PUTS or BEAR SPREADS
  CHOPPY      → HOLD — no options

  RSI 70+ in a BULL trend = momentum continuation NOT reversal
  RSI 30- in a BEAR trend = momentum continuation NOT reversal
  Never fight a strong trend with opposing options

MULTI-TIMEFRAME RULE:
  All 3 timeframes must agree before trading:
  Daily trend + Hourly RSI + 5-minute RSI

  For bearish options: daily BEAR + 1h RSI > 60 + 5m RSI > 65
  For bullish options: daily BULL + 1h RSI < 40 + 5m RSI < 35
  Mixed signals = HOLD and wait for alignment

═══════════════════════════════════════════════════════
OPTION SELECTION
═══════════════════════════════════════════════════════

  Delta     : 0.35 – 0.45
  DTE       : 14 – 30 days
  Spread    : < 15% of mid
  Volume    : > 10 contracts
  Max loss  : < $2,000

EXIT RULES (bot handles automatically):
  Profit target : 80% gain on premium
  Stop loss     : 50% loss on premium
  Time stop     : exit before 5 DTE

═══════════════════════════════════════════════════════
OUTPUT — STRICT JSON ONLY
═══════════════════════════════════════════════════════

Single-leg:
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
  "reason":        "Under 80 words. Must explain trend alignment."
}

Spread:
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
  "reason":       "Under 80 words."
}

No trade:
{
  "action":     "hold",
  "reason":     "Explain specifically: trend conflict / choppy / no setup.",
  "confidence": 0.0
}
"""


class OptionsBrain:
    def __init__(self, config):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def analyze(
        self, ticker: str, stock_data: dict, options_data: dict
    ) -> Optional[dict]:
        """
        Analyze options opportunity.
        Multi-timeframe trend check runs before Claude.
        """

        # ── Multi-timeframe trend check ────────────────────────
        trend_result = self._multi_timeframe_check(ticker, stock_data, options_data)
        if trend_result is not None:
            # Returns hold decision if blocked, or None if allowed
            if trend_result.get("action") == "hold":
                return trend_result
            # trend_result == {} means allowed to proceed

        if not options_data or not options_data.get("recommendations"):
            return None

        prompt = self._build_prompt(ticker, stock_data, options_data)

        for attempt in range(2):
            try:
                resp = self.client.messages.create(
                    model=self.config.MODEL,
                    max_tokens=1000,
                    system=OPTIONS_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = re.sub(r"```json|```", "", resp.content[0].text).strip()
                d = json.loads(raw)
                self._validate(d)

                # Post-Claude safety net
                if d.get("action") != "hold":
                    override = self._post_claude_check(d, stock_data)
                    if override:
                        return override

                return d

            except json.JSONDecodeError:
                if attempt == 0:
                    prompt += "\n\nReturn ONLY the JSON object."
                else:
                    log.error("Options brain JSON parse failed: %s", ticker)
                    return None
            except Exception as e:
                log.error("Options brain error %s: %s", ticker, e)
                return None

    # ─────────────────────────────────────────────────────────────
    # MULTI-TIMEFRAME TREND CHECK
    # ─────────────────────────────────────────────────────────────

    def _multi_timeframe_check(
        self, ticker: str, stock_data: dict, options_data: dict
    ) -> Optional[dict]:
        """
        Score trend across daily, hourly, and 5-minute timeframes.
        Returns:
          hold decision dict → trade blocked
          {} (empty dict)    → trade allowed
          None               → trade allowed
        """
        regime = stock_data.get("market_regime", "CHOPPY")
        rsi_1d = stock_data.get("rsi_14")
        rsi_1h = stock_data.get("rsi_1h")
        rsi_5m = stock_data.get("rsi_5m")
        mom1d = stock_data.get("momentum_1d") or 0.0
        mom5d = stock_data.get("momentum_5d") or 0.0
        signals = stock_data.get("signals", {})
        bias = signals.get("bias", "NEUTRAL")
        signal_bias = options_data.get("signal_bias", "NEUTRAL")

        # ── Score bull vs bear across all timeframes ───────────
        bull_score = 0
        bear_score = 0

        # Daily regime (highest weight)
        if regime == "STRONG_BULL":
            bull_score += 3
        elif regime == "BULL":
            bull_score += 2
        elif regime == "BEAR":
            bear_score += 2
        elif regime == "STRONG_BEAR":
            bear_score += 3
        # CHOPPY = 0

        # Daily RSI
        if rsi_1d:
            if rsi_1d < 40:
                bull_score += 1
            elif rsi_1d > 60:
                bear_score += 1

        # Hourly RSI (medium weight)
        if rsi_1h:
            if rsi_1h < 35:
                bull_score += 2
            elif rsi_1h > 65:
                bear_score += 2
            elif rsi_1h < 45:
                bull_score += 1
            elif rsi_1h > 55:
                bear_score += 1

        # 5-minute RSI (entry timing)
        if rsi_5m:
            if rsi_5m < 30:
                bull_score += 2
            elif rsi_5m > 70:
                bear_score += 2
            elif rsi_5m < 40:
                bull_score += 1
            elif rsi_5m > 60:
                bear_score += 1

        # 5-day momentum
        if mom5d > 5:
            bull_score += 2
        elif mom5d > 2:
            bull_score += 1
        elif mom5d < -5:
            bear_score += 2
        elif mom5d < -2:
            bear_score += 1

        # Today's move
        if mom1d > 3:
            bull_score += 1
        elif mom1d < -3:
            bear_score += 1

        # Signal bias
        if bias == "STRONG_BUY":
            bull_score += 2
        elif bias == "BUY":
            bull_score += 1
        elif bias == "STRONG_SELL":
            bear_score += 2
        elif bias == "SELL":
            bear_score += 1

        score_gap = bull_score - bear_score

        log.info(
            "   📊 %s trend: bull=%d bear=%d gap=%+d regime=%s " "rsi_1h=%s rsi_5m=%s",
            ticker,
            bull_score,
            bear_score,
            score_gap,
            regime,
            rsi_1h,
            rsi_5m,
        )

        # ── Decision logic ─────────────────────────────────────

        # Strong bull consensus (gap >= 4) — block puts
        if score_gap >= 4:
            if signal_bias in ("STRONG_SELL", "SELL"):
                log.info(
                    "   ⚠️  %s bull consensus (gap=%+d) — blocking puts",
                    ticker,
                    score_gap,
                )
                return {
                    "action": "hold",
                    "reason": (
                        f"Bull consensus: {ticker} score gap={score_gap:+d}. "
                        f"Regime={regime} RSI_1h={rsi_1h} RSI_5m={rsi_5m}. "
                        f"Overbought RSI in bull trend = momentum not reversal."
                    ),
                    "confidence": 0.0,
                }
            # Strong bull — allow calls
            return {}

        # Strong bear consensus (gap <= -4) — block calls
        if score_gap <= -4:
            if signal_bias in ("STRONG_BUY", "BUY"):
                log.info(
                    "   ⚠️  %s bear consensus (gap=%+d) — blocking calls",
                    ticker,
                    score_gap,
                )
                return {
                    "action": "hold",
                    "reason": (
                        f"Bear consensus: {ticker} score gap={score_gap:+d}. "
                        f"Regime={regime} RSI_1h={rsi_1h} RSI_5m={rsi_5m}. "
                        f"Oversold RSI in bear trend = momentum not reversal."
                    ),
                    "confidence": 0.0,
                }
            # Strong bear — allow puts
            return {}

        # Mixed signals (gap between -3 and +3)
        # Require BOTH hourly AND 5-minute confirmation
        if abs(score_gap) < 4:

            # Bearish options in mixed market
            if signal_bias in ("STRONG_SELL", "SELL"):
                need_1h = rsi_1h and rsi_1h > 60
                need_5m = rsi_5m and rsi_5m > 65
                if not (need_1h and need_5m):
                    log.info(
                        "   ⚠️  %s mixed signals — puts need 1h>60 AND 5m>65 "
                        "(1h=%s 5m=%s)",
                        ticker,
                        rsi_1h,
                        rsi_5m,
                    )
                    return {
                        "action": "hold",
                        "reason": (
                            f"Mixed signals ({score_gap:+d}): puts need "
                            f"1h RSI>60 AND 5m RSI>65 for confirmation. "
                            f"Current: 1h={rsi_1h} 5m={rsi_5m}"
                        ),
                        "confidence": 0.0,
                    }

            # Bullish options in mixed market
            elif signal_bias in ("STRONG_BUY", "BUY"):
                need_1h = rsi_1h and rsi_1h < 40
                need_5m = rsi_5m and rsi_5m < 35
                if not (need_1h and need_5m):
                    log.info(
                        "   ⚠️  %s mixed signals — calls need 1h<40 AND 5m<35 "
                        "(1h=%s 5m=%s)",
                        ticker,
                        rsi_1h,
                        rsi_5m,
                    )
                    return {
                        "action": "hold",
                        "reason": (
                            f"Mixed signals ({score_gap:+d}): calls need "
                            f"1h RSI<40 AND 5m RSI<35 for confirmation. "
                            f"Current: 1h={rsi_1h} 5m={rsi_5m}"
                        ),
                        "confidence": 0.0,
                    }

        # Pure choppy with no consensus — skip
        if regime == "CHOPPY" and abs(score_gap) < 2:
            log.info("   ⚪ %s CHOPPY no consensus — skipping options", ticker)
            return {
                "action": "hold",
                "reason": "CHOPPY regime with no clear trend consensus.",
                "confidence": 0.0,
            }

        # All checks passed
        return {}

    def _post_claude_check(self, decision: dict, stock_data: dict) -> Optional[dict]:
        """
        Final safety net — override if Claude ignores trend rules.
        """
        strategy = decision.get("strategy", "")
        regime = stock_data.get("market_regime", "CHOPPY")
        bias = stock_data.get("signals", {}).get("bias", "NEUTRAL")

        if strategy in ("LONG_PUT", "BEAR_PUT_SPREAD"):
            if regime in ("STRONG_BULL", "BULL") or bias in ("STRONG_BUY", "BUY"):
                log.warning(
                    "   🚫 Claude chose %s on %s stock — overriding HOLD",
                    strategy,
                    regime,
                )
                return {
                    "action": "hold",
                    "reason": f"Override: {strategy} conflicts with {regime}.",
                    "confidence": 0.0,
                }

        if strategy in ("LONG_CALL", "BULL_CALL_SPREAD"):
            if regime in ("STRONG_BEAR", "BEAR") or bias in ("STRONG_SELL", "SELL"):
                log.warning(
                    "   🚫 Claude chose %s on %s stock — overriding HOLD",
                    strategy,
                    regime,
                )
                return {
                    "action": "hold",
                    "reason": f"Override: {strategy} conflicts with {regime}.",
                    "confidence": 0.0,
                }

        return None

    # ─────────────────────────────────────────────────────────────
    # PROMPT BUILDER
    # ─────────────────────────────────────────────────────────────

    def _build_prompt(self, ticker: str, stock_data: dict, options_data: dict) -> str:
        price = stock_data.get("price", 0)
        regime = stock_data.get("market_regime", "UNKNOWN")
        signals = stock_data.get("signals", {})
        bias = signals.get("bias", "NEUTRAL")
        rsi_1d = stock_data.get("rsi_14", "N/A")
        rsi_1h = stock_data.get("rsi_1h", "N/A")
        rsi_5m = stock_data.get("rsi_5m", "N/A")
        macd = stock_data.get("macd") or {}
        atr = stock_data.get("atr_14", "N/A")
        vr = stock_data.get("vol_ratio") or 1.0
        mom1d = stock_data.get("momentum_1d") or 0.0
        mom5d = stock_data.get("momentum_5d") or 0.0
        iv_rank = options_data.get("iv_rank")
        max_pain = options_data.get("max_pain")
        recs = options_data.get("recommendations", [])
        chains = options_data.get("chains", {})
        buys = signals.get("buy_signals", [])
        sells = signals.get("sell_signals", [])
        signal_bias = options_data.get("signal_bias", "NEUTRAL")

        # Determine allowed strategies
        if regime in ("STRONG_BULL", "BULL") or bias in ("STRONG_BUY", "BUY"):
            allowed = "LONG_CALL or BULL_CALL_SPREAD ONLY"
        elif regime in ("STRONG_BEAR", "BEAR") or bias in ("STRONG_SELL", "SELL"):
            allowed = "LONG_PUT or BEAR_PUT_SPREAD ONLY"
        else:
            allowed = "HOLD — no clear trend direction"

        iv_advice = ""
        if iv_rank is not None:
            if iv_rank < 30:
                iv_advice = "→ BUY outright (cheap)"
            elif iv_rank > 50:
                iv_advice = "→ USE SPREADS (expensive)"
            else:
                iv_advice = "→ Either viable"

        lines = [
            f"═══ OPTIONS: {ticker} ═══",
            f"",
            f"STOCK:",
            f"  Price    : ${price:.2f}  ({mom1d:+.2f}% today  {mom5d:+.2f}% 5d)",
            f"  Regime   : {regime}",
            f"  Bias     : {bias}",
            f"  RSI      : daily={rsi_1d}  1h={rsi_1h}  5m={rsi_5m}",
            f"  MACD     : {'Bullish ✅' if macd.get('bullish') else 'Bearish ❌'}",
            f"  Volume   : {vr:.1f}x avg  ATR=${atr}",
            f"  Buy sigs : {', '.join(buys) if buys else 'none'}",
            f"  Sell sigs: {', '.join(sells) if sells else 'none'}",
            f"",
            f"OPTIONS:",
            f"  IV Rank  : {f'{iv_rank:.1f}%' if iv_rank is not None else 'N/A'}  {iv_advice}",
            f"  Max Pain : ${max_pain:.0f}" if max_pain else "  Max Pain : N/A",
            f"",
            f"⚠️  ALLOWED: {allowed}",
            f"",
            f"CHAIN:",
        ]

        for expiry, cdata in chains.items():
            lines.append(f"\n  Expiry {expiry}:")
            lines.append("  CALLS:")
            for c in cdata.get("calls", []):
                lines.append(
                    f"    ${c['strike']:.0f}C  mid=${c['mid']:.2f}  "
                    f"Δ={c['delta']:.2f}  IV={c['iv']:.0f}%  "
                    f"vol={c['volume']}  {c['moneyness']}"
                )
            lines.append("  PUTS:")
            for p in cdata.get("puts", []):
                lines.append(
                    f"    ${p['strike']:.0f}P  mid=${p['mid']:.2f}  "
                    f"Δ={p['delta']:.2f}  IV={p['iv']:.0f}%  "
                    f"vol={p['volume']}  {p['moneyness']}"
                )

        lines.append("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recs, 1):
            mp = rec.get("max_profit")
            mp_str = mp if isinstance(mp, str) else f"${mp:.0f}"
            lines.append(
                f"  {i}. {rec['strategy']} exp={rec.get('expiry','?')} "
                f"DTE={rec.get('dte','?')}"
            )
            lines.append(f"     {rec['reason']}")
            lines.append(f"     MaxLoss=${rec['max_loss']:.0f}  " f"MaxProfit={mp_str}")

        lines.append(f"""
═══ TASK ═══
Regime={regime} Bias={bias} Signal={signal_bias}
Allowed: {allowed}

Requirements:
  ✅ Strategy matches allowed above
  ✅ MaxProfit:MaxLoss >= 2:1
  ✅ Volume > 10  ✅ DTE 7-45  ✅ Spread < 15%
  ✅ Max loss < $2,000

JSON only.
""")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────────

    def _validate(self, d: dict):
        if "action" not in d:
            raise ValueError("Missing action")
        if d["action"] not in ("buy", "sell", "hold"):
            raise ValueError(f"Bad action: {d['action']}")
        if d["action"] != "hold":
            for f in ("strategy", "ticker", "contracts", "confidence"):
                if f not in d:
                    raise ValueError(f"Missing: {f}")
            d["contracts"] = int(d["contracts"])
            d["confidence"] = float(d["confidence"])
