# 🤖 Claude Trading Bot — Stocks + Options

A dual-layer AI trading bot using **Claude Sonnet** as the decision brain
and **Alpaca** for paper/live trading execution.

---

## 📁 Files

| File | Purpose |
|---|---|
| `bot.py` | Main runner — start here |
| `config.py` | All settings and API keys |
| `data_feed.py` | Market data + 15 technical indicators |
| `claude_brain.py` | Claude AI stock trading decisions |
| `options_brain.py` | Claude AI options trading decisions |
| `options_feed.py` | Options chain, Greeks, IV rank, max pain |
| `options_broker.py` | Options order execution + auto-exit |
| `broker.py` | Stock order execution |
| `risk_manager.py` | Hard risk rules (override Claude) |
| `scanner.py` | Dynamic morning watchlist scan |
| `database.py` | SQLite trade logging |
| `dashboard.py` | Terminal monitoring UI |
| `requirements.txt` | Python dependencies |

---

## ⚡ Quick Start

### 1. Get API Keys (both free to start)
- **Anthropic**: https://console.anthropic.com
- **Alpaca**: https://alpaca.markets (use Paper Trading)

### 2. Add your keys to config.py
```python
ANTHROPIC_API_KEY  = "sk-ant-..."
ALPACA_API_KEY     = "PK..."
ALPACA_SECRET_KEY  = "..."
ALPACA_BASE_URL    = "https://paper-api.alpaca.markets"
```

### 3. Install and run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python bot.py
```

### 4. Monitor in second terminal
```bash
source venv/bin/activate
python dashboard.py
```

---

## 🎯 How It Works

**Every 10 minutes the bot:**
1. Scans market → picks best 12 tickers dynamically
2. Fetches quotes + computes RSI, MACD, Bollinger Bands, EMA, ATR, Volume ratio
3. Generates STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL signal per ticker
4. Sends full technical context to Claude → stock trade decision
5. Sends same signals to options Claude → options trade decision
6. Risk manager checks both (confidence, R:R, daily loss, drawdown)
7. Executes approved trades via Alpaca
8. Checks open options for 80% profit target or 50% stop-loss
9. Trailing stops on stock positions (2x ATR)

---

## 🛡️ Risk Management

| Rule | Setting | Purpose |
|---|---|---|
| Min confidence | 72% | Claude must be sure |
| Daily loss limit | 3% | Halt trading for the day |
| Max drawdown | 8% | Halt if portfolio down 8% from peak |
| Max trades/day | 15 | Prevent overtrading |
| Consecutive losses | 3 | Cool-down after 3 losses in a row |
| Options max loss | 2% per trade | Cap options risk |
| Stock position | 15% max | Diversification |
| Trailing stop | 2x ATR | Lock in profits |

---

## 📊 Options Layer

The options layer amplifies the same stock signals with 5-10x leverage:

| Signal | IV Rank | Strategy | Leverage |
|---|---|---|---|
| STRONG_BUY | < 40% | Long Call | 5-10x |
| BUY | > 50% | Bull Call Spread | 3-5x |
| STRONG_SELL | < 40% | Long Put | 5-10x |
| SELL | > 50% | Bear Put Spread | 3-5x |

Auto-exit rules:
- Close at **+80% gain** on premium
- Close at **-50% loss** on premium
- Close all at **3:45 PM ET** (before market close)

---

## 🔴 Going Live

When ready to trade real money:
1. Paper trade for 30+ days and verify profitability
2. Change `ALPACA_BASE_URL` to `"https://api.alpaca.markets"`
3. Use your **live** Alpaca keys (different from paper keys)
4. Start with $1,000–$5,000 max
5. Scale up only after 60+ days of live profitability

---

## ⚠️ Disclaimer

Educational and paper trading purposes only. Not financial advice.
Past performance does not guarantee future results.
