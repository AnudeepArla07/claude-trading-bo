"""
dashboard.py
============
Terminal dashboard. Run in a second terminal alongside bot.py.
Usage: python dashboard.py
"""
import time
import sqlite3
from datetime import datetime


def clear():
    print("\033[H\033[J", end="")


def fetch():
    try:
        conn = sqlite3.connect("trades.db")
        decisions = conn.execute(
            "SELECT timestamp,action,ticker,quantity,confidence,status,reason "
            "FROM decisions ORDER BY id DESC LIMIT 20"
        ).fetchall()
        stock_trades = conn.execute(
            "SELECT timestamp,symbol,side,qty,order_status "
            "FROM trades ORDER BY id DESC LIMIT 5"
        ).fetchall()
        opt_trades = conn.execute(
            "SELECT timestamp,ticker,strategy,contracts,premium,max_loss,status "
            "FROM options_trades ORDER BY id DESC LIMIT 5"
        ).fetchall()
        conn.close()
        return decisions, stock_trades, opt_trades
    except Exception:
        return [], [], []


def run():
    while True:
        decisions, stocks, opts = fetch()
        clear()
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"╔══════════════════════════════════════════════════════════════╗")
        print(f"║   🤖  Claude Trading Bot  |  Stocks + Options  |  {now}  ║")
        print(f"╚══════════════════════════════════════════════════════════════╝")

        print("\n📋 Recent Decisions:")
        print(f"  {'Time':<20} {'Act':<5} {'Ticker':<6} {'Conf':<6} {'Status':<12} Reason")
        print("  " + "─" * 85)
        for ts, action, ticker, qty, conf, status, reason in decisions:
            icons = {"buy":"🟢","sell":"🔴","hold":"⚪"}
            sicons = {"EXECUTED":"✅","BLOCKED":"🛑","FAILED":"❌",
                      "HOLD":"⚪","EXECUTED_OPTIONS":"🎯","OPTIONS_EXIT":"🔔"}
            a_icon = icons.get(action or "", "❓")
            s_icon = sicons.get(status or "", "")
            c_str  = f"{float(conf):.0%}" if conf else "N/A"
            r_str  = (reason or "")[:50]
            print(f"  {(ts or '')[:19]:<20} {a_icon}{(action or ''):<4} "
                  f"{(ticker or ''):<6} {c_str:<6} {s_icon}{(status or ''):<11} {r_str}")

        print("\n📈 Stock Trades:")
        if stocks:
            for ts, sym, side, qty, ostatus in stocks:
                icon = "🟢" if side == "buy" else "🔴"
                print(f"  {(ts or '')[:19]}  {icon}{side:<5} {sym:<6} x{qty}  {ostatus}")
        else:
            print("  None yet.")

        print("\n🎯 Options Trades:")
        if opts:
            print(f"  {'Time':<20} {'Ticker':<6} {'Strategy':<20} {'Contracts':<10} {'Premium':<10} MaxLoss")
            for ts, ticker, strategy, contracts, premium, max_loss, status in opts:
                icon = "🎯" if "CALL" in (strategy or "") else "🔻"
                print(f"  {(ts or '')[:19]:<20} {icon}{(ticker or ''):<5} "
                      f"{(strategy or ''):<20} {str(contracts):<10} "
                      f"${float(premium or 0):<9.2f} ${float(max_loss or 0):.0f}")
        else:
            print("  None yet.")

        print(f"\n  Refreshing in 30s... (Ctrl+C to exit)")
        time.sleep(30)


if __name__ == "__main__":
    run()
