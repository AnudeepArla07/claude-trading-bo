"""
database.py
===========
SQLite logging for all decisions, trades, and options orders.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)
# Use absolute path for persistence on cloud platforms
DB_FILE = os.path.join(os.path.dirname(__file__), "trades.db")


class TradeDatabase:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT,
                    action      TEXT,
                    ticker      TEXT,
                    quantity    REAL,
                    confidence  REAL,
                    reason      TEXT,
                    stop_loss   REAL,
                    take_profit REAL,
                    status      TEXT,
                    note        TEXT
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT,
                    order_id      TEXT,
                    symbol        TEXT,
                    side          TEXT,
                    qty           REAL,
                    order_status  TEXT,
                    decision_json TEXT
                );
                CREATE TABLE IF NOT EXISTS options_trades (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT,
                    ticker        TEXT,
                    strategy      TEXT,
                    option_symbol TEXT,
                    contracts     INTEGER,
                    premium       REAL,
                    max_loss      REAL,
                    max_profit    REAL,
                    status        TEXT,
                    note          TEXT,
                    decision_json TEXT
                );
            """)
        log.info("💾 Database ready: %s", DB_FILE)

    def _conn(self):
        return sqlite3.connect(DB_FILE)

    def log_decision(self, decision, status, note=""):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO decisions
                  (timestamp,action,ticker,quantity,confidence,reason,
                   stop_loss,take_profit,status,note)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    decision.get("action"),
                    decision.get("ticker"),
                    decision.get("quantity"),
                    decision.get("confidence"),
                    decision.get("reason"),
                    decision.get("stop_loss"),
                    decision.get("take_profit"),
                    status,
                    note,
                ),
            )

    def log_trade(self, decision, order):
        self.log_decision(decision, "EXECUTED")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trades
                  (timestamp,order_id,symbol,side,qty,order_status,decision_json)
                VALUES (?,?,?,?,?,?,?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    order.get("order_id"),
                    order.get("symbol"),
                    order.get("side"),
                    order.get("qty"),
                    order.get("status"),
                    json.dumps(decision),
                ),
            )

    def log_options_trade(self, decision, order):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO options_trades
                  (timestamp,ticker,strategy,option_symbol,contracts,
                   premium,max_loss,max_profit,status,note,decision_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
                (
                    datetime.utcnow().isoformat(),
                    decision.get("ticker"),
                    decision.get("strategy"),
                    decision.get("option_symbol") or decision.get("long_symbol"),
                    decision.get("contracts"),
                    decision.get("limit_price") or decision.get("net_debit"),
                    decision.get("max_loss"),
                    (
                        decision.get("max_profit")
                        if isinstance(decision.get("max_profit"), (int, float))
                        else 0
                    ),
                    order.get("status", "submitted"),
                    "",
                    json.dumps(decision),
                ),
            )

    def print_summary(self):
        with self._conn() as conn:
            stock_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            opt_trades = conn.execute("SELECT COUNT(*) FROM options_trades").fetchone()[
                0
            ]
            blocked = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE status='BLOCKED'"
            ).fetchone()[0]
            total_decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[
                0
            ]
        print(f"\n{'─'*45}")
        print(f"  📊 SESSION SUMMARY")
        print(f"  Decisions analyzed : {total_decisions}")
        print(f"  Trades blocked     : {blocked}")
        print(f"  Stock trades       : {stock_trades}")
        print(f"  Options trades     : {opt_trades}")
        print(f"{'─'*45}\n")
