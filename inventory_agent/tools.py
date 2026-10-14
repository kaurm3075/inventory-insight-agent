"""The tools the agent can call. Every tool returns a plain dict so it can be logged and shown."""
from __future__ import annotations

import sqlite3
from typing import Any, Callable

import pandas as pd

from inventory_agent import config, guardrails
from inventory_agent import queries as q


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _frame_result(df: pd.DataFrame, max_rows: int, sql: str | None = None, note: str | None = None) -> dict:
    truncated = len(df) > max_rows
    df = df.head(max_rows)
    rows = df.astype(object).where(pd.notna(df), None).values.tolist()
    result = {"ok": True, "columns": list(df.columns), "rows": rows,
              "row_count": len(rows), "truncated": truncated}
    if sql:
        result["sql"] = sql
    if note:
        result["note"] = note
    return result


class ToolBox:
    """Holds the read-only database connection and exposes the tools."""

    def __init__(self, con: sqlite3.Connection, max_rows: int = config.MAX_ROWS,
                 timeout_s: float = config.QUERY_TIMEOUT_S):
        self.con = con
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._tools: dict[str, Callable[..., dict]] = {
            "get_schema": self.get_schema,
            "run_sql": self.run_sql,
            "slow_movers": self.slow_movers,
            "cash_blocked_by_category": self.cash_blocked_by_category,
            "reorder_list": self.reorder_list,
            "top_sellers": self.top_sellers,
            "monthly_sales": self.monthly_sales,
        }

    # ---- tool implementations -------------------------------------------------
    def get_schema(self) -> dict:
        tables = {}
        for (name,) in self.con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            cols = [f"{c[1]} ({c[2] or 'TEXT'})" for c in self.con.execute(f"PRAGMA table_info({name})")]
            tables[name] = cols
        return {"ok": True, "as_of_date": q.get_as_of(self.con), "currency": "GBP", "tables": tables,
                "notes": [
                    "sales.sale_date and inventory.last_restock_date are TEXT in YYYY-MM-DD format.",
                    "Stock, cost_price and reorder_level are simulated; sales are real.",
                    "Measure 'last N days' from as_of_date, not from today's date.",
                    "returns holds cancelled invoices with positive quantities.",
                ]}

    def run_sql(self, sql: str) -> dict:
        try:
            cleaned = guardrails.validate_sql(sql)
        except guardrails.GuardrailError as exc:
            return {"ok": False, "error": f"Blocked: {exc}"}
        guardrails.set_timeout(self.con, self.timeout_s)
        try:
            cursor = self.con.execute(cleaned)
            columns = [d[0] for d in cursor.description or []]
            fetched = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            msg = "Query timed out." if "interrupt" in str(exc).lower() else f"SQL error: {exc}"
            return {"ok": False, "error": msg, "sql": cleaned}
        finally:
            guardrails.clear_timeout(self.con)
        truncated = len(fetched) > self.max_rows
        rows = [list(r) for r in fetched[: self.max_rows]]
        return {"ok": True, "columns": columns, "rows": rows, "row_count": len(rows),
                "truncated": truncated, "sql": cleaned}

    def slow_movers(self, days: int = 60, limit: int = 20) -> dict:
        days, limit = _clamp(days, 1, 3650, 60), _clamp(limit, 1, self.max_rows, 20)
        df = q.slow_movers(self.con, days=days, limit=limit)
        return _frame_result(df, self.max_rows, note=f"prebuilt query slow_movers(days={days}, limit={limit})")

    def cash_blocked_by_category(self, days: int = 60) -> dict:
        days = _clamp(days, 1, 3650, 60)
        df = q.cash_blocked_by_category(self.con, days=days)
        return _frame_result(df, self.max_rows, note=f"prebuilt query cash_blocked_by_category(days={days})")

    def reorder_list(self, max_cover_days: int = 14, limit: int = 20) -> dict:
        max_cover_days, limit = _clamp(max_cover_days, 1, 365, 14), _clamp(limit, 1, self.max_rows, 20)
        df = q.reorder_list(self.con, max_cover_days=max_cover_days, limit=limit)
        return _frame_result(df, self.max_rows, note=f"prebuilt query reorder_list(max_cover_days={max_cover_days}, limit={limit})")

    def top_sellers(self, days: int = 30, limit: int = 10) -> dict:
        days, limit = _clamp(days, 1, 3650, 30), _clamp(limit, 1, self.max_rows, 10)
        df = q.top_sellers(self.con, days=days, limit=limit)
        return _frame_result(df, self.max_rows, note=f"prebuilt query top_sellers(days={days}, limit={limit})")

    def monthly_sales(self, months: int = 12) -> dict:
        months = _clamp(months, 1, 36, 12)
        df = q.monthly_sales(self.con, months=months)
        return _frame_result(df, self.max_rows, note=f"prebuilt query monthly_sales(months={months})")

    # ---- plumbing ----------------------------------------------------------------
    def call(self, name: str, args: dict | None) -> dict:
        fn = self._tools.get(name)
        if fn is None:
            return {"ok": False, "error": f"Unknown tool '{name}'. Available: {', '.join(self._tools)}"}
        try:
            return fn(**(args or {}))
        except TypeError as exc:
            return {"ok": False, "error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # never let a tool crash the whole chat
            return {"ok": False, "error": f"{name} failed: {exc}"}

    @staticmethod
    def specs() -> list[dict]:
        """Tool descriptions in a provider-neutral JSON Schema format."""
        def obj(props: dict | None = None, required: list[str] | None = None) -> dict:
            return {"type": "object", "properties": props or {}, "required": required or []}

        integer = lambda d: {"type": "integer", "description": d}
        return [
            {"name": "get_schema", "description": "List the tables and columns, the as-of date and data notes. Call this first if unsure about column names.",
             "parameters": obj()},
            {"name": "slow_movers", "description": "Items in stock that have not sold in the last N days, ranked by cash blocked (quantity on hand times cost price).",
             "parameters": obj({"days": integer("No-sale window in days, default 60"), "limit": integer("Max rows, default 20")})},
            {"name": "cash_blocked_by_category", "description": "Total cash tied up in slow-moving stock, grouped by product category.",
             "parameters": obj({"days": integer("No-sale window in days, default 60")})},
            {"name": "reorder_list", "description": "Items that will run out within N days at the last 30 days' sales pace, most urgent first.",
             "parameters": obj({"max_cover_days": integer("Days of cover threshold, default 14"), "limit": integer("Max rows, default 20")})},
            {"name": "top_sellers", "description": "Best-selling items by revenue in the last N days.",
             "parameters": obj({"days": integer("Window in days, default 30"), "limit": integer("Max rows, default 10")})},
            {"name": "monthly_sales", "description": "Revenue and units sold per calendar month. Note the last month in the data may be partial.",
             "parameters": obj({"months": integer("Number of months, default 12")})},
            {"name": "run_sql", "description": "Run ONE read-only SQLite SELECT query for questions the other tools cannot answer. Max 50 rows are returned.",
             "parameters": obj({"sql": {"type": "string", "description": "A single SELECT (or WITH ... SELECT) statement"}}, ["sql"])},
        ]
