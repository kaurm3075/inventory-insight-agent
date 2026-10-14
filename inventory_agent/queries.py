"""Prebuilt, read-only SQL queries. In Week 2 the agent will call these as tools.

All "last N days" windows are measured from the as-of date stored in the meta
table (the last sale date in the data), not from today's date.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "shop.db"


def connect(db_path: Path = DB_PATH, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the database in read-only mode."""
    if not Path(db_path).exists():
        raise FileNotFoundError("Database not found. Run: python -m inventory_agent.build_db")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=check_same_thread)


def get_as_of(con: sqlite3.Connection) -> str:
    row = con.execute("SELECT value FROM meta WHERE key = 'as_of_date'").fetchone()
    return row[0]


def slow_movers(con: sqlite3.Connection, days: int = 60, limit: int = 20) -> pd.DataFrame:
    """Items in stock with no sale in the last `days` days, ranked by cash blocked."""
    sql = """
    WITH last_sale AS (
        SELECT stock_code, MAX(sale_date) AS last_sale_date FROM sales GROUP BY stock_code
    ), as_of AS (SELECT value AS d FROM meta WHERE key = 'as_of_date')
    SELECT p.stock_code, p.description, p.category, i.quantity_on_hand,
           ls.last_sale_date,
           CAST(julianday((SELECT d FROM as_of)) - julianday(ls.last_sale_date) AS INTEGER) AS days_since_last_sale,
           ROUND(i.quantity_on_hand * p.cost_price, 2) AS cash_blocked
    FROM products p
    JOIN inventory i ON i.stock_code = p.stock_code
    LEFT JOIN last_sale ls ON ls.stock_code = p.stock_code
    WHERE i.quantity_on_hand > 0
      AND (ls.last_sale_date IS NULL
           OR julianday((SELECT d FROM as_of)) - julianday(ls.last_sale_date) > :days)
    ORDER BY cash_blocked DESC
    LIMIT :limit
    """
    return pd.read_sql_query(sql, con, params={"days": days, "limit": limit})


def cash_blocked_by_category(con: sqlite3.Connection, days: int = 60) -> pd.DataFrame:
    """Total cash tied up in slow-moving stock, grouped by category."""
    sql = """
    WITH last_sale AS (
        SELECT stock_code, MAX(sale_date) AS last_sale_date FROM sales GROUP BY stock_code
    ), as_of AS (SELECT value AS d FROM meta WHERE key = 'as_of_date')
    SELECT p.category,
           COUNT(*) AS slow_items,
           SUM(i.quantity_on_hand) AS units_in_stock,
           ROUND(SUM(i.quantity_on_hand * p.cost_price), 2) AS cash_blocked
    FROM products p
    JOIN inventory i ON i.stock_code = p.stock_code
    LEFT JOIN last_sale ls ON ls.stock_code = p.stock_code
    WHERE i.quantity_on_hand > 0
      AND (ls.last_sale_date IS NULL
           OR julianday((SELECT d FROM as_of)) - julianday(ls.last_sale_date) > :days)
    GROUP BY p.category
    ORDER BY cash_blocked DESC
    """
    return pd.read_sql_query(sql, con, params={"days": days})


def reorder_list(con: sqlite3.Connection, max_cover_days: int = 14, limit: int = 20) -> pd.DataFrame:
    """Items that will run out within `max_cover_days`, based on the last 30 days of sales."""
    sql = """
    WITH as_of AS (SELECT value AS d FROM meta WHERE key = 'as_of_date'),
    recent AS (
        SELECT stock_code, SUM(quantity) / 30.0 AS avg_daily_30d
        FROM sales
        WHERE julianday((SELECT d FROM as_of)) - julianday(sale_date) < 30
        GROUP BY stock_code
    )
    SELECT p.stock_code, p.description, p.category, i.quantity_on_hand,
           ROUND(r.avg_daily_30d, 2) AS avg_daily_sales,
           ROUND(i.quantity_on_hand / r.avg_daily_30d, 1) AS days_of_cover,
           i.reorder_level
    FROM products p
    JOIN inventory i ON i.stock_code = p.stock_code
    JOIN recent r ON r.stock_code = p.stock_code
    WHERE r.avg_daily_30d > 0
      AND i.quantity_on_hand / r.avg_daily_30d <= :max_cover_days
    ORDER BY days_of_cover ASC
    LIMIT :limit
    """
    return pd.read_sql_query(sql, con, params={"max_cover_days": max_cover_days, "limit": limit})


def top_sellers(con: sqlite3.Connection, days: int = 30, limit: int = 10) -> pd.DataFrame:
    """Best-selling items by revenue in the last `days` days."""
    sql = """
    WITH as_of AS (SELECT value AS d FROM meta WHERE key = 'as_of_date')
    SELECT p.stock_code, p.description, p.category,
           SUM(s.quantity) AS units_sold,
           ROUND(SUM(s.quantity * s.unit_price), 2) AS revenue
    FROM sales s JOIN products p ON p.stock_code = s.stock_code
    WHERE julianday((SELECT d FROM as_of)) - julianday(s.sale_date) < :days
    GROUP BY p.stock_code
    ORDER BY revenue DESC
    LIMIT :limit
    """
    return pd.read_sql_query(sql, con, params={"days": days, "limit": limit})


def monthly_sales(con: sqlite3.Connection, months: int = 12) -> pd.DataFrame:
    """Revenue and units per calendar month, most recent months last."""
    sql = """
    SELECT strftime('%Y-%m', sale_date) AS month,
           SUM(quantity) AS units_sold,
           ROUND(SUM(quantity * unit_price), 2) AS revenue
    FROM sales
    GROUP BY month
    ORDER BY month DESC
    LIMIT :months
    """
    df = pd.read_sql_query(sql, con, params={"months": months})
    return df.sort_values("month").reset_index(drop=True)
