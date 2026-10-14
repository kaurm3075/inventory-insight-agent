"""Run every prebuilt query and print the results.

Run from the project root:  python -m inventory_agent.run_queries
"""
import pandas as pd

from inventory_agent import queries as q

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 38)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(df.to_string(index=False) if not df.empty else "(no rows)")


def main() -> None:
    con = q.connect()
    print(f"As-of date: {q.get_as_of(con)}  (currency: GBP)")
    show("Slow movers: no sale in 60 days, ranked by cash blocked", q.slow_movers(con, days=60, limit=10))
    show("Cash blocked in slow stock by category (60 days)", q.cash_blocked_by_category(con, days=60))
    show("Reorder list: under 14 days of cover", q.reorder_list(con, max_cover_days=14, limit=10))
    show("Top 10 sellers in the last 30 days", q.top_sellers(con, days=30, limit=10))
    show("Monthly sales, last 12 months", q.monthly_sales(con, months=12))


if __name__ == "__main__":
    main()
