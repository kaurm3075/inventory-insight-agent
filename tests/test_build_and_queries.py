import sqlite3

import pandas as pd
import pytest

from inventory_agent import build_db as b
from inventory_agent import queries as q


def raw_frame():
    rows = [
        # invoice, code, description, qty, date, price, customer, country
        ("1001", "10001", "RED MUG", 5, "2011-11-01", 2.0, 1, "United Kingdom"),
        ("1002", "10001", "RED MUG", 3, "2011-12-01", 2.0, 2, "United Kingdom"),
        ("1003", "20002", "OLD LAMP", 4, "2011-06-01", 10.0, 3, "United Kingdom"),
        ("1004", "30003", "BIG BAG", 50, "2011-12-05", 1.0, 4, "United Kingdom"),
        ("1005", "POST", "POSTAGE", 1, "2011-12-05", 15.0, 5, "United Kingdom"),  # not a product
        ("1006", "10001", "RED MUG", -2, "2011-12-02", 2.0, 6, "United Kingdom"),  # negative quantity
        ("1007", "40004", "WASTE ITEM", 1, "2011-12-02", 0.0, 7, "United Kingdom"),  # zero price
        ("1008", "50005", "GIANT ORDER", 9000, "2011-12-03", 1.0, 8, "United Kingdom"),
        ("C1009", "50005", "GIANT ORDER", 9000, "2011-12-03", 1.0, 8, "United Kingdom"),  # cancels it
        ("C1010", "10001", "RED MUG", 1, "2011-12-04", 2.0, 1, "United Kingdom"),  # real return
    ]
    return pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                       "InvoiceDate", "Price", "Customer ID", "Country"]).assign(
        InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"]))


def test_cleaning_removes_non_products_and_bad_rows():
    sales, returns = b.clean_transactions(raw_frame())
    assert set(sales["stock_code"]) == {"10001", "20002", "30003"}
    assert (sales["quantity"] > 0).all() and (sales["unit_price"] > 0).all()


def test_fully_cancelled_order_is_removed_from_both_tables():
    sales, returns = b.clean_transactions(raw_frame())
    assert "50005" not in set(sales["stock_code"])
    assert "50005" not in set(returns["stock_code"])


def test_real_return_is_kept_with_positive_quantity():
    _, returns = b.clean_transactions(raw_frame())
    assert len(returns) == 1
    assert returns.iloc[0]["stock_code"] == "10001" and returns.iloc[0]["quantity"] == 1


@pytest.fixture()
def con(tmp_path):
    sales, returns = b.clean_transactions(raw_frame())
    as_of = sales["sale_date"].max()
    products = b.build_products(sales)
    inventory = b.build_inventory(sales, products, as_of)
    # Make stock levels predictable for the assertions below.
    inventory.loc[inventory.stock_code == "20002", "quantity_on_hand"] = 10   # no sale since June
    inventory.loc[inventory.stock_code == "30003", "quantity_on_hand"] = 20   # 50 sold in 30 days
    inventory.loc[inventory.stock_code == "10001", "quantity_on_hand"] = 500
    db = tmp_path / "test.db"
    b.write_database(sales, returns, products, inventory, as_of, db_path=db)
    connection = q.connect(db)
    yield connection
    connection.close()


def test_as_of_is_last_sale_date(con):
    assert q.get_as_of(con) == "2011-12-05"


def test_slow_movers_finds_the_old_lamp(con):
    df = q.slow_movers(con, days=60)
    assert list(df["stock_code"]) == ["20002"]
    assert df.iloc[0]["days_since_last_sale"] > 60
    assert df.iloc[0]["cash_blocked"] > 0


def test_reorder_list_flags_fast_seller_with_low_stock(con):
    df = q.reorder_list(con, max_cover_days=14)
    assert "30003" in set(df["stock_code"])
    assert "10001" not in set(df["stock_code"])


def test_database_is_read_only(con):
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM sales")


def test_monthly_sales_totals(con):
    df = q.monthly_sales(con, months=12)
    assert df["units_sold"].sum() == 5 + 3 + 4 + 50
