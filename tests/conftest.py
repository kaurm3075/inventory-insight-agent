import pandas as pd
import pytest

from inventory_agent import build_db as b
from inventory_agent import queries as q


def make_raw():
    rows = [
        ("1001", "10001", "RED MUG", 5, "2011-11-01", 2.0, 1, "United Kingdom"),
        ("1002", "10001", "RED MUG", 3, "2011-12-01", 2.0, 2, "United Kingdom"),
        ("1003", "20002", "OLD LAMP", 4, "2011-06-01", 10.0, 3, "United Kingdom"),
        ("1004", "30003", "BIG BAG", 50, "2011-12-05", 1.0, 4, "United Kingdom"),
    ]
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@pytest.fixture()
def shop_con(tmp_path):
    sales, returns = b.clean_transactions(make_raw())
    as_of = sales["sale_date"].max()
    products = b.build_products(sales)
    inventory = b.build_inventory(sales, products, as_of)
    inventory.loc[inventory.stock_code == "20002", "quantity_on_hand"] = 10
    inventory.loc[inventory.stock_code == "30003", "quantity_on_hand"] = 20
    inventory.loc[inventory.stock_code == "10001", "quantity_on_hand"] = 500
    db = tmp_path / "shop.db"
    b.write_database(sales, returns, products, inventory, as_of, db_path=db)
    con = q.connect(db)
    yield con
    con.close()
