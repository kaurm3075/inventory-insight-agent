"""Clean the Online Retail II sales data and build a SQLite database.

Real data:      sales and returns (UK online gift retailer, Dec 2009 - Dec 2011).
Simulated data: unit cost, stock on hand and reorder levels. The dataset has no
                stock or cost columns, so they are generated with a fixed random
                seed and documented in the README. Currency is GBP.

Run from the project root:  python -m inventory_agent.build_db
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX_PATH = ROOT / "data" / "online_retail_II.xlsx"
DB_PATH = ROOT / "data" / "shop.db"
SEED = 42

# StockCodes that are fees or adjustments, not products.
NON_PRODUCT_CODES = {
    "POST", "D", "M", "BANK CHARGES", "DOT", "CRUK", "C2", "S", "B",
    "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002", "PADS",
}

# Simple keyword rules. The first matching rule wins. Order matters.
CATEGORY_RULES = [
    ("Candles and holders", ["CANDLE", "T-LIGHT", "TEALIGHT", "HOLDER", "LANTERN"]),
    ("Bags", ["BAG", "SHOPPER", "TOTE", "PURSE"]),
    ("Kitchen and dining", ["MUG", "CUP", "PLATE", "BOWL", "TEAPOT", "JAR", "CAKE", "TIN", "SPOON", "COASTER", "KITCHEN", "BOTTLE"]),
    ("Christmas", ["CHRISTMAS", "XMAS", "SANTA", "ADVENT", "REINDEER", "SNOWMAN", "WREATH"]),
    ("Home decor", ["FRAME", "CUSHION", "CLOCK", "SIGN", "DOORMAT", "MIRROR", "WALL", "HEART", "HANGING", "VASE", "LAMP", "LIGHT"]),
    ("Stationery and paper", ["CARD", "NOTEBOOK", "WRAP", "PAPER", "STICKER", "PENCIL", "PEN ", "RIBBON", "TAPE", "BOOK"]),
    ("Toys and games", ["TOY", "GAME", "DOLL", "PUZZLE", "BALLOON", "PLAYING", "CHILDREN", "KIDS"]),
    ("Garden and outdoor", ["GARDEN", "BIRD", "FLOWER", "PLANT", "WATERING", "OUTDOOR", "PICNIC"]),
]


def categorize(description: str) -> str:
    text = f" {description.upper()} "
    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def load_raw(path: Path = XLSX_PATH) -> pd.DataFrame:
    """Read both sheets of the Excel file into one DataFrame."""
    sheets = pd.read_excel(path, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sales, returns) after cleaning.

    sales:   real purchases (positive quantity and price, not a cancellation)
    returns: cancelled invoices (invoice starts with C), quantities made positive
    """
    df = raw.rename(columns={
        "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
        "Quantity": "quantity", "InvoiceDate": "sale_date", "Price": "unit_price",
        "Customer ID": "customer_id", "Country": "country",
    }).copy()
    df["invoice"] = df["invoice"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.normalize()

    df = df[~df["stock_code"].isin(NON_PRODUCT_CODES)]
    # Real product codes start with 5 digits (for example 85123A).
    df = df[df["stock_code"].str.match(r"^\d{5}")]
    df = df[df["unit_price"] > 0]
    df = df[~df["description"].str.upper().str.contains("GIFT VOUCHER", na=False)]

    is_cancel = df["invoice"].str.upper().str.startswith("C")
    returns = df[is_cancel].copy()
    returns["quantity"] = returns["quantity"].abs()
    sales = df[~is_cancel & (df["quantity"] > 0)].copy()

    sales, returns = remove_cancelled_orders(sales, returns)
    keep = ["invoice", "stock_code", "sale_date", "quantity", "unit_price", "country"]
    return sales[keep + ["description"]].reset_index(drop=True), returns[keep].reset_index(drop=True)


def remove_cancelled_orders(sales: pd.DataFrame, returns: pd.DataFrame,
                            window_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop sale lines that were fully cancelled shortly after, together with their cancellation.

    Example: invoice 581483 sold 80,995 units of one item and invoice C581484 cancelled the
    same 80,995 units the same day. Left in, that single order would dominate every ranking.
    A cancellation matches a sale of the same product and quantity within `window_days` days.
    """
    s = sales[["stock_code", "quantity", "sale_date"]].reset_index().rename(columns={"index": "sale_idx"})
    r = returns[["stock_code", "quantity", "sale_date"]].reset_index().rename(columns={"index": "ret_idx"})
    pairs = s.merge(r, on=["stock_code", "quantity"], suffixes=("_sale", "_ret"))
    gap = (pairs["sale_date_ret"] - pairs["sale_date_sale"]).dt.days
    pairs = pairs[(gap >= 0) & (gap <= window_days)]
    pairs = pairs.sort_values("sale_idx").drop_duplicates("ret_idx").drop_duplicates("sale_idx")
    return sales.drop(index=pairs["sale_idx"]), returns.drop(index=pairs["ret_idx"])


def build_products(sales: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One row per product with a clean name, category and a simulated unit cost."""
    rng = np.random.default_rng(seed)
    named = sales[sales["description"] != ""]
    # Most frequent description per product, to handle spelling variants.
    name = named.groupby("stock_code")["description"].agg(lambda s: s.mode().iloc[0])
    price = sales.groupby("stock_code")["unit_price"].median().rename("selling_price")
    products = pd.concat([name, price], axis=1).reset_index()
    products["description"] = products["description"].fillna("UNKNOWN ITEM")
    products["category"] = products["description"].map(categorize)
    products = products.sort_values("stock_code").reset_index(drop=True)
    # Simulated cost: 45 to 70 percent of the selling price.
    products["cost_price"] = (products["selling_price"] * rng.uniform(0.45, 0.70, len(products))).round(2)
    return products[["stock_code", "description", "category", "cost_price", "selling_price"]]


def build_inventory(sales: pd.DataFrame, products: pd.DataFrame, as_of: pd.Timestamp,
                    seed: int = SEED) -> pd.DataFrame:
    """Simulate stock on hand for every product as of the last date in the data.

    Normal products hold 10 to 60 days of recent demand. About 15 percent are made
    overstocked (150 to 400 days of demand) so there are slow movers to find.
    Products with no recent sales get a small random stock, so they show up as dead stock.
    """
    rng = np.random.default_rng(seed + 1)
    recent = sales[sales["sale_date"] > as_of - pd.Timedelta(days=90)]
    daily = (recent.groupby("stock_code")["quantity"].sum() / 90).rename("avg_daily_90d")
    inv = products[["stock_code"]].merge(daily, on="stock_code", how="left")
    inv["avg_daily_90d"] = inv["avg_daily_90d"].fillna(0.0)

    n = len(inv)
    cover_days = rng.uniform(10, 60, n)
    overstocked = rng.random(n) < 0.15
    cover_days[overstocked] = rng.uniform(150, 400, n)[overstocked]

    qty = np.ceil(inv["avg_daily_90d"].to_numpy() * cover_days)
    no_demand = inv["avg_daily_90d"].to_numpy() == 0
    qty[no_demand] = rng.integers(5, 120, no_demand.sum())
    inv["quantity_on_hand"] = qty.astype(int)
    # Reorder when stock drops under 14 days of demand (minimum 5 units).
    inv["reorder_level"] = np.maximum(5, np.ceil(inv["avg_daily_90d"] * 14)).astype(int)
    inv["last_restock_date"] = (as_of - pd.to_timedelta(rng.integers(5, 120, n), unit="D")).strftime("%Y-%m-%d")
    return inv[["stock_code", "quantity_on_hand", "reorder_level", "last_restock_date"]]


def write_database(sales: pd.DataFrame, returns: pd.DataFrame, products: pd.DataFrame,
                   inventory: pd.DataFrame, as_of: pd.Timestamp, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    out_sales = sales.drop(columns=["description"]).copy()
    out_sales["sale_date"] = out_sales["sale_date"].dt.strftime("%Y-%m-%d")
    out_returns = returns.copy()
    out_returns["sale_date"] = out_returns["sale_date"].dt.strftime("%Y-%m-%d")
    meta = pd.DataFrame({"key": ["as_of_date", "currency", "data_note"],
                         "value": [as_of.strftime("%Y-%m-%d"), "GBP",
                                   "Sales are real (Online Retail II). Cost, stock and reorder levels are simulated."]})
    with sqlite3.connect(db_path) as con:
        products.to_sql("products", con, index=False)
        inventory.to_sql("inventory", con, index=False)
        out_sales.to_sql("sales", con, index=False)
        out_returns.to_sql("returns", con, index=False)
        meta.to_sql("meta", con, index=False)
        con.execute("CREATE INDEX idx_sales_code ON sales(stock_code)")
        con.execute("CREATE INDEX idx_sales_date ON sales(sale_date)")


def main() -> None:
    if not XLSX_PATH.exists():
        raise SystemExit("Dataset not found. Run: python data/download_data.py")
    print("Reading Excel file (this takes 1 to 2 minutes)...")
    raw = load_raw()
    sales, returns = clean_transactions(raw)
    as_of = sales["sale_date"].max()
    products = build_products(sales)
    inventory = build_inventory(sales, products, as_of)
    write_database(sales, returns, products, inventory, as_of)
    print(f"Database written: {DB_PATH}")
    print(f"  as-of date : {as_of.date()}")
    print(f"  products   : {len(products):,}")
    print(f"  sales rows : {len(sales):,}")
    print(f"  return rows: {len(returns):,}")


if __name__ == "__main__":
    main()
