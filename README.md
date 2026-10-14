# Retail Inventory Insight Agent

An AI assistant that answers a shop owner's stock questions in plain English
("Which items have not sold in 60 days, and how much cash is stuck in them?").

**Status: Week 1 of 6 (data and SQL queries).** The agent, forecasting, evaluation and app
come in later weeks. Full project documentation is in the shared Google Doc / Claude doc.

## What is real and what is simulated

| Data | Source |
|---|---|
| Sales and returns | Real: UCI Online Retail II (UK online gift retailer, Dec 2009 to Dec 2011) |
| Product categories | Derived from product names with keyword rules (about a third land in "Other") |
| Unit cost, stock on hand, reorder levels | **Simulated** with a fixed random seed (the dataset has no stock or cost data) |
| Currency | GBP |

"Last 60 days" always means 60 days before the as-of date (the last sale in the data,
2011-12-09), not today. December 2011 is a partial month.

Cleaning rules: non-product codes (postage, fees) removed, zero prices removed, gift
vouchers removed, and orders that were fully cancelled within 7 days are removed together
with their cancellation (this removes two single orders of 74,215 and 80,995 units that
would otherwise dominate every ranking).

## Run it

```bash
pip install -r requirements.txt
python data/download_data.py            # about 45 MB
python -m inventory_agent.build_db      # 1 to 2 minutes, creates data/shop.db
python -m inventory_agent.run_queries   # prints the 5 prebuilt queries
pytest -q                               # unit tests
```

## Project layout

```
data/download_data.py          downloads the dataset
inventory_agent/build_db.py   cleaning, simulated stock, SQLite database
inventory_agent/queries.py    prebuilt read-only SQL queries (the agent's tools in Week 2)
inventory_agent/run_queries.py prints the query results
tests/                        unit tests
```
