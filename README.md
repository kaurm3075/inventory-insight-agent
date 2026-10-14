# Retail Inventory Insight Agent

An AI assistant that answers a shop owner's stock questions in plain English
("Which items have not sold in 60 days, and how much cash is stuck in them?").

**Status: Week 2 of 6 (agent, chat app, guardrails, logging).** Forecasting, multi-step reasoning and
the evaluation comparison come in later weeks.

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

## Week 2: the agent and chat app

The agent is an LLM (Google Gemini, free tier) that picks from seven tools and never touches the data directly.

```
question -> agent (LLM + tool calling) -> tools (read-only SQL) -> database
```

Tools: `get_schema`, `slow_movers`, `cash_blocked_by_category`, `reorder_list`, `top_sellers`,
`monthly_sales`, and `run_sql` for anything else.

Guardrails (`inventory_agent/guardrails.py`, tested in `tests/test_guardrails.py`):
only a single SELECT is allowed; writes, comments, PRAGMA, ATTACH and system tables are rejected;
the database connection is opened read-only; at most 50 rows are returned; each query is stopped after 5 seconds;
the agent gives up after 6 tool-call rounds. The SQL behind every answer is shown in the app.

Logging: every question is appended to `logs/agent_log.jsonl` (question, answer, tools called, SQL,
latency, token counts).

### Setup

1. Create a free API key in [Google AI Studio](https://aistudio.google.com) and store it as a secret named
   `GEMINI_API_KEY` (Codespaces secret, or `export GEMINI_API_KEY=...` for one session). Never commit it.
2. Check it works: `python -m inventory_agent.check_llm`
   (if the default model is not available to your key, it lists the ones that are; set `GEMINI_MODEL` to one of them).
3. Start the chat app: `streamlit run app/streamlit_app.py`

The tests use a scripted fake model, so `pytest -q` needs no key and no network.

## Project layout

```
data/download_data.py          downloads the dataset
inventory_agent/build_db.py   cleaning, simulated stock, SQLite database
inventory_agent/queries.py    prebuilt read-only SQL queries (the agent's tools in Week 2)
inventory_agent/run_queries.py prints the query results
inventory_agent/tools.py      the agent's tools (prebuilt queries and guarded run_sql)
inventory_agent/guardrails.py SQL validation, timeout
inventory_agent/llm.py        Gemini adapter behind a small provider-neutral interface
inventory_agent/agent.py      agent loop, system prompt, logging
inventory_agent/check_llm.py  checks the API key and model
app/streamlit_app.py          chat interface
tests/                        unit tests (45)
```
