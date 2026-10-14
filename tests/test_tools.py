import sqlite3

from inventory_agent.tools import ToolBox


def test_prebuilt_tool_returns_rows_and_columns(shop_con):
    result = ToolBox(shop_con).call("slow_movers", {"days": 60})
    assert result["ok"] and result["columns"][0] == "stock_code"
    assert [r[0] for r in result["rows"]] == ["20002"]


def test_run_sql_returns_sql_and_rows(shop_con):
    result = ToolBox(shop_con).call("run_sql", {"sql": "SELECT COUNT(*) AS n FROM products"})
    assert result["ok"] and result["rows"] == [[3]] and "COUNT" in result["sql"]


def test_run_sql_blocks_writes(shop_con):
    result = ToolBox(shop_con).call("run_sql", {"sql": "DELETE FROM sales"})
    assert not result["ok"] and "Blocked" in result["error"]
    count = shop_con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    assert count == 4  # nothing was deleted


def test_run_sql_limits_rows(shop_con):
    result = ToolBox(shop_con, max_rows=2).call("run_sql", {"sql": "SELECT * FROM sales"})
    assert result["row_count"] == 2 and result["truncated"] is True


def test_run_sql_reports_sql_errors(shop_con):
    result = ToolBox(shop_con).call("run_sql", {"sql": "SELECT nope FROM sales"})
    assert not result["ok"] and "SQL error" in result["error"]


def test_run_sql_timeout(shop_con):
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000) "
            "SELECT COUNT(*) FROM c")
    result = ToolBox(shop_con, timeout_s=0.2).call("run_sql", {"sql": slow})
    assert not result["ok"] and "timed out" in result["error"]


def test_unknown_tool_and_bad_args_do_not_crash(shop_con):
    box = ToolBox(shop_con)
    assert not box.call("delete_everything", {})["ok"]
    assert not box.call("slow_movers", {"bogus": 1})["ok"]


def test_arguments_are_clamped(shop_con):
    result = ToolBox(shop_con).call("slow_movers", {"days": -5, "limit": 100000})
    assert result["ok"]


def test_get_schema_lists_tables_and_as_of(shop_con):
    result = ToolBox(shop_con).call("get_schema", {})
    assert {"products", "inventory", "sales", "returns", "meta"} <= set(result["tables"])
    assert result["as_of_date"] == "2011-12-05"


def test_every_spec_has_a_matching_tool(shop_con):
    box = ToolBox(shop_con)
    assert {s["name"] for s in box.specs()} == set(box._tools)
