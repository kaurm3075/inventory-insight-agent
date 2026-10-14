import pytest

from inventory_agent.guardrails import GuardrailError, validate_sql


@pytest.mark.parametrize("sql", [
    "SELECT * FROM products",
    "select count(*) from sales;",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
])
def test_allows_read_only_queries(sql):
    assert validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM sales",
    "DROP TABLE products",
    "UPDATE inventory SET quantity_on_hand = 0",
    "INSERT INTO products VALUES (1)",
    "SELECT 1; DROP TABLE sales",
    "SELECT * FROM sales -- sneaky",
    "SELECT /* hi */ 1",
    "PRAGMA table_info(sales)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT name FROM sqlite_master",
    "",
    "   ",
    "EXPLAIN SELECT 1",
])
def test_blocks_everything_else(sql):
    with pytest.raises(GuardrailError):
        validate_sql(sql)
