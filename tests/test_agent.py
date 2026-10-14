import json

from inventory_agent.agent import InventoryAgent
from inventory_agent.llm import LLMError, LLMResponse, ToolCall
from inventory_agent.tools import ToolBox


class FakeLLM:
    """Plays back scripted responses and records what it was sent."""
    model = "fake-model"

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def step(self, messages, tools, system):
        self.calls.append([dict(m) for m in messages])
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(shop_con, script, tmp_path, **kw):
    llm = FakeLLM(script)
    return InventoryAgent(llm, ToolBox(shop_con), log_path=tmp_path / "log.jsonl", **kw), llm


def test_agent_calls_tool_then_answers(shop_con, tmp_path):
    agent, llm = make_agent(shop_con, [
        LLMResponse(tool_calls=[ToolCall("slow_movers", {"days": 60})], usage={"input_tokens": 10, "output_tokens": 2}),
        LLMResponse(text="One item is slow: OLD LAMP.", usage={"input_tokens": 20, "output_tokens": 8}),
    ], tmp_path)
    result = agent.ask("What is not selling?")
    assert result.answer == "One item is slow: OLD LAMP."
    assert [s["tool"] for s in result.steps] == ["slow_movers"]
    assert result.input_tokens == 30 and result.output_tokens == 10
    # the second model call saw the tool result
    assert llm.calls[1][-1]["role"] == "tool" and llm.calls[1][-1]["result"]["ok"]


def test_agent_stops_after_max_steps(shop_con, tmp_path):
    looping = [LLMResponse(tool_calls=[ToolCall("get_schema", {})]) for _ in range(5)]
    agent, _ = make_agent(shop_con, looping, tmp_path, max_steps=3)
    result = agent.ask("loop forever")
    assert result.error == "max_steps_reached" and len(result.steps) == 3


def test_agent_survives_llm_error(shop_con, tmp_path):
    agent, _ = make_agent(shop_con, [LLMError("The free-tier rate limit was reached.")], tmp_path)
    result = agent.ask("anything")
    assert result.error == "llm_error" and "rate limit" in result.answer


def test_blocked_sql_goes_back_to_the_model(shop_con, tmp_path):
    agent, llm = make_agent(shop_con, [
        LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "DROP TABLE sales"})]),
        LLMResponse(text="I can only run read-only queries."),
    ], tmp_path)
    result = agent.ask("delete all sales")
    assert not result.steps[0]["result"]["ok"]
    assert shop_con.execute("SELECT COUNT(*) FROM sales").fetchone()[0] == 4
    assert "read-only" in result.answer


def test_history_is_sent_for_follow_up_questions(shop_con, tmp_path):
    agent, llm = make_agent(shop_con, [LLMResponse(text="ok")], tmp_path)
    agent.ask("and by category?", history=[{"role": "user", "text": "slow stock?"},
                                          {"role": "assistant", "text": "One item."}])
    assert [m["role"] for m in llm.calls[0]] == ["user", "assistant", "user"]


def test_every_question_is_logged(shop_con, tmp_path):
    agent, _ = make_agent(shop_con, [
        LLMResponse(tool_calls=[ToolCall("run_sql", {"sql": "SELECT 1 AS x"})]),
        LLMResponse(text="done"),
    ], tmp_path)
    agent.ask("log me")
    record = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert record["question"] == "log me" and record["model"] == "fake-model"
    assert record["tool_calls"][0]["tool"] == "run_sql" and record["tool_calls"][0]["sql"] == "SELECT 1 AS x"
    assert "latency_s" in record
