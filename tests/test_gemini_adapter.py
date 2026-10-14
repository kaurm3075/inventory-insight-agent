"""Tests the Gemini adapter against fake SDK responses, so no API key or network is needed."""
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")
from google.genai import types

from inventory_agent.llm import GeminiClient, LLMError, ToolCall

TOOLS = [{"name": "slow_movers", "description": "x", "parameters": {"type": "object", "properties": {}, "required": []}}]


class StubModels:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.sent = response, error, None

    def generate_content(self, model, contents, config):
        self.sent = {"model": model, "contents": contents, "config": config}
        if self.error:
            raise self.error
        return self.response


def client_with(models):
    c = GeminiClient.__new__(GeminiClient)
    c.model, c._client = "test-model", SimpleNamespace(models=models)
    return c


def response(parts, usage=(5, 3)):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=usage[0], candidates_token_count=usage[1]))


def test_parses_function_calls_and_usage():
    parts = [types.Part(function_call=types.FunctionCall(name="slow_movers", args={"days": 30}))]
    out = client_with(StubModels(response(parts))).step([{"role": "user", "text": "hi"}], TOOLS, "sys")
    assert out.tool_calls == [ToolCall("slow_movers", {"days": 30})]
    assert out.usage == {"input_tokens": 5, "output_tokens": 3} and out.raw is not None


def test_parses_plain_text_and_ignores_thoughts():
    parts = [types.Part(text="thinking...", thought=True), types.Part(text="The answer.")]
    out = client_with(StubModels(response(parts))).step([{"role": "user", "text": "hi"}], TOOLS, "sys")
    assert out.text == "The answer." and not out.tool_calls


def test_tool_results_are_grouped_and_raw_content_is_reused():
    raw = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="slow_movers", args={}))])
    messages = [
        {"role": "user", "text": "q"},
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("slow_movers", {})], "raw": raw},
        {"role": "tool", "name": "slow_movers", "result": {"ok": True}},
        {"role": "tool", "name": "get_schema", "result": {"ok": True}},
    ]
    models = StubModels(response([types.Part(text="done")]))
    client_with(models).step(messages, TOOLS, "sys")
    sent = models.sent["contents"]
    assert sent[1] is raw
    assert len(sent) == 3 and len(sent[2].parts) == 2  # two tool results in one turn


def test_rate_limit_error_is_friendly():
    models = StubModels(error=RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded"))
    with pytest.raises(LLMError, match="rate limit"):
        client_with(models).step([{"role": "user", "text": "hi"}], TOOLS, "sys")


def test_bad_key_error_is_friendly():
    models = StubModels(error=RuntimeError("400 API key not valid"))
    with pytest.raises(LLMError, match="API key"):
        client_with(models).step([{"role": "user", "text": "hi"}], TOOLS, "sys")
