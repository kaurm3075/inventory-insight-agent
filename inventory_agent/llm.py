"""LLM adapters. The agent talks to a small neutral interface, so the provider can be swapped.

Neutral message format (a list of dicts):
  {"role": "user", "text": "..."}
  {"role": "assistant", "text": "...", "tool_calls": [ToolCall, ...], "raw": <provider object or None>}
  {"role": "tool", "name": "slow_movers", "result": {...}}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from inventory_agent import config


class LLMError(RuntimeError):
    """Raised with a message that is safe and helpful to show to the user."""


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    usage: dict = field(default_factory=dict)  # input_tokens, output_tokens when available


class LLMClient(Protocol):
    model: str

    def step(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse: ...


class GeminiClient:
    """Google Gemini through the google-genai SDK (free tier available in Google AI Studio)."""

    def __init__(self, api_key: str, model: str = config.GEMINI_MODEL):
        try:
            from google import genai
        except ImportError as exc:
            raise LLMError("The google-genai package is missing. Run: pip install -r requirements.txt") from exc
        self.model = model
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _to_contents(self, messages: list[dict]) -> list:
        from google.genai import types

        contents: list = []
        pending_tool_parts: list = []

        def flush_tools() -> None:
            if pending_tool_parts:
                contents.append(types.Content(role="user", parts=list(pending_tool_parts)))
                pending_tool_parts.clear()

        for m in messages:
            if m["role"] == "tool":
                pending_tool_parts.append(
                    types.Part.from_function_response(name=m["name"], response={"result": m["result"]}))
                continue
            flush_tools()
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["text"])]))
            elif m.get("raw") is not None:
                contents.append(m["raw"])  # keeps provider-specific fields such as thought signatures
            else:
                parts = [types.Part(text=m["text"])] if m.get("text") else []
                for call in m.get("tool_calls", []):
                    parts.append(types.Part(function_call=types.FunctionCall(name=call.name, args=call.args)))
                contents.append(types.Content(role="model", parts=parts))
        flush_tools()
        return contents

    def step(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        from google.genai import types

        declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                                  parameters_json_schema=t["parameters"]) for t in tools]
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=self._to_contents(messages), config=cfg)
        except Exception as exc:
            raise LLMError(_friendly_error(exc, self.model)) from exc

        if not response.candidates or response.candidates[0].content is None:
            raise LLMError("The model returned no answer (it may have been blocked). Try rephrasing the question.")
        content = response.candidates[0].content
        text, calls = [], []
        for part in content.parts or []:
            if part.function_call:
                calls.append(ToolCall(name=part.function_call.name, args=dict(part.function_call.args or {})))
            elif part.text and not part.thought:
                text.append(part.text)
        usage = {}
        meta = response.usage_metadata
        if meta:
            usage = {"input_tokens": meta.prompt_token_count, "output_tokens": meta.candidates_token_count}
        return LLMResponse(text="".join(text).strip(), tool_calls=calls, raw=content, usage=usage)


def _friendly_error(exc: Exception, model: str) -> str:
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low:
        return "The free-tier rate limit was reached. Wait a minute and try again."
    if "api key" in low or "api_key" in low or "permission" in low or "401" in msg or "403" in msg:
        return "The API key was rejected. Check that GEMINI_API_KEY is set correctly."
    if "404" in msg or "not found" in low:
        return (f"Model '{model}' was not found. Run `python -m inventory_agent.check_llm` to see which "
                "models your key can use, then set GEMINI_MODEL to one of them.")
    return f"The model call failed: {msg[:300]}"


def make_llm() -> LLMClient:
    """Build the configured client. Raises LLMError with setup instructions if no key is set."""
    key = config.get_api_key()
    if not key:
        raise LLMError("No API key found. Set GEMINI_API_KEY (see README: Week 2 setup).")
    if config.LLM_PROVIDER == "gemini":
        return GeminiClient(api_key=key)
    raise LLMError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Supported: gemini.")
