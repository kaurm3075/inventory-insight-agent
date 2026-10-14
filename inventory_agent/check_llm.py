"""Check that your API key works and which model to use.

Run from the project root:  python -m inventory_agent.check_llm
"""
from inventory_agent import config
from inventory_agent.llm import LLMError, make_llm


def main() -> None:
    try:
        llm = make_llm()
    except LLMError as exc:
        raise SystemExit(f"Setup problem: {exc}")
    print(f"Key found. Testing model: {llm.model}")
    try:
        response = llm.step([{"role": "user", "text": "Reply with the single word: ready"}], [
            {"name": "noop", "description": "Does nothing.", "parameters": {"type": "object", "properties": {}, "required": []}}
        ], "You are a test assistant.")
        print("Model replied:", response.text or f"(tool call: {[c.name for c in response.tool_calls]})")
        print("OK. Setup works.")
    except LLMError as exc:
        print("Model call failed:", exc)
        try:
            names = [m.name for m in llm._client.models.list() if "generateContent" in (m.supported_actions or [])]
            print("\nModels your key can use (set GEMINI_MODEL to one of these):")
            for n in names[:25]:
                print("  ", n.replace("models/", ""))
        except Exception as list_exc:
            print("Could not list models:", str(list_exc)[:200])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
