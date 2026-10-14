"""Chat interface for the inventory agent.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from inventory_agent import config
from inventory_agent import queries as q
from inventory_agent.agent import InventoryAgent
from inventory_agent.llm import LLMError, make_llm
from inventory_agent.tools import ToolBox

st.set_page_config(page_title="Inventory Insight Agent", page_icon="📦", layout="wide")

EXAMPLES = [
    "Which items have not sold in 60 days, and how much money is stuck in them?",
    "Which categories have the most cash blocked in slow stock?",
    "What will run out in the next 7 days?",
    "What were my top 5 sellers in the last 30 days?",
    "How did monthly sales trend over the last 12 months?",
]


def load_secret_key() -> None:
    """Streamlit Community Cloud keeps secrets in st.secrets; copy the key into the environment."""
    if config.get_api_key():
        return
    try:
        if "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass


@st.cache_resource
def get_agent() -> InventoryAgent:
    con = q.connect(check_same_thread=False)
    return InventoryAgent(make_llm(), ToolBox(con))


def render_steps(steps: list[dict]) -> None:
    for i, step in enumerate(steps, 1):
        result = step["result"]
        st.markdown(f"**Step {i}: `{step['tool']}`** {step['args'] or ''}")
        if result.get("sql"):
            st.code(result["sql"], language="sql")
        elif result.get("note"):
            st.caption(result["note"])
        if not result.get("ok"):
            st.error(result.get("error", "Unknown error"))
        elif result.get("columns"):
            st.dataframe(pd.DataFrame(result["rows"], columns=result["columns"]), width="stretch", hide_index=True)
            if result.get("truncated"):
                st.caption(f"Showing the first {result['row_count']} rows.")


def main() -> None:
    st.title("📦 Retail Inventory Insight Agent")
    st.caption("Ask about slow-moving stock, reorders and sales in plain English. "
               "Sales are real (Online Retail II); cost, stock levels and reorder levels are simulated. Amounts in GBP.")

    if not config.DB_PATH.exists():
        st.error("Database not found. Run `python -m inventory_agent.build_db` first.")
        st.stop()
    load_secret_key()
    try:
        agent = get_agent()
    except LLMError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Try a question")
        clicked = None
        for example in EXAMPLES:
            if st.button(example, width="stretch"):
                clicked = example
        st.divider()
        st.caption(f"Model: {agent.llm.model}")
        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()

    messages = st.session_state.setdefault("messages", [])
    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("steps"):
                with st.expander(f"How I got this ({len(m['steps'])} tool call(s))"):
                    render_steps(m["steps"])

    question = st.chat_input("Ask about your stock...") or clicked
    if question:
        history = [{"role": m["role"], "text": m["text"]} for m in messages][-6:]
        messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Checking the data..."):
                result = agent.ask(question, history=history)
            st.markdown(result.answer)
            if result.steps:
                with st.expander(f"How I got this ({len(result.steps)} tool call(s))"):
                    render_steps(result.steps)
            st.caption(f"{result.latency_s}s")
        messages.append({"role": "assistant", "text": result.answer, "steps": result.steps})


main()
