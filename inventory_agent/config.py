"""Settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "shop.db"
LOG_PATH = ROOT / "logs" / "agent_log.jsonl"

# LLM settings. The key is read from the environment, never stored in the code.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def get_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


# Safety limits for every query the agent runs.
MAX_ROWS = 50            # rows returned to the model and the user
QUERY_TIMEOUT_S = 5.0    # a query running longer than this is stopped
MAX_AGENT_STEPS = 6      # tool-call rounds per question before the agent gives up
