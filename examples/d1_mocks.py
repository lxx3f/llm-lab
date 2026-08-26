"""Pure-Python mock tools for D1 dataset integration tests.

These deterministic mocks back the D1 generator's expected_tool_calls so the
canonical samples can be end-to-end executed by ``MockExecutor`` (no
real network, file, or subprocess I/O).

Registered tools:

- ``d1_calculate``: evaluates simple arithmetic expressions on integers.
- ``d1_get_weather``: returns a deterministic weather string keyed by city.
- ``d1_web_search``: returns a deterministic search summary keyed by query.
"""

from __future__ import annotations


def d1_calculate(expression: str) -> int:
    """Evaluate simple arithmetic on integers (``+ - * / **``).

    Limited to integer arithmetic to keep results deterministic and JSON-safe.
    Division rounds toward zero.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    # Whitelist digits, spaces, and the operators below.
    allowed = set("0123456789+-*/%() ")
    if not set(expression) <= allowed:
        raise ValueError(f"expression contains unsupported characters: {expression!r}")
    result = eval(expression, {"__builtins__": {}}, {})
    if not isinstance(result, (int, float)):
        raise ValueError(f"expression did not evaluate to a number: {result!r}")
    return int(result)


def d1_get_weather(city: str) -> str:
    """Return a deterministic weather string for ``city``."""
    if not isinstance(city, str) or not city.strip():
        raise ValueError("city must be a non-empty string")
    # Deterministic but plausible response.
    return f"{city} 当前天气：晴，22°C，微风。"


def d1_web_search(query: str, limit: int | None = None) -> str:
    """Return a deterministic search summary for ``query``."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    n = limit if isinstance(limit, int) and limit > 0 else 3
    return f"关于「{query}」找到 {n} 条结果。"


# Tools that always return an error response, used to exercise the
# ``tool_error_response`` task_type (model should call the tool but detect
# that the response itself is an error and report it, not use the bad result).
TRANSLATE_TOOL = {
    "type": "function",
    "function": {
        "name": "d1_translate",
        "description": "Translate text from one language to another.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}},
            "required": ["text", "target_lang"],
        },
    },
}


def d1_translate(text: str, target_lang: str) -> str:
    """Always returns an error response (simulates an unavailable service).

    The tool is registered with valid arguments and parses correctly; the
    correct model behaviour is to call it, observe the error response, and
    report the failure rather than propagate the bad result.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(target_lang, str) or not target_lang.strip():
        raise ValueError("target_lang must be a non-empty string")
    return f"ERROR: translation service unavailable (text={text!r}, target_lang={target_lang!r})"
