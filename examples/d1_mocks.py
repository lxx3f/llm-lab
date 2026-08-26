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
