"""Pure-Python mock tools for the P1-01 mock executor example.

These are in-process functions with no external side effects (no subprocess,
filesystem, or network). Each one is registered via ``MockExecutor.register_mock``
together with its arguments JSON Schema.
"""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b; raises ValueError when b == 0 (used for the
    ``mock_exception`` path)."""
    if b == 0:
        raise ValueError("division by zero")
    return a / b
