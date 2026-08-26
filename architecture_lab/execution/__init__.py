"""Tool execution for the evaluation pipeline.

P1-01 ships the in-process ``MockExecutor`` (zero external side effects);
sandboxed subprocess execution is deferred to P1-02.
"""

from architecture_lab.execution.mock_executor import MockExecutor, validate_execution_result

__all__ = ["MockExecutor", "validate_execution_result"]
