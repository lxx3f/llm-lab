"""P1-05 failure-layer classification evaluator.

Evaluates a model's tool-calling transcript against a D0/D1 sample through
eight failure layers:

    parse_success          transcript has a structured tool_calls list
    schema_valid           every call's arguments match the named tool's schema
    tool_name_correct      every call's tool name matches the expected plan
    argument_value_correct every call's arguments match the expected plan
    call_plan_matches      calls match expected order/dependencies/quantity
    execution_success      MockExecutor ran all calls successfully
    result_grounded        actual results match expected results (where declared)
    task_success           final answer contains expected_answer (where declared)

``final_answer_correct`` is the 8-level name for the last layer (same value as
``task_success``, kept as a backward-compatible alias).

Layers that are undefined for a sample (e.g. ``task_success`` when
``expected_answer`` is null, ``result_grounded`` when no ``expected_result``
is declared) are reported as ``None`` (not applicable) and never count as the
first failure.

For ``no_tool`` samples (empty ``expected_tool_calls``), the correct output is
an empty tool-call list; parse/schema/name/args/execution/grounding are
trivially true.

Usage:
    .venv/python.exe scripts/classify_tool_failure.py \\
        --sample datasets/tool-calling-d1/train/d1-0001.json \\
        --transcript artifacts/transcript.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator  # noqa: E402


def _norm_name(name: Any) -> str:
    """Normalize a tool-call name for multiset comparison. Non-string names
    (None, int, dict, ...) are mapped to a stable, sortable placeholder so
    mixed-type lists never crash sorted(). Expected names are always strings,
    so a non-string actual name always mismatches the expected plan."""
    if isinstance(name, str):
        return name
    return f"<non-str:{type(name).__name__}>"


def _norm_args(arguments: Any) -> str:
    """Serialize tool-call arguments to a stable string for comparison.
    Handles non-dict arguments (None, str, int, list, ...) and dicts with
    non-string keys without crashing."""
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(arguments)


def classify(sample: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    """Classify a model transcript against the sample through the layers.

    Returns ``{"layers": {...}, "first_failure": str | None}`` where layer
    values are True / False / None (None = not applicable).
    """
    layers: dict[str, bool | None] = {}
    expected_calls: list[dict[str, Any]] = sample.get("expected_tool_calls", [])

    # Normalize tool_calls: missing key, None, non-list, OR a list whose
    # elements are not all structured mappings (e.g. ``[None]``, ``[1]``,
    # ``["x"]``) is treated as a malformed transcript. ``parse_success``
    # reports the parse outcome; subsequent layers fall back to empty lists
    # so they never crash on NoneType / non-dict members. (Auditor rounds 7-8.)
    raw_calls = transcript.get("tool_calls")
    is_list_of_dicts = isinstance(raw_calls, list) and all(
        isinstance(c, dict) for c in raw_calls
    )
    transcript_calls: list[dict[str, Any]] = raw_calls if is_list_of_dicts else []

    # Layer 1: parse_success — transcript is structurally parseable into a
    # list of structured tool-call objects (an empty list is valid; e.g.
    # no_tool samples). Any non-list or non-dict member is a parse failure.
    layers["parse_success"] = is_list_of_dicts

    if not is_list_of_dicts:
        # Malformed transcript: all lower layers are N/A; first_failure is
        # parse_success. ``task_success`` is evaluated independently from
        # ``final_answer`` so the final-answer check still works.
        layers["schema_valid"] = None
        layers["tool_name_correct"] = None
        layers["argument_value_correct"] = None
        layers["call_plan_matches"] = None
        layers["execution_success"] = None
        layers["result_grounded"] = None
        expected_answer = sample.get("expected_answer")
        if expected_answer is None:
            layers["task_success"] = None
            layers["final_answer_correct"] = None
        else:
            answer_ok = str(expected_answer) in (transcript.get("final_answer", "") or "")
            layers["task_success"] = answer_ok
            layers["final_answer_correct"] = answer_ok
        return {"layers": layers, "first_failure": "parse_success"}

    # Layer 2: schema_valid — every transcript call's arguments satisfy the
    # named tool's declared parameters schema. A call whose tool name does
    # not exist in the declared tool registry cannot be schema-checked; the
    # layer reports None (not False) so the wrong-name case is localized to
    # tool_name_correct (auditor round 1 for this stage).
    schema_valid: bool | None = True
    schema_checkable = True
    if layers["parse_success"]:
        for call in transcript_calls:
            tool = next(
                (t for t in sample.get("tools", []) if t["function"]["name"] == call.get("name")),
                None,
            )
            if tool is None:
                # Unknown tool name: schema layer is N/A for this call; the
                # name mismatch is handled by tool_name_correct below.
                schema_checkable = False
                continue
            validator = Draft202012Validator(tool["function"]["parameters"])
            if list(validator.iter_errors(call.get("arguments", {}))):
                schema_valid = False
                break
    if not schema_checkable:
        schema_valid = None
    layers["schema_valid"] = schema_valid

    # Layer 3: tool_name_correct — the transcript's tool names, as a
    # multiset, match the expected names. Order is NOT considered here (a
    # wrong-order transcript still has the right names → tool_name_correct
    # stays True and the failure is localized to call_plan_matches). Names
    # are normalized for comparison so mixed/missing/non-string names never
    # crash sorted() (auditor round 1 for this stage).
    name_ok: bool | None = True
    if len(expected_calls) == 0:
        # no_tool: no calls expected; any transcript call is a wrong name.
        name_ok = len(transcript_calls) == 0
    elif isinstance(transcript_calls, list):
        exp_names = sorted(_norm_name(e.get("name")) for e in expected_calls)
        act_names = sorted(_norm_name(c.get("name")) for c in transcript_calls)
        name_ok = exp_names == act_names
    else:
        name_ok = None
    layers["tool_name_correct"] = name_ok

    # Layer 4: argument_value_correct — the transcript's (name, arguments)
    # pairs, as a multiset, match the expected pairs. Order not considered.
    # Names are normalized (see _norm_name) and arguments are serialized
    # safely so mixed/missing names or non-dict arguments never crash.
    args_ok: bool | None = True
    if len(expected_calls) == 0:
        args_ok = len(transcript_calls) == 0
    elif isinstance(transcript_calls, list):
        exp_pairs = sorted((_norm_name(e.get("name")), _norm_args(e.get("arguments")))
                          for e in expected_calls)
        act_pairs = sorted((_norm_name(c.get("name")), _norm_args(c.get("arguments")))
                          for c in transcript_calls)
        args_ok = exp_pairs == act_pairs
    else:
        args_ok = None
    layers["argument_value_correct"] = args_ok

    # Layer 5: call_plan_matches — transcript calls match the expected plan:
    # same order, same depends_on edges, no dangling dependency references,
    # and execution order respects deps. (Name/argument correctness is now
    # localized to tool_name_correct / argument_value_correct layers above,
    # but this layer retains the full plan comparison as an overall check.)
    plan_matches: bool | None = True
    if len(expected_calls) == 0:
        # no_tool: expect no calls.
        plan_matches = len(transcript_calls) == 0
    elif not isinstance(transcript_calls, list):
        plan_matches = None
    else:
        if len(transcript_calls) != len(expected_calls):
            plan_matches = False
        else:
            # Normalize call_ids (may be non-string / unhashable dicts) so
            # set/sorted operations never crash (auditor round 2). Expected
            # call_ids are strings, so any non-string actual id mismatches.
            all_ids = {_norm_name(c.get("call_id")) for c in transcript_calls}
            for exp, act in zip(expected_calls, transcript_calls):
                if exp.get("name") != act.get("name"):
                    plan_matches = False
                    break
                # Every transcript call_id must match the corresponding
                # expected call_id (position-paired). Non-string / missing /
                # mismatched ids fail the plan layer (auditor round 3).
                if _norm_name(act.get("call_id")) != _norm_name(exp.get("call_id")):
                    plan_matches = False
                    break
                if exp.get("arguments") != act.get("arguments"):
                    plan_matches = False
                    break
                # Normalize depends_on values (may be mixed str/int/unhashable)
                # so sorted() never crashes.
                exp_deps = sorted(_norm_name(d) for d in (exp.get("depends_on") or []))
                act_deps = sorted(_norm_name(d) for d in (act.get("depends_on") or []))
                if exp_deps != act_deps:
                    plan_matches = False
                    break
                # Dangling dependency references: every depends_on id must
                # exist in the call sequence.
                act_id = _norm_name(act.get("call_id"))
                for dep in act_deps:
                    if dep not in all_ids or dep == act_id:
                        plan_matches = False
                        break
                # Execution must respect dependency order: a dependency must
                # appear earlier in the transcript than the dependent call.
                act_pos = next(i for i, c in enumerate(transcript_calls)
                               if _norm_name(c.get("call_id")) == act_id)
                for dep in act_deps:
                    dep_pos = next((i for i, c in enumerate(transcript_calls)
                                    if _norm_name(c.get("call_id")) == dep), None)
                    if dep_pos is None or dep_pos >= act_pos:
                        plan_matches = False
                        break
                if plan_matches is False:
                    break
    layers["call_plan_matches"] = plan_matches

    # Layer 4: execution_success — every transcript call executed without a
    # mock error. Undefined (None) when there are no calls.
    if not isinstance(transcript_calls, list) or len(transcript_calls) == 0:
        layers["execution_success"] = None if len(expected_calls) > 0 else True
    else:
        layers["execution_success"] = all(
            call.get("execution_outcome") == "success" for call in transcript_calls
        )

    # Layer 5: result_grounded — actual results match expected results where
    # expected_result is declared. Pairs with expected calls by POSITION
    # (same as call_plan_matches) so a transcript that passes call_plan_matches
    # is always paired with the corresponding expected result. Falling back
    # to call_id matching alone would silently produce a false pass when the
    # transcript call_id differs from the expected call_id (auditor round 10).
    # For no_tool (zero expected and zero actual calls) grounding trivially
    # passes (True). None only when there ARE expected calls but none declares
    # expected_result.
    grounded: bool | None = True
    grounded_declared = False
    if len(expected_calls) == 0 and isinstance(transcript_calls, list) and len(transcript_calls) == 0:
        # no_tool: nothing to ground — trivially passes.
        grounded = True
    elif isinstance(transcript_calls, list):
        # Position-based pairing: i-th transcript call ↔ i-th expected call.
        for i, call in enumerate(transcript_calls):
            if i >= len(expected_calls):
                break
            exp = expected_calls[i]
            if "expected_result" not in exp:
                continue
            grounded_declared = True
            if call.get("result") != exp["expected_result"]:
                grounded = False
                break
        if not grounded_declared and len(expected_calls) > 0:
            grounded = None
    layers["result_grounded"] = grounded

    # Layer 6/7: task_success + final_answer_correct — final answer contains
    # expected_answer where declared. ``final_answer_correct`` is the
    # 8-level name for this layer; ``task_success`` is kept as a
    # backward-compatible alias (same value). None when expected_answer is
    # null (unverifiable).
    expected_answer = sample.get("expected_answer")
    if expected_answer is None:
        layers["task_success"] = None
        layers["final_answer_correct"] = None
    else:
        final_text = transcript.get("final_answer", "") or ""
        answer_ok = str(expected_answer) in final_text
        layers["task_success"] = answer_ok
        layers["final_answer_correct"] = answer_ok

    # First failing layer (skip None / not-applicable). Order follows the
    # 8-level taxonomy; ``final_answer_correct`` is reported instead of its
    # ``task_success`` alias when the final answer is wrong.
    first_failure = next((name for name, ok in layers.items() if ok is False), None)
    if first_failure == "task_success":
        first_failure = "final_answer_correct"
    return {"layers": layers, "first_failure": first_failure}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    result = classify(sample, transcript)
    if args.output:
        payload = {
            "schema_version": "1.0",
            "sample_id": sample.get("id"),
            "layers": result["layers"],
            "first_failure": result["first_failure"],
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[classify_tool_failure] wrote {args.output}")
    print(f"[classify_tool_failure] layers={json.dumps(result['layers'])} first_failure={result['first_failure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
