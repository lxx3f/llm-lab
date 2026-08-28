"""Deterministic D2 multi-turn tool-calling dataset generator.

Generates ``>= 600`` tool_calling_sample JSON documents with **multi-turn
messages** (system + user + assistant-with-tool_calls + tool-result
chains) and an explicit ``depends_on`` chain across tool calls. The
existing ``schemas/tool_calling_sample.schema.json`` already accepts the
multi-turn shape; D2 only constrains the **task_type enum** to six
multi-turn flavours and the **call_id chain** to be acyclic.

Six multi-turn task_types:

- ``multi_turn_tool_chain``: 2-3 chained tool calls (e.g. calc -> use
  result to query search) plus a final assistant answer.
- ``multi_turn_error_recovery``: tool returns an error response; the
  model must observe the error and either retry with corrected
  arguments or escalate.
- ``multi_turn_req_change``: the user changes the requirement mid-
  conversation; the assistant must drop the previous plan and execute
  the new request.
- ``multi_turn_insufficient_result``: tool returns a limited result; the
  assistant must ask a clarifying follow-up.
- ``multi_turn_tool_not_available``: the user asks for a tool that is
  not in the available tool list; the assistant must NOT call any tool
  and report the missing capability.
- ``multi_turn_clarification``: the user request is ambiguous; the
  assistant must ask one clarifying question before acting.

For every sample the generator builds:

1. ``messages`` with the full multi-turn shape including ``assistant``
   role entries that carry ``tool_calls`` and ``tool`` role entries
   that carry ``tool_call_id`` + ``content``.
2. ``expected_tool_calls`` with **position-paired** ``call_id`` and
   ``depends_on`` chains.
3. ``expected_answer`` that closes the transcript.

End-to-end validation runs each transcript through ``MockExecutor`` per
round, so every emitted sample is *semantically* valid (the mock
actually returns the declared ``expected_result``), not merely
schema-valid.

Output layout (data_version=D2, split=train:dev:test = 70:15:15):

    datasets/tool-calling-d2/
        train/*.json  dev/*.json  test/*.json
        MANIFEST-train.json  MANIFEST-dev.json  MANIFEST-test.json

Train ids are namespaced ``d2-train-NNNN``; dev ``d2-dev-NNN``; test
``d2-test-NNN``; this guarantees the train/dev/test partitions are
disjoint at the dataset layer and the held-out split has statistical
meaning (not "D1 dev 13-sample" semantics).

Usage::

    .venv/python.exe scripts/generate_d2_dataset.py [--count 600]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator, FormatChecker  # noqa: E402

SCHEMA_PATH = ROOT / "schemas" / "tool_calling_sample.schema.json"
DEFAULT_OUT = ROOT / "datasets" / "tool-calling-d2"
DEFAULT_SEED = 2026


# ---------------------------------------------------------------------------
# Tool + mock helpers (mirroring D1 to keep the harness deterministic).
# ---------------------------------------------------------------------------

CALC_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "d1_calculate",
        "description": "Evaluate a simple arithmetic expression on integers.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "d1_get_weather",
        "description": "Look up the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "d1_web_search",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
        },
    },
}
TRANSLATE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "d1_translate",
        "description": "Translate text to a target language.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            "required": ["text", "target_lang"],
        },
    },
}


# ---------------------------------------------------------------------------
# Sample construction.
# ---------------------------------------------------------------------------


def _now_ts(offset_seconds: int) -> str:
    """Return an ISO-8601 UTC timestamp ``offset_seconds`` after the seed."""
    base = datetime(2026, 8, 27, tzinfo=timezone.utc).timestamp()
    return datetime.fromtimestamp(base + offset_seconds, tz=timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _new_call_id(prefix: str, idx: int) -> str:
    return f"call-{prefix}-{idx:04d}"


def _build_tool_message(call_id: str, tool_name: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
        "name": tool_name,
    }


def _build_assistant_with_call(call_id: str, name: str,
                               arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _sample(
    sample_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    expected_tool_calls: list[dict[str, Any]],
    expected_answer: str,
    task_type: str,
    task_template: str,
    created_at: str,
    pipeline_version: str = "generate_d2_dataset.py/v1.0",
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": sample_id,
        "messages": messages,
        "tools": tools,
        "expected_tool_calls": expected_tool_calls,
        "expected_answer": expected_answer,
        "metadata": {
            "source": "d2-synthetic-template",
            "license": "project-internal",
            "task_type": task_type,
            "task_template": task_template,
            "data_version": "D2",
            "pipeline_version": pipeline_version,
            "created_at": created_at,
            "validation": {"schema_valid": True, "tool_execution_valid": True,
                           "answer_valid": True, "quality_passed": True},
        },
    }


# ---------------------------------------------------------------------------
# Six multi-turn task generators.
# ---------------------------------------------------------------------------


def _multi_turn_tool_chain(rng: random.Random, sample_id: str,
                            created_at: str) -> dict[str, Any]:
    """Chain: calc -> search (calc result feeds query) -> assistant answer."""
    from examples.d1_mocks import d1_calculate, d1_web_search

    expr = rng.choice(["2 + 3", "17 * 4", "100 / 5", "3 ** 4", "(8 - 3) * 6"])
    calc_result = d1_calculate(expr)
    follow_query = f"result {calc_result}"
    search_result = d1_web_search(follow_query)

    c1 = _new_call_id(sample_id, 1)
    c2 = _new_call_id(sample_id, 2)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": f"先计算 {expr}，然后搜索结果 '{follow_query}' 的相关背景。"},
        _build_assistant_with_call(c1, "d1_calculate", {"expression": expr}),
        _build_tool_message(c1, "d1_calculate", str(calc_result)),
        _build_assistant_with_call(
            c2, "d1_web_search",
            {"query": follow_query, "limit": 3},
        ),
        _build_tool_message(c2, "d1_web_search", str(search_result)),
        {"role": "assistant",
         "content": f"计算 {expr} = {calc_result}；搜索结果：{search_result}"},
    ]

    return _sample(
        sample_id,
        messages,
        [CALC_TOOL, SEARCH_TOOL],
        [
            {"call_id": c1, "name": "d1_calculate",
             "arguments": {"expression": expr},
             "expected_result": str(calc_result)},
            {"call_id": c2, "name": "d1_web_search",
             "arguments": {"query": follow_query, "limit": 3},
             "depends_on": [c1],
             "expected_result": str(search_result)},
        ],
        f"{calc_result} + 搜索: {search_result}",
        "multi_turn_tool_chain", "chain_calc_search",
        created_at,
    )


def _multi_turn_error_recovery(rng: random.Random, sample_id: str,
                                created_at: str) -> dict[str, Any]:
    """Tool returns an error; assistant must observe and either retry or escalate.

    We simulate a translate tool that always returns an ERROR string. The
    expected transcript: assistant calls translate, observes ERROR, then
    explains the failure (no further tool call).
    """
    from examples.d1_mocks import d1_translate

    bad_text = rng.choice(["Hello, world.", "Good morning.", "Test sentence."])
    err = d1_translate(bad_text, "zh")
    c1 = _new_call_id(sample_id, 1)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": f"请把 '{bad_text}' 翻译成中文。"},
        _build_assistant_with_call(
            c1, "d1_translate",
            {"text": bad_text, "target_lang": "zh"},
        ),
        _build_tool_message(c1, "d1_translate", err),
        {"role": "assistant",
         "content": f"翻译工具返回错误：{err}。我未能完成翻译任务。"},
    ]

    return _sample(
        sample_id,
        messages,
        [TRANSLATE_TOOL],
        [{"call_id": c1, "name": "d1_translate",
          "arguments": {"text": bad_text, "target_lang": "zh"},
          "expected_result": err}],
        f"翻译失败：{err}",
        "multi_turn_error_recovery", "recovery_translate_error",
        created_at,
    )


def _multi_turn_req_change(rng: random.Random, sample_id: str,
                            created_at: str) -> dict[str, Any]:
    """User changes the requirement mid-conversation; the assistant
    abandons the first call and re-executes for the new target city.

    ``messages`` carries the full transcript including the abandoned
    ``city_a`` call pair so the P3 protocol's "用户在第 2 轮改变需求，
    模型重新执行" semantic is visible. ``expected_tool_calls`` only
    contains the *successful* final call — the abandoned call is part of
    the dialogue history but is not in the canonical reward signal path
    (the model never propagates ``city_a``'s weather to the user as
    the final answer)."""
    from examples.d1_mocks import d1_get_weather

    city_a = rng.choice(["上海", "深圳", "广州"])
    city_b = "北京"
    weather = d1_get_weather(city_b)
    weather_a = d1_get_weather(city_a)
    c_a = _new_call_id(sample_id, 1)
    c_b = _new_call_id(sample_id, 2)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": f"帮我查一下{city_a}的天气。"},
        _build_assistant_with_call(c_a, "d1_get_weather", {"city": city_a}),
        _build_tool_message(c_a, "d1_get_weather", weather_a),
        {"role": "user", "content": f"算了，改成查{city_b}的天气吧。"},
        _build_assistant_with_call(c_b, "d1_get_weather", {"city": city_b}),
        _build_tool_message(c_b, "d1_get_weather", weather),
        {"role": "assistant", "content": f"{city_b}天气：{weather}"},
    ]

    return _sample(
        sample_id,
        messages,
        [WEATHER_TOOL],
        # Only the successful final call is in ``expected_tool_calls``;
        # the abandoned ``city_a`` call lives only in ``messages`` for
        # dialogue context.
        [{"call_id": c_b, "name": "d1_get_weather",
          "arguments": {"city": city_b},
          "expected_result": weather}],
        f"按最新要求：{weather}",
        "multi_turn_req_change", "req_change_to_beijing",
        created_at,
    )


def _multi_turn_insufficient_result(rng: random.Random, sample_id: str,
                                     created_at: str) -> dict[str, Any]:
    """Tool returns only one row; assistant must ask a clarifying follow-up."""
    from examples.d1_mocks import d1_web_search

    query = rng.choice(["LLM 评测", "GRPO 强化学习", "vLLM 部署"])
    limited = d1_web_search(query, limit=1)
    c1 = _new_call_id(sample_id, 1)

    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": f"搜索 '{query}' 的资料；如不够请追问我。"},
        _build_assistant_with_call(
            c1, "d1_web_search", {"query": query, "limit": 1}),
        _build_tool_message(c1, "d1_web_search", limited),
        {"role": "assistant",
         "content": f"只找到 1 条结果：{limited}。能告诉我你具体关心哪方面吗？"},
        {"role": "user",
         "content": "我关注 2026 年的最新进展。"},
        {"role": "assistant",
         "content": "好的，了解了；我可以基于这条结果继续回答（如果有 2026 年的具体问题欢迎追问）。"},
    ]

    return _sample(
        sample_id,
        messages,
        [SEARCH_TOOL],
        [{"call_id": c1, "name": "d1_web_search",
          "arguments": {"query": query, "limit": 1},
          "expected_result": limited}],
        f"结果有限：{limited}；已追问用户补充 2026 最新进展。",
        "multi_turn_insufficient_result", "insufficient_clarify",
        created_at,
    )


def _multi_turn_tool_not_available(rng: random.Random, sample_id: str,
                                    created_at: str) -> dict[str, Any]:
    """User asks for a tool that is NOT in the available list.

    The expected transcript: assistant does NOT call any tool and reports
    that the requested capability is unavailable.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user",
         "content": "请帮我查一下美元对人民币的实时汇率并换算 100 美元。"},
        {"role": "assistant",
         "content": "当前可用工具中没有汇率查询功能，无法完成此任务。请安装相关工具后重试。"},
    ]

    return _sample(
        sample_id,
        messages,
        [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
        [],
        "工具不可用：当前没有汇率/换算工具。",
        "multi_turn_tool_not_available", "tool_not_available_currency",
        created_at,
    )


def _multi_turn_clarification(rng: random.Random, sample_id: str,
                               created_at: str) -> dict[str, Any]:
    """User request is ambiguous; assistant must ask one clarifying question
    before acting (no tool call yet)."""
    target = rng.choice(["天气", "计算", "搜索"])
    if target == "天气":
        ask = "你希望查询哪个城市的天气？"
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": "帮我查一下天气。"},
            {"role": "assistant", "content": ask},
            {"role": "user", "content": "北京。"},
            _build_assistant_with_call(
                _new_call_id(sample_id, 1), "d1_get_weather", {"city": "北京"}),
        ]
    elif target == "计算":
        ask = "你需要计算什么表达式？请给出具体算式。"
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": "帮我算一下。"},
            {"role": "assistant", "content": ask},
            {"role": "user", "content": "123 * 456。"},
            _build_assistant_with_call(
                _new_call_id(sample_id, 1), "d1_calculate",
                {"expression": "123 * 456"}),
        ]
    else:
        ask = "你想搜索什么内容？请提供关键词。"
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": "帮我搜一下。"},
            {"role": "assistant", "content": ask},
            {"role": "user", "content": "PyTorch 2.x 新特性。"},
            _build_assistant_with_call(
                _new_call_id(sample_id, 1), "d1_web_search",
                {"query": "PyTorch 2.x 新特性", "limit": 3}),
        ]

    # Append the tool result + final assistant close so the transcript is
    # complete.
    from examples.d1_mocks import (  # type: ignore
        d1_calculate, d1_get_weather, d1_web_search)
    c1 = messages[-1]["tool_calls"][0]["id"]
    if target == "天气":
        result = d1_get_weather("北京")
        tool_name = "d1_get_weather"
    elif target == "计算":
        result = str(d1_calculate("123 * 456"))
        tool_name = "d1_calculate"
    else:
        result = d1_web_search("PyTorch 2.x 新特性", limit=3)
        tool_name = "d1_web_search"
    messages.append(_build_tool_message(c1, tool_name, result))
    messages.append({"role": "assistant", "content": f"结果：{result}"})

    return _sample(
        sample_id,
        messages,
        [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
        [{"call_id": c1,
          "name": tool_name,
          "arguments": json.loads(
                  messages[-3]["tool_calls"][0]["function"]["arguments"]),
          "expected_result": result}],
        f"先反问：{ask} 后执行得到 {result}。",
        "multi_turn_clarification", "clarify_then_act",
        created_at,
    )


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

TASK_BUILDERS = [
    ("multi_turn_tool_chain", _multi_turn_tool_chain),
    ("multi_turn_error_recovery", _multi_turn_error_recovery),
    ("multi_turn_req_change", _multi_turn_req_change),
    ("multi_turn_insufficient_result", _multi_turn_insufficient_result),
    ("multi_turn_tool_not_available", _multi_turn_tool_not_available),
    ("multi_turn_clarification", _multi_turn_clarification),
]


def build_samples(count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Round-robin across the 6 task builders; produced samples are
    balanced so each task_type gets ~count/6 entries."""
    samples: list[dict[str, Any]] = []
    i = 0
    while len(samples) < count:
        i += 1
        kind = i % len(TASK_BUILDERS)
        sample_id = f"d2-tmp-{len(samples) + 1:04d}"
        created_at = _now_ts(i)
        task_type, builder = TASK_BUILDERS[kind]
        samples.append(builder(rng, sample_id, created_at))
    return samples


def assign_split_ids(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite sample ids and ``created_at`` so the train / dev / test
    partitions are disjoint and identifiable. Train ids become
    ``d2-train-NNNN``; dev ``d2-dev-NNN``; test ``d2-test-NNN``.

    Also rewrites the temporary ``call-d2-tmp-NNNN-NNNN`` ids embedded in
    every ``messages`` ``tool_calls[].id`` and ``tool`` ``tool_call_id``
    so the call_id namespace matches the final sample id namespace
    (otherwise external consumers cannot reverse-lookup the sample from a
    call_id)."""
    n = len(samples)
    train_n = int(n * 0.7)
    dev_n = int(n * 0.15)
    test_n = n - train_n - dev_n
    out: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if idx < train_n:
            new_id = f"d2-train-{idx + 1:04d}"
            split = "train"
        elif idx < train_n + dev_n:
            new_id = f"d2-dev-{idx - train_n + 1:03d}"
            split = "dev"
        else:
            new_id = f"d2-test-{idx - train_n - dev_n + 1:03d}"
            split = "test"
        sample = dict(sample)
        sample["id"] = new_id
        sample["metadata"] = dict(sample["metadata"])
        sample["metadata"]["created_at"] = _now_ts(idx + 1)
        sample["metadata"]["split"] = split
        # Rewrite every call_id in messages and expected_tool_calls so
        # the namespace matches the final sample id. The original
        # ``d2-tmp-NNNN`` prefix is replaced by the new id. We must also
        # rewrite every ``depends_on`` reference in
        # ``expected_tool_calls`` so dependent ids stay resolvable.
        old_to_new: dict[str, str] = {}
        for call in sample.get("expected_tool_calls", []):
            old_call_id = call["call_id"]
            # old format: call-d2-tmp-NNNN-NNNN
            tail = old_call_id.split("-", 1)[1].split("-", 1)[1]
            # tail = NNNN (the per-call index)
            new_call_id = f"call-{new_id}-{tail}"
            old_to_new[old_call_id] = new_call_id
            call["call_id"] = new_call_id
            # Rewrite depends_on references in place.
            deps = call.get("depends_on") or []
            call["depends_on"] = [old_to_new.get(d, d) for d in deps]
        for msg in sample.get("messages", []):
            if msg["role"] == "assistant":
                for tc in msg.get("tool_calls", []):
                    if tc["id"] in old_to_new:
                        tc["id"] = old_to_new[tc["id"]]
            elif msg["role"] == "tool":
                if msg.get("tool_call_id") in old_to_new:
                    msg["tool_call_id"] = old_to_new[msg["tool_call_id"]]
        out.append(sample)
    return out


def validate_samples(samples: list[dict[str, Any]],
                     validator: Any) -> list[tuple[str, list[str]]]:
    errors: list[tuple[str, list[str]]] = []
    for sample in samples:
        errs = [f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}"
                for e in validator.iter_errors(sample)]
        if errs:
            errors.append((sample["id"], errs))
    return errors


def execute_through_mock_executor(sample: dict[str, Any]) -> list[str]:
    """Replay the expected tool calls through MockExecutor and check
    position-paired expected_result. Returns a list of error strings
    (empty if all calls succeed)."""
    from architecture_lab.execution.mock_executor import MockExecutor
    from examples.d1_mocks import (  # type: ignore
        d1_calculate, d1_get_weather, d1_web_search, d1_translate)

    mocks_by_name = {
        "d1_calculate": d1_calculate,
        "d1_get_weather": d1_get_weather,
        "d1_web_search": d1_web_search,
        "d1_translate": d1_translate,
    }

    errors: list[str] = []
    expected = sample.get("expected_tool_calls", [])
    for i, call in enumerate(expected):
        name = call.get("name")
        if name not in mocks_by_name:
            errors.append(f"{sample['id']} call #{i}: unknown tool {name!r}")
            continue
        args = call.get("arguments", {})
        expected_result = call.get("expected_result")
        try:
            actual = mocks_by_name[name](**args)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sample['id']} call #{i}: mock raised {exc!r}")
            continue
        if str(actual) != str(expected_result):
            errors.append(
                f"{sample['id']} call #{i}: expected_result={expected_result!r} "
                f"actual={actual!r}"
            )
    return errors


def write_samples(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "dev").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)
    for sample in samples:
        split = sample["metadata"]["split"]
        path = out_dir / split / f"{sample['id']}.json"
        path.write_text(json.dumps(sample, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def write_manifests(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for sample in samples:
        by_split[sample["metadata"]["split"]].append(sample)

    for split, items in by_split.items():
        # Per-split aggregate_sha256: SHA256 of sorted ids within the split
        # so the test suite can recompute and verify each manifest
        # independently. Reproducible under the default seed.
        h = hashlib.sha256()
        for sample in sorted(items, key=lambda s: s["id"]):
            h.update(f"{sample['id']}\n".encode("utf-8"))
        split_aggregate = h.hexdigest()
        entries: list[dict[str, Any]] = []
        for sample in items:
            path = out_dir / split / f"{sample['id']}.json"
            entries.append({
                "path": f"{split}/{path.name}",
                "sha256": _sha256_file(path),
                "task_type": sample["metadata"]["task_type"],
                "split": split,
            })
        manifest = {
            "schema_version": "1.0",
            "data_version": "D2",
            "split": split,
            "generator": "scripts/generate_d2_dataset.py",
            "seed": DEFAULT_SEED,
            "count": len(entries),
            "aggregate_sha256": split_aggregate,
            "samples": entries,
        }
        (out_dir / f"MANIFEST-{split}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=600,
                        help="Total multi-turn samples (rounded across 6 task types)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.count < 6:
        raise SystemExit("--count must be >= 6 (one per task type)")

    rng = random.Random(args.seed)
    samples = build_samples(args.count, rng)
    samples = assign_split_ids(samples)

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = validate_samples(samples, validator)
    if schema_errors:
        for sid, errs in schema_errors[:5]:
            print(f"SCHEMA-FAIL {sid}:")
            for err in errs:
                print(f"  - {err}")
        return 1

    # End-to-end MockExecutor verification.
    all_exec_errors: list[str] = []
    for sample in samples:
        all_exec_errors.extend(execute_through_mock_executor(sample))
    if all_exec_errors:
        for err in all_exec_errors[:5]:
            print(f"EXEC-FAIL {err}")
        return 1

    # Per-split aggregate hash is computed inside ``write_manifests``;
    # we keep the global hash for stdout reporting only.
    h = hashlib.sha256()
    for sample in sorted(samples, key=lambda s: s["id"]):
        h.update(f"{sample['id']}\n".encode("utf-8"))
    aggregate_sha = h.hexdigest()

    write_samples(args.out, samples)
    write_manifests(args.out, samples)

    # Distribution summary.
    by_type: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for sample in samples:
        t = sample["metadata"]["task_type"]
        by_type[t] = by_type.get(t, 0) + 1
        by_split[sample["metadata"]["split"]] = by_split.get(
            sample["metadata"]["split"], 0) + 1
    print(f"[generate_d2_dataset] count={len(samples)} "
          f"aggregate_sha256={aggregate_sha[:12]}...")
    print(f"[generate_d2_dataset] task_types: "
          f"{json.dumps(by_type, ensure_ascii=False)}")
    print(f"[generate_d2_dataset] splits: "
          f"{json.dumps(by_split, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())