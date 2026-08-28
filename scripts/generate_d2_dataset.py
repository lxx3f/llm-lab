"""Generate the deterministic D2 multi-turn tool-calling dataset.

D2 is deliberately separate from the D1 sample contract.  The generator
uses ``schemas/d2_multi_turn_sample.schema.json`` and the six canonical task
names from the P3 objective:

``tool_not_available``, ``tool_error_response``, ``insufficient_result_search``,
``req_change_city``, ``multi_tool_sequential``, and ``error_recovery``.

Every expected call is registered in :class:`MockExecutor` and replayed by
``execute_sequence`` before any sample is written.  Dependency references are
validated for existence, acyclic ordering, and successful step-by-step
execution.  No network or external side effect is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jsonschema import Draft202012Validator, FormatChecker  # noqa: E402

from architecture_lab.execution.mock_executor import MockExecutor  # noqa: E402
from examples.d1_mocks import (  # noqa: E402
    d1_calculate,
    d1_get_weather,
    d1_translate,
    d1_web_search,
)

SCHEMA_PATH = ROOT / "schemas" / "d2_multi_turn_sample.schema.json"
DEFAULT_OUT = ROOT / "datasets" / "tool-calling-d2"
DEFAULT_SEED = 2026

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

TASK_TYPES = (
    "tool_not_available",
    "tool_error_response",
    "insufficient_result_search",
    "req_change_city",
    "multi_tool_sequential",
    "error_recovery",
)


def _new_executor() -> MockExecutor:
    """Create the canonical executor used by generation and replay."""
    executor = MockExecutor()
    executor.register_mock(
        "d1_calculate", d1_calculate, CALC_TOOL["function"]["parameters"])
    executor.register_mock(
        "d1_get_weather", d1_get_weather, WEATHER_TOOL["function"]["parameters"])
    executor.register_mock(
        "d1_web_search", d1_web_search, SEARCH_TOOL["function"]["parameters"])
    executor.register_mock(
        "d1_translate", d1_translate, TRANSLATE_TOOL["function"]["parameters"])
    return executor


def _mock_result(name: str, arguments: dict[str, Any]) -> Any:
    """Get one deterministic result through MockExecutor, never directly."""
    result = _new_executor().execute({
        "tool_name": name,
        "call_id": "call-d2-build-0001",
        "arguments": arguments,
    })
    if result["outcome"] != "success":
        raise ValueError(f"mock failed for {name}: {result}")
    return result["result"]


# Canonical D2 timestamp contract (mirror scripts/generate_d1_dataset.py
# manifest field): ``created_at`` is deterministic from the generator seed
# and the sample's 1-based index so a fresh checkout with the same seed
# yields byte-identical timestamps. The formula is:
#
#     timestamp = 1785000000 + seed + index
#     isoformat = datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
#                 .replace("+00:00", "Z")
D2_TIMESTAMP_EPOCH = 1785000000


def _now_ts(seed: int, index: int) -> str:
    """Deterministic UTC timestamp for the (seed, index) pair.

    index is 1-based (matches the ``offset+1`` convention previously used
    by this generator and the per-sample ordering contract). Same seed +
    same index always yield the exact same string, byte-for-byte.
    """
    ts = D2_TIMESTAMP_EPOCH + seed + index
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_id(sample_id: str, number: int) -> str:
    return f"call-{sample_id}-{number:04d}"


def _assistant_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
    }


def _tool_message(call_id: str, name: str, result: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": str(result),
        "tool_call_id": call_id,
        "name": name,
    }


def _sample(
    sample_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
    expected_answer: str,
    task_type: str,
    template: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": sample_id,
        "messages": messages,
        "tools": tools,
        "expected_tool_calls": expected_calls,
        "expected_answer": expected_answer,
        "metadata": {
            "source": "d2-synthetic-template",
            "license": "project-internal",
            "task_type": task_type,
            "task_template": template,
            "data_version": "D2",
            "pipeline_version": "generate_d2_dataset.py/v2.0",
            "created_at": created_at,
            "validation": {
                "schema_valid": True,
                "tool_execution_valid": True,
                "answer_valid": True,
                "quality_passed": True,
            },
        },
    }


def _multi_tool_sequential(rng: random.Random, sid: str, created: str) -> dict[str, Any]:
    expr = rng.choice(["2 + 3", "17 * 4", "100 / 5", "3 ** 4", "(8 - 3) * 6"])
    first_args = {"expression": expr}
    first_result = _mock_result("d1_calculate", first_args)
    second_args = {"query": f"result {first_result}", "limit": 3}
    second_result = _mock_result("d1_web_search", second_args)
    c1, c2 = _call_id(sid, 1), _call_id(sid, 2)
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": f"先计算 {expr}，再搜索结果 {second_args['query']}。"},
            _assistant_call(c1, "d1_calculate", first_args),
            _tool_message(c1, "d1_calculate", first_result),
            _assistant_call(c2, "d1_web_search", second_args),
            _tool_message(c2, "d1_web_search", second_result),
            {"role": "assistant", "content": f"计算结果为 {first_result}；搜索结果：{second_result}"},
        ],
        [CALC_TOOL, SEARCH_TOOL],
        [
            {"call_id": c1, "name": "d1_calculate", "arguments": first_args,
             "expected_result": first_result},
            {"call_id": c2, "name": "d1_web_search", "arguments": second_args,
             "depends_on": [c1], "expected_result": second_result},
        ],
        f"计算结果为 {first_result}；搜索结果：{second_result}",
        "multi_tool_sequential", "calc_then_search", created,
    )


def _tool_error_response(rng: random.Random, sid: str, created: str) -> dict[str, Any]:
    text = rng.choice(["Hello, world.", "Good morning.", "Test sentence."])
    args = {"text": text, "target_lang": "zh"}
    result = _mock_result("d1_translate", args)
    c1 = _call_id(sid, 1)
    answer = f"翻译工具返回错误：{result}。无法完成翻译。"
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": f"请把 '{text}' 翻译成中文。"},
            _assistant_call(c1, "d1_translate", args),
            _tool_message(c1, "d1_translate", result),
            {"role": "assistant", "content": answer},
        ],
        [TRANSLATE_TOOL],
        [{"call_id": c1, "name": "d1_translate", "arguments": args,
          "expected_result": result}],
        answer, "tool_error_response", "translate_error_report", created,
    )


def _insufficient_result_search(rng: random.Random, sid: str, created: str) -> dict[str, Any]:
    query = rng.choice(["LLM 评测", "GRPO 强化学习", "vLLM 部署"])
    args = {"query": query, "limit": 1}
    result = _mock_result("d1_web_search", args)
    c1 = _call_id(sid, 1)
    answer = f"仅找到 1 条结果：{result}。请补充你关注的方向。"
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": f"搜索 '{query}' 的资料，如果不够请追问。"},
            _assistant_call(c1, "d1_web_search", args),
            _tool_message(c1, "d1_web_search", result),
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "我关注 2026 年的最新进展。"},
            {"role": "assistant", "content": "收到，我会按这个方向继续。"},
        ],
        [SEARCH_TOOL],
        [{"call_id": c1, "name": "d1_web_search", "arguments": args,
          "expected_result": result}],
        answer, "insufficient_result_search", "search_then_clarify", created,
    )


def _req_change_city(rng: random.Random, sid: str, created: str) -> dict[str, Any]:
    old_city = rng.choice(["上海", "深圳", "广州"])
    new_city = "北京"
    old_args, new_args = {"city": old_city}, {"city": new_city}
    old_result = _mock_result("d1_get_weather", old_args)
    new_result = _mock_result("d1_get_weather", new_args)
    c1, c2 = _call_id(sid, 1), _call_id(sid, 2)
    answer = f"按最新要求，{new_city}天气：{new_result}"
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": f"帮我查一下{old_city}的天气。"},
            _assistant_call(c1, "d1_get_weather", old_args),
            _tool_message(c1, "d1_get_weather", old_result),
            {"role": "user", "content": f"改成查{new_city}的天气吧。"},
            _assistant_call(c2, "d1_get_weather", new_args),
            _tool_message(c2, "d1_get_weather", new_result),
            {"role": "assistant", "content": answer},
        ],
        [WEATHER_TOOL],
        [{"call_id": c1, "name": "d1_get_weather", "arguments": old_args,
          "expected_result": old_result},
         {"call_id": c2, "name": "d1_get_weather", "arguments": new_args,
          "depends_on": [c1], "expected_result": new_result}],
        answer, "req_change_city", "abandoned_old_city_then_retry", created,
    )


_NOT_AVAILABLE_VARIANTS: tuple[tuple[str, str, str], ...] = (
    # (missing capability, user request, refusal rationale)
    ("实时汇率查询", "请帮我查一下美元兑人民币的实时汇率。",
     "当前可用工具中没有汇率查询功能，无法完成此任务。"),
    ("股票行情查询", "帮我查一下特斯拉今天的股价。",
     "当前可用工具中没有股票行情查询功能，无法获取实时股价。"),
    ("发送邮件", "请帮我给 luna@example.com 发一封提醒邮件。",
     "当前可用工具中没有邮件发送功能，我无法代你发邮件。"),
    ("创建日历日程", "帮我在明天下午 3 点创建一个会议日程。",
     "当前可用工具中没有日历日程功能，无法创建日程。"),
    ("航班查询", "查一下后天北京到上海的航班。",
     "当前可用工具中没有航班查询功能，无法获取航班信息。"),
    ("食谱查询", "帮我找一份番茄炒蛋的做法。",
     "当前可用工具中没有食谱查询功能，无法提供菜谱。"),
    ("停车位查询", "帮我看看公司楼下有没有停车位。",
     "当前可用工具中没有停车位查询功能，无法查询。"),
    ("工单创建", "请帮我创建一个新的 IT 工单。",
     "当前可用工具中没有工单创建功能，无法创建工单。"),
    ("加密货币价格", "查一下比特币现在的价格。",
     "当前可用工具中没有加密货币价格查询功能，无法获取行情。"),
    ("天气预警", "帮我订阅明天北京的大风预警。",
     "当前可用工具中没有天气预警订阅功能，无法完成订阅。"),
    ("翻译成英文", "请帮我把这份合同翻译成英文。",
     "当前可用工具中没有合同翻译功能，无法完成翻译。"),
    ("图片生成", "帮我生成一张秋天森林的图片。",
     "当前可用工具中没有图片生成功能，无法生成图片。"),
)


def _tool_not_available(rng: random.Random, sid: str, created: str) -> dict[str, Any]:
    """User asks for a capability that is NOT among the available tools.

    The expected transcript: assistant does NOT call any tool and reports
    that the requested capability is unavailable. Each sample draws from
    a pool of (capability, request, rationale) triples so the task keeps
    its semantics while the user request, available-tool set, and refusal
    wording differ across samples (auditor round 7 diversity fix)."""
    capability, request, rationale = rng.choice(_NOT_AVAILABLE_VARIANTS)
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": request},
            {"role": "assistant", "content": rationale},
        ],
        [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL], [], rationale,
        "tool_not_available", f"missing_{capability[:8]}", created,
    )


def _error_recovery(rng: random.Random, sid: str, created: str) -> dict[str, Any]:
    first_args = {"query": "PyTorch", "limit": 1}
    second_args = {"query": "PyTorch 2.x 新特性", "limit": 3}
    first_result = _mock_result("d1_web_search", first_args)
    second_result = _mock_result("d1_web_search", second_args)
    c1, c2 = _call_id(sid, 1), _call_id(sid, 2)
    answer = f"初始搜索过窄，已重试细化关键词：{second_result}"
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": "搜索 PyTorch 的最新特性；结果不足时请调整查询。"},
            _assistant_call(c1, "d1_web_search", first_args),
            _tool_message(c1, "d1_web_search", first_result),
            {"role": "assistant", "content": "结果过少，我会细化关键词后重试。"},
            _assistant_call(c2, "d1_web_search", second_args),
            _tool_message(c2, "d1_web_search", second_result),
            {"role": "assistant", "content": answer},
        ],
        [SEARCH_TOOL],
        [
            {"call_id": c1, "name": "d1_web_search", "arguments": first_args,
             "expected_result": first_result},
            {"call_id": c2, "name": "d1_web_search", "arguments": second_args,
             "depends_on": [c1], "expected_result": second_result},
        ],
        answer, "error_recovery", "refine_query_after_insufficient_result", created,
    )


BUILDERS: tuple[tuple[str, Callable[[random.Random, str, str], dict[str, Any]]], ...] = (
    ("tool_not_available", _tool_not_available),
    ("tool_error_response", _tool_error_response),
    ("insufficient_result_search", _insufficient_result_search),
    ("req_change_city", _req_change_city),
    ("multi_tool_sequential", _multi_tool_sequential),
    ("error_recovery", _error_recovery),
)


def build_samples(count: int, rng: random.Random, *, seed: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in range(count):
        sid = f"d2-local-{index + 1:04d}"
        task_type, builder = BUILDERS[index % len(BUILDERS)]
        sample = builder(rng, sid, _now_ts(seed, index + 1))
        if sample["metadata"]["task_type"] != task_type:
            raise AssertionError(f"builder/type mismatch: {task_type} != {sample['metadata']}")
        samples.append(sample)
    return samples


def assign_split_ids(samples: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    total = len(samples)
    train_n = int(total * 0.7)
    dev_n = int(total * 0.15)
    output: list[dict[str, Any]] = []
    for index, original in enumerate(samples):
        if index < train_n:
            split, number = "train", index + 1
            final_id = f"d2-train-{number:04d}"
        elif index < train_n + dev_n:
            split, number = "dev", index - train_n + 1
            final_id = f"d2-dev-{number:03d}"
        else:
            split, number = "test", index - train_n - dev_n + 1
            final_id = f"d2-test-{number:03d}"
        sample = json.loads(json.dumps(original, ensure_ascii=False))
        old_id = sample["id"]
        sample["id"] = final_id
        sample["metadata"]["split"] = split
        sample["metadata"]["created_at"] = _now_ts(seed, index + 1)

        old_to_new: dict[str, str] = {}
        all_call_ids: list[str] = []
        for call in sample["expected_tool_calls"]:
            all_call_ids.append(call["call_id"])
        for message in sample["messages"]:
            if message["role"] == "assistant":
                all_call_ids.extend(
                    tool_call["id"] for tool_call in message.get("tool_calls", []))
        for old_call_id in dict.fromkeys(all_call_ids):
            suffix = old_call_id.rsplit("-", 1)[-1]
            old_to_new[old_call_id] = _call_id(final_id, int(suffix))
        for call in sample["expected_tool_calls"]:
            call["call_id"] = old_to_new[call["call_id"]]
            call["depends_on"] = [old_to_new.get(dep, dep) for dep in call.get("depends_on", [])]
        for message in sample["messages"]:
            if message["role"] == "assistant":
                for tool_call in message.get("tool_calls", []):
                    if tool_call["id"] in old_to_new:
                        tool_call["id"] = old_to_new[tool_call["id"]]
            elif message["role"] == "tool":
                if message.get("tool_call_id") in old_to_new:
                    message["tool_call_id"] = old_to_new[message["tool_call_id"]]
        # ``old_id`` is intentionally retained only in this local variable;
        # no original identifier is written into D2 artifacts.
        del old_id
        output.append(sample)
    return output


def dependency_errors(calls: list[dict[str, Any]]) -> list[str]:
    """Validate existence, strict earlier ordering, and acyclicity."""
    errors: list[str] = []
    ids = [call.get("call_id") for call in calls]
    known = set(ids)
    positions = {call_id: index for index, call_id in enumerate(ids)}
    if len(ids) != len(known):
        errors.append("duplicate call_id")
    for index, call in enumerate(calls):
        call_id = call.get("call_id")
        for dep in call.get("depends_on", []) or []:
            if dep not in known:
                errors.append(f"{call_id}: dangling dependency {dep}")
            elif dep == call_id:
                errors.append(f"{call_id}: self dependency")
            elif positions[dep] >= index:
                errors.append(f"{call_id}: dependency {dep} is not earlier")
    # Kahn-style cycle check over the dependency graph.
    remaining = set(known)
    while remaining:
        ready = {
            call_id for call_id in remaining
            if all(dep not in remaining for dep in next(
                call.get("depends_on", []) or [] for call in calls
                if call.get("call_id") == call_id
            ))
        }
        if not ready:
            errors.append("dependency cycle")
            break
        remaining -= ready
    return errors


def validate_semantics(sample: dict[str, Any]) -> list[str]:
    errors = dependency_errors(sample.get("expected_tool_calls", []))
    expected_ids = {call["call_id"] for call in sample.get("expected_tool_calls", [])}
    transcript_ids = {
        call["id"]
        for message in sample.get("messages", [])
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    }
    if not expected_ids.issubset(transcript_ids):
        errors.append("expected call_id missing from assistant transcript")
    return errors


def execute_through_mock_executor(sample: dict[str, Any]) -> list[str]:
    """Replay all expected calls through the real executor sequence API."""
    errors = dependency_errors(sample.get("expected_tool_calls", []))
    if errors:
        return errors
    calls = [
        {
            "tool_name": call["name"],
            "call_id": call["call_id"],
            "arguments": call["arguments"],
            "depends_on": call.get("depends_on", []),
        }
        for call in sample.get("expected_tool_calls", [])
    ]
    results = _new_executor().execute_sequence(calls)
    if len(results) != len(calls):
        return [f"{sample['id']}: executor returned {len(results)} results for {len(calls)} calls"]
    errors = []
    for call, result in zip(sample.get("expected_tool_calls", []), results):
        if result["outcome"] != "success":
            errors.append(f"{sample['id']} {call['call_id']}: {result['outcome']}")
        elif str(result["result"]) != str(call.get("expected_result")):
            errors.append(
                f"{sample['id']} {call['call_id']}: expected {call.get('expected_result')!r}, "
                f"got {result['result']!r}"
            )
    return errors


def _validate_samples(samples: list[dict[str, Any]]) -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for sample in samples:
        errors.extend(f"{sample['id']}: {error.message}" for error in validator.iter_errors(sample))
        errors.extend(f"{sample['id']}: {error}" for error in validate_semantics(sample))
        errors.extend(f"{sample['id']}: {error}" for error in execute_through_mock_executor(sample))
    return errors


def _write_samples(out_dir: Path, samples: list[dict[str, Any]]) -> None:
    for split in ("train", "dev", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    for sample in samples:
        path = out_dir / sample["metadata"]["split"] / f"{sample['id']}.json"
        path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifests(out_dir: Path, samples: list[dict[str, Any]], seed: int) -> None:
    for split in ("train", "dev", "test"):
        items = [sample for sample in samples if sample["metadata"]["split"] == split]
        entries = []
        for sample in items:
            path = out_dir / split / f"{sample['id']}.json"
            entries.append({
                "path": f"{split}/{path.name}",
                "sha256": _sha256_file(path),
                "task_type": sample["metadata"]["task_type"],
                "split": split,
            })
        digest = hashlib.sha256()
        for sample in sorted(items, key=lambda item: item["id"]):
            digest.update(f"{sample['id']}\n".encode("utf-8"))
        manifest = {
            "schema_version": "1.0",
            "data_version": "D2",
            "split": split,
            "generator": "scripts/generate_d2_dataset.py",
            "seed": seed,
            "created_at": _now_ts(seed, 0),
            "count": len(items),
            "aggregate_sha256": digest.hexdigest(),
            "task_types": {task: sum(1 for item in items if item["metadata"]["task_type"] == task)
                           for task in TASK_TYPES},
            "samples": entries,
        }
        (out_dir / f"MANIFEST-{split}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.count < 6:
        raise SystemExit("--count must be >= 6")

    samples = assign_split_ids(
        build_samples(args.count, random.Random(args.seed), seed=args.seed),
        seed=args.seed,
    )
    errors = _validate_samples(samples)
    if errors:
        for error in errors[:20]:
            print(f"FAIL {error}")
        return 1
    _write_samples(args.out, samples)
    _write_manifests(args.out, samples, args.seed)
    counts = {task: sum(1 for sample in samples if sample["metadata"]["task_type"] == task)
              for task in TASK_TYPES}
    splits = {split: sum(1 for sample in samples if sample["metadata"]["split"] == split)
              for split in ("train", "dev", "test")}
    print(f"[generate_d2_dataset] count={len(samples)} task_types={json.dumps(counts)} splits={splits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
