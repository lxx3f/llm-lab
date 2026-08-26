"""Deterministic D1 tool-calling dataset generator.

Generates >=100 tool_calling_sample JSON documents from templates with
deterministic pseudo-randomness (fixed seed), covering all six task_types:

- no_tool: answer without calling a tool
- single_tool: exactly one tool call
- multi_tool: >=2 dependent/independent calls
- tool_error: the model should detect a tool error (bad input, missing arg)
- insufficient_result: tool result is insufficient → ask follow-up
- requirement_change: user changes the requirement mid-conversation

Output layout (data_version=D1, split=train/dev/test 80/10/10):

    datasets/tool-calling-d1/
        train/*.json  dev/*.json  test/*.json
        MANIFEST.json

Usage:
    .venv/python.exe scripts/generate_d1_dataset.py [--count N]
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

OUT_DIR = ROOT / "datasets" / "tool-calling-d1"
SCHEMA_PATH = ROOT / "schemas" / "tool_calling_sample.schema.json"

# Tool definitions shared by templates.
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a simple arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1}},
            "required": ["query"],
        },
    },
}


def _tool(name: str) -> dict[str, Any]:
    for tool in (CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL):
        if tool["function"]["name"] == name:
            return tool
    raise KeyError(name)


def _sample(
    sample_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    answer: str | None,
    task_type: str,
    task_template: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "id": sample_id,
        "messages": messages,
        "tools": tools,
        "expected_tool_calls": expected,
        "expected_answer": answer,
        "metadata": {
            "source": "d1-synthetic-template",
            "license": "project-internal",
            "task_type": task_type,
            "task_template": task_template,
            "data_version": "D1",
            "pipeline_version": "p1-05-d1-generator",
            "created_at": "2026-08-27T00:00:00Z",
            "validation": {"schema_valid": True, "tool_execution_valid": None, "answer_valid": None, "quality_passed": None},
        },
    }


def _messages(user_text: str, tool_outputs: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": user_text},
    ]
    if tool_outputs:
        for call_id, output in tool_outputs:
            msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "calculate", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "content": output, "tool_call_id": call_id, "name": "calculate"})
    return msgs


def build_samples(count: int, rng: random.Random) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cities = ["北京", "上海", "深圳", "广州", "杭州", "成都", "武汉", "西安", "重庆", "南京"]
    expressions = ["2 + 3", "17 * 4", "100 / 5", "3 ** 4", "(8 - 3) * 6", "99 - 45", "12 + 87", "56 / 7", "9 * 11", "250 / 25"]
    queries = ["LLM 评测 2026", "RTX 5070 Ti 规格", "RoPE 旋转位置编码", "MoE 稀疏专家模型", "PyTorch 2.x 新特性", "LoRA 微调原理", "vLLM 部署", "GRPO 强化学习"]
    i = 0
    while len(samples) < count:
        i += 1
        kind = i % 7
        sid = f"d1-{len(samples) + 1:04d}"
        if kind == 0:  # no_tool
            city = rng.choice(cities)
            samples.append(_sample(
                sid,
                _messages(f"今天{city}的天气怎么样？（不需要工具）", []),
                [WEATHER_TOOL],
                [],
                f"直接回答{city}天气，不调用工具。",
                "no_tool", "no_tool_greeting",
            ))
        elif kind == 1:  # single_tool calculator
            expr = rng.choice(expressions)
            samples.append(_sample(
                sid,
                _messages(f"请计算 {expr} 的结果。", []),
                [CALC_TOOL],
                [{"call_id": f"call-{sid}", "name": "calculate", "arguments": {"expression": expr}}],
                None,
                "single_tool", "single_calc",
            ))
        elif kind == 2:  # single_tool weather
            city = rng.choice(cities)
            samples.append(_sample(
                sid,
                _messages(f"帮我查一下{city}现在的天气。", []),
                [WEATHER_TOOL],
                [{"call_id": f"call-{sid}", "name": "get_weather", "arguments": {"city": city}}],
                None,
                "single_tool", "single_weather",
            ))
        elif kind == 3:  # multi_tool
            expr = rng.choice(expressions)
            query = rng.choice(queries)
            samples.append(_sample(
                sid,
                _messages(f"先计算 {expr}，再搜索一下 {query}。", []),
                [CALC_TOOL, SEARCH_TOOL],
                [
                    {"call_id": f"call-{sid}-a", "name": "calculate", "arguments": {"expression": expr}},
                    {"call_id": f"call-{sid}-b", "name": "web_search", "arguments": {"query": query}, "depends_on": [f"call-{sid}-a"]},
                ],
                None,
                "multi_tool", "multi_calc_search",
            ))
        elif kind == 4:  # tool_error (missing required argument)
            expr = rng.choice(expressions)
            samples.append(_sample(
                sid,
                _messages(f"请计算 {expr}，如果工具缺少必要参数就说明。", []),
                [CALC_TOOL],
                [{"call_id": f"call-{sid}", "name": "calculate", "arguments": {}}],
                "工具缺少必要参数 expression，应报告错误。",
                "tool_error", "tool_missing_arg",
            ))
        elif kind == 5:  # insufficient_result
            query = rng.choice(queries)
            samples.append(_sample(
                sid,
                _messages(f"搜索 '{query}'，如果结果不足就追问用户补充。", []),
                [SEARCH_TOOL],
                [{"call_id": f"call-{sid}", "name": "web_search", "arguments": {"query": query}}],
                "搜索结果可能不足，需追问用户补充。",
                "insufficient_result", "insufficient_search",
            ))
        else:  # requirement_change
            city = rng.choice(cities)
            samples.append(_sample(
                sid,
                _messages(f"先查{city}天气，然后我改主意了：改成查北京。", []),
                [WEATHER_TOOL],
                [{"call_id": f"call-{sid}", "name": "get_weather", "arguments": {"city": "北京"}}],
                "最终以最新要求（北京）为准。",
                "requirement_change", "req_change_city",
            ))
    return samples


def validate(sample: dict[str, Any], validator: Any) -> list[str]:
    return [f"{'.'.join(str(p) for p in e.path)}: {e.message}" for e in validator.iter_errors(sample)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.count < 3:
        raise SystemExit("--count must be >= 3")

    rng = random.Random(args.seed)
    samples = build_samples(args.count, rng)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for sample in samples:
        errs = validate(sample, validator)
        if errs:
            errors.append(f"{sample['id']}: {errs}")

    # Deterministic split: sort by id, then 80/10/10.
    samples.sort(key=lambda s: s["id"])
    n = len(samples)
    train, dev, test = samples[: int(n * 0.8)], samples[int(n * 0.8): int(n * 0.9)], samples[int(n * 0.9):]

    manifest_samples: list[dict[str, Any]] = []
    for split_name, split_samples in (("train", train), ("dev", dev), ("test", test)):
        split_dir = OUT_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for sample in split_samples:
            path = split_dir / f"{sample['id']}.json"
            path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest_samples.append({
                "path": str(path.relative_to(OUT_DIR)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "task_type": sample["metadata"]["task_type"],
                "split": split_name,
            })
    manifest = {
        "schema_version": "1.0",
        "data_version": "D1",
        "split": "train/dev/test",
        "generator": "scripts/generate_d1_dataset.py",
        "seed": args.seed,
        "count": len(samples),
        "samples": manifest_samples,
        "samples_sha256": hashlib.sha256("".join(s["sha256"] for s in manifest_samples).encode("ascii")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (OUT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    task_types: dict[str, int] = {}
    for sample in samples:
        task_types[sample["metadata"]["task_type"]] = task_types.get(sample["metadata"]["task_type"], 0) + 1
    print(f"[generate_d1_dataset] {len(samples)} samples (train={len(train)}, dev={len(dev)}, test={len(test)})")
    print(f"[generate_d1_dataset] task_types: {json.dumps(task_types, ensure_ascii=False)}")
    if errors:
        print(f"[generate_d1_dataset] {len(errors)} schema errors!", file=sys.stderr)
        for e in errors[:5]:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"[generate_d1_dataset] wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
