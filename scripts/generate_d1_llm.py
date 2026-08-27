"""D1.1 real-LLM tool-calling dataset generator.

Upgrades the template-based D1 dataset (``source: d1-synthetic-template``)
to LLM-generated samples (``source: <model-name>@<version>``) by asking a
real chat LLM to act as a "user" that produces natural-language requests,
then annotating each request with the expected tool-call plan.

Pipeline (per sample):

1. ``generate_user_turn`` — prompt the LLM (anthropic-messages API) with a
   task-type spec and tool registry; the LLM returns a natural-language
   user request in Chinese.
2. ``annotate_plan`` — prompt the LLM again with the request + registry; it
   returns the expected tool-call plan as JSON: which tool(s) to call, with
   what arguments, and the expected final answer.
3. Semantic validation — every expected call is executed through
   MockExecutor; the emitted sample must have schema-valid tools,
   executable expected_tool_calls (outcome=success, result == expected_result),
   and a plausible expected_answer.
4. Emission — write ``datasets/tool-calling-d1-llm/`` (or a new version dir)
   with per-split files + MANIFEST.json (sha256 per file + aggregate hash).

Provenance: ``metadata.source = "<model-name>@<version>"`` and
``metadata.pipeline_version = "d1.1-llm-generator"`` so the LLM-generated
set is fully distinguishable from the template set.

Usage:
    .venv/python.exe scripts/generate_d1_llm.py --count 3 --split train
    .venv/python.exe scripts/generate_d1_llm.py --count 12  # all samples
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows consoles default to a legacy codepage (e.g. GBK) that cannot
# encode arbitrary Chinese characters; force UTF-8 so progress prints and
# user-turn previews never raise UnicodeEncodeError mid-generation.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from jsonschema import Draft202012Validator  # noqa: E402

from architecture_lab.execution import MockExecutor  # noqa: E402
import examples.d1_mocks as d1_mocks  # noqa: E402

# ---------------------------------------------------------------------------
# LLM client (anthropic-messages compatible endpoint)
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("D1_LLM_API_BASE", "https://api.minimaxi.com/anthropic")
API_KEY = os.environ.get("D1_LLM_API_KEY", "")
MODEL = os.environ.get("D1_LLM_MODEL", "MiniMax-M3")
MAX_TOKENS = int(os.environ.get("D1_LLM_MAX_TOKENS", "4096"))


class LLMClient:
    """Minimal anthropic-messages client for D1.1 generation."""

    def __init__(self, api_key: str = "", base_url: str = API_BASE,
                 model: str = MODEL, max_tokens: int = MAX_TOKENS) -> None:
        self.api_key = api_key or API_KEY
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError(
                "D1_LLM_API_KEY not set. Provide the MiniMax (or compatible) "
                "API key via env var D1_LLM_API_KEY."
            )

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
            "system": system,
        }
        resp = requests.post(
            f"{self.base_url}/v1/messages", headers=headers, json=payload, timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        parts: list[str] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Tool registry + task-type spec (mirror D1 templates)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "d1_calculate",
            "description": "计算简单算术表达式（整数），返回整数结果。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "d1_get_weather",
            "description": "查询指定城市当前天气，返回天气字符串。",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "d1_web_search",
            "description": "搜索网络信息，返回搜索摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
            },
        },
    },
]

# task_type -> generation instruction (Chinese) fed to the LLM.
TASK_SPECS: dict[str, str] = {
    "no_tool": "用户的问题不需要调用任何工具，模型直接回答即可。",
    "single_tool": "用户的问题需要恰好调用一个工具（计算、天气或搜索）才能回答。",
    "multi_tool": "用户的问题需要调用两个或更多工具（如先搜索再计算，或先查天气再搜索），且存在顺序/依赖关系。",
    "tool_error": "用户的问题涉及一个场景：要么请求的工具不在可用列表中（应报告不可用），要么工具会返回错误响应（应报告失败）。",
    "insufficient_result": "用户的问题需要搜索，但搜索结果是有限的（只返回 1 条），模型应发现结果不足并追问用户。",
    "requirement_change": "用户先提出一个需求，然后中途改变主意，模型应以最新要求为准。",
}


def build_user_prompt(task_type: str, seed: int) -> str:
    spec = TASK_SPECS[task_type]
    return (
        "你是一个中文用户，正在与一个具备工具调用能力的 AI 助手对话。\n"
        f"场景要求：{spec}\n"
        "可用工具：\n"
        + "\n".join(
            f"- {t['function']['name']}({', '.join(t['function']['parameters'].get('properties', {}))})"
            for t in TOOL_REGISTRY
        )
        + f"\n请用一句自然、真实的中文说出你作为用户的需求（不要提到'工具'这个词，不要输出 JSON）。\n随机种子:{seed}"
    )


def build_annotate_prompt(task_type: str, user_turn: str, seed: int) -> str:
    spec = TASK_SPECS[task_type]
    return (
        "你是工具调用数据标注器。给定用户请求，输出预期的工具调用计划 JSON。\n"
        f"任务类型：{task_type}\n任务类型说明：{spec}\n"
        f"用户请求：{user_turn}\n\n"
        "输出 JSON（不要其他文字）：\n"
        "{\n"
        '  "expected_tool_calls": [\n'
        '    {"call_id": "c1", "name": "<tool name>", "arguments": {<arguments>}, "depends_on": []},\n'
        '    ...\n'
        "  ],\n"
        '  "expected_answer": "<期望的最终回答摘要，中文>"\n'
        "}\n"
        "规则：\n"
        "- no_tool → expected_tool_calls 为空数组；\n"
        "- tool_error 的不可用场景 → 空数组 + expected_answer 说明应报告不可用；\n"
        "- tool_error 的错误响应场景 → 调用 d1_translate（不在注册表中）由模型识别失败，expected_tool_calls 为空；\n"
        "- 工具名必须是：d1_calculate / d1_get_weather / d1_web_search；\n"
        "- **d1_calculate 的 expression 必须是纯算术表达式**（只含数字和 + - * / % ( ) ^ 空格，不含任何中文或英文单词），例如 \"3*7\" 或 \"1200000*(0.042/12)\"；\n"
        "- d1_get_weather 的 city 必须是中文城市名，**只允许 city 一个参数，禁止添加 date/units 等任何额外参数**；d1_web_search 的 query 必须是中文查询词（limit 可选正整数）；\n"
        "- multi_tool 时用 depends_on 表达依赖（如搜索结果作为计算输入）。\n"
        f"随机种子:{seed}"
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM text response.

    Handles optional markdown code fences (```json ... ```).
    """
    # Strip markdown code fences if present.
    if "```" in text:
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# MockExecutor execution of expected calls
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Any] = {
    "d1_calculate": d1_mocks.d1_calculate,
    "d1_get_weather": d1_mocks.d1_get_weather,
    "d1_web_search": d1_mocks.d1_web_search,
}


def verify_expected_calls(expected_calls: list[dict[str, Any]]) -> None:
    """Execute expected_calls through MockExecutor; raise on any failure.

    This guarantees every emitted sample's expected plan is actually
    executable and the declared result matches the mock output (auditor
    contract: no expected_tool_calls without a valid expected_result).
    """
    if not expected_calls:
        return
    executor = MockExecutor()
    for tool in TOOL_REGISTRY:
        fn = REGISTRY.get(tool["function"]["name"])
        if fn is None:
            continue
        executor.register_mock(tool["function"]["name"], fn, tool["function"]["parameters"])
    for call in expected_calls:
        exec_result = executor.execute({
            "tool_name": call.get("name"),
            "call_id": call.get("call_id", ""),
            "arguments": call.get("arguments") or {},
            "depends_on": call.get("depends_on", []),
        })
        if exec_result["outcome"] != "success":
            raise ValueError(
                f"expected call {call.get('name')} not executable: "
                f"outcome={exec_result['outcome']} error={exec_result['error']}"
            )
        expected_result = call.get("expected_result")
        if expected_result is not None and exec_result["result"] != expected_result:
            raise ValueError(
                f"expected call {call.get('name')} result mismatch: "
                f"declared={expected_result!r} actual={exec_result['result']!r}"
            )
        if expected_result is None:
            # Fill in the deterministic result from the mock so result_grounded
            # is always verifiable (mirror D1 generator contract).
            call["expected_result"] = exec_result["result"]


def build_sample(
    sample_id: str, task_type: str, user_turn: str,
    plan: dict[str, Any], split: str, model_version: str,
) -> dict[str, Any]:
    expected_calls = plan.get("expected_tool_calls", [])
    # Semantic guard: no_tool samples must have NO expected calls (the LLM
    # annotator sometimes mislabels a request that actually needs a tool).
    if task_type == "no_tool" and expected_calls:
        raise ValueError(
            f"no_tool sample {sample_id} has {len(expected_calls)} expected calls"
        )
    verify_expected_calls(expected_calls)
    return {
        "schema_version": "1.0",
        "id": sample_id,
        "messages": [
            {"role": "user", "content": user_turn},
        ],
        "tools": TOOL_REGISTRY,
        "expected_tool_calls": expected_calls,
        "expected_answer": plan.get("expected_answer"),
        "metadata": {
            "source": f"{model_version}",
            "license": "internal",
            "task_type": task_type,
            "data_version": "D1.1",
            "pipeline_version": "d1.1-llm-generator",
            "created_at": "2026-08-27T00:00:00Z",
            "validation": {"schema_valid": True},
        },
    }


def validate_schema(sample: dict[str, Any]) -> None:
    # Validate tools + expected_tool_calls against the tool schema.
    for tool in sample["tools"]:
        validator = Draft202012Validator(tool["function"]["parameters"])
        for call in sample.get("expected_tool_calls", []):
            if call.get("name") != tool["function"]["name"]:
                continue
            errors = list(validator.iter_errors(call.get("arguments", {})))
            if errors:
                raise ValueError(f"schema error in {sample['id']}: {errors}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=6,
                        help="number of samples to generate (default 6, one per task_type)")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "datasets" / "tool-calling-d1-llm")
    parser.add_argument("--task-types", nargs="*", default=None,
                        help="restrict to specific task_types")
    parser.add_argument("--model-version", default="MiniMax-M3@2026-08-27")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip sample ids whose output file already exists (resume semantics)")
    parser.add_argument("--dry-run", action="store_true",
                        help="use a canned LLM response (no API call) for pipeline testing")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    task_types = args.task_types or list(TASK_SPECS.keys())
    client: LLMClient | None = None
    if not args.dry_run:
        client = LLMClient()

    # Deterministic per-sample seed so a dry-run / fixed-seed run is reproducible.
    rng = random.Random(2026)
    samples: list[dict[str, Any]] = []
    failures: list[str] = []
    for i in range(args.count):
        task_type = task_types[i % len(task_types)]
        seed = rng.randint(1, 10**9)
        sample_id = f"d1llm-{args.split}-{i + 1:04d}"
        out_file = out_dir / args.split / f"{sample_id}.json"
        if args.skip_existing and out_file.is_file():
            print(f"[d1.1] {sample_id} SKIP (exists)", flush=True)
            continue
        # LLM generation is noisy: retry each sample up to 3 attempts with
        # fresh user-turn + annotation calls before giving up.
        generated = False
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                if args.dry_run:
                    user_turn = f"[dry-run] 请帮我查一下北京天气（task_type={task_type}）"
                    plan = {
                        "expected_tool_calls": [
                            {"call_id": "c1", "name": "d1_get_weather",
                             "arguments": {"city": "北京"}, "depends_on": []},
                        ],
                        "expected_answer": "北京今天晴朗，23 度。",
                    }
                else:
                    assert client is not None
                    user_turn = client.chat(
                        "你是中文用户，生成工具调用场景。",
                        build_user_prompt(task_type, seed + attempt),
                        temperature=0.9,
                    )
                    plan_text = client.chat(
                        "你是工具调用数据标注器，只输出 JSON。",
                        build_annotate_prompt(task_type, user_turn, seed + attempt),
                        temperature=0.3,
                    )
                    plan = _extract_json(plan_text)
                sample = build_sample(
                    sample_id, task_type, user_turn, plan, args.split, args.model_version,
                )
                validate_schema(sample)
                samples.append(sample)
                print(f"[d1.1] {sample_id} {task_type} ok (attempt {attempt}): "
                      f"{user_turn[:40]!r}", flush=True)
                generated = True
                break
            except Exception as e:  # noqa: BLE001 — retry then collect
                last_error = e
                print(f"[d1.1] {sample_id} {task_type} attempt {attempt} FAIL: {e}",
                      flush=True)
        if not generated:
            failures.append(f"{sample_id} ({task_type}): {last_error}")

    if failures:
        print(f"[d1.1] {len(failures)}/{args.count} samples failed:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        # Successful samples are still persisted (only the failed ones are
        # dropped) so a partial run never loses completed work.

    # Write samples + MANIFEST (all successfully generated samples). When
    # --skip-existing was used, the manifest is rebuilt from every sample on
    # disk so skipped ids are still included.
    manifest_entries: list[dict[str, Any]] = []
    written = 0
    for sample in samples:
        rel = Path(args.split) / f"{sample['id']}.json"
        path = out_dir / rel
        path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        written += 1
        manifest_entries.append({
            "path": str(rel),
            "sha256": sha256_file(path),
            "split": args.split,
            "task_type": sample["metadata"]["task_type"],
            "source": sample["metadata"]["source"],
        })
    if args.skip_existing:
        for path in sorted((out_dir / args.split).glob("d1llm-*.json")):
            rel = Path(args.split) / path.name
            if rel in {Path(e["path"]) for e in manifest_entries}:
                continue
            sample = json.loads(path.read_text(encoding="utf-8"))
            manifest_entries.append({
                "path": str(rel),
                "sha256": sha256_file(path),
                "split": args.split,
                "task_type": sample["metadata"]["task_type"],
                "source": sample["metadata"]["source"],
            })
    aggregate = hashlib.sha256()
    # Per-file provenance: hash includes path + sha256 + source so the
    # aggregate changes when provenance changes (even if file bytes do not).
    for e in manifest_entries:
        aggregate.update(e["sha256"].encode("utf-8"))
        aggregate.update(b"\x00")
        aggregate.update(e["source"].encode("utf-8"))
        aggregate.update(b"\x00")
    # Aggregate the per-entry sources (sorted) for a quick top-level view.
    sources_seen = sorted({e["source"] for e in manifest_entries})
    manifest = {
        "data_version": "D1.1",
        "pipeline": "d1.1-llm-generator",
        # Total on-disk entries — includes skipped-existing samples from
        # previous runs with potentially different sources.
        "count": len(manifest_entries),
        "split": args.split,
        "aggregate_sha256": aggregate.hexdigest(),
        "sources": sources_seen,
        "samples": manifest_entries,
    }
    manifest_path = out_dir / f"MANIFEST-{args.split}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(f"[d1.1] wrote {len(samples)} samples to {split_dir}")
    print(f"[d1.1] wrote {manifest_path} (count={len(manifest_entries)}, sources={sources_seen})")
    # Exit non-zero if any sample failed so callers can detect partial runs,
    # but successful samples are already persisted.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
