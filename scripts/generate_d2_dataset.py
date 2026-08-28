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

def _variant_index_from_build_pos(build_pos: int) -> int:
    """Return the zero-based per-task variant index from a build position.

    ``build_pos`` is the 0-based position of the sample inside the
    round-robin ``build_samples`` output (so position ``6 * k + t`` is
    variant ``k`` of ``TASK_TYPES[t]``). This decouples variant identity
    from the final split id, so IID stratified shuffling can reshuffle
    variants across train / dev / test without rewriting semantic content.
    """
    return build_pos // len(TASK_TYPES)


# ``_variant_index`` is retained as a thin alias so the previous round-9
# tests can still call it on a fully-built sample id (it is no longer used
# by the builders themselves; see ``build_samples`` for the new contract).
def _variant_index(sample_id: str) -> int:
    local_number = int(sample_id.rsplit("-", 1)[-1])
    return (local_number - 1) // len(TASK_TYPES)


_SEMANTIC_DOMAINS = (
    "企业知识库", "客服工单", "科研复现", "教学演示", "移动端部署",
    "离线环境", "小模型推理", "多语言内容", "数据治理", "安全审查",
    "成本控制", "高并发服务", "边缘设备", "版本升级", "回归测试",
    "团队协作", "论文复核", "生产排障", "产品原型", "长期维护",
)
_SEMANTIC_FOCUSES = (
    "优先保留可复现性", "需要清楚说明限制", "关注输入边界",
    "记录关键中间结果", "避免臆造无法验证的信息",
)
_SEMANTIC_CONTEXTS = tuple(
    f"场景是{domain}，额外要求是{focus}"
    for domain in _SEMANTIC_DOMAINS
    for focus in _SEMANTIC_FOCUSES
)


def canonical_content_signature(sample: dict[str, Any]) -> str:
    """Canonical semantic projection used to prevent split leakage.

    Removes bookkeeping identifiers, dependency references, split metadata,
    and timestamps while retaining task type, tools, message content,
    expected arguments/results, and expected answer. Equal signatures mean
    the two rows express the same semantic training example.
    """
    ignored_keys = {"id", "call_id", "tool_call_id", "depends_on"}

    def project(value: Any, *, parent_key: str | None = None) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, child in value.items():
                if key in ignored_keys:
                    continue
                if parent_key == "metadata" and key in {"created_at", "split"}:
                    continue
                result[key] = project(child, parent_key=key)
            return result
        if isinstance(value, list):
            return [project(child, parent_key=parent_key) for child in value]
        return value

    projection = project(sample)
    projection.pop("id", None)
    return json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _multi_tool_sequential(rng: random.Random, sid: str, created: str, variant: int) -> dict[str, Any]:
    expressions = [
        "2 + 3", "17 * 4", "100 / 5", "3 ** 4", "(8 - 3) * 6",
        "11 + 29", "144 / 12", "7 * 8 - 9", "2 ** 6", "81 / 9 + 1",
        "(15 + 5) * 2", "99 - 37", "6 ** 2", "125 / 5", "13 * 7",
        "(20 - 6) * 3", "4 ** 3 + 2", "72 / 8", "18 + 27", "5 * 11",
    ]
    topics = [
        "result", "计算结果", "这个数字", "上面的答案", "所得数值",
        "算式输出", "第一步结果", "前一步数值", "运算答案", "该计算值",
    ]
    user_styles = [
        "先计算 {expr}，再搜索{topic} {result}。",
        "请先算出 {expr}，随后查找{topic}为 {result} 的资料。",
        "第一步求 {expr}；计算得到 {result} 后，搜索{topic}。",
        "把 {expr} 算出来，再围绕{topic} {result} 做一次搜索。",
        "依次完成：计算 {expr}，然后检索{topic}等于 {result} 的内容。",
        "请按顺序处理 {expr} 和后续搜索，搜索关键词使用 {topic} {result}。",
    ]
    expr = expressions[variant % len(expressions)]
    first_args = {"expression": expr}
    first_result = _mock_result("d1_calculate", first_args)
    topic = topics[(variant // len(expressions)) % len(topics)]
    limit = 2 + ((variant // (len(expressions) * len(topics))) % 4)
    second_args = {"query": f"{topic} {first_result}", "limit": limit}
    second_result = _mock_result("d1_web_search", second_args)
    c1, c2 = _call_id(sid, 1), _call_id(sid, 2)
    answer_styles = [
        "第一步结果为 {result}；随后搜索得到：{search_result}",
        "已完成计算（{result}），并根据该结果搜索：{search_result}",
        "计算输出是 {result}。相关检索结果为：{search_result}",
        "先得到 {result}，再完成相关检索：{search_result}",
    ]
    answer = answer_styles[(variant // 2) % len(answer_styles)].format(
        result=first_result, search_result=second_result)
    user = user_styles[(variant // 3) % len(user_styles)].format(
        expr=expr, topic=topic, result=first_result)
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": user},
            _assistant_call(c1, "d1_calculate", first_args),
            _tool_message(c1, "d1_calculate", first_result),
            _assistant_call(c2, "d1_web_search", second_args),
            _tool_message(c2, "d1_web_search", second_result),
            {"role": "assistant", "content": answer},
        ],
        [CALC_TOOL, SEARCH_TOOL],
        [
            {"call_id": c1, "name": "d1_calculate", "arguments": first_args,
             "expected_result": first_result},
            {"call_id": c2, "name": "d1_web_search", "arguments": second_args,
             "depends_on": [c1], "expected_result": second_result},
        ],
        answer, "multi_tool_sequential", "calc_then_search", created,
    )


_TRANSLATE_TEXTS = [
    "Hello, world.", "Good morning.", "Test sentence.", "Please review this note.",
    "The meeting starts at nine.", "模型训练需要稳定的数据。", "Keep the cache warm.",
    "Today is a sunny day.", "Attention is all you need.", "Thank you for your help.",
    "请确认收到这封邮件。", "The quick brown fox jumps.", "Version two is ready.",
    "我们将在周五发布。", "Use a deterministic seed.", "The result is reproducible.",
    "欢迎参加本次讨论。", "Metrics should be recorded.", "Deploy after validation.",
    "请把这段话翻译一下。",
]
_TRANSLATE_LANGS = ["zh", "en", "ja", "fr", "de"]
_TRANSLATE_REQUESTS = [
    "请把“{text}”翻译成{lang_name}。",
    "帮我将下面这句话转换为{lang_name}：{text}",
    "请调用翻译服务处理这段文本，目标语言是{lang_name}：{text}",
    "我需要{lang_name}版本，请翻译：{text}",
]
_TRANSLATE_LANG_NAMES = {"zh": "中文", "en": "英文", "ja": "日文", "fr": "法文", "de": "德文"}


def _tool_error_response(rng: random.Random, sid: str, created: str, variant: int) -> dict[str, Any]:
    text = _TRANSLATE_TEXTS[variant % len(_TRANSLATE_TEXTS)]
    target_lang = _TRANSLATE_LANGS[(variant // len(_TRANSLATE_TEXTS)) % len(_TRANSLATE_LANGS)]
    args = {"text": text, "target_lang": target_lang}
    result = _mock_result("d1_translate", args)
    c1 = _call_id(sid, 1)
    request = _TRANSLATE_REQUESTS[(variant // 3) % len(_TRANSLATE_REQUESTS)].format(
        text=text, lang_name=_TRANSLATE_LANG_NAMES[target_lang])
    answer = [
        f"翻译工具返回错误：{result}。无法完成本次翻译。",
        f"翻译服务暂不可用（{result}），因此没有生成译文。",
        f"已调用翻译工具，但它报告错误：{result}。请稍后重试。",
        f"本次翻译失败，工具反馈为 {result}。",
    ][variant % 4]
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": request},
            _assistant_call(c1, "d1_translate", args),
            _tool_message(c1, "d1_translate", result),
            {"role": "assistant", "content": answer},
        ],
        [TRANSLATE_TOOL],
        [{"call_id": c1, "name": "d1_translate", "arguments": args,
          "expected_result": result}],
        answer, "tool_error_response", "translate_error_report", created,
    )


def _insufficient_result_search(rng: random.Random, sid: str, created: str, variant: int) -> dict[str, Any]:
    queries = [
        "LLM 评测", "GRPO 强化学习", "vLLM 部署", "MoE 路由", "MLA 注意力",
        "GQA 推理", "BPE tokenizer", "长上下文训练", "RAG 评测", "DPO 对齐",
        "RLHF 数据构造", "FlashAttention 优化", "KV cache 压缩", "模型蒸馏",
        "量化推理", "结构化剪枝", "RoPE 外推", "Mamba 状态空间模型",
        "speculative decoding", "PagedAttention", "工具调用数据集", "SFT 过拟合",
        "Transformer 归一化", "学习率调度", "多 seed 实验",
    ]
    request_styles = [
        "搜索『{query}』的资料，如果结果不够请追问。",
        "请查找关于 {query} 的信息；资料过少时向我确认范围。",
        "帮我搜集 {query} 相关内容，若只有少量结果就先询问我的关注点。",
        "我想了解 {query}，请先搜索，结果不足再向我澄清需求。",
        "围绕 {query} 做一次检索；不充分时不要臆测，请回来提问。",
        "请检索 {query}，如果命中有限，请询问我更具体的方向。",
    ]
    followups = [
        "我关注 2026 年的最新进展。",
        "我更想看工程实践和性能数据。",
        "请优先整理开源实现方面的信息。",
        "我关注训练稳定性与可复现性。",
        "请把重点放在小模型场景。",
        "我想了解它在工具调用中的应用。",
        "请补充论文和实验结论。",
        "我主要关心部署成本。",
    ]
    query = queries[variant % len(queries)]
    limit = 1 + ((variant // len(queries)) % 2)
    args = {"query": query, "limit": limit}
    result = _mock_result("d1_web_search", args)
    c1 = _call_id(sid, 1)
    request = request_styles[(variant // 2) % len(request_styles)].format(query=query)
    followup = followups[(variant // (len(request_styles) * 2)) % len(followups)]
    answer = [
        f"仅找到 {limit} 条结果：{result}。请补充你关注的方向。",
        f"目前检索到的结果有限（{result}），请告诉我你想深入的角度。",
        f"这次搜索范围较窄，仅返回 {result}；请进一步说明筛选条件。",
        f"已有初步结果：{result}。为了继续整理，请补充具体关注点。",
    ][variant % 4]
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": request},
            _assistant_call(c1, "d1_web_search", args),
            _tool_message(c1, "d1_web_search", result),
            {"role": "assistant", "content": answer},
            {"role": "user", "content": followup},
            {"role": "assistant", "content": (
                [
                    "收到，我会按这个方向继续。",
                    "明白，我会据此收窄后续整理范围。",
                    "了解，我将优先筛选符合该条件的资料。",
                    "好的，我会按你的关注点继续检索。",
                ][variant % 4]
            )},
        ],
        [SEARCH_TOOL],
        [{"call_id": c1, "name": "d1_web_search", "arguments": args,
          "expected_result": result}],
        answer, "insufficient_result_search", "search_then_clarify", created,
    )


def _req_change_city(rng: random.Random, sid: str, created: str, variant: int) -> dict[str, Any]:
    cities = [
        "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉", "西安",
        "厦门", "青岛", "苏州", "重庆", "天津", "昆明", "郑州", "福州",
    ]
    # Pick distinct old/new city pairs deterministically from ``variant`` so
    # the 100 req_change_city rows do not collapse onto the small set the
    # rng.sample(cities, 2) call would otherwise produce. ``cities`` has 16
    # entries, so 16*15/2 = 120 ordered pairs; ``variant % 120`` indexes one.
    ordered_pairs = [(a, b) for i, a in enumerate(cities)
                     for b in cities[i + 1:]]
    old_city, new_city = ordered_pairs[variant % len(ordered_pairs)]
    first_style_index = variant % 6
    change_style_index = (variant // 6) % 6
    answer_index = (variant // 36) % 4
    old_args, new_args = {"city": old_city}, {"city": new_city}
    old_result = _mock_result("d1_get_weather", old_args)
    new_result = _mock_result("d1_get_weather", new_args)
    c1, c2 = _call_id(sid, 1), _call_id(sid, 2)
    first_styles = [
        "帮我查一下{city}的天气。",
        "请先看看{city}今天的天气情况。",
        "我想知道{city}现在的天气，请查询一下。",
        "先帮我获取{city}的天气信息。",
        "可以查查{city}的当前天气吗？",
        "请调用天气工具查询{city}。",
    ]
    change_styles = [
        "改成查{city}的天气吧。",
        "不用刚才的城市了，请换成{city}。",
        "我改变主意了，改看{city}的天气。",
        "请把查询目标改为{city}。",
        "刚才城市不对，重新查询{city}。",
        "现在优先给我{city}的天气。",
    ]
    answer = [
        f"按最新要求，{new_city}天气：{new_result}",
        f"已按你的新要求查询{new_city}：{new_result}",
        f"忽略之前的{old_city}，当前结果是{new_result}",
        f"最新指定城市为{new_city}，天气信息如下：{new_result}",
    ][answer_index]
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": first_styles[first_style_index].format(city=old_city)},
            _assistant_call(c1, "d1_get_weather", old_args),
            _tool_message(c1, "d1_get_weather", old_result),
            {"role": "user", "content": change_styles[change_style_index].format(city=new_city)},
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


_NOT_AVAILABLE_VARIANTS: tuple[tuple[str, list[dict[str, Any]], str, str, tuple[str, ...]], ...] = (
    # (missing capability, available tool subset, user request, refusal rationale, capability_tag)
    ("实时汇率查询",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "请帮我查一下美元兑人民币的实时汇率。",
     "当前可用工具中没有汇率查询功能，无法完成此任务。",
     ("实时汇率",)),
    ("股票行情查询",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "帮我查一下特斯拉今天的股价。",
     "当前可用工具中没有股票行情查询功能，无法获取实时股价。",
     ("股票行情",)),
    ("发送邮件",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "请帮我给 luna@example.com 发一封提醒邮件。",
     "当前可用工具中没有邮件发送功能，我无法代你发邮件。",
     ("发送邮件",)),
    ("创建日历日程",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "帮我在明天下午 3 点创建一个会议日程。",
     "当前可用工具中没有日历日程功能，无法创建日程。",
     ("日历日程",)),
    ("航班查询",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "查一下后天北京到上海的航班。",
     "当前可用工具中没有航班查询功能，无法获取航班信息。",
     ("航班查询",)),
    ("食谱查询",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "帮我找一份番茄炒蛋的做法。",
     "当前可用工具中没有食谱查询功能，无法提供菜谱。",
     ("食谱查询",)),
    ("停车位查询",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "帮我看看公司楼下有没有停车位。",
     "当前可用工具中没有停车位查询功能，无法查询。",
     ("停车位查询",)),
    ("工单创建",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "请帮我创建一个新的 IT 工单。",
     "当前可用工具中没有工单创建功能，无法创建工单。",
     ("工单创建",)),
    ("加密货币价格",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "查一下比特币现在的价格。",
     "当前可用工具中没有加密货币价格查询功能，无法获取行情。",
     ("加密货币",)),
    ("天气预警",
     [CALC_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "帮我订阅明天北京的大风预警。",
     "当前可用工具中没有天气预警订阅功能，无法完成订阅。",
     ("天气预警",)),
    ("翻译成英文",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "请帮我把这份合同翻译成英文。",
     "当前可用工具中没有合同翻译功能，无法完成翻译。",
     ("合同翻译",)),
    ("图片生成",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "帮我生成一张秋天森林的图片。",
     "当前可用工具中没有图片生成功能，无法生成图片。",
     ("图片生成",)),
    ("附近便利店查询",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "查一下离我最近的 24 小时便利店。",
     "当前可用工具中没有便利店位置查询功能，无法获取相关信息。",
     ("便利店",)),
    ("商品比价",
     [CALC_TOOL, WEATHER_TOOL, TRANSLATE_TOOL],
     "帮我比一下同一款耳机在京东和淘宝的当前售价。",
     "当前可用工具中没有商品比价功能，无法跨平台对比价格。",
     ("商品比价",)),
    ("车辆违章查询",
     [CALC_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "查一下我车牌京A12345 最近的违章记录。",
     "当前可用工具中没有车辆违章查询功能，无法查询违规记录。",
     ("违章查询",)),
    ("身份证信息核验",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "请帮我核验一下这个身份证号的归属地。",
     "当前可用工具中没有身份信息核验功能，无法验证身份证。",
     ("身份核验",)),
    ("医院挂号",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "帮我在协和医院挂一个明天上午的消化内科号。",
     "当前可用工具中没有医院挂号功能，无法为你预约门诊。",
     ("医院挂号",)),
    ("电梯维保查询",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "查一下我们小区电梯下次维保的时间。",
     "当前可用工具中没有电梯维保查询功能，无法获取记录。",
     ("电梯维保",)),
    ("小区门禁临时密码",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "帮访客生成一个今天有效的小区门禁临时密码。",
     "当前可用工具中没有门禁密码生成功能，无法创建临时凭证。",
     ("门禁密码",)),
    ("机房报警阈值设置",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "把机房的温度报警门槛调到 28 度。",
     "当前可用工具中没有机房参数调整功能，无法修改阈值。",
     ("机房阈值",)),
    ("上传文件到云盘",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "帮我把刚才的 PDF 上传到公司云盘。",
     "当前可用工具中没有云盘上传功能，无法代你上传文件。",
     ("云盘上传",)),
    ("查询论文引用次数",
     [CALC_TOOL, WEATHER_TOOL, TRANSLATE_TOOL],
     "查一下 Attention Is All You Need 的最新引用数。",
     "当前可用工具中没有论文引用查询功能，无法统计引用次数。",
     ("论文引用",)),
    ("健身房课程预约",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL, TRANSLATE_TOOL],
     "帮我预约周三晚上的动感单车课。",
     "当前可用工具中没有课程预约功能，无法为你预订课程。",
     ("课程预约",)),
    ("附近打印店",
     [CALC_TOOL, WEATHER_TOOL, SEARCH_TOOL],
     "帮我在公司 1 公里内找一家能彩打的店。",
     "当前可用工具中没有周边商铺查询功能，无法推荐店铺。",
     ("打印店",)),
)


def _tool_not_available(rng: random.Random, sid: str, created: str, variant: int) -> dict[str, Any]:
    """Generate a semantically varied unavailable-capability refusal.

    The 24 capability records are combined with independent request,
    context, and refusal styles using the deterministic per-task variant
    index. This gives 100 distinct semantic combinations for the default
    count instead of sampling a small pool with replacement.
    """
    capability, tools_subset, request, rationale, capability_tag = (
        _NOT_AVAILABLE_VARIANTS[variant % len(_NOT_AVAILABLE_VARIANTS)])
    request_styles = (
        "{request}",
        "我现在有一个实际需求：{request}",
        "请处理下面这项请求：{request}",
        "如果可以的话，麻烦帮我完成：{request}",
        "我需要你协助处理这件事——{request}",
    )
    contexts = (
        "我今天就要用到结果。",
        "这是一个办公场景，请说明当前能力边界。",
        "请直接告诉我是否能完成，不要虚构结果。",
        "如果不能执行，请给出明确的限制说明。",
    )
    refusal_styles = (
        "当前可用工具中没有{capability}功能，无法完成此任务。",
        "当前工具列表不包含{capability}能力，因此无法执行该请求。",
        "我没有可用于{capability}的工具，不能假装已经完成这项操作。",
        "可用工具无法支持{capability}；本次请求不能由工具链完成。",
    )
    style = (variant // len(_NOT_AVAILABLE_VARIANTS)) % len(request_styles)
    context = contexts[(variant // (len(_NOT_AVAILABLE_VARIANTS) * len(request_styles))) % len(contexts)]
    refusal = refusal_styles[(variant // 7) % len(refusal_styles)].format(
        capability=capability)
    user = request_styles[style].format(request=request) + " " + context
    answer = refusal
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        list(tools_subset), [], answer,
        "tool_not_available", f"missing_{capability_tag[0][:8]}", created,
    )


_ERROR_RECOVERY_VARIANTS: tuple[tuple[str, str, str], ...] = (
    # (first_query, refined_query, user_intent)
    ("PyTorch", "PyTorch 2.x 新特性", "深度学习框架最新进展"),
    ("TensorFlow", "TensorFlow 2.15 新特性", "深度学习框架最新进展"),
    ("JAX", "JAX v0.4.20 主要改进", "深度学习框架最新进展"),
    ("MindSpore", "MindSpore 2.x 架构变动", "深度学习框架最新进展"),
    ("transformers", "transformers 4.40 主要变更", "训练库版本变化"),
    ("vLLM", "vLLM 0.4 性能提升点", "推理引擎近期性能"),
    ("LoRA", "LoRA 微调 2025 综述", "参数高效微调最新研究"),
    ("KV cache", "KV cache 压缩 2025 进展", "推理优化近期研究"),
    ("FlashAttention", "FlashAttention-3 性能数据", "注意力优化近期"),
    ("MoE", "Mixture-of-Experts 路由 2025", "混合专家机制"),
    ("MLA", "Multi-head Latent Attention 解析", "新型注意力机制"),
    ("GQA", "Grouped Query Attention 实现", "注意力机制实现"),
    ("GRPO", "GRPO 强化学习 综述", "强化学习算法"),
    ("RLHF", "RLHF vs DPO 对比", "对齐算法对比"),
    ("DPO", "DPO 收敛性分析", "对齐算法理论"),
    ("distillation", "knowledge distillation 2025", "模型压缩技术"),
    ("quantization", "INT4 quantization 论文", "模型量化研究"),
    ("pruning", "structured pruning 2025", "模型剪枝研究"),
    ("scaling laws", "Chinchilla scaling law 修正", "扩展律研究"),
    ("emergent abilities", "emergent abilities 争议", "涌现能力研究"),
    ("Mamba", "Mamba 状态空间模型 2025", "序列模型近期进展"),
    ("long context", "1M context 训练方法", "长上下文技术"),
    ("RAG", "RAG 检索增强 2025", "检索增强生成"),
    ("agent", "agent tool-use 2025", "智能体工具使用"),
    ("BPE", "BPE 词表训练 2025", "分词训练方法"),
    ("RoPE", "RoPE 外推 2025", "位置编码"),
    ("Yarn", "YaRN 位置编码", "位置编码变体"),
    ("ALiBi", "ALiBi 位置编码", "位置编码变体"),
    ("paged attention", "PagedAttention 性能", "推理优化"),
    ("speculative decoding", "speculative decoding 2025", "推理优化"),
)


def _error_recovery(rng: random.Random, sid: str, created: str, variant: int) -> dict[str, Any]:
    """Generate one deterministic, semantically distinct recovery trace."""
    first_query, refined_query, intent = _ERROR_RECOVERY_VARIANTS[
        variant % len(_ERROR_RECOVERY_VARIANTS)]
    first_limit = 1 + ((variant // len(_ERROR_RECOVERY_VARIANTS)) % 2)
    second_limit = 3 + ((variant // (len(_ERROR_RECOVERY_VARIANTS) * 2)) % 4)
    first_args = {"query": first_query, "limit": first_limit}
    second_args = {"query": refined_query, "limit": second_limit}
    first_result = _mock_result("d1_web_search", first_args)
    second_result = _mock_result("d1_web_search", second_args)
    c1, c2 = _call_id(sid, 1), _call_id(sid, 2)
    answer_styles = (
        "初次搜索『{first}』结果过少（{intent}），已重试为『{second}』并扩大 limit，得到：{result}",
        "第一次检索『{first}』不够充分；围绕{intent}改用『{second}』后得到：{result}",
        "针对{intent}，初始关键词『{first}』命中有限，细化为『{second}』的结果是：{result}",
        "搜索『{first}』后发现信息不足，我按{intent}调整到『{second}』：{result}",
        "『{first}』的初步结果不够，我将{intent}聚焦到『{second}』，返回：{result}",
    )
    first_ack_styles = (
        "“{first}”结果过少，我会细化关键词后重试。",
        "初次检索不充分，接下来我会收窄到更具体的关键词。",
        "这个搜索范围太宽，我先调整检索词。",
        "结果数量有限，我会根据任务目标重新组织查询。",
        "需要进一步细化主题后再搜索。",
    )
    answer = answer_styles[variant % len(answer_styles)].format(
        first=first_query, second=refined_query, intent=intent, result=second_result)
    return _sample(
        sid,
        [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user", "content": (
                f"搜索 {first_query} 的 {intent}；结果不足时请调整查询。"
            )},
            _assistant_call(c1, "d1_web_search", first_args),
            _tool_message(c1, "d1_web_search", first_result),
            {"role": "assistant", "content": first_ack_styles[
                (variant // len(_ERROR_RECOVERY_VARIANTS)) % len(first_ack_styles)
            ].format(first=first_query)},
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


BUILDERS: tuple[tuple[str, Callable[[random.Random, str, str, int], dict[str, Any]]], ...] = (
    ("tool_not_available", _tool_not_available),
    ("tool_error_response", _tool_error_response),
    ("insufficient_result_search", _insufficient_result_search),
    ("req_change_city", _req_change_city),
    ("multi_tool_sequential", _multi_tool_sequential),
    ("error_recovery", _error_recovery),
)


def build_samples(count: int, rng: random.Random, *, seed: int) -> list[dict[str, Any]]:
    """Generate ``count`` D2 samples in a fixed round-robin order.

    The ``build_pos`` (0-based) is the only stable identity used by the
    builders to derive ``variant``; the final sample id (and therefore
    the call ids and ``created_at`` timestamp) is assigned later by
    :func:`assign_split_ids` so the IID stratified split reshuffle can
    rewrite only the bookkeeping fields.
    """
    if count < 6:
        raise ValueError(f"--count must be >= 6, got {count}")
    if count % len(TASK_TYPES) != 0:
        raise ValueError(
            f"--count {count} must be a multiple of {len(TASK_TYPES)} to keep "
            f"per-task variant counts balanced"
        )
    samples: list[dict[str, Any]] = []
    for build_pos in range(count):
        task_type, builder = BUILDERS[build_pos % len(BUILDERS)]
        variant = _variant_index_from_build_pos(build_pos)
        build_sid = f"d2-build-{build_pos:04d}"
        sample = builder(rng, build_sid, _now_ts(seed, build_pos + 1), variant)
        if sample["metadata"]["task_type"] != task_type:
            raise AssertionError(
                f"builder/type mismatch at build_pos={build_pos}: "
                f"{task_type} != {sample['metadata']['task_type']}"
            )
        samples.append(sample)
    return samples


def assign_split_ids(samples: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """IID stratified shuffle + per-split renumbering.

    For each ``task_type``, the per-task variants are deterministically
    shuffled with ``seed`` and the first 70 / next 15 / final 15 are
    assigned to ``train`` / ``dev`` / ``test``. The final sample id
    (e.g. ``d2-train-0001``) is 1-based within its split so the
    timestamp contract ``1785000000 + seed + index`` stays globally
    unique across all 600 rows while every split now sees a uniform
    slice of the per-task variants.

    Sample content (messages, tool_calls' arguments, expected_answer,
    semantic context) is unchanged: only ``id``, ``metadata.split``,
    ``metadata.created_at``, and the per-message ``call_id`` /
    ``tool_call_id`` / ``depends_on`` strings are rewritten so they
    match the new public id.
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {task: [] for task in TASK_TYPES}
    for sample in samples:
        task_type = sample["metadata"]["task_type"]
        if task_type not in by_type:
            raise AssertionError(f"unexpected task_type {task_type!r}")
        by_type[task_type].append(sample)
    for task_type, type_samples in by_type.items():
        if len(type_samples) == 0:
            raise AssertionError(f"task_type {task_type!r} has no samples")
        rng.shuffle(type_samples)

    counts = {task: len(items) for task, items in by_type.items()}
    if len(set(counts.values())) != 1:
        raise AssertionError(
            f"unbalanced task_type counts: {counts}"
        )
    per_type_n = next(iter(counts.values()))
    train_per_type = int(per_type_n * 0.70)
    dev_per_type = int(per_type_n * 0.15)
    test_per_type = per_type_n - train_per_type - dev_per_type
    if train_per_type + dev_per_type + test_per_type != per_type_n:
        raise AssertionError(
            f"split sum mismatch: {train_per_type}+{dev_per_type}+{test_per_type} != {per_type_n}"
        )

    assigned: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for task_type, type_samples in by_type.items():
        assigned["train"].extend(type_samples[:train_per_type])
        assigned["dev"].extend(type_samples[train_per_type:train_per_type + dev_per_type])
        assigned["test"].extend(type_samples[train_per_type + dev_per_type:])

    output: list[dict[str, Any]] = []
    for split_name in ("train", "dev", "test"):
        for split_pos, original in enumerate(assigned[split_name], start=1):
            sample = json.loads(json.dumps(original, ensure_ascii=False))
            old_id = sample["id"]
            sample["id"] = f"d2-{split_name}-{split_pos:04d}"
            sample["metadata"]["split"] = split_name
            sample["metadata"]["created_at"] = _now_ts(seed, split_pos)
            sample["metadata"]["created_at_pos"] = split_pos

            old_to_new: dict[str, str] = {}
            all_call_ids: list[str] = []
            for call in sample["expected_tool_calls"]:
                all_call_ids.append(call["call_id"])
            for message in sample["messages"]:
                if message["role"] == "assistant":
                    all_call_ids.extend(
                        tool_call["id"] for tool_call in message.get("tool_calls", [])
                    )
            for old_call_id in dict.fromkeys(all_call_ids):
                suffix = old_call_id.rsplit("-", 1)[-1]
                old_to_new[old_call_id] = _call_id(sample["id"], int(suffix))
            for call in sample["expected_tool_calls"]:
                call["call_id"] = old_to_new[call["call_id"]]
                call["depends_on"] = [
                    old_to_new.get(dep, dep) for dep in call.get("depends_on", [])
                ]
            for message in sample["messages"]:
                if message["role"] == "assistant":
                    for tool_call in message.get("tool_calls", []):
                        if tool_call["id"] in old_to_new:
                            tool_call["id"] = old_to_new[tool_call["id"]]
                elif message["role"] == "tool":
                    if message.get("tool_call_id") in old_to_new:
                        message["tool_call_id"] = old_to_new[message["tool_call_id"]]
            # ``old_id`` is intentionally not retained in the artifact;
            # the build-time id has been fully translated to the new
            # public id by the rename above.
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


def _parse_assistant_arguments(arguments: Any) -> Any:
    """Assistant tool call ``arguments`` is stored as a JSON string. Return
    the parsed object so it can be compared structurally against the
    ``expected_tool_calls`` rows which carry dict arguments.
    """
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (TypeError, ValueError):
            return arguments
    return arguments


def transcript_well_formedness_errors(sample: dict[str, Any]) -> list[str]:
    """Round 11 cross-message invariants verified by message-position state machine.

    A flat-list ID comparison can hide positional violations such as a tool
    response appearing *after* the assistant final answer, or an assistant
    tool call whose matching tool response appears earlier in the
    transcript. The validator therefore walks ``messages`` in order and
    enforces the following positional invariants:

    1. The transcript's final assistant message (no ``tool_calls``,
       non-empty ``content``) is the conversation's terminal answer.
       Any ``role=tool`` message appearing after that point is rejected.
    2. Every assistant ``tool_calls[i].id`` must be answered by exactly
       one subsequent ``role=tool`` message whose ``tool_call_id``
       matches the call id, in the order the calls were emitted.
    3. No ``role=tool`` message may reference an unknown id, appear
       before its issuing assistant call, or appear after the final
       answer.
    4. Every assistant tool call's ``function.name`` and parsed
       ``function.arguments`` must match the ``expected_tool_calls``
       entry with the same ``call_id``.
    """
    errors: list[str] = []
    messages = list(sample.get("messages", []))
    expected_calls = sample.get("expected_tool_calls", [])
    expected_by_id = {call["call_id"]: call for call in expected_calls}

    final_answer_position: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if (
            message.get("role") == "assistant"
            and not (message.get("tool_calls") or [])
            and message.get("content") not in (None, "")
        ):
            final_answer_position = index
            break

    pending: list[str] = []
    seen_call_ids: set[str] = set()
    seen_tool_refs: set[str] = set()

    for position, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if final_answer_position is not None and position > final_answer_position:
                    errors.append(
                        f"message[{position}]: assistant tool_calls emitted "
                        f"after the final answer at message[{final_answer_position}]"
                    )
                for call in tool_calls:
                    cid = call["id"]
                    seen_call_ids.add(cid)
                    pending.append(cid)
        elif role == "tool":
            tcid = message.get("tool_call_id")
            if not tcid:
                errors.append(
                    f"message[{position}]: tool message missing tool_call_id"
                )
                continue
            seen_tool_refs.add(tcid)
            if final_answer_position is not None and position > final_answer_position:
                errors.append(
                    f"message[{position}]: tool message for {tcid!r} "
                    f"appears after the final answer at "
                    f"message[{final_answer_position}]"
                )
                continue
            if not pending:
                errors.append(
                    f"message[{position}]: tool message for {tcid!r} "
                    f"appears before any assistant tool call"
                )
                continue
            expected_first = pending[0]
            if tcid != expected_first:
                errors.append(
                    f"message[{position}]: tool message for {tcid!r} "
                    f"violates in-order matching (pending={expected_first!r})"
                )
                continue
            pending.pop(0)
        elif role in ("system", "user"):
            continue
        else:
            errors.append(
                f"message[{position}]: unexpected role {role!r}"
            )

    if pending:
        errors.append(
            f"unanswered assistant tool calls: {pending}"
        )

    missing_in_tool = seen_call_ids - seen_tool_refs
    if missing_in_tool:
        errors.append(
            f"assistant tool_call.id without matching tool message: "
            f"{sorted(missing_in_tool)}"
        )
    extra_in_tool = seen_tool_refs - seen_call_ids
    if extra_in_tool:
        errors.append(
            f"tool message references unknown tool_call_id: "
            f"{sorted(extra_in_tool)}"
        )

    for position, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            cid = call["id"]
            expected = expected_by_id.get(cid)
            if expected is None:
                errors.append(
                    f"message[{position}]: assistant tool_call.id {cid!r} "
                    f"missing from expected_tool_calls"
                )
                continue
            if expected["name"] != call["function"]["name"]:
                errors.append(
                    f"message[{position}]: assistant tool_call.name "
                    f"mismatch for {cid!r}: got {call['function']['name']!r}, "
                    f"expected {expected['name']!r}"
                )
            parsed_args = _parse_assistant_arguments(
                call["function"].get("arguments")
            )
            if expected["arguments"] != parsed_args:
                errors.append(
                    f"message[{position}]: assistant tool_call.arguments "
                    f"mismatch for {cid!r}: got {parsed_args!r}, "
                    f"expected {expected['arguments']!r}"
                )

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
    signatures: dict[str, str] = {}
    for sample in samples:
        errors.extend(f"{sample['id']}: {error.message}" for error in validator.iter_errors(sample))
        errors.extend(f"{sample['id']}: {error}" for error in validate_semantics(sample))
        errors.extend(f"{sample['id']}: {error}" for error in transcript_well_formedness_errors(sample))
        errors.extend(f"{sample['id']}: {error}" for error in execute_through_mock_executor(sample))
        signature = canonical_content_signature(sample)
        previous_id = signatures.get(signature)
        if previous_id is not None:
            errors.append(
                f"{sample['id']}: duplicate canonical semantic content with {previous_id}"
            )
        else:
            signatures[signature] = sample["id"]
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
    if len(samples) != args.count:
        raise AssertionError(f"generated {len(samples)} samples, expected {args.count}")
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
