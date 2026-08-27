"""Deterministic data augmentation for SFT tool-calling samples.

The D1 + D1.1 datasets contain only 252 samples — too few to fine-tune a
Transformer without heavy overfitting (observed: train 0.69 / val 2.29 at
2000 steps with degenerate EOS-only generations). This module expands the
training set by deterministic, semantically-safe rewrites:

- weather calls:  replace the city with another from a fixed list and
  recompute ``expected_result`` via the mock (results are city-dependent);
- calculate calls:  rewrite the expression with small perturbations and
  recompute the result via the mock;
- search calls:  replace the query text and keep a mock-derived result;
- no_tool / tool_error:  copy verbatim (no safe rewrite).

Augmentation is seeded so a fixed run is reproducible; each original sample
can yield up to ``k`` variants (k=1..3).

Usage (from train_sft.py):
    from architecture_lab.training.sft_augment import augment_samples
    samples = augment_samples(base_samples, tokenizer_agnostic=False, k=2, seed=42)
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any

from examples.d1_mocks import d1_calculate, d1_get_weather, d1_web_search

CITIES = ["北京", "上海", "深圳", "广州", "杭州", "成都", "武汉", "西安", "重庆", "南京"]

# Simple numeric perturbations for arithmetic expressions: replace an integer
# literal with a nearby one (modulo 3..13) keeping the expression valid.
_INT_RE = re.compile(r"\b(\d+)\b")


def _perturb_expression(expr: str, delta: int) -> str:
    def repl(match: re.Match) -> str:
        value = int(match.group(1))
        return str(max(1, value + delta))

    # Perturb only a few literals (first occurrence) to keep semantics sane.
    parts = _INT_RE.split(expr)
    replaced = 0
    out: list[str] = []
    for i, part in enumerate(parts):
        if _INT_RE.fullmatch(part) and replaced < 2:
            value = int(part)
            out.append(str(max(1, value + delta)))
            replaced += 1
        else:
            out.append(part)
    return "".join(out)


def _augment_weather(call: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    city = call.get("arguments", {}).get("city")
    if not isinstance(city, str) or not city:
        return None
    choices = [c for c in CITIES if c != city]
    if not choices:
        return None
    new_city = rng.choice(choices)
    new_call = copy.deepcopy(call)
    new_call["arguments"]["city"] = new_city
    new_call["expected_result"] = d1_get_weather(new_city)
    return new_call


def _augment_calculate(call: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    expr = call.get("arguments", {}).get("expression")
    if not isinstance(expr, str) or not expr:
        return None
    delta = rng.choice([3, 5, 7, 11, 13])
    new_expr = _perturb_expression(expr, delta)
    if new_expr == expr:
        return None
    try:
        new_result = d1_calculate(new_expr)
    except ValueError:
        return None
    new_call = copy.deepcopy(call)
    new_call["arguments"]["expression"] = new_expr
    new_call["expected_result"] = new_result
    return new_call


def _augment_search(call: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    query = call.get("arguments", {}).get("query")
    if not isinstance(query, str) or not query:
        return None
    suffixes = ["最新", "2026", "推荐", "对比"]
    suffix = rng.choice(suffixes)
    new_query = query + suffix
    new_call = copy.deepcopy(call)
    new_call["arguments"]["query"] = new_query
    new_call["expected_result"] = d1_web_search(new_query)
    return new_call


def _augment_sample(sample: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    task_type = sample.get("metadata", {}).get("task_type", "")
    if task_type == "no_tool":
        return None  # no safe rewrite
    calls = sample.get("expected_tool_calls", [])
    if not calls:
        return None
    new_calls: list[dict[str, Any]] = []
    for call in calls:
        name = call.get("name", "")
        if name == "d1_get_weather":
            aug = _augment_weather(call, rng)
        elif name == "d1_calculate":
            aug = _augment_calculate(call, rng)
        elif name == "d1_web_search":
            aug = _augment_search(call, rng)
        else:
            aug = None
        if aug is None:
            new_calls.append(call)
        else:
            new_calls.append(aug)
    if new_calls == calls:
        return None  # no call was actually rewritten
    new_sample = copy.deepcopy(sample)
    new_sample["expected_tool_calls"] = new_calls
    new_sample["id"] = f"{sample.get('id', 'aug')}-aug"
    new_sample["metadata"] = dict(new_sample.get("metadata", {}))
    new_sample["metadata"]["pipeline_version"] = "sft-augment"
    return new_sample


def augment_samples(
    samples: list[dict[str, Any]], *, k: int = 2, seed: int = 42
) -> list[dict[str, Any]]:
    """Return original samples plus up to k deterministic variants each."""
    rng = random.Random(seed)
    out = list(samples)
    for sample in samples:
        for _ in range(k):
            variant = _augment_sample(sample, rng)
            if variant is None:
                break
            out.append(variant)
    return out
