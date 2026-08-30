"""Guard test: GQA vs MHA 公开模型对比的 no-go 结论。

不直接写可行性结论（结论在 docs/experiments/gqa-vs-mha/feasibility.md），
本测试只校验：

1. 该结论文件存在且含 no-go 关键短语；
2. 已扫描的"主要 LLM 家族 + research artifacts"候选 list 不为空；
3. P5-04 已下载的 5 个公开 model 至少存在 1 个 GQA + 1 个 MHA，证明
   backend 兼容性已隐式验证，但 size / layers / hidden 不同 → 不能归因 GQA。

如果未来 P5-04 升级到某个真正"同 base MHA/GQA 双版本"的公开模型，
本测试需同步更新（这就是 no-go 结论被推翻的信号）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FEASIBILITY_DOC = ROOT / "docs" / "experiments" / "gqa-vs-mha" / "feasibility.md"
STAGE_REVIEW = ROOT / "docs" / "plans" / "reviews" / "stage-gqa-vs-mha-no-go.md"
MODELS_ROOT = ROOT / "artifacts" / "owt-real-eval" / "models" / "models"


def discover_model_configs():
    """Discover HF cache configs under any snapshot subdir.

    HF cache layout can be ``snapshots/<commit-sha>/config.json`` (P5-04
    default with exact revision) or ``snapshots/master/config.json`` (when
    only the master branch was fetched). Use ``rglob`` so both layouts are
    covered portably across Linux/Windows.
    """
    if not MODELS_ROOT.exists():
        return []
    return sorted(MODELS_ROOT.rglob("config.json"))


@pytest.fixture(scope="module")
def feasibility_text() -> str:
    if not FEASIBILITY_DOC.exists():
        pytest.skip(f"feasibility doc not found at {FEASIBILITY_DOC}")
    return FEASIBILITY_DOC.read_text(encoding="utf-8")


def test_feasibility_doc_present() -> None:
    assert FEASIBILITY_DOC.exists(), f"missing {FEASIBILITY_DOC}"


def test_feasibility_doc_states_no_go(feasibility_text: str) -> None:
    assert "no-go" in feasibility_text.lower()
    assert "找不到" in feasibility_text or "not find" in feasibility_text.lower()


def test_feasibility_doc_lists_main_families(feasibility_text: str) -> None:
    """no-go 结论必须基于对主要 LLM 家族的扫描，否则不充分。"""
    required_families = ["LLaMA", "Qwen", "Mistral", "Gemma", "Phi", "DeepSeek"]
    missing = [f for f in required_families if f not in feasibility_text]
    assert not missing, f"feasibility doc missing families: {missing}"


def test_feasibility_doc_cites_ainslie_uptraining(feasibility_text: str) -> None:
    """uptraining 是 GQA 论文的关键概念，必须被引用。"""
    assert "Ainslie" in feasibility_text or "uptraining" in feasibility_text.lower()


def test_p5_04_models_have_both_attention_types() -> None:
    """P5-04 5 个公开 model 中必须既有 GQA 也有 MHA，证明 backend 能处理两种路径。

    Portably discovers config.json under either ``snapshots/master/`` or
    ``snapshots/<commit-sha>/`` layouts via ``rglob``.
    """
    configs = discover_model_configs()
    assert configs, "no P5-04 models discovered under artifacts/owt-real-eval/models/models/"
    gqa_models = []
    mha_models = []
    seen = set()
    for cfg in configs:
        # Deduplicate: one canonical config per repo (the snapshot dir may
        # contain multiple ``snapshots/<sha>/`` symlinks pointing to the same
        # blob; they share the underlying config.json).
        try:
            if cfg.resolve() in seen:
                continue
            seen.add(cfg.resolve())
        except OSError:
            seen.add(cfg)
        c = json.loads(cfg.read_text(encoding="utf-8"))
        h = c.get("num_attention_heads")
        kv = c.get("num_key_value_heads", h)
        if h is None or kv is None:
            continue
        # Per-repo short name: ``HuggingFaceTB--SmolLM2-360M-Instruct``
        # = ``<owner>--<repo>``. Use the directory two levels above
        # ``config.json`` (snapshots/<sha>/config.json → repo dir).
        short = cfg.parent.parent.parent.name
        if kv == h:
            mha_models.append(short)
        else:
            gqa_models.append(short)
    assert gqa_models, f"no GQA model found in P5-04 set: {gqa_models=} {mha_models=}"
    assert mha_models, f"no MHA model found in P5-04 set: {gqa_models=} {mha_models=}"


def test_same_base_rule_documented(feasibility_text: str) -> None:
    """硬性判定标准 #1-#3 必须出现在文档中。"""
    assert "同 base" in feasibility_text or "same base" in feasibility_text.lower()
    assert "num_key_value_heads" in feasibility_text


def test_roadmap_no_go_status() -> None:
    """roadmap 候选 1 行必须标记为 no-go / 不可行。"""
    roadmap = ROOT / "docs" / "plans" / "roadmap.md"
    if not roadmap.exists():
        pytest.skip("roadmap not found")
    text = roadmap.read_text(encoding="utf-8")
    # 候选 1 行上下文（GQA vs MHA）应标记为 no-go
    assert "GQA" in text
    no_go_markers = ["no-go", "不可行", "infeasible", "未找到"]
    # 不强制每行都有 no-go（roadmap 可能先列目标再下结论），但至少出现一次
    assert any(m in text for m in no_go_markers), f"roadmap missing no-go marker; got: {[m for m in no_go_markers if m in text]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
