"""知识库校验测试(V0 验收标准 2)。

正向:仓库里那份真实的知识库必须能加载通过。
反向:每一类写错的 YAML 都必须在**加载时**失败,而不是在报告里变成一个错误的数字。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from rca.errors import (
    DimensionNotFoundError,
    KnowledgeError,
    MetricNotFoundError,
)
from rca.knowledge import load_knowledge


# ---------------------------------------------------------------------------
# 正向
# ---------------------------------------------------------------------------
def test_repository_knowledge_loads(knowledge):
    assert set(knowledge.metrics) == {"gmv", "orders", "uv", "cvr", "aov"}
    assert set(knowledge.tables) == {
        "fact_orders",
        "fact_sessions",
        "fact_ad_spend",
        "dim_product",
        "dim_user",
    }


def test_gmv_declares_the_multiplicative_identity(knowledge):
    gmv = knowledge.metric("gmv")
    assert gmv.decomposition is not None
    assert gmv.decomposition.type == "multiplicative"
    assert gmv.decomposition.factors == ["uv", "cvr", "aov"]


def test_ratio_metrics_resolve_to_additive_base_metrics(knowledge):
    assert knowledge.base_metrics_of("cvr") == ["orders", "uv"]
    assert knowledge.base_metrics_of("aov") == ["gmv", "orders"]
    assert knowledge.base_metrics_of("gmv") == ["gmv"]


def test_cvr_spans_both_fact_tables(knowledge):
    """CVR 的口径横跨两张表 —— 这正是 filters 必须对齐的原因。"""
    assert knowledge.tables_of("cvr") == {"fact_orders", "fact_sessions"}
    assert knowledge.tables_of("gmv") == {"fact_orders"}


def test_dimension_priority_defines_drill_order(knowledge):
    assert knowledge.ordered_dimensions("gmv")[:3] == ["channel", "device", "market"]
    assert knowledge.ordered_dimensions("uv") == [
        "channel",
        "device",
        "market",
        "region",
        "user_tier",
    ]


def test_ratio_metrics_are_not_additive(knowledge):
    assert knowledge.metric("gmv").is_additive
    assert knowledge.metric("uv").is_additive
    assert not knowledge.metric("cvr").is_additive
    assert not knowledge.metric("aov").is_additive


# ---------------------------------------------------------------------------
# 反向:每一类错误都必须被拦下
# ---------------------------------------------------------------------------
@pytest.fixture()
def broken(tmp_path: Path, knowledge_dir: Path):
    """把真实知识库拷到临时目录,允许测试改坏其中一个文件。"""
    target = tmp_path / "knowledge"
    shutil.copytree(knowledge_dir, target)

    def edit(relative: str, mutate) -> Path:
        path = target / relative
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        return target

    return edit


def test_missing_required_field_is_rejected(broken):
    root = broken("metrics/gmv.yaml", lambda m: m.pop("display_name"))
    with pytest.raises(KnowledgeError, match="display_name"):
        load_knowledge(root)


def test_unknown_field_is_rejected(broken):
    """拼错的 key 必须报错,不能被静默忽略。"""
    root = broken("metrics/gmv.yaml", lambda m: m.update({"dimenions": ["channel"]}))
    with pytest.raises(KnowledgeError, match="dimenions"):
        load_knowledge(root)


def test_unknown_dimension_is_rejected(broken):
    root = broken("metrics/gmv.yaml", lambda m: m["dimensions"].append("weather"))
    with pytest.raises(DimensionNotFoundError, match="weather"):
        load_knowledge(root)


def test_dimension_unavailable_on_source_table_is_rejected(broken):
    """campaign 只存在于 fact_ad_spend,GMV 按它分解是无从谈起的。"""
    root = broken("metrics/gmv.yaml", lambda m: m["dimensions"].append("campaign"))
    with pytest.raises(KnowledgeError, match="campaign"):
        load_knowledge(root)


def test_unknown_metric_reference_is_rejected(broken):
    root = broken("metrics/cvr.yaml", lambda m: m["ratio"].update({"numerator": "revenue"}))
    with pytest.raises(MetricNotFoundError, match="revenue"):
        load_knowledge(root)


def test_broken_multiplicative_identity_is_rejected(broken):
    """漏掉一个因子(GMV = UV × CVR)—— 恒等式立刻不成立,必须在加载时就失败。

    这是本项目最重要的一条校验:分解公式一旦错了,
    残差就不再是「未解释的部分」,而是「公式本身的错」,整个系统失去判据。
    没有它,一个漏了 AOV 的知识库会安静地把 AOV 的全部变化算进残差里,
    然后被 §3.2 的阈值误判成「数据质量问题」。
    """
    root = broken(
        "metrics/gmv.yaml", lambda m: m["decomposition"].update({"factors": ["uv", "cvr"]})
    )
    with pytest.raises(KnowledgeError, match="恒等式不成立"):
        load_knowledge(root)


def test_inconsistent_default_filters_are_rejected(broken):
    """orders 与 gmv 同出自 fact_orders,却用了不同口径 —— AOV 会算错。"""
    root = broken(
        "metrics/orders.yaml",
        lambda m: m["filters_default"].update({"order_status": "CANCELLED"}),
    )
    with pytest.raises(KnowledgeError, match="filters_default"):
        load_knowledge(root)


def test_filter_on_nonexistent_column_is_rejected(broken):
    root = broken(
        "metrics/gmv.yaml", lambda m: m["filters_default"].update({"is_fraud": "false"})
    )
    with pytest.raises(KnowledgeError, match="is_fraud"):
        load_knowledge(root)


def test_dimension_order_must_match_global_priority(broken):
    """维度顺序是 L3 唯一被允许改的东西,所以它必须只有一个真源。"""

    def swap(metric):
        metric["dimensions"][0], metric["dimensions"][1] = (
            metric["dimensions"][1],
            metric["dimensions"][0],
        )

    root = broken("metrics/gmv.yaml", swap)
    with pytest.raises(KnowledgeError, match="priority"):
        load_knowledge(root)


def test_metric_must_be_either_additive_or_ratio(broken):
    root = broken(
        "metrics/cvr.yaml",
        lambda m: m.update({"sql": "COUNT(*)", "source_table": "fact_orders"}),
    )
    with pytest.raises(KnowledgeError, match="二选一"):
        load_knowledge(root)


def test_dimension_referencing_missing_column_is_rejected(broken):
    def rename(payload):
        for dim in payload["dimensions"]:
            if dim["name"] == "channel":
                dim["availability"]["fact_orders"]["column"] = "traffic_source"

    root = broken("dimensions.yaml", rename)
    with pytest.raises(KnowledgeError, match="traffic_source"):
        load_knowledge(root)


def test_duplicate_priority_is_rejected(broken):
    def clash(payload):
        payload["dimensions"][1]["priority"] = payload["dimensions"][0]["priority"]

    root = broken("dimensions.yaml", clash)
    with pytest.raises(KnowledgeError, match="priority"):
        load_knowledge(root)


def test_join_declared_unique_must_match_primary_key(broken):
    def bad_join(payload):
        for dim in payload["dimensions"]:
            if dim["name"] == "category":
                dim["availability"]["fact_orders"]["join"]["right_key"] = "category"

    root = broken("dimensions.yaml", bad_join)
    with pytest.raises(KnowledgeError, match="fan-out"):
        load_knowledge(root)
