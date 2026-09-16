"""**V0 验收标准 4**:各分项贡献之和与总变化的差值必须小于 0.1%。

这是整个 V0 里最重要的一组测试。简报 §3.2 把残差定为「唯一的确定性信号」,
而残差之所以能当信号用,前提是「分解本身不会引入误差」。
若这些测试挂了,后续所有版本的结论都不可信。

阈值说明:简报要求 0.1%,而实际实现的误差应当只有浮点量级(~1e-13%)。
因此每条断言都同时检查两个阈值 —— 先卡死浮点量级,再复核简报的 0.1%。
"""

from __future__ import annotations

import math

import pytest

BRIEF_TOLERANCE_PCT = 0.1
"""简报 V0 验收标准 4 的阈值。"""

FLOAT_TOLERANCE_PCT = 1e-6
"""实现层面应达到的阈值。远严于简报要求,任何真实的算术错误都会被它抓住。"""


# ---------------------------------------------------------------------------
# 乘法(因子)分解
# ---------------------------------------------------------------------------
def test_factor_contributions_sum_to_total_log_change(engine, last_two_weeks):
    """Δln(GMV) = Δln(UV) + Δln(CVR) + Δln(AOV),精确成立。"""
    period_a, period_b = last_two_weeks
    result = engine.decompose_factors("gmv", period_a, period_b)

    total_from_parts = sum(f.log_delta for f in result.factors)
    assert result.log_delta == pytest.approx(total_from_parts, abs=1e-12)
    assert abs(result.residual_pct) < FLOAT_TOLERANCE_PCT
    assert abs(result.residual_pct) < BRIEF_TOLERANCE_PCT
    assert result.is_closed


def test_factor_shares_sum_to_one(engine, last_two_weeks):
    period_a, period_b = last_two_weeks
    result = engine.decompose_factors("gmv", period_a, period_b)
    assert sum(f.share_of_change for f in result.factors) == pytest.approx(1.0, abs=1e-12)


def test_factor_percentage_points_sum_to_total_pct_change(engine, last_two_weeks):
    """百分点换算后仍然精确闭合(简报 §13.3)。"""
    period_a, period_b = last_two_weeks
    result = engine.decompose_factors("gmv", period_a, period_b)

    total_pp = sum(f.contribution_pct_points for f in result.factors)
    assert total_pp == pytest.approx(result.pct_change, abs=1e-12)

    # 相对误差也要满足简报的 0.1%
    relative_gap_pct = abs(total_pp - result.pct_change) / abs(result.pct_change) * 100
    assert relative_gap_pct < BRIEF_TOLERANCE_PCT


def test_uv_times_cvr_times_aov_equals_gmv(engine, last_two_weeks):
    """恒等式在**数据层**也成立 —— 这一条验的是口径,不是算术。"""
    period_a, period_b = last_two_weeks
    result = engine.decompose_factors("gmv", period_a, period_b)

    product_a = math.prod(f.value_a for f in result.factors)
    product_b = math.prod(f.value_b for f in result.factors)
    assert product_a == pytest.approx(result.value_a, rel=1e-12)
    assert product_b == pytest.approx(result.value_b, rel=1e-12)
    assert result.identity_gap_pct < FLOAT_TOLERANCE_PCT


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"channel": "Paid Search"},
        {"channel": "Organic"},
        {"device": "mobile"},
        {"device": "desktop"},
        {"market": "US"},
        {"market": "DE"},
        {"region": "EMEA"},
        {"user_tier": "RETURNING"},
        {"channel": "Paid Search", "device": "mobile"},
        {"channel": "Organic", "device": "desktop", "market": "US"},
    ],
    ids=lambda f: "+".join(f"{k}={v}" for k, v in f.items()) or "no_filter",
)
def test_factor_closure_holds_on_every_slice(engine, last_two_weeks, filters):
    """闭合性与切片无关 —— 这是 L1 循环能层层下钻的前提。"""
    period_a, period_b = last_two_weeks
    result = engine.decompose_factors("gmv", period_a, period_b, filters)

    assert result.log_delta == pytest.approx(
        sum(f.log_delta for f in result.factors), abs=1e-12
    )
    assert abs(result.residual_pct) < FLOAT_TOLERANCE_PCT
    assert result.identity_gap_pct < FLOAT_TOLERANCE_PCT


def test_factor_closure_holds_on_every_week_pair(engine, weeks):
    """8 周里任取相邻两周,闭合性都成立。"""
    for period_a, period_b in zip(weeks, weeks[1:]):
        result = engine.decompose_factors("gmv", period_a, period_b)
        assert abs(result.residual_pct) < FLOAT_TOLERANCE_PCT, f"{period_a} -> {period_b}"
        assert result.identity_gap_pct < FLOAT_TOLERANCE_PCT, f"{period_a} -> {period_b}"


# ---------------------------------------------------------------------------
# 加法(维度)分解
# ---------------------------------------------------------------------------
def _additive_cases(knowledge) -> list[tuple[str, str]]:
    return [
        (name, dimension)
        for name, metric in sorted(knowledge.metrics.items())
        if metric.is_additive
        for dimension in metric.dimensions
    ]


def test_dimension_closure_for_every_metric_and_dimension(engine, knowledge, last_two_weeks):
    """每个可加指标 × 每个白名单维度,残差都必须小于 0.1%。

    其中 category / sub_category / price_tier / user_tier 需要 LEFT JOIN 维表,
    因此这条测试同时验证了 join 没有丢行、也没有 fan-out。
    """
    period_a, period_b = last_two_weeks
    cases = _additive_cases(knowledge)
    assert cases, "知识库里应当至少有一个可加指标"

    for metric, dimension in cases:
        result = engine.decompose_by_dimension(metric, dimension, period_a, period_b)
        label = f"{metric} by {dimension}"

        recomputed = sum(part.delta for part in result.parts)
        assert result.delta == pytest.approx(recomputed, rel=1e-12, abs=1e-9), label
        assert abs(result.residual_pct) < FLOAT_TOLERANCE_PCT, label
        assert abs(result.residual_pct) < BRIEF_TOLERANCE_PCT, label
        assert result.is_closed, label

        # 两个窗口各自的「总量 vs 分组求和」也必须对得上:
        # 只看 Δ 的话,丢行造成的水位错位可能在两期之间互相抵消。
        assert result.level_residual_a_pct < FLOAT_TOLERANCE_PCT, label
        assert result.level_residual_b_pct < FLOAT_TOLERANCE_PCT, label


def test_dimension_parts_partition_the_total(engine, last_two_weeks):
    """各分项的基期值之和等于基期总量(划分完整)。"""
    period_a, period_b = last_two_weeks
    result = engine.decompose_by_dimension("gmv", "channel", period_a, period_b)

    assert sum(p.value_a for p in result.parts) == pytest.approx(result.value_a, rel=1e-12)
    assert sum(p.value_b for p in result.parts) == pytest.approx(result.value_b, rel=1e-12)
    assert sum(p.share_of_change for p in result.parts) == pytest.approx(1.0, abs=1e-12)


def test_dimension_closure_holds_under_filters(engine, last_two_weeks):
    """在过滤后的子集里,闭合性同样成立(L1 下钻时会不断加过滤)。"""
    period_a, period_b = last_two_weeks
    result = engine.decompose_by_dimension(
        "gmv", "category", period_a, period_b, {"channel": "Paid Search", "device": "mobile"}
    )
    assert abs(result.residual_pct) < FLOAT_TOLERANCE_PCT
    assert result.value_a > 0


def test_no_unknown_or_null_buckets_in_clean_sample_data(engine, knowledge, last_two_weeks):
    """样例数据是干净的,因此不应出现 ``__UNKNOWN__`` / ``__NULL__`` 桶。

    这两个桶的**存在**本身是诊断信号(join 丢行 / 列有 NULL);
    在合成数据上出现它们,说明生成器坏了。
    """
    period_a, period_b = last_two_weeks
    for dimension in knowledge.metric("gmv").dimensions:
        result = engine.decompose_by_dimension("gmv", dimension, period_a, period_b)
        assert result.unknown_share_pct == 0.0, f"{dimension} 出现了未知/空值分组"


# ---------------------------------------------------------------------------
# 跨层:因子分解与维度分解必须互相印证
# ---------------------------------------------------------------------------
def test_dimension_totals_match_factor_totals(engine, last_two_weeks):
    """同一个 GMV,两种分解方式算出的总量必须一致。

    两条路径走的是不同的 SQL(一条带 join 分组,一条不带),
    它们对不上就意味着某处口径漂了。
    """
    period_a, period_b = last_two_weeks
    factor = engine.decompose_factors("gmv", period_a, period_b)
    dimensional = engine.decompose_by_dimension("gmv", "channel", period_a, period_b)

    assert factor.value_a == pytest.approx(dimensional.value_a, rel=1e-12)
    assert factor.value_b == pytest.approx(dimensional.value_b, rel=1e-12)
    assert factor.pct_change == pytest.approx(dimensional.pct_change, rel=1e-12)


def test_uv_dimension_sum_matches_factor_uv(engine, last_two_weeks):
    """UV 也要能两边对上 —— 它来自另一张事实表。"""
    period_a, period_b = last_two_weeks
    factor = engine.decompose_factors("gmv", period_a, period_b)
    uv_factor = next(f for f in factor.factors if f.factor == "uv")
    uv_dim = engine.decompose_by_dimension("uv", "channel", period_a, period_b)

    assert uv_factor.value_a == pytest.approx(uv_dim.value_a, rel=1e-12)
    assert uv_factor.value_b == pytest.approx(uv_dim.value_b, rel=1e-12)


# ---------------------------------------------------------------------------
# 预算:一次分解花了几条查询
# ---------------------------------------------------------------------------
def test_query_budget_per_decomposition(engine, target, last_two_weeks):
    """两个窗口合并成一条查询 —— Free Edition 的配额经不起翻倍。"""
    period_a, period_b = last_two_weeks

    factor = engine.decompose_factors("gmv", period_a, period_b)
    assert len(factor.queries) == 2, "GMV 的因子分解应当只查 fact_orders 与 fact_sessions 各一次"

    dimensional = engine.decompose_by_dimension("gmv", "channel", period_a, period_b)
    assert len(dimensional.queries) == 2, "维度分解 = 独立总量 + 分组求和,共两条"

    # 每条查询都留下可复现的凭证(简报约束 3)
    for record in (*factor.queries, *dimensional.queries):
        assert len(record.fingerprint) == 16
        assert record.row_count > 0
        assert record.dialect == target.dialect.name


def test_money_values_come_back_as_float(engine, last_two_weeks):
    """金额列是 ``DECIMAL(12,2)``,两种驱动都会还原成 ``Decimal``。

    对数分解需要 float;漏了这层转换的话,``math.log`` 会在真机上直接抛 TypeError,
    而本地如果恰好用了 DOUBLE 就发现不了。
    """
    period_a, period_b = last_two_weeks
    result = engine.decompose_factors("gmv", period_a, period_b)

    assert isinstance(result.value_a, float)
    assert isinstance(result.value_b, float)
    assert result.value_a > 0

    dimensional = engine.decompose_by_dimension("gmv", "channel", period_a, period_b)
    for part in dimensional.parts:
        assert isinstance(part.value_a, float)
        assert isinstance(part.delta, float)
