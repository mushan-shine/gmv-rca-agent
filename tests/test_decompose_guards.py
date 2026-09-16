"""分解引擎的「拒绝」行为。

简报 §5 给组件划了死线。这组测试验证越线时引擎会**报错**,
而不是返回一个看起来合理、实际上错误的数字 ——
归因系统最危险的失败模式不是崩溃,是自信地给出错误结论。
"""

from __future__ import annotations

from datetime import date

import pytest

from rca.decompose import Period, residual_severity
from rca.errors import (
    DecompositionError,
    DimensionNotFoundError,
    FilterNotSupportedError,
    MetricNotAdditiveError,
    PeriodError,
)


# ---------------------------------------------------------------------------
# 时间窗口
# ---------------------------------------------------------------------------
def test_period_rejects_reversed_range():
    with pytest.raises(PeriodError):
        Period(date(2025, 6, 10), date(2025, 6, 1))


def test_overlapping_periods_are_rejected(engine):
    a = Period(date(2025, 6, 1), date(2025, 6, 10))
    b = Period(date(2025, 6, 8), date(2025, 6, 17))
    with pytest.raises(PeriodError, match="重叠"):
        engine.decompose_factors("gmv", a, b)


def test_unequal_periods_are_rejected_by_default(engine):
    """7 天对 10 天,变化里会混进「天数差」这个伪因子。"""
    a = Period(date(2025, 6, 1), date(2025, 6, 7))
    b = Period(date(2025, 6, 8), date(2025, 6, 17))
    with pytest.raises(PeriodError, match="长度不同"):
        engine.decompose_factors("gmv", a, b)


def test_unequal_periods_allowed_with_explicit_opt_in(engine):
    a = Period(date(2025, 6, 1), date(2025, 6, 7))
    b = Period(date(2025, 6, 8), date(2025, 6, 17))
    result = engine.decompose_factors("gmv", a, b, allow_unequal_periods=True)
    assert abs(result.residual_pct) < 1e-6


# ---------------------------------------------------------------------------
# 指标形态
# ---------------------------------------------------------------------------
def test_ratio_metric_cannot_be_decomposed_by_dimension(engine, last_two_weeks):
    """各渠道的 CVR 相加不等于总体 CVR —— 必须拒绝,不能默默相加。"""
    a, b = last_two_weeks
    with pytest.raises(MetricNotAdditiveError, match="比率"):
        engine.decompose_by_dimension("cvr", "channel", a, b)


def test_aov_cannot_be_decomposed_by_dimension(engine, last_two_weeks):
    a, b = last_two_weeks
    with pytest.raises(MetricNotAdditiveError):
        engine.decompose_by_dimension("aov", "category", a, b)


def test_metric_without_declared_factors_cannot_be_factor_decomposed(engine, last_two_weeks):
    a, b = last_two_weeks
    with pytest.raises(DecompositionError, match="未声明乘法分解"):
        engine.decompose_factors("uv", a, b)


# ---------------------------------------------------------------------------
# 维度与过滤
# ---------------------------------------------------------------------------
def test_dimension_outside_metric_whitelist_is_rejected(engine, last_two_weeks):
    """LLM(V2)只能在白名单里选维度,不能凭空造一个。"""
    a, b = last_two_weeks
    with pytest.raises(DimensionNotFoundError, match="campaign"):
        engine.decompose_by_dimension("gmv", "campaign", a, b)


def test_unknown_dimension_is_rejected(engine, last_two_weeks):
    a, b = last_two_weeks
    with pytest.raises(DimensionNotFoundError, match="weather"):
        engine.decompose_by_dimension("gmv", "weather", a, b)


def test_factor_filter_must_be_available_on_every_involved_table(engine, last_two_weeks):
    """按 category 过滤去做 UV×CVR×AOV 分解是无解的。

    category 只存在于 dim_product(经 fact_orders 关联),
    fact_sessions 上根本没有这个概念 —— UV 无法按同口径过滤,
    恒等式会破裂。宁可报错,也不能返回一个「差不多」的结果。
    """
    a, b = last_two_weeks
    with pytest.raises(FilterNotSupportedError, match="category"):
        engine.decompose_factors("gmv", a, b, {"category": "Apparel"})


def test_factor_filter_via_join_is_allowed_when_aligned(engine, last_two_weeks):
    """user_tier 在两张事实表上都能经 dim_user 取到,因此口径是对齐的。"""
    a, b = last_two_weeks
    result = engine.decompose_factors("gmv", a, b, {"user_tier": "VIP"})
    assert abs(result.residual_pct) < 1e-6
    assert result.identity_gap_pct < 1e-6


def test_filter_on_unknown_dimension_is_rejected(engine, last_two_weeks):
    a, b = last_two_weeks
    with pytest.raises(DimensionNotFoundError):
        engine.decompose_by_dimension("gmv", "channel", a, b, {"weather": "rain"})


def test_periods_with_no_data_are_reported_as_a_data_problem(engine):
    """窗口内一行数据都没有时,必须明确说这是数据问题,而不是算出一堆 0。

    真实工作中相当比例的「GMV 暴跌」真相是某个分区没写进来(简报 §9 的 L0)。
    V0 还没有观察层 L0,但分解引擎至少不能把「没有数据」伪装成「指标为 0」。
    """
    a = Period(date(2019, 1, 7), date(2019, 1, 13))
    b = Period(date(2019, 1, 14), date(2019, 1, 20))
    with pytest.raises(DecompositionError, match="没有任何数据"):
        engine.decompose_factors("gmv", a, b)


# ---------------------------------------------------------------------------
# 残差判定(观察层 L2,确定性)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("residual_pct", "expected"),
    [
        (0.0, "clean"),
        (0.9, "clean"),
        (-0.9, "clean"),
        (1.0, "structural"),
        (12.0, "structural"),
        (-15.0, "structural"),
        (20.0, "structural"),
        (20.1, "data_quality"),
        (-45.0, "data_quality"),
    ],
)
def test_residual_severity_matches_the_brief_thresholds(residual_pct, expected):
    """简报 §3.2 的三档阈值,完全确定性,不经 LLM。"""
    assert residual_severity(residual_pct) == expected


# ---------------------------------------------------------------------------
# SQL 注入
# ---------------------------------------------------------------------------
def test_filter_values_are_escaped(engine, last_two_weeks):
    """过滤值将来会来自 LLM 的结构化输出,必须转义。"""
    a, b = last_two_weeks
    result = engine.decompose_by_dimension(
        "gmv", "channel", a, b, {"device": "mobile'; DROP TABLE fact_orders; --"}
    )
    assert result.value_a == 0.0
    # 表还在
    table = engine.dialect.qualify("fact_orders")
    assert engine.executor.run(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] > 0
