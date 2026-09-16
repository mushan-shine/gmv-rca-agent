"""用手算得出的数字验证分解算术,**不经过数据库**。

DuckDB 上的闭合性测试证明「各分项之和 = 总变化」,但它无法证明
*每一项各自的数值* 是对的 —— 一个整体错位的分解也能完美闭合。
这里用一组可以手算的数字把每个字段钉死。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pytest

from rca.decompose import DecompositionEngine, Period
from rca.dialects import DuckDBDialect
from rca.errors import NonPositiveValueError


class StubExecutor:
    """返回预设聚合结果的执行器。

    键为表名,值为 ``{'A': {...}, 'B': {...}}``。
    """

    dialect_name = "duckdb"

    def __init__(self, data: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.data = data
        self.executed: list[str] = []

    def run(self, sql: str) -> list[dict[str, Any]]:
        self.executed.append(sql)
        table = next(name for name in self.data if f"main.{name} f" in sql)
        return [
            {"period": period, **values} for period, values in self.data[table].items()
        ]


A = Period(date(2025, 1, 1), date(2025, 1, 7), "基期")
B = Period(date(2025, 1, 8), date(2025, 1, 14), "当期")


@pytest.fixture()
def stub_engine(knowledge):
    """一组刻意挑选的数字:

        UV     1000 -> 1100   (+10%)
        orders   50 ->   44   (-12%)
        GMV    5000 -> 4400   (-12%)

    于是 CVR = 0.050 -> 0.040 (-20%),AOV = 100.0 -> 100.0 (不变)。
    GMV 整体 -12%,AOV 的贡献必须恰好是 0。
    """
    executor = StubExecutor(
        {
            "fact_orders": {"A": {"gmv": 5000.0, "orders": 50}, "B": {"gmv": 4400.0, "orders": 44}},
            "fact_sessions": {"A": {"uv": 1000}, "B": {"uv": 1100}},
        }
    )
    return DecompositionEngine(knowledge, executor, DuckDBDialect())


def test_factor_values_are_derived_exactly_as_the_yaml_declares(stub_engine):
    result = stub_engine.decompose_factors("gmv", A, B)
    by_name = {f.factor: f for f in result.factors}

    assert by_name["uv"].value_a == pytest.approx(1000.0)
    assert by_name["uv"].value_b == pytest.approx(1100.0)
    assert by_name["cvr"].value_a == pytest.approx(0.05)      # 50 / 1000
    assert by_name["cvr"].value_b == pytest.approx(0.04)      # 44 / 1100
    assert by_name["aov"].value_a == pytest.approx(100.0)     # 5000 / 50
    assert by_name["aov"].value_b == pytest.approx(100.0)     # 4400 / 44

    assert result.value_a == pytest.approx(5000.0)
    assert result.value_b == pytest.approx(4400.0)
    assert result.pct_change == pytest.approx(-0.12)


def test_unchanged_factor_contributes_exactly_zero(stub_engine):
    """AOV 两期都是 100,它的贡献必须是 0,而不是「很小的数」。"""
    result = stub_engine.decompose_factors("gmv", A, B)
    aov = next(f for f in result.factors if f.factor == "aov")

    assert aov.log_delta == pytest.approx(0.0, abs=1e-15)
    assert aov.share_of_change == pytest.approx(0.0, abs=1e-15)
    assert aov.contribution_pct_points == pytest.approx(0.0, abs=1e-15)
    assert aov.isolated_pct_change == pytest.approx(0.0, abs=1e-15)


def test_log_deltas_match_hand_computation(stub_engine):
    result = stub_engine.decompose_factors("gmv", A, B)
    by_name = {f.factor: f for f in result.factors}

    assert by_name["uv"].log_delta == pytest.approx(math.log(1.1))
    assert by_name["cvr"].log_delta == pytest.approx(math.log(0.8))
    assert result.log_delta == pytest.approx(math.log(0.88))
    # ln(1.1) + ln(0.8) = ln(0.88)
    assert math.log(1.1) + math.log(0.8) == pytest.approx(math.log(0.88))


def test_shares_match_hand_computation(stub_engine):
    result = stub_engine.decompose_factors("gmv", A, B)
    by_name = {f.factor: f for f in result.factors}

    expected_uv_share = math.log(1.1) / math.log(0.88)
    expected_cvr_share = math.log(0.8) / math.log(0.88)
    assert by_name["uv"].share_of_change == pytest.approx(expected_uv_share)
    assert by_name["cvr"].share_of_change == pytest.approx(expected_cvr_share)
    # UV 上涨、GMV 下跌 => UV 的份额为负(它在抵消跌幅)
    assert by_name["uv"].share_of_change < 0
    assert by_name["cvr"].share_of_change > 1


def test_isolated_change_differs_from_allocated_points(stub_engine):
    """简报 §13.3 要求说明的换算误差,在这里被量化钉住。

    CVR 自身下跌 20.00%(``isolated_pct_change``);
    而按对数份额分配到 GMV 跌幅上的是 −20.95 个百分点
    (``contribution_pct_points`` = ln0.8/ln0.88 × −12%)。

    两个数都对,含义不同:前者回答「CVR 掉了多少」,
    后者回答「GMV 这 −12% 里有多少记在 CVR 头上」。
    此例中二者相差 0.95 个百分点(相对约 4.7%),报告里混用就会得出错误结论。
    """
    result = stub_engine.decompose_factors("gmv", A, B)
    cvr = next(f for f in result.factors if f.factor == "cvr")

    assert cvr.isolated_pct_change == pytest.approx(-0.20)
    assert cvr.contribution_pct_points == pytest.approx(
        math.log(0.8) / math.log(0.88) * -0.12
    )
    assert cvr.contribution_pct_points == pytest.approx(-0.2095, abs=1e-4)

    gap = abs(cvr.isolated_pct_change - cvr.contribution_pct_points)
    assert gap == pytest.approx(0.0095, abs=1e-3)
    assert gap / abs(cvr.isolated_pct_change) > 0.04


def test_zero_change_leaves_shares_undefined(knowledge):
    """无异常时不得凭空造出「贡献」—— 这是 V1 无异常对照组的底座。"""
    same = {"gmv": 5000.0, "orders": 50}
    executor = StubExecutor(
        {
            "fact_orders": {"A": dict(same), "B": dict(same)},
            "fact_sessions": {"A": {"uv": 1000}, "B": {"uv": 1000}},
        }
    )
    engine = DecompositionEngine(knowledge, executor, DuckDBDialect())
    result = engine.decompose_factors("gmv", A, B)

    assert result.pct_change == pytest.approx(0.0)
    assert result.log_delta == pytest.approx(0.0)
    assert all(f.share_of_change is None for f in result.factors)
    assert all(f.contribution_pct_points is None for f in result.factors)
    assert result.residual_pct == 0.0


def test_non_positive_value_is_rejected_not_silently_handled(knowledge):
    """某个切片当期为 0(常见于数据缺失)时必须报错,不能返回 -inf。"""
    executor = StubExecutor(
        {
            "fact_orders": {"A": {"gmv": 5000.0, "orders": 50}, "B": {"gmv": 0.0, "orders": 0}},
            "fact_sessions": {"A": {"uv": 1000}, "B": {"uv": 1100}},
        }
    )
    engine = DecompositionEngine(knowledge, executor, DuckDBDialect())
    with pytest.raises(NonPositiveValueError):
        engine.decompose_factors("gmv", A, B)
