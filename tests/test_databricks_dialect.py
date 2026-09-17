"""**目标平台是 Databricks**,所以针对 Databricks 生成的 SQL 也必须被验证。

本地跑不起 Databricks Runtime,但可以把生成的 SQL 交给 ``sqlglot`` 的
Databricks 方言解析器 —— 它能抓住语法层面的方言错误
(函数名写错、DuckDB 的 ``::`` 强转漏进来、``unnest`` 混进 Spark SQL 等)。

**这不能替代在真实 warehouse 上跑一次。** 语法过了不代表语义对
(比如 generator 函数的位置限制、Delta 的 DDL 约束)。
真实验证在 ``tests/test_databricks_live.py`` 与 ``notebooks/``。
这一层的定位是:让方言错误在**提交前**就暴露,而不是等到连上 workspace。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from rca.decompose import DecompositionEngine, Period
from rca.dialects import DatabricksDialect
from rca.sampledata import (
    SampleDataConfig,
    build_setup_statements,
    build_verification_sql,
)

sqlglot = pytest.importorskip("sqlglot", reason="需要 sqlglot 做静态方言校验")
from sqlglot.errors import ParseError  # noqa: E402

DATABRICKS = DatabricksDialect(catalog="workspace", schema="gmv_rca")


def _parse(sql: str, dialect: str, label: str) -> Any:
    try:
        return sqlglot.parse_one(sql, dialect=dialect)
    except ParseError as exc:
        pytest.fail(f"{label} 无法按 {dialect} 方言解析:\n{exc}\n--- SQL ---\n{sql}")


# ---------------------------------------------------------------------------
# 建库 SQL
# ---------------------------------------------------------------------------
def test_setup_statements_parse_as_databricks(knowledge):
    statements = build_setup_statements(knowledge, DATABRICKS, SampleDataConfig())
    assert statements, "应当生成建库语句"
    for statement in statements:
        _parse(statement.sql, "databricks", statement.label)


def test_verification_sql_parses_as_databricks():
    _parse(build_verification_sql(DATABRICKS), "databricks", "verify")


def test_databricks_sql_contains_no_duckdb_only_syntax(knowledge):
    """DuckDB 的写法混进 Databricks 产物,是最容易发生也最难发现的漂移。"""
    forbidden = ("unnest(", "generate_series(", "::DOUBLE", "::BIGINT", "datediff('day'")
    for statement in build_setup_statements(knowledge, DATABRICKS, SampleDataConfig()):
        lowered = statement.sql.lower()
        for token in forbidden:
            assert token.lower() not in lowered, f"{statement.label} 混入了 DuckDB 写法 {token}"


# ---------------------------------------------------------------------------
# 分析 SQL(decompose 生成的那几条)
# ---------------------------------------------------------------------------
class RecordingExecutor:
    """记录 SQL 并返回结构正确的假结果,用于在不连库的情况下拿到生成的 SQL。"""

    dialect_name = "databricks"

    def __init__(self) -> None:
        self.executed: list[str] = []

    def run(self, sql: str) -> list[dict[str, Any]]:
        self.executed.append(sql)
        payload = {"gmv": 1000.0, "orders": 10, "uv": 500, "dim_value": "Paid Search"}
        return [{"period": "A", **payload}, {"period": "B", **payload}]


@pytest.fixture()
def databricks_engine(knowledge):
    return DecompositionEngine(knowledge, RecordingExecutor(), DATABRICKS)


A = Period(date(2025, 6, 16), date(2025, 6, 22), "上周")
B = Period(date(2025, 6, 23), date(2025, 6, 29), "本周")


def test_factor_decomposition_sql_parses_as_databricks(databricks_engine):
    databricks_engine.decompose_factors("gmv", A, B)
    executed = databricks_engine.executor.executed
    assert len(executed) == 2, "GMV 因子分解应当只查两张事实表各一次"
    for index, sql in enumerate(executed):
        _parse(sql, "databricks", f"factor_base[{index}]")
        assert "workspace.gmv_rca." in sql, "表名必须完全限定到 catalog.schema"


@pytest.mark.parametrize(
    "dimension",
    ["channel", "device", "market", "region", "category", "sub_category", "price_tier", "user_tier"],
)
def test_dimension_decomposition_sql_parses_as_databricks(databricks_engine, dimension):
    databricks_engine.decompose_by_dimension("gmv", dimension, A, B)
    executed = databricks_engine.executor.executed
    assert len(executed) == 2, "维度分解 = 独立总量 + 分组求和"
    for index, sql in enumerate(executed):
        _parse(sql, "databricks", f"dimension[{dimension}][{index}]")


def test_filtered_decomposition_sql_parses_as_databricks(databricks_engine):
    databricks_engine.decompose_factors("gmv", A, B, {"channel": "Paid Search", "device": "mobile"})
    for index, sql in enumerate(databricks_engine.executor.executed):
        _parse(sql, "databricks", f"filtered[{index}]")


def test_join_based_dimension_emits_a_projected_subquery(databricks_engine):
    """维表关联必须包成只投影所需列的子查询。

    否则 ``dim_user.market`` 这类与事实表同名的列会让指标 YAML 里
    不带前缀的聚合表达式产生歧义 —— 在 Spark 上是直接报错的。
    """
    databricks_engine.decompose_by_dimension("gmv", "user_tier", A, B)
    parts_sql = databricks_engine.executor.executed[-1]
    assert "LEFT JOIN (SELECT" in parts_sql
    assert "workspace.gmv_rca.dim_user" in parts_sql
    assert "_jk" in parts_sql
    _parse(parts_sql, "databricks", "user_tier parts")
