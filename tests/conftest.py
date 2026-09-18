"""测试夹具:**同一套验收测试,可以跑在任意一个运行目标上**。

    pytest -q                              # 默认 duckdb,秒级,用于改代码时的快速回归
    pytest -q --rca-target=spark           # 在 Databricks Notebook / Job 里跑(见 notebooks/02_run_tests.py)
    pytest -q --rca-target=databricks      # 从本机/CI 连 SQL Warehouse 跑

这一点很重要:简报的 V0 验收标准是「有测试证明闭合性」,而目标平台是 Databricks。
如果验收测试只能在本地 DuckDB 上跑,那它证明的就只是本地的算术,
而不是**这套 SQL 在目标引擎上确实跑得通且算得对**。

远程目标会写入独立的 ``gmv_rca_test`` schema(``RCA_TEST_SCHEMA`` 可改),
不碰 ``rca.cli setup`` 建出来的正式 schema。
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from rca.config import TARGET_NAMES, build_target
from rca.decompose import DecompositionEngine, Period
from rca.knowledge import Knowledge, load_knowledge
from rca.nl2sql.cases import EvalCase, build_reference_sql, load_cases, validate_cases
from rca.nl2sql.generate import ReferenceGenerator
from rca.nl2sql.guard import SqlGuard
from rca.nl2sql.loop import AnswerLoop
from rca.nl2sql.state import (
    ATTEMPTS_TABLE,
    EVAL_RUNS_TABLE,
    FEEDBACK_TABLE,
    LEDGER_TABLE,
    REQUESTS_TABLE,
    StateStore,
)
from rca.sampledata import SampleDataConfig, build_setup_statements

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"
CASES_PATH = REPO_ROOT / "eval" / "cases.yaml"

TEST_CONFIG = SampleDataConfig(daily_sessions=1200)
"""测试规模:约 7 万行会话。

足以让每个 (日期 × 渠道 × 设备 × 市场) 切片都有数据,
又小到在本地是毫秒级、在 Free Edition 的 2X-Small 上是几十秒。
"""

TEST_SCHEMA = os.environ.get("RCA_TEST_SCHEMA", "gmv_rca_test")

BUSINESS_TABLES = (
    "fact_orders",
    "fact_sessions",
    "fact_ad_spend",
    "dim_product",
    "dim_user",
)


# ---------------------------------------------------------------------------
# 命令行选项
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--rca-target",
        default=os.environ.get("RCA_TARGET", "duckdb"),
        choices=list(TARGET_NAMES),
        help="跑测试的运行目标(默认 duckdb;远程目标会写入 %s)" % TEST_SCHEMA,
    )
    parser.addoption(
        "--rca-skip-setup",
        action="store_true",
        default=False,
        help="假设目标 schema 里已有样例数据,跳过建库(重复跑测试时省时间)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "duckdb_only: 只在本地 DuckDB 目标上有意义的测试"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--rca-target") == "duckdb":
        return
    skip = pytest.mark.skip(reason="只在 --rca-target=duckdb 下运行")
    for item in items:
        if "duckdb_only" in item.keywords:
            item.add_marker(skip)


def pytest_report_header(config: pytest.Config) -> str:
    return f"rca target: {config.getoption('--rca-target')}  (schema: {TEST_SCHEMA})"


# ---------------------------------------------------------------------------
# 知识库(不需要数据库)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def knowledge_dir() -> Path:
    return KNOWLEDGE_DIR


@pytest.fixture(scope="session")
def knowledge() -> Knowledge:
    return load_knowledge(KNOWLEDGE_DIR)


# ---------------------------------------------------------------------------
# 运行目标
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def target_name(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--rca-target"))


@pytest.fixture(scope="session")
def target(knowledge: Knowledge, target_name: str, pytestconfig: pytest.Config):
    """装配目标并灌好样例数据。整个测试会话只建一次库。"""
    schema = None if target_name == "duckdb" else TEST_SCHEMA
    built = build_target(target_name, schema=schema)
    try:
        if not pytestconfig.getoption("--rca-skip-setup"):
            for statement in build_setup_statements(knowledge, built.dialect, TEST_CONFIG):
                built.executor.run(statement.sql)
        if target_name != "duckdb":
            # 状态表在生产里是 append-only 的(循环的记忆),但测试 schema 是一次性的。
            # 不清掉的话,第二次跑测试时固定 run_id 的记录会重复,聚合断言全部翻倍。
            for table in (ATTEMPTS_TABLE, EVAL_RUNS_TABLE, LEDGER_TABLE, REQUESTS_TABLE, FEEDBACK_TABLE):
                built.executor.run(f"DROP TABLE IF EXISTS {built.dialect.qualify(table)}")
        yield built
    finally:
        built.close()


@pytest.fixture()
def dialect(target):
    return target.dialect


@pytest.fixture()
def executor(target):
    return target.executor


@pytest.fixture()
def engine(knowledge: Knowledge, target):
    return DecompositionEngine(knowledge, target.executor, target.dialect)


@pytest.fixture()
def setup_statements(knowledge: Knowledge, target):
    """本目标下的整套建库语句(用于幂等性测试)。"""
    return build_setup_statements(knowledge, target.dialect, TEST_CONFIG)


# ---------------------------------------------------------------------------
# 跨方言的取数助手
# ---------------------------------------------------------------------------
@pytest.fixture()
def scalar(target):
    """执行一条 SQL 并返回第一行第一列。

    SQL 里用 ``{fact_orders}`` 这样的占位符引用表,由方言补全成完全限定名 ——
    同一条断言因此可以跑在 ``main.fact_orders`` 和 ``main.gmv_rca_test.fact_orders`` 上。
    """
    names = {table: target.dialect.qualify(table) for table in BUSINESS_TABLES}

    def run(sql_template: str) -> Any:
        rows = target.executor.run(sql_template.format(**names))
        assert rows, f"查询没有返回任何行:\n{sql_template}"
        return next(iter(rows[0].values()))

    return run


# ---------------------------------------------------------------------------
# 时间窗口
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def weeks() -> list[Period]:
    """把样例数据切成 8 个等长的自然周(周一起)。"""
    start = TEST_CONFIG.start_date
    out = []
    for index in range(TEST_CONFIG.weeks):
        week_start = date.fromordinal(start.toordinal() + index * 7)
        week_end = date.fromordinal(week_start.toordinal() + 6)
        out.append(Period(week_start, week_end, f"W{index + 1}"))
    return out


@pytest.fixture()
def last_two_weeks(weeks: list[Period]) -> tuple[Period, Period]:
    """(基期, 当期) —— 最常用的环比窗口。"""
    return weeks[-2], weeks[-1]


# ---------------------------------------------------------------------------
# text-to-SQL 循环
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def cases(knowledge: Knowledge) -> list[EvalCase]:
    loaded = load_cases(CASES_PATH)
    validate_cases(loaded, knowledge)
    return loaded


@pytest.fixture()
def guard(knowledge: Knowledge, target) -> SqlGuard:
    return SqlGuard(knowledge, dialect=target.dialect.name)


@pytest.fixture()
def store(target) -> StateStore:
    built = StateStore(target.executor, target.dialect)
    built.ensure_tables()
    return built


@pytest.fixture()
def reference_generator(cases, knowledge: Knowledge, target) -> ReferenceGenerator:
    """永远答对的生成器 —— 用来校验评估器本身,而不是刷分。"""
    answers, refusals = {}, {}
    for case in cases:
        if case.gold.kind.value == "unanswerable":
            refusals[case.question] = case.gold.reason or ""
        else:
            answers[case.question] = build_reference_sql(
                case, knowledge, target.dialect
            )
    return ReferenceGenerator(answers, refusals)


@pytest.fixture()
def make_loop(knowledge: Knowledge, target, guard: SqlGuard):
    """按需造一个接了指定生成器的循环。"""

    def build(generator, **kwargs) -> AnswerLoop:
        return AnswerLoop(
            knowledge=knowledge,
            dialect=target.dialect,
            executor=target.executor,
            generator=generator,
            guard=guard,
            **kwargs,
        )

    return build
