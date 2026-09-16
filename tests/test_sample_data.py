"""样例数据的不变量(V0 验收标准 1)。

样例数据不是「随便造点数」—— 它是后续所有验证的地基:
V1 的合成异常注入需要一个干净的对照底座,
V0 的闭合性测试需要每个切片都有数据。这里把这些前提逐条钉死。

除标注 ``duckdb_only`` 的两条之外,本文件的断言在**任何运行目标**上都成立 ——
用 ``--rca-target=spark`` 就能在 Databricks 上复核同一批不变量。
两个方言的 ``hash()`` 实现不同,数据并不相同,但这些不变量与具体数值无关。
"""

from __future__ import annotations

import pytest

from rca.sampledata import (
    CATEGORIES,
    CHANNELS,
    DEVICES,
    MARKETS,
    SampleDataConfig,
    run_verification,
)

from conftest import BUSINESS_TABLES, TEST_CONFIG


# ---------------------------------------------------------------------------
# 覆盖度
# ---------------------------------------------------------------------------
def test_eight_weeks_of_daily_data(scalar):
    assert str(scalar("SELECT MIN(dt) FROM {fact_sessions}")).startswith(
        TEST_CONFIG.start_date.isoformat()
    )
    assert str(scalar("SELECT MAX(dt) FROM {fact_sessions}")).startswith(
        TEST_CONFIG.end_date.isoformat()
    )
    assert scalar("SELECT COUNT(DISTINCT dt) FROM {fact_sessions}") == TEST_CONFIG.n_days == 56


def test_all_five_business_tables_are_populated(scalar):
    for table in BUSINESS_TABLES:
        assert scalar("SELECT COUNT(*) FROM {%s}" % table) > 0, table


def test_dimension_cardinalities_match_the_brief(scalar):
    assert scalar("SELECT COUNT(DISTINCT channel) FROM {fact_sessions}") == len(CHANNELS) == 5
    assert scalar("SELECT COUNT(DISTINCT device) FROM {fact_sessions}") == len(DEVICES) == 2
    assert scalar("SELECT COUNT(DISTINCT market) FROM {fact_sessions}") == len(MARKETS) == 3
    categories = scalar("SELECT COUNT(DISTINCT category) FROM {dim_product}")
    assert 5 <= categories <= 8 and categories == len(CATEGORIES)


def test_every_channel_device_market_cell_has_data_every_day(scalar):
    """每个切片每天都有数据 —— 否则下钻到第三层就会遇到空集。"""
    cells = scalar(
        "SELECT COUNT(*) FROM ("
        " SELECT dt, channel, device, market FROM {fact_sessions}"
        " GROUP BY dt, channel, device, market) g"
    )
    assert cells == TEST_CONFIG.n_days * len(CHANNELS) * len(DEVICES) * len(MARKETS)


def test_sample_size_stays_within_free_edition_budget():
    """简报 §13.1:样例数据规模控制在百万行以内。"""
    default = SampleDataConfig()
    assert default.daily_sessions * default.n_days < 1_000_000


# ---------------------------------------------------------------------------
# 一致性
# ---------------------------------------------------------------------------
def test_converted_sessions_equal_completed_orders(executor, dialect):
    """CVR 的两侧口径必须严丝合缝。

    ``fact_sessions.converted`` 与 ``fact_orders`` 是两套独立生成的判定,
    它们相等才说明「一个转化会话恰好一个订单」这条前提成立 ——
    这是 UV × CVR × AOV = GMV 在**数据层**无歧义的根据。
    """
    checks = run_verification(executor, dialect)
    assert checks["sessions_converted"] == checks["completed_orders"]
    assert int(checks["completed_orders"]) > 0
    assert checks["distinct_days"] == str(TEST_CONFIG.n_days)
    assert int(checks["products"]) == TEST_CONFIG.n_products
    assert int(checks["users"]) == TEST_CONFIG.n_users


def test_no_orphan_foreign_keys(scalar):
    """join 必须不丢行 —— 否则维度分解的残差会被数据缺陷污染。"""
    assert (
        scalar(
            "SELECT COUNT(*) FROM {fact_orders} o"
            " LEFT JOIN {dim_product} p ON o.product_id = p.product_id"
            " WHERE p.product_id IS NULL"
        )
        == 0
    )
    assert (
        scalar(
            "SELECT COUNT(*) FROM {fact_sessions} s"
            " LEFT JOIN {dim_user} u ON s.user_id = u.user_id"
            " WHERE u.user_id IS NULL"
        )
        == 0
    )


def test_session_market_matches_user_market(scalar):
    assert (
        scalar(
            "SELECT COUNT(*) FROM {fact_sessions} s"
            " JOIN {dim_user} u ON s.user_id = u.user_id"
            " WHERE s.market <> u.market"
        )
        == 0
    )


def test_every_order_traces_back_to_a_session(scalar):
    assert (
        scalar(
            "SELECT COUNT(*) FROM {fact_orders} o"
            " LEFT JOIN {fact_sessions} s ON o.session_id = s.session_id"
            " WHERE s.session_id IS NULL"
        )
        == 0
    )


def test_order_ids_are_unique(scalar):
    total = scalar("SELECT COUNT(*) FROM {fact_orders}")
    distinct = scalar("SELECT COUNT(DISTINCT order_id) FROM {fact_orders}")
    assert total == distinct


def test_region_is_a_strict_parent_of_market(scalar):
    """region ⊃ market:每个 market 只能落在一个 region 下。"""
    violations = scalar(
        "SELECT COUNT(*) FROM ("
        " SELECT market FROM {fact_sessions}"
        " GROUP BY market HAVING COUNT(DISTINCT region) > 1) v"
    )
    assert violations == 0


# ---------------------------------------------------------------------------
# 真实感:数据要有波动,但不能有断崖
# ---------------------------------------------------------------------------
def test_daily_gmv_fluctuates_without_structural_breaks(executor, dialect):
    """有自然波动(否则 V1 注入的异常无从对比),但没有人为断崖。"""
    rows = executor.run(
        f"SELECT dt, SUM(order_amount) AS gmv FROM {dialect.qualify('fact_orders')}"
        f" WHERE order_status = 'COMPLETED' GROUP BY dt ORDER BY dt"
    )
    values = [float(row["gmv"]) for row in rows]
    assert len(values) == TEST_CONFIG.n_days

    average = sum(values) / len(values)
    spread = (max(values) - min(values)) / average
    assert 0.05 < spread < 1.5, f"日 GMV 波动幅度 {spread:.2%} 不真实"
    assert min(values) > 0


def test_order_status_distribution_is_realistic(scalar):
    completed = scalar(
        "SELECT COUNT(*) FROM {fact_orders} WHERE order_status = 'COMPLETED'"
    )
    total = scalar("SELECT COUNT(*) FROM {fact_orders}")
    assert 0.85 < completed / total < 0.97


# ---------------------------------------------------------------------------
# 确定性与幂等性(简报约束 3)
# ---------------------------------------------------------------------------
def test_rerunning_setup_reproduces_identical_data(executor, dialect, setup_statements):
    """整套建库脚本重跑一次,自检结果必须逐项相同。

    生成器不使用 ``rand()``,全部随机量派生自行的业务键 ——
    这是「报告里每个数字都能重跑复现」的最底层保证。
    在 Databricks 上跑这条,同时也验证了 ``CREATE OR REPLACE`` 的幂等性。
    """
    before = run_verification(executor, dialect)
    for statement in setup_statements:
        executor.run(statement.sql)
    after = run_verification(executor, dialect)
    assert before == after


@pytest.mark.duckdb_only
def test_generated_rows_are_byte_identical_across_runs(knowledge, dialect):
    """逐行校验确定性 —— 建两个独立的库,比对内容哈希。"""
    import duckdb

    from rca.sampledata import build_setup_statements

    config = SampleDataConfig(daily_sessions=300)
    checksums = []
    for _ in range(2):
        connection = duckdb.connect()
        for statement in build_setup_statements(knowledge, dialect, config):
            connection.execute(statement.sql)
        checksums.append(
            connection.execute(
                "SELECT SUM(hash(session_id || order_id || CAST(order_amount AS VARCHAR)"
                " || order_status)) FROM fact_orders"
            ).fetchone()[0]
        )
        connection.close()
    assert checksums[0] == checksums[1]


def test_start_date_must_be_a_monday():
    """周内节律靠「距起始日的天数 % 7」索引,起点不是周一就会整体错位。"""
    from datetime import date

    with pytest.raises(ValueError, match="周一"):
        SampleDataConfig(start_date=date(2025, 5, 6))
