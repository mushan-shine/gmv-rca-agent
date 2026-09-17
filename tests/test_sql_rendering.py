"""建库脚本的渲染与防漂移。

``sql/setup/*.sql`` 是产物,不是手写件。这组测试保证:

* 产物与生成器一致 —— 有人手改了 .sql 或改了生成器却忘了重新渲染,立刻失败;
* Databricks 版本确实用的是 Databricks 的写法(而不是不小心渲染成了 DuckDB);
* DDL 与 ``knowledge/tables.yaml`` 逐列一致。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rca.dialects import DatabricksDialect, DuckDBDialect
from rca.sampledata import render_create_table

SETUP_DIR = Path(__file__).resolve().parents[1] / "sql" / "setup"
RENDER_SCRIPT = SETUP_DIR / "render.py"


@pytest.mark.parametrize("dialect_name", ["databricks", "duckdb"])
def test_checked_in_sql_matches_the_generator(dialect_name, tmp_path):
    """产物没有漂移。失败时的修复动作:重新跑 ``python sql/setup/render.py``。"""
    committed = SETUP_DIR / f"{dialect_name}_setup.sql"
    assert committed.exists(), f"缺少产物 {committed.name},请运行 sql/setup/render.py"

    regenerated = tmp_path / "out.sql"
    subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), "--dialect", dialect_name, "--out", str(regenerated)],
        check=True,
        capture_output=True,
    )
    assert regenerated.read_text(encoding="utf-8") == committed.read_text(encoding="utf-8"), (
        f"{committed.name} 与生成器不一致 —— 请重新运行 "
        f"`python sql/setup/render.py --dialect {dialect_name}`"
    )


def test_databricks_script_uses_databricks_syntax():
    sql = (SETUP_DIR / "databricks_setup.sql").read_text(encoding="utf-8")
    assert "USING DELTA" in sql
    assert "explode(sequence(" in sql
    assert "pmod(hash(" in sql
    assert "workspace.gmv_rca.fact_orders" in sql
    # 绝不能混进 DuckDB 的写法
    assert "unnest(generate_series" not in sql
    assert "::DOUBLE" not in sql


def test_duckdb_script_uses_duckdb_syntax():
    sql = (SETUP_DIR / "duckdb_setup.sql").read_text(encoding="utf-8")
    assert "unnest(generate_series" in sql
    assert "USING DELTA" not in sql
    assert "explode(sequence(" not in sql


def test_generator_never_emits_nondeterministic_rand():
    """简报约束 3:``rand()`` 会让数据不可复现,生成器里不许出现。"""
    for name in ("databricks_setup.sql", "duckdb_setup.sql"):
        sql = (SETUP_DIR / name).read_text(encoding="utf-8")
        assert "rand()" not in sql.lower(), name
        assert "random()" not in sql.lower(), name
        assert "current_timestamp" not in sql.lower(), name


def test_every_business_table_is_created_and_populated():
    sql = (SETUP_DIR / "databricks_setup.sql").read_text(encoding="utf-8")
    for table in ("fact_orders", "fact_sessions", "fact_ad_spend", "dim_product", "dim_user"):
        assert f"-- [create:{table}]" in sql, table
        assert f"-- [insert:{table}]" in sql, table


@pytest.mark.parametrize("dialect", [DatabricksDialect(), DuckDBDialect()], ids=["databricks", "duckdb"])
def test_ddl_matches_tables_yaml(knowledge, dialect):
    """DDL 由 tables.yaml 渲染,因此不可能少列或列名拼错。"""
    for table in knowledge.tables.values():
        ddl = render_create_table(table, dialect)
        for column, definition in table.columns.items():
            assert f"  {column} {dialect.type_sql(definition.type)}" in ddl
            if not definition.nullable:
                assert f"{column} {dialect.type_sql(definition.type)} NOT NULL" in ddl


def test_create_or_replace_keeps_setup_idempotent():
    for name in ("databricks_setup.sql", "duckdb_setup.sql"):
        sql = (SETUP_DIR / name).read_text(encoding="utf-8")
        assert sql.count("CREATE OR REPLACE TABLE") == 5, name
        assert "CREATE TABLE IF NOT EXISTS" not in sql, name
