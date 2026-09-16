"""把 ``rca.sampledata`` 的 SQL 生成器渲染成可直接粘贴执行的 .sql 文件。

    python sql/setup/render.py                 # 渲染 Databricks 版本(默认)
    python sql/setup/render.py --dialect duckdb

``sql/setup/*.sql`` 是**产物**,不要手工编辑 —— 改 ``knowledge/tables.yaml``
或 ``src/rca/sampledata.py`` 之后重新跑本脚本。
``tests/test_sql_setup_rendered.py`` 会校验产物与生成器一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rca.dialects import DIALECTS  # noqa: E402
from rca.knowledge import load_knowledge  # noqa: E402
from rca.sampledata import (  # noqa: E402
    SampleDataConfig,
    build_setup_statements,
    build_verification_sql,
)

HEADER = """-- ===========================================================================
-- 本文件由 sql/setup/render.py 自动生成,请勿手工编辑。
-- 生成器:src/rca/sampledata.py    Schema 真源:knowledge/tables.yaml
-- 方言:{dialect}    目标:{target}
--
-- 使用方式(Databricks):在 SQL Editor 中按顺序整段执行。
-- 全部语句均为幂等:CREATE OR REPLACE + 确定性 INSERT,重跑得到完全相同的数据。
-- 规模:{weeks} 周日粒度,日均 {daily} 会话,约 {rows} 行会话数据。
-- ===========================================================================
"""


def render(dialect_name: str, catalog: str | None, schema: str) -> str:
    knowledge = load_knowledge(REPO_ROOT / "knowledge")
    dialect_cls = DIALECTS[dialect_name]
    dialect = (
        dialect_cls(catalog=catalog, schema=schema)  # type: ignore[call-arg]
        if catalog is not None
        else dialect_cls(schema=schema)  # type: ignore[call-arg]
    )
    cfg = SampleDataConfig()
    parts = [
        HEADER.format(
            dialect=dialect_name,
            target=dialect.qualify("<table>"),
            weeks=cfg.weeks,
            daily=cfg.daily_sessions,
            rows=f"{cfg.daily_sessions * cfg.n_days:,}",
        )
    ]
    for stmt in build_setup_statements(knowledge, dialect, cfg):
        parts.append(f"\n-- [{stmt.label}]\n{stmt.sql};\n")
    parts.append("\n-- [verify] 自检:sessions_converted 必须等于 completed_orders\n")
    parts.append(f"{build_verification_sql(dialect)};\n")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialect", default="databricks", choices=sorted(DIALECTS))
    parser.add_argument("--catalog", default=None, help="Unity Catalog 目录名(默认 main)")
    parser.add_argument("--schema", default=None, help="Schema 名(默认 gmv_rca)")
    parser.add_argument("--out", default=None, help="输出文件(默认 sql/setup/<dialect>_setup.sql)")
    args = parser.parse_args()

    catalog = args.catalog
    schema = args.schema
    if args.dialect == "databricks":
        catalog = catalog or "main"
        schema = schema or "gmv_rca"
    else:
        schema = schema or "main"

    sql = render(args.dialect, catalog, schema)
    out = Path(args.out) if args.out else Path(__file__).parent / f"{args.dialect}_setup.sql"
    out.write_text(sql, encoding="utf-8")
    print(f"已写出 {out}  ({len(sql):,} 字符)")


if __name__ == "__main__":
    main()
