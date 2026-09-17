"""命令行入口:同一套代码,三个运行目标。

    python -m rca.cli setup  --target databricks     # 建表 + 灌数 + 自检
    python -m rca.cli verify --target databricks     # 只跑自检
    python -m rca.cli demo   --target databricks     # 跑一次完整分解
    python -m rca.cli sql    --target databricks     # 打印/导出 SQL,不连库

``--target duckdb`` 把同样的命令跑在本地内存库上,用于快速自测。

凭据来自环境变量(见 :mod:`rca.config`),不接受命令行传 token ——
命令行参数会进入 shell 历史和进程列表。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from .config import TARGET_NAMES, Target, build_target  # noqa: E402
from .decompose import DecompositionEngine, Period, residual_severity  # noqa: E402
from .errors import RcaError  # noqa: E402
from .knowledge import load_knowledge  # noqa: E402
from .sampledata import (  # noqa: E402
    SampleDataConfig,
    build_setup_statements,
    build_verification_sql,
    run_verification,
)


def _rule(title: str) -> None:
    print(f"\n{title}\n" + "─" * 78)


def _knowledge(args: argparse.Namespace):
    return load_knowledge(Path(args.knowledge))


def _config(args: argparse.Namespace) -> SampleDataConfig:
    return SampleDataConfig(daily_sessions=args.daily_sessions)


def _target(args: argparse.Namespace) -> Target:
    return build_target(
        args.target,
        catalog=args.catalog,
        schema=args.schema,
        duckdb_path=getattr(args, "duckdb_path", ":memory:"),
    )


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def cmd_setup(args: argparse.Namespace) -> int:
    knowledge = _knowledge(args)
    config = _config(args)
    target = _target(args)
    statements = build_setup_statements(knowledge, target.dialect, config)

    _rule(f"建库灌数 → {target.description}")
    print(
        f"   规模:{config.weeks} 周 × 日均 {config.daily_sessions:,} 会话 "
        f"≈ {config.daily_sessions * config.n_days:,} 行\n"
    )
    try:
        for index, statement in enumerate(statements, 1):
            started = time.time()
            target.executor.run(statement.sql)
            print(
                f"   [{index:>2}/{len(statements)}] {statement.label:<26}"
                f"{time.time() - started:>7.2f}s"
            )
        return _verify(target)
    finally:
        target.close()


def _verify(target: Target) -> int:
    _rule("自检")
    checks = run_verification(target.executor, target.dialect)
    for name, value in checks.items():
        print(f"   {name:<20} {value}")

    converted = checks.get("sessions_converted")
    completed = checks.get("completed_orders")
    if converted != completed:
        print(
            f"\n   ✗ 自检失败:sessions_converted({converted}) "
            f"≠ completed_orders({completed})"
            f"\n     两处转化判定已漂移,UV × CVR × AOV = GMV 在数据层不再成立。"
        )
        return 1
    print(f"\n   ✓ sessions_converted == completed_orders == {completed}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    target = _target(args)
    try:
        return _verify(target)
    finally:
        target.close()


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def cmd_demo(args: argparse.Namespace) -> int:
    knowledge = _knowledge(args)
    config = _config(args)
    target = _target(args)
    try:
        if args.target == "duckdb" and args.duckdb_path == ":memory:":
            # 内存库里什么都没有,先建
            for statement in build_setup_statements(knowledge, target.dialect, config):
                target.executor.run(statement.sql)

        engine = DecompositionEngine(knowledge, target.executor, target.dialect)
        current = Period(config.end_date - timedelta(days=6), config.end_date, "本周")
        baseline = Period(
            current.start - timedelta(days=7), current.start - timedelta(days=1), "上周"
        )

        _rule(f"乘法分解  GMV = UV × CVR × AOV   ({baseline} vs {current})")
        print(f"   目标:{target.description}")
        factor = engine.decompose_factors("gmv", baseline, current)
        print(
            f"   GMV {factor.value_a:>14,.2f} → {factor.value_b:>14,.2f}"
            f"   {factor.pct_change:+.2%}\n"
        )
        header = ("因子", "基期", "当期", "自身变化", "占变化份额", "分配百分点")
        print(f"   {header[0]:<8}{header[1]:>14}{header[2]:>14}{header[3]:>12}{header[4]:>12}{header[5]:>12}")
        for contribution in factor.factors:
            share = _fmt_pct(contribution.share_of_change, 1)
            points = _fmt_pct(contribution.contribution_pct_points, 2)
            print(
                f"   {contribution.factor:<10}{contribution.value_a:>14,.4f}"
                f"{contribution.value_b:>14,.4f}{contribution.pct_change:>12.2%}"
                f"{share:>12}{points:>12}"
            )
        print(
            f"\n   残差 {factor.residual_pct:.3e}%    "
            f"恒等式偏差 {factor.identity_gap_pct:.3e}%    "
            f"闭合 {'是' if factor.is_closed else '否'}"
        )

        _rule("加法分解  按 dimensions.yaml 的 priority 逐个维度")
        print("   (顺序是写死的默认优先级,不是「根据上一轮结果决定下一轮」—— 那是 V2)")
        for dimension in knowledge.ordered_dimensions("gmv"):
            result = engine.decompose_by_dimension("gmv", dimension, baseline, current)
            print(
                f"\n   [{dimension}]  ΔGMV = {result.delta:+,.2f}   "
                f"残差 {result.residual_pct:.2e}%   "
                f"判定 {residual_severity(result.residual_pct)}   "
                f"未知桶 {result.unknown_share_pct:.2f}%"
            )
            for part in result.parts[: args.top]:
                print(
                    f"      {part.value:<16}{part.value_a:>12,.0f} → {part.value_b:>12,.0f}"
                    f"   Δ={part.delta:>+12,.0f}   占变化 {_fmt_pct(part.share_of_change, 1):>8}"
                )

        _rule("可复现性:每个数字都对应一条可重跑的 SQL")
        for record in factor.queries:
            print(f"   {record.fingerprint}  {record.purpose}  ({record.row_count} 行)")
        if args.show_sql:
            for record in factor.queries:
                print(f"\n--- {record.purpose} [{record.fingerprint}] ---\n{record.sql}")
        return 0
    finally:
        target.close()


def _fmt_pct(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:+.{digits}%}"


# ---------------------------------------------------------------------------
# sql
# ---------------------------------------------------------------------------
def cmd_sql(args: argparse.Namespace) -> int:
    """只渲染 SQL,不连库 —— 无凭据时也能用。"""
    from .dialects import DIALECTS

    knowledge = _knowledge(args)
    config = _config(args)
    dialect_name = "databricks" if args.target == "spark" else args.target
    dialect_cls = DIALECTS[dialect_name]
    if dialect_name == "databricks":
        dialect = dialect_cls(catalog=args.catalog or "workspace", schema=args.schema or "gmv_rca")
    else:
        dialect = dialect_cls(schema=args.schema or "main")

    chunks = [
        f"-- [{statement.label}]\n{statement.sql};"
        for statement in build_setup_statements(knowledge, dialect, config)
    ]
    chunks.append(f"-- [verify]\n{build_verification_sql(dialect)};")
    text = "\n\n".join(chunks) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写出 {args.out}({len(text):,} 字符)")
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rca", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--target",
            default="databricks",
            choices=TARGET_NAMES,
            help="运行目标(默认 databricks)",
        )
        p.add_argument("--catalog", default=None, help="Unity Catalog 目录名")
        p.add_argument("--schema", default=None, help="Schema 名")
        p.add_argument(
            "--knowledge", default=str(REPO_ROOT / "knowledge"), help="知识库目录"
        )
        p.add_argument(
            "--daily-sessions",
            type=int,
            default=SampleDataConfig().daily_sessions,
            help="样例数据的日均会话数",
        )
        p.add_argument(
            "--duckdb-path",
            default=":memory:",
            help="target=duckdb 时的库文件路径(默认内存库)",
        )

    p_setup = sub.add_parser("setup", help="建表 + 灌数 + 自检")
    common(p_setup)
    p_setup.set_defaults(func=cmd_setup)

    p_verify = sub.add_parser("verify", help="只跑自检")
    common(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_demo = sub.add_parser("demo", help="跑一次完整分解")
    common(p_demo)
    p_demo.add_argument("--top", type=int, default=3, help="每个维度打印前 N 个分项")
    p_demo.add_argument("--show-sql", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_sql = sub.add_parser("sql", help="渲染建库 SQL(不连库)")
    common(p_sql)
    p_sql.add_argument("--out", default=None, help="输出文件;省略则打印到标准输出")
    p_sql.set_defaults(func=cmd_sql)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except RcaError as exc:
        # 配置缺失、SQL 执行失败、分解请求不合法 —— 这些都是「用户看得懂的错」,
        # 打成干净的一段话,不要甩一页 traceback。
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
