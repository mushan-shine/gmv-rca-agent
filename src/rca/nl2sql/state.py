"""Delta 状态表:**下一轮读上一轮写下的东西**。

原简报 §3.1 给了「循环」一个可证伪的判据:

> 一个循环之所以成立,在于「下一轮会读取上一轮写下的状态」。
> 只写不读的定时任务是复读机,不是循环。

本模块就是这条判据的落点。两张 append-only 的 Delta 表:

``nl2sql_attempts``
    每一次生成尝试:SQL、指纹、失败类型、守卫报错、引擎报错、耗时。
    改进循环从这里聚合失败模式 —— 「哪类问题总在第几轮才修好」。

``nl2sql_eval_runs``
    每一次评估跑批的指标快照,带 ``knowledge_version``。
    **回归闸门**读它:新知识版本的留出集指标不得低于上一版,否则不许合并。

刻意用 ``CREATE TABLE IF NOT EXISTS`` 而不是 ``CREATE OR REPLACE`` ——
这两张表的价值全在于**跨运行累积**,重建一次就等于把循环的记忆抹掉了。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..dialects import SqlDialect
from ..warehouse import SqlExecutor

ATTEMPTS_TABLE = "nl2sql_attempts"
EVAL_RUNS_TABLE = "nl2sql_eval_runs"

# 列名 -> 逻辑类型。顺序即建表顺序,也是 INSERT 的列顺序。
ATTEMPTS_SCHEMA: dict[str, str] = {
    "run_id": "string",
    "case_id": "string",
    "verdict": "string",
    "attempt_no": "int",
    "generator": "string",
    "model": "string",
    "sql": "string",
    "fingerprint": "string",
    "failure_kind": "string",
    "guard_errors": "string",
    "execution_error": "string",
    "generation_error": "string",
    "row_count": "int",
    "prompt_tokens": "int",
    "completion_tokens": "int",
    "cached": "boolean",
    "latency_ms": "int",
    "created_at": "timestamp",
}

EVAL_RUNS_SCHEMA: dict[str, str] = {
    "run_id": "string",
    "created_at": "timestamp",
    "generator": "string",
    "model": "string",
    "knowledge_version": "string",
    "n_cases": "int",
    "n_answerable": "int",
    "n_unanswerable": "int",
    "pass_at_1": "double",
    "pass_final": "double",
    "avg_attempts": "double",
    "hallucination_rate": "double",
    "correct_refusal_rate": "double",
    "false_refusal_rate": "double",
    "holdout_pass_final": "double",
    "llm_calls": "int",
    "cached_calls": "int",
    "total_tokens": "int",
    "notes": "string",
}


def _literal(value: Any, logical_type: str, dialect: SqlDialect) -> str:
    if value is None:
        return "NULL"
    if logical_type == "timestamp":
        stamp = (
            value
            if isinstance(value, str)
            else value.isoformat(sep=" ", timespec="microseconds")
        )
        return f"CAST({SqlDialect.string_literal(stamp)} AS {dialect.type_sql('timestamp')})"
    if logical_type == "string":
        return SqlDialect.string_literal(str(value))
    if logical_type == "double":
        return repr(float(value))
    if logical_type == "boolean":
        return "TRUE" if value else "FALSE"
    if logical_type in ("int", "bigint"):
        return str(int(value))
    return SqlDialect.string_literal(str(value))


def _create_table_sql(
    table: str, schema: Mapping[str, str], dialect: SqlDialect
) -> str:
    columns = ",\n  ".join(
        f"{name} {dialect.type_sql(logical)}" for name, logical in schema.items()
    )
    suffix = dialect.create_table_suffix()
    tail = f"\n{suffix}" if suffix else ""
    return (
        f"CREATE TABLE IF NOT EXISTS {dialect.qualify(table)} (\n  {columns}\n){tail}"
    )


def _insert_sql(
    table: str,
    schema: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    dialect: SqlDialect,
) -> str:
    columns = list(schema)
    values = []
    for row in rows:
        rendered = ", ".join(
            _literal(row.get(column), schema[column], dialect) for column in columns
        )
        values.append(f"({rendered})")
    return (
        f"INSERT INTO {dialect.qualify(table)} ({', '.join(columns)})\nVALUES\n  "
        + ",\n  ".join(values)
    )


@dataclass
class StateStore:
    """两张状态表的读写入口。"""

    executor: SqlExecutor
    dialect: SqlDialect

    # -- 建表 --------------------------------------------------------------
    def ensure_tables(self) -> None:
        """建表(已存在则什么都不做)。**绝不 DROP** —— 循环的记忆在这里。"""
        for table, schema in (
            (ATTEMPTS_TABLE, ATTEMPTS_SCHEMA),
            (EVAL_RUNS_TABLE, EVAL_RUNS_SCHEMA),
        ):
            self.executor.run(_create_table_sql(table, schema, self.dialect))

    # -- 写 ----------------------------------------------------------------
    def record_attempts(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """批量写入尝试记录。一条 INSERT 写完,不逐行插。"""
        if not rows:
            return 0
        now = datetime.now(timezone.utc)
        stamped = [{**row, "created_at": row.get("created_at") or now} for row in rows]
        self.executor.run(_insert_sql(ATTEMPTS_TABLE, ATTEMPTS_SCHEMA, stamped, self.dialect))
        return len(stamped)

    def record_eval_run(self, row: Mapping[str, Any]) -> None:
        stamped = {**row, "created_at": row.get("created_at") or datetime.now(timezone.utc)}
        self.executor.run(_insert_sql(EVAL_RUNS_TABLE, EVAL_RUNS_SCHEMA, [stamped], self.dialect))

    # -- 读(改进循环的输入)-----------------------------------------------
    def recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """最近几次评估,新的在前。回归闸门比的就是这个序列。"""
        return self.executor.run(
            f"SELECT * FROM {self.dialect.qualify(EVAL_RUNS_TABLE)}"
            f" ORDER BY created_at DESC, run_id DESC LIMIT {int(limit)}"
        )

    def previous_run(
        self, before_run_id: str, *, same_generator: bool = True
    ) -> dict[str, Any] | None:
        """紧挨在 ``before_run_id`` **之前**的那次评估。

        刻意不是「除它以外最新的一条」—— 那个语义在并发跑批或时间戳撞车时
        会拿到错误的基线。这里取出本次的时间戳,再找严格早于它的最后一条。

        ``same_generator=True``(默认)时只和**同一个生成器**比。真实踩过的坑:
        接上智谱之后的第一次评估,被拿去和之前 ``ReferenceGenerator``(永远正确)
        的 100% 比,闸门报「全量正确率退步 100% → 0%」—— 结论毫无意义。
        生成器名里带模型名(``llm:glm-4-flash``),所以换模型也不会互相比。
        """
        table = self.dialect.qualify(EVAL_RUNS_TABLE)
        target = SqlDialect.string_literal(before_run_id)
        generator_filter = " AND r.generator = a.generator" if same_generator else ""
        rows = self.executor.run(
            f"WITH anchor AS ("
            f" SELECT created_at, generator FROM {table} WHERE run_id = {target}"
            f" ORDER BY created_at DESC LIMIT 1)"
            f" SELECT r.* FROM {table} r CROSS JOIN anchor a"
            f" WHERE r.created_at < a.created_at AND r.run_id <> {target}{generator_filter}"
            f" ORDER BY r.created_at DESC, r.run_id DESC LIMIT 1"
        )
        return rows[0] if rows else None

    def failures_of_run(self, run_id: str) -> list[dict[str, Any]]:
        """某次评估里**答错**的尝试 —— 改进循环的原始素材。

        判据是 ``verdict`` 而不是 ``failure_kind``:一次正确拒答的
        failure_kind 是 ``refused``,但它并不是失败。
        """
        return self.executor.run(
            f"SELECT case_id, verdict, attempt_no, failure_kind, guard_errors,"
            f" execution_error, sql"
            f" FROM {self.dialect.qualify(ATTEMPTS_TABLE)}"
            f" WHERE run_id = {SqlDialect.string_literal(run_id)}"
            f"   AND verdict NOT IN ('correct', 'correct_refusal')"
            f" ORDER BY case_id, attempt_no"
        )

    def failure_profile(self, run_id: str) -> list[dict[str, Any]]:
        """按 (评估判定 × 循环失败类型) 聚合。决定「下一步该补哪类知识」。

        **刻意不过滤掉 ``failure_kind = 'none'``。** 一条语法正确、执行成功、
        但口径错了的 SQL,在循环看来毫无异常(failure_kind 就是 ``none``),
        而它恰恰是最危险、最需要补知识的失败模式 —— 把它过滤掉,
        改进循环就会对「静悄悄答错」这一类完全失明。
        """
        return self.executor.run(
            f"SELECT verdict, failure_kind, COUNT(*) AS n,"
            f" COUNT(DISTINCT case_id) AS cases"
            f" FROM {self.dialect.qualify(ATTEMPTS_TABLE)}"
            f" WHERE run_id = {SqlDialect.string_literal(run_id)}"
            f"   AND verdict NOT IN ('correct', 'correct_refusal')"
            f" GROUP BY verdict, failure_kind"
            f" ORDER BY n DESC, verdict"
        )

    def cases_needing_help(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """这一轮里**答错了**的 case(含误拒与该拒未拒)。

        改进循环下一步要做的事就是给它们补知识 ——
        补完之后这份名单应当变短,而留出集指标不许变差。
        """
        attempts = self.dialect.qualify(ATTEMPTS_TABLE)
        return self.executor.run(
            f"SELECT case_id, MIN(verdict) AS verdict, COUNT(*) AS attempts,"
            f" MAX(failure_kind) AS last_failure_kind"
            f" FROM {attempts}"
            f" WHERE run_id = {SqlDialect.string_literal(run_id)}"
            f"   AND verdict NOT IN ('correct', 'correct_refusal')"
            f" GROUP BY case_id"
            f" ORDER BY attempts DESC, case_id"
            f" LIMIT {int(limit)}"
        )
