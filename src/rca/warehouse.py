"""查询执行层与「可复现」的定义。

简报约束 3:**报告中每一个数字都必须可复现** —— 能回溯到一条确定性生成的
SQL 并重跑得到相同结果。本模块把这条约束落成三件具体的东西:

1. :class:`SqlExecutor` —— 唯一的执行入口。分解引擎只生成 SQL 文本,
   自己不碰连接,因此同一段 SQL 可以在 Databricks 上跑,也可以在本地 DuckDB 上跑;
2. :func:`sql_fingerprint` —— SQL 的内容指纹,写进
   ``decomposition_steps.sql_fingerprint``(V2 状态表),让任何一个数字
   都能被原样重放;
3. :class:`ExecutedQuery` —— 每次执行留下的凭证(SQL 原文 + 指纹 + 返回行数),
   报告里的每条证据都指向它。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, Sequence, runtime_checkable

from .errors import QueryExecutionError

_WHITESPACE = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    """折叠空白后的 SQL,用于计算指纹。

    刻意**不做**大小写归一:字符串字面量(如 ``'Paid Search'``)大小写敏感,
    归一化会让两条语义不同的 SQL 撞上同一个指纹。
    """
    return _WHITESPACE.sub(" ", sql).strip()


def sql_fingerprint(sql: str) -> str:
    """SQL 的内容指纹(sha256 前 16 位十六进制)。"""
    return hashlib.sha256(normalize_sql(sql).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ExecutedQuery:
    """一次查询的可审计凭证。"""

    purpose: str
    """这条查询在分解中扮演的角色,例如 ``"factor_base:fact_orders"``。"""

    sql: str
    fingerprint: str
    row_count: int
    dialect: str = "unknown"

    @classmethod
    def of(cls, purpose: str, sql: str, rows: Sequence[Any], dialect: str) -> "ExecutedQuery":
        return cls(
            purpose=purpose,
            sql=sql,
            fingerprint=sql_fingerprint(sql),
            row_count=len(rows),
            dialect=dialect,
        )


@runtime_checkable
class SqlExecutor(Protocol):
    """执行一条 SQL 并返回 ``list[dict]``。"""

    dialect_name: str

    def run(self, sql: str) -> list[dict[str, Any]]:
        ...


@dataclass
class DuckDBExecutor:
    """本地 DuckDB 执行器(开发与测试用)。"""

    connection: Any
    dialect_name: str = "duckdb"
    executed: list[str] = field(default_factory=list)

    def run(self, sql: str) -> list[dict[str, Any]]:
        self.executed.append(sql)
        try:
            cursor = self.connection.execute(sql)
        except Exception as exc:  # noqa: BLE001 - 统一包装成观察层 L1 失败
            raise QueryExecutionError(f"DuckDB 执行失败: {exc}\n--- SQL ---\n{sql}") from exc
        if cursor.description is None:
            return []
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


@dataclass
class DatabricksExecutor:
    """从本机/CI 连 Databricks SQL Warehouse。

    依赖 ``databricks-sql-connector``(可选依赖 ``[databricks]``)。

    **连接是复用的。** Free Edition 只有一个 2X-Small warehouse,
    每条语句重建一次连接会把大量时间耗在握手与 warehouse 唤醒上;
    而且一次分解本来就只有 2 条查询,重连的开销会盖过查询本身。
    用完调 :meth:`close`,或者用 ``with`` 语句。
    """

    server_hostname: str
    http_path: str
    access_token: str = field(repr=False)
    dialect_name: str = "databricks"
    executed: list[str] = field(default_factory=list)
    _connection: Any = field(default=None, repr=False)

    def _connect(self) -> Any:
        if self._connection is not None:
            return self._connection
        try:
            from databricks import sql as dbsql  # 延迟导入:本地测试不需要它
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise QueryExecutionError(
                "缺少 databricks-sql-connector。安装:pip install '.[databricks]'"
            ) from exc
        try:
            self._connection = dbsql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token,
            )
        except Exception as exc:  # noqa: BLE001
            raise QueryExecutionError(
                f"无法连接 Databricks SQL Warehouse ({self.server_hostname}{self.http_path}): {exc}"
            ) from exc
        return self._connection

    def run(self, sql: str) -> list[dict[str, Any]]:
        self.executed.append(sql)
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                if cursor.description is None:
                    return []
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as exc:  # noqa: BLE001
            raise QueryExecutionError(
                f"Databricks 执行失败: {exc}\n--- SQL ---\n{sql}"
            ) from exc

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def __enter__(self) -> "DatabricksExecutor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class SparkExecutor:
    """代码跑在 Databricks Notebook / Job 内部时的执行器。

    与 :class:`DatabricksExecutor` 执行的是**完全相同的 SQL** ——
    区别只是走 SparkSession 而不是 Thrift 连接,因此不需要 token,
    也不受出站网络限制(简报约束 2)。
    """

    spark: Any
    dialect_name: str = "databricks"
    executed: list[str] = field(default_factory=list)

    def run(self, sql: str) -> list[dict[str, Any]]:
        self.executed.append(sql)
        try:
            frame = self.spark.sql(sql)
            columns = list(frame.columns)
            if not columns:
                return []
            return [dict(zip(columns, row)) for row in frame.collect()]
        except Exception as exc:  # noqa: BLE001
            raise QueryExecutionError(
                f"Spark 执行失败: {exc}\n--- SQL ---\n{sql}"
            ) from exc


def to_float(value: Any) -> float:
    """把仓库返回的数值统一成 float。

    金额列用 ``DECIMAL(12,2)`` 存储(避免二进制浮点在求和时引入抖动),
    两种驱动都会把它还原成 :class:`decimal.Decimal`;对数分解需要 float。
    ``NULL``(某个分组在某个窗口内没有任何数据)按 0 处理 —— 这是可加指标的
    正确语义,并且让「新增/消失的分组」自然体现为一条完整的贡献项。
    """
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)
