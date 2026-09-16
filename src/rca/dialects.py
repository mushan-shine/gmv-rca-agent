"""SQL 方言层。

项目只有**一套** SQL 生成逻辑(见 ``decompose.py`` 与 ``sampledata.py``),
方言层负责把其中少数几处写法差异抹平,使同一段逻辑既能跑在
Databricks SQL Warehouse 上,也能跑在本地 DuckDB 上。

为什么需要本地方言:简报要求「各分项贡献之和与总变化的差值小于 0.1%」
必须有测试证明。若测试只能连上 Databricks 才能跑,它就不会被经常跑。
DuckDB 让**同一份 SQL 生成器**在本地毫秒级完成算术闭合性验证。

**跨方言的数据并不相同** —— ``hash()`` 的实现不同,因此伪随机序列不同。
确定性保证的是「同一方言下重跑结果完全一致」,这已满足简报约束 3;
而算术闭合性本来就与具体数值无关,所以本地验证是充分的。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

# 逻辑类型 -> 各方言物理类型
_LOGICAL_TYPES = ("string", "bigint", "int", "double", "boolean", "date", "timestamp")


class SqlDialect(ABC):
    """一个 SQL 方言。只暴露两边写法真正不同的那几个原语。"""

    name: str
    catalog: str | None
    schema: str

    def __init__(self, catalog: str | None, schema: str) -> None:
        self.catalog = catalog
        self.schema = schema

    # -- 标识符与字面量 ----------------------------------------------------
    def qualify(self, table: str) -> str:
        """把表名补全为完全限定名。"""
        parts = [p for p in (self.catalog, self.schema, table) if p]
        return ".".join(parts)

    @staticmethod
    def string_literal(value: str) -> str:
        """安全地把 Python 字符串变成 SQL 字面量。

        过滤值由调用方(``decompose``)提供,可能来自 LLM 或用户输入,
        因此必须转义。列名与表名走白名单校验,不走这里。
        """
        if "\x00" in value:
            raise ValueError("字符串字面量不得包含 NUL")
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def date_literal(value: date) -> str:
        return f"DATE '{value.isoformat()}'"

    @abstractmethod
    def type_sql(self, logical_type: str) -> str:
        """逻辑类型 -> 物理类型。``decimal(p,s)`` 原样透传。"""

    # -- 建表 --------------------------------------------------------------
    @abstractmethod
    def create_table_suffix(self) -> str:
        """DDL 尾部(Databricks 需要 ``USING DELTA``,DuckDB 不需要)。"""

    # -- 生成器与确定性伪随机 ----------------------------------------------
    @abstractmethod
    def rand01(self, seed_expr: str) -> str:
        """由种子表达式派生 [0, 1) 上的确定性伪随机数。

        刻意不使用 ``rand()`` —— 它不可复现,会直接违反简报约束 3。
        """

    @abstractmethod
    def day_index(self, dt_expr: str, start: date) -> str:
        """``dt_expr`` 相对 ``start`` 的天数差(整数)。"""

    @abstractmethod
    def date_series(self, start: date, end: date, alias: str = "dt") -> str:
        """产生 [start, end] 闭区间日序列的 SELECT 语句。"""

    @abstractmethod
    def explode_int_range(self, bound_expr: str) -> str:
        """在 SELECT 列表中把每行展开成 ``1..bound_expr`` 行的表达式。"""

    @abstractmethod
    def date_shift(self, date_expr: str, days_expr: str) -> str:
        """``date_expr`` 加上 ``days_expr`` 天(可为负)。"""

    def cast_string(self, expr: str) -> str:
        return f"CAST({expr} AS {self.type_sql('string')})"


class DatabricksDialect(SqlDialect):
    """Databricks SQL Warehouse(Unity Catalog + Delta)。"""

    name = "databricks"

    def __init__(self, catalog: str = "main", schema: str = "gmv_rca") -> None:
        super().__init__(catalog, schema)

    def type_sql(self, logical_type: str) -> str:
        if logical_type.startswith("decimal"):
            return logical_type.upper()
        mapping = {
            "string": "STRING",
            "bigint": "BIGINT",
            "int": "INT",
            "double": "DOUBLE",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "timestamp": "TIMESTAMP",
        }
        return _lookup(mapping, logical_type)

    def create_table_suffix(self) -> str:
        return "USING DELTA"

    def rand01(self, seed_expr: str) -> str:
        # pmod 保证非负;hash 是 Murmur3,返回 INT。
        return f"(pmod(hash({seed_expr}), 1000000) / 1000000.0)"

    def day_index(self, dt_expr: str, start: date) -> str:
        return f"datediff({dt_expr}, {self.date_literal(start)})"

    def date_series(self, start: date, end: date, alias: str = "dt") -> str:
        return (
            f"SELECT explode(sequence({self.date_literal(start)}, "
            f"{self.date_literal(end)}, INTERVAL 1 DAY)) AS {alias}"
        )

    def explode_int_range(self, bound_expr: str) -> str:
        return f"explode(sequence(1, {bound_expr}))"

    def date_shift(self, date_expr: str, days_expr: str) -> str:
        return f"date_add({date_expr}, {days_expr})"


class DuckDBDialect(SqlDialect):
    """本地 DuckDB(仅用于开发与测试)。"""

    name = "duckdb"

    def __init__(self, catalog: str | None = None, schema: str = "main") -> None:
        super().__init__(catalog, schema)

    def type_sql(self, logical_type: str) -> str:
        if logical_type.startswith("decimal"):
            return logical_type.upper()
        mapping = {
            "string": "VARCHAR",
            "bigint": "BIGINT",
            "int": "INTEGER",
            "double": "DOUBLE",
            "boolean": "BOOLEAN",
            "date": "DATE",
            "timestamp": "TIMESTAMP",
        }
        return _lookup(mapping, logical_type)

    def create_table_suffix(self) -> str:
        return ""

    def rand01(self, seed_expr: str) -> str:
        # DuckDB 的 hash() 返回 UBIGINT,取模后天然非负。
        return f"((hash({seed_expr}) % 1000000)::DOUBLE / 1000000.0)"

    def day_index(self, dt_expr: str, start: date) -> str:
        return f"datediff('day', {self.date_literal(start)}, {dt_expr})"

    def date_series(self, start: date, end: date, alias: str = "dt") -> str:
        return (
            f"SELECT CAST(unnest(generate_series({self.date_literal(start)}, "
            f"{self.date_literal(end)}, INTERVAL 1 DAY)) AS DATE) AS {alias}"
        )

    def explode_int_range(self, bound_expr: str) -> str:
        return f"unnest(generate_series(1, {bound_expr}))"

    def date_shift(self, date_expr: str, days_expr: str) -> str:
        return f"({date_expr} + CAST({days_expr} AS INTEGER))"


def _lookup(mapping: dict[str, str], logical_type: str) -> str:
    try:
        return mapping[logical_type]
    except KeyError:
        raise ValueError(
            f"未知逻辑类型 {logical_type!r};支持:{_LOGICAL_TYPES} 或 decimal(p,s)"
        ) from None


DIALECTS: dict[str, type[SqlDialect]] = {
    "databricks": DatabricksDialect,
    "duckdb": DuckDBDialect,
}
