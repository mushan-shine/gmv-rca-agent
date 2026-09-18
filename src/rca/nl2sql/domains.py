"""维度取值检查:把「跑得通但必然查空」变成 L1 能修的错误。

第二轮真实评估(glm-4-flash + 报告日历)剩下 11 道错题,其中 5 道是同一种错::

    market = 'Germany'      实际取值是 DE
    region = 'US'           region 的取值是 NA / EMEA,US 在 market 列
    device = 'DESKTOP'      实际是小写 desktop
    category = 'Paid Search'   把渠道值填进了品类列
    order_status = 'RETURNED'  这个状态根本不存在(不可答题上编造出来的)

这些 SQL 语法正确、执行成功、返回 NULL 或空集 —— 守卫只认列名,引擎不报错,
自修复循环完全发现不了。但只要知道每个维度列**实际有哪些值**,就能在执行前给出
可以照着改的反馈:「market 没有 'Germany',实际取值是 US, UK, DE」。

取值**从仓库实时查出来**,不写死在代码或 YAML 里:数据换了,检查跟着变。
只查低基数的字符串列(默认 ≤ 50 个取值),ID 这类高基数列直接跳过。

这里刻意**不把取值写进 prompt**。一上来就告诉模型所有取值,当然能少错几道,
但那是人替系统做了改进。放在反馈里,模型只在真的写错时才看到 ——
它能不能从反馈里学会,以及改进循环能不能把「market 用国家代码」沉淀成一条提示,
正是要验证的东西。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from ..dialects import SqlDialect
from ..knowledge import Knowledge
from ..warehouse import SqlExecutor

DEFAULT_MAX_CARDINALITY = 50

VIOLATION_MARKER = "没有取值"
"""每条取值违规反馈都含这几个字。改进循环据此把失败归到「维度取值」一类。"""


@dataclass(frozen=True)
class ValueDomains:
    """列名 -> 该列在仓库里实际出现过的取值。

    按**列名**合并,不区分表:``market`` 在 fact_orders / fact_sessions / dim_user 上
    取值相同,模型写 ``fo.market`` 还是 ``du.market`` 都用同一份取值检查。
    """

    values: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def __contains__(self, column: str) -> bool:
        return column.lower() in self.values

    def allowed(self, column: str) -> frozenset[str]:
        return self.values.get(column.lower(), frozenset())

    def check(self, sql: str, dialect: str = "databricks") -> tuple[str, ...]:
        """找出 ``列 = '字面量'`` / ``列 IN (...)`` 里不存在的取值。

        只检查「列直接和字符串字面量比较」这一种形态。``LOWER(col) = 'x'``、
        ``col LIKE '%x%'`` 之类写法语义各异,判错的代价(给出错误反馈)比漏判高,一律放过。
        """
        if not self.values:
            return ()
        try:
            tree = sqlglot.parse_one(sql, dialect=dialect)
        except ParseError:
            return ()  # 解析不了是守卫的事,这里不重复报
        if tree is None:
            return ()

        problems: dict[tuple[str, str], str] = {}
        for column, literal in _column_literal_pairs(tree):
            allowed = self.allowed(column)
            if not allowed or literal in allowed:
                continue
            problems.setdefault((column.lower(), literal), self._message(column, literal, allowed))
        return tuple(problems.values())

    def _message(self, column: str, literal: str, allowed: frozenset[str]) -> str:
        shown = ", ".join(sorted(allowed))
        same_but_case = [value for value in allowed if value.lower() == literal.lower()]
        if same_but_case:
            return (
                f"列 {column} {VIOLATION_MARKER} '{literal}'(注意大小写,实际是 '{same_but_case[0]}')。"
                f"{column} 的全部取值:{shown}。"
            )
        elsewhere = self.columns_containing(literal)
        hint = f"'{literal}' 实际出现在列:{', '.join(elsewhere)}。" if elsewhere else ""
        return f"列 {column} {VIOLATION_MARKER} '{literal}'。{column} 的全部取值:{shown}。{hint}"

    def columns_containing(self, literal: str) -> list[str]:
        """哪些列里有这个值(不区分大小写)。

        「region = 'US'」这种错,最有用的反馈是「US 在 market 列里」。
        """
        needle = literal.lower()
        return sorted(
            column for column, allowed in self.values.items()
            if any(value.lower() == needle for value in allowed)
        )


def _column_literal_pairs(tree: exp.Expression):
    for node in tree.find_all(exp.EQ, exp.NEQ):
        left, right = node.left, node.right
        if isinstance(right, exp.Column) and _is_string(left):
            left, right = right, left
        if isinstance(left, exp.Column) and _is_string(right):
            yield left.name, right.this
    for node in tree.find_all(exp.In):
        column = node.this
        if not isinstance(column, exp.Column):
            continue
        for item in node.expressions:
            if _is_string(item):
                yield column.name, item.this


def _is_string(node: exp.Expression) -> bool:
    return isinstance(node, exp.Literal) and node.is_string


def dimension_columns(knowledge: Knowledge) -> dict[str, set[str]]:
    """知识库里所有「可能被拿来过滤」的字符串列:表名 -> 列名集合。

    来源:维度的原生列、经 join 取到的维表列、指标默认过滤用到的列(如 order_status)。
    """
    wanted: dict[str, set[str]] = {}

    def add(table: str, column: str) -> None:
        definition = knowledge.table(table)
        if column in definition.columns and definition.columns[column].type == "string":
            wanted.setdefault(table, set()).add(column)

    for dimension in knowledge.dimensions.values():
        for table, availability in dimension.availability.items():
            if availability.join is None:
                add(table, availability.column)
            else:
                add(availability.join.table, availability.column)
    for metric in knowledge.metrics.values():
        if metric.source_table:
            for column in metric.filters_default:
                add(metric.source_table, column)
    return wanted


def load_value_domains(
    knowledge: Knowledge,
    executor: SqlExecutor,
    dialect: SqlDialect,
    *,
    max_cardinality: int = DEFAULT_MAX_CARDINALITY,
) -> ValueDomains:
    """从仓库里查出每个维度列的实际取值。

    每张表一条查询,每列取至多 ``max_cardinality + 1`` 个值 ——
    超过上限的列视为高基数,不参与检查(报它「没有这个值」大概率是误报)。
    """
    merged: dict[str, set[str]] = {}
    too_many: set[str] = set()
    for table, columns in sorted(dimension_columns(knowledge).items()):
        qualified = dialect.qualify(table)
        parts = [
            f"SELECT '{column}' AS col, v FROM ("
            f"SELECT DISTINCT CAST({column} AS {dialect.type_sql('string')}) AS v "
            f"FROM {qualified} WHERE {column} IS NOT NULL LIMIT {max_cardinality + 1}) d_{column}"
            for column in sorted(columns)
        ]
        for row in executor.run(" UNION ALL ".join(parts)):
            merged.setdefault(str(row["col"]).lower(), set()).add(str(row["v"]))

    for column, values in merged.items():
        if len(values) > max_cardinality:
            too_many.add(column)
    return ValueDomains(
        {column: frozenset(values) for column, values in merged.items() if column not in too_many}
    )
