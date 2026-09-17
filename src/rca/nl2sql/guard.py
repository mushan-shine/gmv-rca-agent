"""生成 SQL 的**静态守卫**:执行之前先做确定性检查。

为什么需要它,而不是「直接跑一下看报不报错」:

1. **安全。** LLM 生成的文本会被送进仓库执行。只放行只读查询,
   `DROP` / `INSERT` / `MERGE` 一律在执行前就拦掉。
2. **诊断质量。** 「列 ``revenue`` 不存在,fact_orders 上有 order_amount」
   比仓库返回的 ``[UNRESOLVED_COLUMN]`` 更能帮模型下一轮改对 ——
   自修复循环的修复信号就来自这里。
3. **可度量。** 「幻觉列率」是评估指标之一。只有静态解析才能把
   「引用了不存在的列」与「SQL 语法错」「结果算错了」区分开。

守卫本身**完全确定性,不经 LLM** —— 沿用简报 §5 的边界:
LLM 不判断自己的结果对不对。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from ..knowledge import Knowledge

# 只读语句之外的一切都不放行。用词法而非语义判断,宁可误杀。
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|TRUNCATE|ALTER|CREATE|REPLACE|GRANT|REVOKE|"
    r"COPY|REFRESH|VACUUM|OPTIMIZE|SET|USE|CALL)\b",
    re.IGNORECASE,
)


_DATE_PARTS = frozenset(
    {
        "year", "years", "quarter", "quarters", "month", "months", "week", "weeks",
        "day", "days", "dayofweek", "dayofyear", "hour", "hours", "minute", "minutes",
        "second", "seconds", "millisecond", "milliseconds", "microsecond", "microseconds",
    }
)
"""日期函数的时间单位参数。见 :meth:`SqlGuard._allowed_columns`。"""


class Violation(str, Enum):
    """守卫能给出的判定。每一种都对应一条可以喂回给模型的具体反馈。"""

    EMPTY = "empty"
    PARSE_ERROR = "parse_error"
    NOT_READ_ONLY = "not_read_only"
    MULTIPLE_STATEMENTS = "multiple_statements"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"


@dataclass(frozen=True)
class GuardReport:
    """一次静态检查的结果。"""

    sql: str
    violations: tuple[Violation, ...] = ()
    messages: tuple[str, ...] = ()
    tables: frozenset[str] = frozenset()
    columns: frozenset[str] = frozenset()
    unknown_tables: frozenset[str] = frozenset()
    unknown_columns: frozenset[str] = frozenset()

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def hallucinated(self) -> bool:
        """是否引用了 schema 里根本不存在的表或列。

        这是评估里的「幻觉率」指标 —— 与「语法错」「答案错」是三件不同的事。
        """
        return bool(self.unknown_tables or self.unknown_columns)

    def feedback(self) -> str:
        """喂回给生成器的反馈文本。

        **必须具体到可以照着改**:说清哪个标识符不存在、这张表上有哪些列。
        含糊的「SQL 有误,请重试」会让自修复循环空转。
        """
        return "\n".join(self.messages)


@dataclass
class SqlGuard:
    """按知识库里的 schema 校验一条生成出来的 SQL。"""

    knowledge: Knowledge
    dialect: str = "databricks"
    max_suggestions: int = 6
    _known_columns: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._known_columns = {
            name: frozenset(table.columns) for name, table in self.knowledge.tables.items()
        }

    @property
    def all_columns(self) -> frozenset[str]:
        return frozenset().union(*self._known_columns.values()) if self._known_columns else frozenset()

    def check(self, sql: str) -> GuardReport:
        """静态检查一条 SQL,不执行它。"""
        text = (sql or "").strip().rstrip(";").strip()
        if not text:
            return GuardReport(
                sql=sql or "",
                violations=(Violation.EMPTY,),
                messages=("生成结果为空:没有产出任何 SQL。",),
            )

        try:
            statements = [s for s in sqlglot.parse(text, dialect=self.dialect) if s is not None]
        except ParseError as exc:
            return GuardReport(
                sql=text,
                violations=(Violation.PARSE_ERROR,),
                messages=(f"SQL 无法解析({self.dialect} 方言):{_first_line(str(exc))}",),
            )

        if len(statements) != 1:
            return GuardReport(
                sql=text,
                violations=(Violation.MULTIPLE_STATEMENTS,),
                messages=(
                    f"只允许一条语句,实际解析出 {len(statements)} 条。"
                    "请把答案写成单条 SELECT。",
                ),
            )

        statement = statements[0]
        violations: list[Violation] = []
        messages: list[str] = []

        if not isinstance(statement, (exp.Select, exp.Union, exp.Subquery)) or _FORBIDDEN.search(
            _strip_literals(text)
        ):
            violations.append(Violation.NOT_READ_ONLY)
            messages.append(
                "只允许只读查询(SELECT / WITH ... SELECT)。"
                "不要写 INSERT / UPDATE / DELETE / DROP / CREATE 等语句。"
            )

        tables = _referenced_tables(statement)
        cte_names = {cte.alias_or_name for cte in statement.find_all(exp.CTE)}
        unknown_tables = frozenset(
            name for name in tables if name not in self._known_columns and name not in cte_names
        )
        if unknown_tables:
            violations.append(Violation.UNKNOWN_TABLE)
            messages.append(
                f"引用了不存在的表:{sorted(unknown_tables)}。"
                f"可用的表只有:{sorted(self._known_columns)}。"
            )

        # 只在表都认识的前提下查列 —— 表就错了的话,列的报错只是噪音
        columns = _referenced_columns(statement)
        unknown_columns: frozenset[str] = frozenset()
        if not unknown_tables:
            allowed = self._allowed_columns(tables, statement)
            unknown_columns = frozenset(name for name in columns if name not in allowed)
            if unknown_columns:
                violations.append(Violation.UNKNOWN_COLUMN)
                messages.append(
                    f"引用了不存在的列:{sorted(unknown_columns)}。"
                    + self._suggest(tables)
                )

        return GuardReport(
            sql=text,
            violations=tuple(violations),
            messages=tuple(messages),
            tables=frozenset(tables),
            columns=frozenset(columns),
            unknown_tables=unknown_tables,
            unknown_columns=unknown_columns,
        )

    def _allowed_columns(self, tables: set[str], statement: exp.Expression) -> set[str]:
        allowed: set[str] = set()
        for table in tables:
            allowed |= set(self._known_columns.get(table, ()))
        # SELECT 里的别名可以在 GROUP BY / ORDER BY / HAVING 里被再次引用
        for alias in statement.find_all(exp.Alias):
            allowed.add(alias.alias_or_name)
        # 时间单位关键字(DATEADD(day, -7, ...) 里的 day)。有的方言解析器会把它当成列,
        # 误判成「幻觉列」并反馈「列 day 不存在」—— 反馈错了,自修复就会往错的方向改。
        # 真实列名优先:只有 schema 里没有同名列时才放行。
        allowed |= _DATE_PARTS
        return allowed

    def _suggest(self, tables: set[str]) -> str:
        parts = []
        for table in sorted(tables):
            columns = sorted(self._known_columns.get(table, ()))[: self.max_suggestions]
            if columns:
                parts.append(f"{table} 上有:{', '.join(columns)}")
        return "".join(f"\n  {p}" for p in parts)


def _referenced_tables(statement: exp.Expression) -> set[str]:
    return {table.name for table in statement.find_all(exp.Table) if table.name}


def _referenced_columns(statement: exp.Expression) -> set[str]:
    return {column.name for column in statement.find_all(exp.Column) if column.name}


def _strip_literals(sql: str) -> str:
    """去掉字符串字面量,免得 ``WHERE channel = 'Paid Search'`` 里的词被误判。"""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else text
