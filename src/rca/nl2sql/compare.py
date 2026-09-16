"""**执行式比对**(execution match):比结果集,不比 SQL 文本。

同一个问题有无数种写法都是对的 —— 子查询还是 JOIN、`COUNT(DISTINCT id)` 还是
`COUNT(*)`、列取什么别名。按字符串比 SQL 会把这些全判成错,
评估指标就变成了「像不像我写的那条」,而不是「答得对不对」。

所以标准答案存的是**结果集**,由确定性参考 SQL 执行得来(见 :mod:`rca.nl2sql.cases`)。
比对规则:

* **忽略列名** —— `AS total` 和 `AS value` 是同一个答案;
* **忽略行序** —— 除非 case 明确声明答案是有序的(问「排名前三」时才有序);
* **数值带容差** —— `DECIMAL` / `DOUBLE` / `int` 统一成 float 后按相对误差比;
* **列数必须一致** —— 多选了一列(比如顺手把 `dt` 也选出来)算错。
  这是 Spider / BIRD 等基准的通行做法,也符合直觉:答案的形状本身就是答案的一部分。

失败时给出**分类**而不只是 True/False —— 改进循环要靠这个分类做失败归因。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping, Sequence

NUMERIC_RELATIVE_TOLERANCE = 1e-6
NUMERIC_ABSOLUTE_TOLERANCE = 1e-9


class MatchFailure(str, Enum):
    """比对失败的类型。改进循环按这个分类做失败归因。"""

    COLUMN_COUNT = "column_count"
    ROW_COUNT = "row_count"
    VALUE_MISMATCH = "value_mismatch"
    EXPECTED_EMPTY = "expected_empty"
    GOT_EMPTY = "got_empty"


@dataclass(frozen=True)
class MatchResult:
    match: bool
    failure: MatchFailure | None = None
    detail: str = ""

    def __bool__(self) -> bool:
        return self.match


def _normalize_value(value: Any) -> Any:
    """把仓库返回的值归一到可比较的形式。

    ``Decimal``(金额)、``int``、``float`` 统一成 float;日期统一成 ISO 字符串;
    字符串去首尾空白。``None`` 保持为 ``None`` —— NULL 与 0 是不同的答案。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (Decimal, int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _values_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, float) and isinstance(right, float):
        if left == right:
            return True
        scale = max(abs(left), abs(right))
        return abs(left - right) <= max(
            NUMERIC_ABSOLUTE_TOLERANCE, NUMERIC_RELATIVE_TOLERANCE * scale
        )
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _row_tuple(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(_normalize_value(value) for value in row.values())


def _sort_key(row: tuple[Any, ...]) -> tuple[str, ...]:
    """行序无关比对用的排序键。转成字符串,免得 None 与数字比大小时抛错。"""
    return tuple("\x00" if value is None else repr(value) for value in row)


def compare_result_sets(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    *,
    ordered: bool = False,
) -> MatchResult:
    """比较两个结果集。

    Args:
        expected: 标准答案(由确定性参考 SQL 执行得来)。
        actual: 被评估的 SQL 执行结果。
        ordered: 答案是否有序。问「排名前三」时传 ``True``,
            否则默认行序无关。

    Returns:
        :class:`MatchResult`;不匹配时带上失败分类与人类可读的差异说明。
    """
    expected_rows = [_row_tuple(row) for row in expected]
    actual_rows = [_row_tuple(row) for row in actual]

    if not expected_rows and not actual_rows:
        return MatchResult(True)
    if not expected_rows:
        return MatchResult(
            False,
            MatchFailure.EXPECTED_EMPTY,
            f"标准答案是空结果集,实际返回了 {len(actual_rows)} 行。",
        )
    if not actual_rows:
        return MatchResult(
            False,
            MatchFailure.GOT_EMPTY,
            f"实际返回空结果集,标准答案有 {len(expected_rows)} 行。",
        )

    expected_width = len(expected_rows[0])
    actual_width = len(actual_rows[0])
    if expected_width != actual_width:
        return MatchResult(
            False,
            MatchFailure.COLUMN_COUNT,
            f"列数不一致:标准答案 {expected_width} 列,实际 {actual_width} 列。"
            f"多选或少选列都算答错 —— 答案的形状也是答案的一部分。",
        )

    if len(expected_rows) != len(actual_rows):
        return MatchResult(
            False,
            MatchFailure.ROW_COUNT,
            f"行数不一致:标准答案 {len(expected_rows)} 行,实际 {len(actual_rows)} 行。",
        )

    left = expected_rows if ordered else sorted(expected_rows, key=_sort_key)
    right = actual_rows if ordered else sorted(actual_rows, key=_sort_key)

    for index, (expected_row, actual_row) in enumerate(zip(left, right)):
        for column, (want, got) in enumerate(zip(expected_row, actual_row)):
            if not _values_equal(want, got):
                return MatchResult(
                    False,
                    MatchFailure.VALUE_MISMATCH,
                    f"第 {index + 1} 行第 {column + 1} 列不一致:"
                    f"期望 {want!r},实际 {got!r}"
                    + ("" if ordered else "(已按行内容排序后比对)"),
                )
    return MatchResult(True)
