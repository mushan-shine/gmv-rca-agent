"""评估集:问题、以及**程序化生成**的标准答案。

这是整个 text-to-SQL 项目最关键的一块 —— 也是绝大多数同类项目缺的一块。

常见做法是手写二三十条「问题 + 我认为对的 SQL」。问题有三:
规模上不去、人写的答案本身可能是错的、数据一变答案就全废。

本项目的做法:标准答案由**确定性参考 SQL** 执行得来,而参考 SQL 由
``knowledge/`` 里的指标与维度定义按模板生成(复用 :class:`rca.decompose.FromBuilder`,
join 路径只有一份实现)。于是:

* case 只需要声明「问的是哪个指标、哪个维度、哪个窗口」,不需要写 SQL;
* 样例数据重建之后,标准答案自动跟着变,不需要重新标注;
* 口径(比如 GMV 只算 COMPLETED)由知识库统一保证,不会出现
  「标注者以为的口径」和「系统以为的口径」不一致。

四类 gold
---------

``aggregate``      一个标量。支持可加指标与比率指标(CVR/AOV 用 CTE 拼分子分母)。
``group_by``       按某维度分组。仅限可加指标 —— 比率指标按维度分组会有口径歧义。
``change``         两个窗口之差。支持可加与比率。
``unanswerable``   **答案是「答不了」。** 数据里根本没有这个字段,正确行为是拒答。

最后一类是刻意设计的对照组,呼应原简报 §3.3 的「无异常对照组权重最高」:
用来测系统会不会在没有依据时编一个看起来很像样的答案。
一个 pass@1 很高但对不可答问题照样编 SQL 的模型,在生产里是危险的。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..decompose import FromBuilder, Period
from ..dialects import SqlDialect
from ..errors import RcaError
from ..knowledge import Knowledge
from ..sampledata import SampleDataConfig
from ..warehouse import ExecutedQuery, SqlExecutor

_STRICT = ConfigDict(extra="forbid", frozen=True)


class EvalCaseError(RcaError):
    """评估集定义不合法。"""


class GoldKind(str, Enum):
    AGGREGATE = "aggregate"
    GROUP_BY = "group_by"
    CHANGE = "change"
    UNANSWERABLE = "unanswerable"


class GoldSpec(BaseModel):
    """标准答案的**声明**,而不是标准答案本身。

    答案由 :func:`build_reference_sql` 按这份声明生成 SQL、再执行得来。
    """

    model_config = _STRICT

    kind: GoldKind
    metric: str | None = None
    dimension: str | None = None
    period: str | None = None
    """窗口标签,如 ``W8``。由 :func:`resolve_period` 解析成具体日期。"""

    baseline: str | None = None
    """``change`` 用的基期窗口标签。"""

    filters: dict[str, str] = Field(default_factory=dict)
    order: Literal["desc", "asc"] | None = None
    limit: int | None = None
    reason: str | None = None
    """``unanswerable`` 必填:为什么答不了。会作为「正确拒答理由」写进报告。"""

    @model_validator(mode="after")
    def _check(self) -> "GoldSpec":
        if self.kind is GoldKind.UNANSWERABLE:
            if not self.reason:
                raise ValueError("unanswerable 必须写明 reason(为什么答不了)")
            if self.metric or self.dimension or self.period:
                raise ValueError("unanswerable 不应声明 metric / dimension / period")
            return self
        if not self.metric or not self.period:
            raise ValueError(f"{self.kind.value} 必须声明 metric 与 period")
        if self.kind is GoldKind.GROUP_BY and not self.dimension:
            raise ValueError("group_by 必须声明 dimension")
        if self.kind is not GoldKind.GROUP_BY and self.dimension:
            raise ValueError(f"{self.kind.value} 不应声明 dimension")
        if self.kind is GoldKind.CHANGE and not self.baseline:
            raise ValueError("change 必须声明 baseline")
        if self.kind is not GoldKind.CHANGE and self.baseline:
            raise ValueError(f"{self.kind.value} 不应声明 baseline")
        if (self.order or self.limit) and self.kind is not GoldKind.GROUP_BY:
            raise ValueError("order / limit 只对 group_by 有意义")
        return self


class EvalCase(BaseModel):
    """一条评估 case。"""

    model_config = _STRICT

    id: str
    question: str
    """英文问题 —— 生成器看到的就是这一句。"""

    gold: GoldSpec
    question_zh: str = ""
    """中文对照,仅供人阅读,不进 prompt。"""

    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    holdout: bool = False
    """留出集。改进循环不得用它来调知识 —— 它专门用来验证改进没有过拟合。"""

    @property
    def ordered(self) -> bool:
        """答案是否有序(问「前三名」时比对要保序)。"""
        return self.gold.order is not None


def load_cases(path: str | Path) -> list[EvalCase]:
    """加载 ``eval/cases.yaml``。"""
    path = Path(path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvalCaseError(f"评估集文件不存在:{path}") from None
    except yaml.YAMLError as exc:
        raise EvalCaseError(f"YAML 解析失败 {path}: {exc}") from None

    if not isinstance(payload, dict) or "cases" not in payload:
        raise EvalCaseError("cases.yaml 顶层必须是 {cases: [...]}")

    cases: list[EvalCase] = []
    seen: set[str] = set()
    for item in payload["cases"] or []:
        identifier = item.get("id", "?") if isinstance(item, dict) else "?"
        try:
            case = EvalCase.model_validate(item)
        except ValidationError as exc:
            raise EvalCaseError(f"case {identifier!r} 校验失败:\n{exc}") from None
        if case.id in seen:
            raise EvalCaseError(f"case id 重复:{case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise EvalCaseError(f"{path} 里没有任何 case")
    return cases


def validate_cases(cases: Sequence[EvalCase], knowledge: Knowledge) -> None:
    """交叉校验:case 引用的指标与维度必须真的存在且可用。

    评估集里一个拼错的指标名,会让那条 case 永远失败,
    而失败原因看起来像是模型的问题 —— 必须在加载阶段就拦下。
    """
    for case in cases:
        gold = case.gold
        if gold.kind is GoldKind.UNANSWERABLE:
            continue
        metric = knowledge.metric(gold.metric)  # type: ignore[arg-type]
        if gold.kind is GoldKind.GROUP_BY:
            if not metric.is_additive:
                raise EvalCaseError(
                    f"case {case.id}:{gold.metric} 是比率指标,不能按维度分组作为标准答案"
                    f"(各分组的比率之和不等于总体比率)"
                )
            if gold.dimension not in metric.dimensions:
                raise EvalCaseError(
                    f"case {case.id}:维度 {gold.dimension!r} 不在 {gold.metric} 的白名单内;"
                    f"可用:{knowledge.ordered_dimensions(gold.metric)}"  # type: ignore[arg-type]
                )
        for dimension in gold.filters:
            knowledge.dimension(dimension)


# ---------------------------------------------------------------------------
# 窗口
# ---------------------------------------------------------------------------
def resolve_period(label: str, config: SampleDataConfig | None = None) -> Period:
    """把 ``W1``~``W8`` 这样的标签解析成具体日期窗口。

    用标签而不是硬编码日期,是为了让样例数据的起始日改变时,
    整个评估集不需要逐条改。
    """
    config = config or SampleDataConfig()
    text = label.strip().upper()
    if not text.startswith("W") or not text[1:].isdigit():
        raise EvalCaseError(f"窗口标签只支持 W1..W{config.weeks},收到 {label!r}")
    index = int(text[1:])
    if not 1 <= index <= config.weeks:
        raise EvalCaseError(f"窗口 {label!r} 超出样例数据范围(W1..W{config.weeks})")
    start = date.fromordinal(config.start_date.toordinal() + (index - 1) * 7)
    end = date.fromordinal(start.toordinal() + 6)
    return Period(start, end, text)


# ---------------------------------------------------------------------------
# 参考 SQL
# ---------------------------------------------------------------------------
def _period_predicate(column: str, period: Period, dialect: SqlDialect) -> str:
    return (
        f"{column} BETWEEN {dialect.date_literal(period.start)} "
        f"AND {dialect.date_literal(period.end)}"
    )


def _scalar_ctes(
    knowledge: Knowledge,
    dialect: SqlDialect,
    prefix: str,
    metric_name: str,
    period: Period,
    filters: Mapping[str, str],
) -> tuple[list[str], str]:
    """把一个指标展开成若干个「一行一列」的 CTE,外加引用它们的标量表达式。

    比率指标递归展开成分子 / 分母两个 CTE,最外层用 ``CROSS JOIN`` 拼起来 ——
    避免「无 FROM 的标量子查询」这种在各引擎上支持度参差的写法。
    """
    metric = knowledge.metric(metric_name)
    if metric.is_additive:
        table = metric.source_table
        assert table is not None
        builder = FromBuilder(knowledge, dialect, table)
        partition = knowledge.table(table).partition_column
        assert partition is not None

        conditions = [_period_predicate(builder.fact_column(partition), period, dialect)]
        for column in sorted(metric.filters_default):
            conditions.append(
                f"{builder.fact_column(column)} = "
                f"{SqlDialect.string_literal(metric.filters_default[column])}"
            )
        for dimension in sorted(filters):
            expr, _ = builder.dimension_column(dimension)
            conditions.append(f"{expr} = {SqlDialect.string_literal(filters[dimension])}")

        cte = (
            f"{prefix} AS (\n  SELECT {metric.sql} AS v\n  "
            + builder.render()
            + "\n  WHERE "
            + "\n    AND ".join(conditions)
            + "\n)"
        )
        return [cte], f"{prefix}.v"

    assert metric.ratio is not None
    numerator_ctes, numerator = _scalar_ctes(
        knowledge, dialect, f"{prefix}_n", metric.ratio.numerator, period, filters
    )
    denominator_ctes, denominator = _scalar_ctes(
        knowledge, dialect, f"{prefix}_d", metric.ratio.denominator, period, filters
    )
    return numerator_ctes + denominator_ctes, f"({numerator} / {denominator})"


def _cte_names(ctes: Sequence[str]) -> list[str]:
    return [cte.split(" AS ", 1)[0].strip() for cte in ctes]


def build_reference_sql(
    case: EvalCase,
    knowledge: Knowledge,
    dialect: SqlDialect,
    config: SampleDataConfig | None = None,
) -> str:
    """为一条 case 生成确定性的参考 SQL。

    Args:
        case: 评估 case;``unanswerable`` 没有参考 SQL,会抛错。
        knowledge: 已校验的知识库 —— 指标口径与 join 路径都来自它。
        dialect: 目标方言。
        config: 样例数据配置,用于解析 ``W1``~``W8`` 标签。

    Returns:
        一条可直接执行的 SQL 文本。

    Raises:
        EvalCaseError: case 是 ``unanswerable``,或声明了不可用的指标/维度。
    """
    gold = case.gold
    if gold.kind is GoldKind.UNANSWERABLE:
        raise EvalCaseError(
            f"case {case.id} 是 unanswerable,没有参考 SQL —— 正确答案是拒答"
        )

    assert gold.metric is not None and gold.period is not None
    period = resolve_period(gold.period, config)

    if gold.kind is GoldKind.AGGREGATE:
        ctes, expression = _scalar_ctes(
            knowledge, dialect, "m", gold.metric, period, gold.filters
        )
        joined = " CROSS JOIN ".join(_cte_names(ctes))
        return f"WITH {', '.join(ctes)}\nSELECT {expression} AS value\nFROM {joined}"

    if gold.kind is GoldKind.CHANGE:
        assert gold.baseline is not None
        baseline = resolve_period(gold.baseline, config)
        current_ctes, current = _scalar_ctes(
            knowledge, dialect, "cur", gold.metric, period, gold.filters
        )
        base_ctes, base = _scalar_ctes(
            knowledge, dialect, "base", gold.metric, baseline, gold.filters
        )
        ctes = current_ctes + base_ctes
        joined = " CROSS JOIN ".join(_cte_names(ctes))
        return (
            f"WITH {', '.join(ctes)}\n"
            f"SELECT ({current} - {base}) AS value\nFROM {joined}"
        )

    # group_by
    metric = knowledge.metric(gold.metric)
    table = metric.source_table
    assert table is not None
    builder = FromBuilder(knowledge, dialect, table)
    partition = knowledge.table(table).partition_column
    assert partition is not None

    assert gold.dimension is not None
    dimension_expr, _ = builder.dimension_column(gold.dimension)
    conditions = [_period_predicate(builder.fact_column(partition), period, dialect)]
    for column in sorted(metric.filters_default):
        conditions.append(
            f"{builder.fact_column(column)} = "
            f"{SqlDialect.string_literal(metric.filters_default[column])}"
        )
    for name in sorted(gold.filters):
        expr, _ = builder.dimension_column(name)
        conditions.append(f"{expr} = {SqlDialect.string_literal(gold.filters[name])}")

    sql = (
        f"SELECT {dimension_expr} AS {gold.dimension}, {metric.sql} AS value\n"
        + builder.render()
        + "\nWHERE "
        + "\n  AND ".join(conditions)
        + "\nGROUP BY 1"
    )
    if gold.order:
        sql += f"\nORDER BY 2 {gold.order.upper()}"
    if gold.limit:
        sql += f"\nLIMIT {gold.limit}"
    return sql


@dataclass(frozen=True)
class Golden:
    """一条 case 的标准答案(已执行)。"""

    case_id: str
    answerable: bool
    rows: tuple[dict[str, Any], ...] = ()
    reference_sql: str = ""
    reason: str = ""
    query: ExecutedQuery | None = None


def build_golden(
    case: EvalCase,
    knowledge: Knowledge,
    dialect: SqlDialect,
    executor: SqlExecutor,
    config: SampleDataConfig | None = None,
) -> Golden:
    """生成并执行参考 SQL,得到标准答案。

    ``unanswerable`` 的 case 不执行任何 SQL —— 它的标准答案是「拒答」。
    """
    if case.gold.kind is GoldKind.UNANSWERABLE:
        return Golden(case_id=case.id, answerable=False, reason=case.gold.reason or "")

    sql = build_reference_sql(case, knowledge, dialect, config)
    rows = executor.run(sql)
    return Golden(
        case_id=case.id,
        answerable=True,
        rows=tuple(rows),
        reference_sql=sql,
        query=ExecutedQuery.of(f"golden:{case.id}", sql, rows, dialect.name),
    )
