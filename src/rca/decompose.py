"""分解引擎:参数化 SQL 模板 + 残差计算。

这是 V0 的核心,也是整个项目唯一**不允许出错**的部分 ——
简报 §3.2 把残差定为「唯一的确定性信号」,若分解算术本身有偏差,
上层的停止条件、评估、报告全部失效。

本模块严格遵守简报 §5 的职责边界:

* 只按**参数化模板**生成 SQL,模板由 ``knowledge/`` 下的 YAML 驱动;
* 不调用 LLM,不猜测下一个该钻取的维度;
* 不判断结果「对不对」,只如实返回贡献与残差。

两种分解
--------

**乘法(因子)分解** —— ``decompose_factors``

    GMV = UV · CVR · AOV
    取对数:    ln GMV = ln UV + ln CVR + ln AOV
    两窗口作差:Δln GMV = Δln UV + Δln CVR + Δln AOV      ← 精确恒等,无近似

    记 L = Δln GMV = ln(GMV_b / GMV_a),r = GMV_b / GMV_a − 1(百分比变化)。
    则 L = ln(1 + r)。

    各因子的**份额**:  share_f = Δln f / L,       Σ share_f ≡ 1
    换算成**百分点**:  pp_f    = share_f · r,     Σ pp_f    ≡ r

    两个求和都是**恒等式**,不是近似 —— 因此「各分项之和等于总变化」
    在因子分解里天然成立,残差只反映浮点误差(~1e-16)。

    ⚠ 换算的语义代价(简报 §13.3 要求说明):
    ``pp_f`` 是把总百分比变化 r 按对数份额**分配**给各因子,
    它**不等于**「只有该因子变化时 GMV 会变多少」——
    后者是 ``exp(Δln f) − 1``。两者仅在小变化时接近:
    |Δln f| ≤ 0.10 时相对差约 5%,|Δln f| ≤ 0.20 时约 10%。
    报告中若要说「CVR 单独导致 GMV 下跌 x%」,必须用 ``isolated_pct_change``,
    而不是 ``contribution_pct_points``。

**加法(维度)分解** —— ``decompose_by_dimension``

    ΔGMV = Σ_d [GMV_b(d) − GMV_a(d)]

    只要维度取值构成完整划分,等式精确成立。因此残差在这里是一个**诊断信号**:
    引擎**独立地**算一次总量(不带 join),再算一次分组求和(带 join),
    两者之差即残差。

    * 残差 ≈ 0        → 划分完整
    * 残差 < 0        → join fan-out(维表主键不唯一),行被放大
    * ``__UNKNOWN__`` 桶不为空 → join 丢行(事实表有维表里不存在的键)
    * ``__NULL__``    桶不为空 → 原生列本身有 NULL

    比率指标(CVR / AOV)会被直接拒绝:各分项比率之和不等于总体比率,
    强行相加会得到一个算术上错误、却很像结论的数字。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from .dialects import SqlDialect
from .errors import (
    DecompositionError,
    DimensionNotFoundError,
    FilterNotSupportedError,
    MetricNotAdditiveError,
    NonPositiveValueError,
    PeriodError,
)
from .knowledge import Knowledge, MetricDef
from .warehouse import ExecutedQuery, SqlExecutor, to_float

NULL_BUCKET = "__NULL__"
"""原生列取值为 NULL 的分组。"""

UNKNOWN_BUCKET = "__UNKNOWN__"
"""join 未命中(事实表的键在维表中不存在)的分组。"""

_EPS = 1e-12
"""判定「变化为零」的阈值。低于它时份额没有定义,而不是等于 0。"""


# ---------------------------------------------------------------------------
# 时间窗口
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Period:
    """闭区间 ``[start, end]`` 的日期窗口。"""

    start: date
    end: date
    label: str = ""

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise PeriodError(f"窗口起止颠倒:{self.start} > {self.end}")

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def overlaps(self, other: "Period") -> bool:
        return self.start <= other.end and other.start <= self.end

    def __str__(self) -> str:
        return self.label or f"{self.start}~{self.end}"


def _check_periods(a: Period, b: Period, allow_unequal: bool) -> None:
    if a.overlaps(b):
        raise PeriodError(
            f"两个窗口重叠({a} 与 {b})。重叠期的行会被算进其中一个窗口,"
            f"分解结果无法解释。"
        )
    if not allow_unequal and a.days != b.days:
        raise PeriodError(
            f"窗口长度不同({a} 为 {a.days} 天,{b} 为 {b.days} 天)。"
            f"长度不同的窗口做同比/环比,变化里会混入「天数差」这个伪因子。"
            f"确需如此(例如 MTD 对比上月整月)请显式传 allow_unequal_periods=True。"
        )


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorContribution:
    """乘法分解中一个因子的贡献。"""

    factor: str
    display_name: str
    value_a: float
    value_b: float
    pct_change: float
    """该因子自身的百分比变化 ``value_b / value_a − 1``。"""

    log_delta: float
    """``ln(value_b) − ln(value_a)``。各因子的这一项精确相加等于总体的对数变化。"""

    share_of_change: float | None
    """占总对数变化的份额。总变化为零时无定义(``None``)。"""

    contribution_pct_points: float | None
    """分配到的百分点。各因子之和精确等于总百分比变化。"""

    isolated_pct_change: float
    """只有该因子变化时,指标会变化的百分比 ``exp(log_delta) − 1``。

    与 ``contribution_pct_points`` 是两个不同的量,报告里不可混用。
    """


@dataclass(frozen=True)
class FactorDecomposition:
    """``decompose_factors`` 的返回值。"""

    metric: str
    period_a: Period
    period_b: Period
    filters: Mapping[str, str]
    value_a: float
    value_b: float
    pct_change: float
    log_delta: float
    factors: tuple[FactorContribution, ...]
    residual_log: float
    residual_pct: float
    """未被因子解释的部分,占总对数变化的百分比。数学上应为 0,实际为浮点误差。"""

    identity_gap_pct: float
    """两个窗口中 ``Π factors`` 与指标自身值的最大相对偏差(%)。

    这是对知识库的**运行期复核**:若它明显不为 0,说明 YAML 里的因子口径
    与指标本身对不上(例如 orders 与 gmv 用了不同的 order_status 过滤),
    此时任何解释都不可信。
    """

    queries: tuple[ExecutedQuery, ...] = field(default_factory=tuple)

    @property
    def is_closed(self) -> bool:
        """残差是否小于 0.1%(简报 V0 验收标准 4)。"""
        return abs(self.residual_pct) < 0.1


@dataclass(frozen=True)
class DimensionContribution:
    """加法分解中一个维度取值的贡献。"""

    value: str
    value_a: float
    value_b: float
    delta: float
    share_of_change: float | None
    contribution_pct_points: float
    """该分项的变化占**基期总量**的百分点。各分项之和等于总体百分比变化。"""


@dataclass(frozen=True)
class DimensionDecomposition:
    """``decompose_by_dimension`` 的返回值。"""

    metric: str
    dimension: str
    period_a: Period
    period_b: Period
    filters: Mapping[str, str]
    value_a: float
    value_b: float
    delta: float
    pct_change: float | None
    parts: tuple[DimensionContribution, ...]
    """按 ``|delta|`` 降序排列。"""

    residual_abs: float
    residual_pct: float
    """(总变化 − 各分项变化之和) / |总变化| × 100。"""

    level_residual_a_pct: float
    level_residual_b_pct: float
    """两个窗口各自的「总量 vs 分组求和」偏差(%)。

    即使 ΔA 与 ΔB 的偏差恰好抵消,这两个数也会暴露 join 造成的水位错位。
    """

    queries: tuple[ExecutedQuery, ...] = field(default_factory=tuple)

    @property
    def is_closed(self) -> bool:
        return abs(self.residual_pct) < 0.1

    @property
    def unknown_share_pct(self) -> float:
        """落入 ``__UNKNOWN__`` / ``__NULL__`` 桶的基期占比(%)。"""
        if abs(self.value_a) < _EPS:
            return 0.0
        bad = sum(
            p.value_a for p in self.parts if p.value in (UNKNOWN_BUCKET, NULL_BUCKET)
        )
        return bad / abs(self.value_a) * 100.0


# ---------------------------------------------------------------------------
# SQL 构造
# ---------------------------------------------------------------------------
@dataclass
class _JoinPlan:
    """一次维表关联。

    被关联的维表包在子查询里,只投影出 join key 与所需列,并统一改名 ——
    这样维表上与事实表同名的列(例如 ``dim_user.market``)绝不可能造成歧义,
    指标 YAML 里的聚合表达式可以放心地写不带前缀的列名。
    """

    alias: str
    table: str
    left_key: str
    right_key: str
    columns: dict[str, str] = field(default_factory=dict)  # 维表列名 -> 子查询内别名

    def column_alias(self, column: str) -> str:
        if column not in self.columns:
            self.columns[column] = f"_c{len(self.columns)}"
        return self.columns[column]


class FromBuilder:
    """为一张事实表构造 FROM 子句,按需追加维表关联。

    :mod:`rca.nl2sql.cases` 生成评估集的参考 SQL 时也复用它 ——
    join 路径与消歧逻辑只能有一份实现,否则标准答案和分解结果会对不上。
    """

    FACT_ALIAS = "f"

    def __init__(self, knowledge: Knowledge, dialect: SqlDialect, fact_table: str) -> None:
        self.k = knowledge
        self.d = dialect
        self.fact_table = fact_table
        self._joins: dict[tuple[str, str, str], _JoinPlan] = {}

    def fact_column(self, column: str) -> str:
        table = self.k.table(self.fact_table)
        if column not in table.columns:
            raise DecompositionError(
                f"表 {self.fact_table} 上没有列 {column!r}"
            )
        return f"{self.FACT_ALIAS}.{column}"

    def dimension_column(self, dimension: str) -> tuple[str, str]:
        """返回 (该维度在本表上的取值表达式, 空值语义的占位符)。"""
        dim = self.k.dimension(dimension)
        avail = dim.on(self.fact_table)
        if avail is None:
            raise DimensionNotFoundError(
                f"维度 {dimension!r} 在表 {self.fact_table} 上不可用;"
                f"可用表:{sorted(dim.availability)}"
            )
        if avail.join is None:
            return self.fact_column(avail.column), NULL_BUCKET

        key = (avail.join.table, avail.join.left_key, avail.join.right_key)
        plan = self._joins.get(key)
        if plan is None:
            plan = _JoinPlan(
                alias=f"j{len(self._joins)}",
                table=avail.join.table,
                left_key=avail.join.left_key,
                right_key=avail.join.right_key,
            )
            self._joins[key] = plan
        return f"{plan.alias}.{plan.column_alias(avail.column)}", UNKNOWN_BUCKET

    def bucket_expr(self, dimension: str) -> str:
        """分组表达式:NULL 折成语义明确的占位桶,保证划分完整。"""
        expr, sentinel = self.dimension_column(dimension)
        cast = self.d.cast_string(expr)
        return f"COALESCE({cast}, {SqlDialect.string_literal(sentinel)})"

    def render(self) -> str:
        lines = [f"FROM {self.d.qualify(self.fact_table)} {self.FACT_ALIAS}"]
        for plan in self._joins.values():
            key_alias = "_jk"
            projected = ", ".join(
                [f"{plan.right_key} AS {key_alias}"]
                + [f"{col} AS {alias}" for col, alias in plan.columns.items()]
            )
            lines.append(
                f"LEFT JOIN (SELECT {projected} FROM {self.d.qualify(plan.table)}) {plan.alias}"
                f" ON {self.FACT_ALIAS}.{plan.left_key} = {plan.alias}.{key_alias}"
            )
        return "\n".join(lines)


def _period_predicate(column: str, period: Period, d: SqlDialect) -> str:
    return (
        f"{column} BETWEEN {d.date_literal(period.start)} AND {d.date_literal(period.end)}"
    )


def _build_aggregate_sql(
    knowledge: Knowledge,
    dialect: SqlDialect,
    fact_table: str,
    metrics: Sequence[MetricDef],
    filters: Mapping[str, str],
    period_a: Period,
    period_b: Period,
    group_dimension: str | None = None,
) -> str:
    """一条 SQL 同时算出两个窗口(可选:按某维度分组)的多个可加指标。

    两个窗口合并成一条查询,是为了省 warehouse 配额:
    Free Edition 只有一个 2X-Small,每多一次扫描都算在预算里。
    """
    builder = FromBuilder(knowledge, dialect, fact_table)
    dt_col = knowledge.table(fact_table).partition_column
    assert dt_col is not None
    dt = builder.fact_column(dt_col)

    pred_a = _period_predicate(dt, period_a, dialect)
    pred_b = _period_predicate(dt, period_b, dialect)
    period_expr = f"CASE WHEN {pred_a} THEN 'A' WHEN {pred_b} THEN 'B' END"

    select_items = [f"{period_expr} AS period"]
    group_by = ["1"]
    if group_dimension is not None:
        select_items.append(f"{builder.bucket_expr(group_dimension)} AS dim_value")
        group_by.append("2")
    for metric in metrics:
        select_items.append(f"{metric.sql} AS {metric.name}")

    # 过滤条件有两个来源:
    #   1. 指标 YAML 的 filters_default(键是事实表的**列名**,如 order_status)
    #      —— knowledge 已保证同表基础指标的默认过滤完全一致;
    #   2. 调用方传入的 filters(键是**维度名**)。
    # 若调用方的键恰好命中某个默认过滤的列名,视为显式覆盖该列。
    defaults: dict[str, str] = {}
    for metric in metrics:
        defaults.update(metric.filters_default)

    where = [f"(({pred_a}) OR ({pred_b}))"]
    for column in sorted(defaults):
        value = filters.get(column, defaults[column])
        where.append(f"{builder.fact_column(column)} = {SqlDialect.string_literal(value)}")
    for dim_name in sorted(filters):
        if dim_name in defaults:
            continue  # 已作为列覆盖处理
        expr, _ = builder.dimension_column(dim_name)
        where.append(f"{expr} = {SqlDialect.string_literal(filters[dim_name])}")

    # render() 必须在所有 dimension_column 调用之后 —— join 计划此时才完整
    from_clause = builder.render()
    return (
        "SELECT\n  "
        + ",\n  ".join(select_items)
        + "\n"
        + from_clause
        + "\nWHERE "
        + "\n  AND ".join(where)
        + "\nGROUP BY "
        + ", ".join(group_by)
    )


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------
class DecompositionEngine:
    """把知识库 + 执行器组合成可用的分解引擎。"""

    def __init__(
        self,
        knowledge: Knowledge,
        executor: SqlExecutor,
        dialect: SqlDialect,
    ) -> None:
        self.knowledge = knowledge
        self.executor = executor
        self.dialect = dialect

    # -- 内部:执行一条聚合查询 -------------------------------------------
    def _run(self, purpose: str, sql: str) -> tuple[list[dict[str, Any]], ExecutedQuery]:
        rows = self.executor.run(sql)
        return rows, ExecutedQuery.of(purpose, sql, rows, self.dialect.name)

    def _base_values(
        self,
        base_metrics: Sequence[str],
        filters: Mapping[str, str],
        period_a: Period,
        period_b: Period,
    ) -> tuple[dict[str, dict[str, float]], list[ExecutedQuery]]:
        """按表分组查询所有基础指标,返回 ``{'A'|'B': {metric: value}}``。"""
        by_table: dict[str, list[MetricDef]] = {}
        for name in base_metrics:
            metric = self.knowledge.metric(name)
            by_table.setdefault(metric.source_table, []).append(metric)  # type: ignore[arg-type]

        values: dict[str, dict[str, float]] = {"A": {}, "B": {}}
        queries: list[ExecutedQuery] = []
        for table, metrics in sorted(by_table.items()):
            sql = _build_aggregate_sql(
                self.knowledge, self.dialect, table, metrics, filters, period_a, period_b
            )
            rows, record = self._run(f"factor_base:{table}", sql)
            queries.append(record)
            seen = {row["period"] for row in rows if row["period"] is not None}
            for period_key in ("A", "B"):
                if period_key not in seen:
                    raise DecompositionError(
                        f"表 {table} 在窗口 "
                        f"{period_a if period_key == 'A' else period_b} 内没有任何数据。"
                        f"这通常不是业务问题而是数据问题(分区缺失/ETL 未跑完),"
                        f"应由观察层 L0 先行拦截。"
                    )
            for row in rows:
                key = row["period"]
                if key is None:
                    continue
                for metric in metrics:
                    values[key][metric.name] = to_float(row[metric.name])
        return values, queries

    def _metric_value(self, name: str, values: Mapping[str, float]) -> float:
        """按 YAML 的定义把基础指标组合成任意指标的值。"""
        metric = self.knowledge.metric(name)
        if metric.is_additive:
            return values[name]
        assert metric.ratio is not None
        denominator = self._metric_value(metric.ratio.denominator, values)
        if abs(denominator) < _EPS:
            raise DecompositionError(
                f"指标 {name} 的分母 {metric.ratio.denominator} 为 0,比率无定义"
            )
        return self._metric_value(metric.ratio.numerator, values) / denominator

    # -- 乘法(因子)分解 --------------------------------------------------
    def decompose_factors(
        self,
        metric: str,
        period_a: Period,
        period_b: Period,
        filters: Mapping[str, str] | None = None,
        *,
        allow_unequal_periods: bool = False,
    ) -> FactorDecomposition:
        """按指标树里声明的乘法因子分解两个窗口之间的变化。

        Args:
            metric: 指标名,其 YAML 必须含 ``decomposition.type: multiplicative``。
            period_a: **基期**(baseline)。
            period_b: **当期**(current)。变化 = B − A。
            filters: 额外的等值过滤,键为维度名。该维度必须在本次分解涉及的
                **每一张**事实表上都可用,否则抛 :class:`FilterNotSupportedError` ——
                否则 UV 与 GMV 会落在不同口径上,恒等式失效。
            allow_unequal_periods: 允许两个窗口天数不同(默认拒绝)。

        Returns:
            :class:`FactorDecomposition`。其 ``residual_pct`` 在数学上恒为 0,
            实测为浮点误差量级;``identity_gap_pct`` 则会暴露知识库口径错误。

        Raises:
            DecompositionError: 指标未声明乘法分解。
            FilterNotSupportedError: 过滤维度在某张涉及的表上不可用。
            NonPositiveValueError: 任一因子在任一窗口内非正(对数无定义)。
            PeriodError: 窗口重叠或长度不一致。
        """
        filters = dict(filters or {})
        _check_periods(period_a, period_b, allow_unequal_periods)

        definition = self.knowledge.metric(metric)
        if definition.decomposition is None:
            raise DecompositionError(
                f"指标 {metric} 未声明乘法分解;"
                f"可分解的指标:"
                f"{sorted(m for m, d in self.knowledge.metrics.items() if d.decomposition)}"
            )
        factors = list(definition.decomposition.factors)

        involved_tables = set(self.knowledge.tables_of(metric))
        for factor in factors:
            involved_tables |= self.knowledge.tables_of(factor)
        self._check_filters_alignment(filters, involved_tables)

        base_metrics = sorted(
            set(self.knowledge.base_metrics_of(metric)).union(
                *(set(self.knowledge.base_metrics_of(f)) for f in factors)
            )
        )
        values, queries = self._base_values(base_metrics, filters, period_a, period_b)

        value_a = self._metric_value(metric, values["A"])
        value_b = self._metric_value(metric, values["B"])
        _require_positive(metric, value_a, value_b)

        log_total = math.log(value_b) - math.log(value_a)
        pct_change = value_b / value_a - 1.0
        shares_defined = abs(log_total) > _EPS

        contributions: list[FactorContribution] = []
        identity_a, identity_b = 1.0, 1.0
        for factor in factors:
            fa = self._metric_value(factor, values["A"])
            fb = self._metric_value(factor, values["B"])
            _require_positive(factor, fa, fb)
            identity_a *= fa
            identity_b *= fb
            log_delta = math.log(fb) - math.log(fa)
            share = log_delta / log_total if shares_defined else None
            contributions.append(
                FactorContribution(
                    factor=factor,
                    display_name=self.knowledge.metric(factor).display_name,
                    value_a=fa,
                    value_b=fb,
                    pct_change=fb / fa - 1.0,
                    log_delta=log_delta,
                    share_of_change=share,
                    contribution_pct_points=None if share is None else share * pct_change,
                    isolated_pct_change=math.expm1(log_delta),
                )
            )

        residual_log = log_total - sum(c.log_delta for c in contributions)
        residual_pct = (
            residual_log / abs(log_total) * 100.0 if shares_defined else 0.0
        )
        identity_gap_pct = max(
            _relative_gap(identity_a, value_a), _relative_gap(identity_b, value_b)
        )

        return FactorDecomposition(
            metric=metric,
            period_a=period_a,
            period_b=period_b,
            filters=filters,
            value_a=value_a,
            value_b=value_b,
            pct_change=pct_change,
            log_delta=log_total,
            factors=tuple(contributions),
            residual_log=residual_log,
            residual_pct=residual_pct,
            identity_gap_pct=identity_gap_pct,
            queries=tuple(queries),
        )

    def _check_filters_alignment(
        self, filters: Mapping[str, str], tables: Iterable[str]
    ) -> None:
        tables = sorted(tables)
        for dim_name in filters:
            dim = self.knowledge.dimension(dim_name)
            missing = [t for t in tables if dim.on(t) is None]
            if missing:
                raise FilterNotSupportedError(
                    f"过滤维度 {dim_name!r} 在 {missing} 上不可用,"
                    f"而本次分解需要同时查询 {tables}。"
                    f"若强行只在部分表上过滤,UV 与 GMV 会落在不同口径上,"
                    f"UV × CVR × AOV = GMV 立刻失效。"
                )

    # -- 加法(维度)分解 --------------------------------------------------
    def decompose_by_dimension(
        self,
        metric: str,
        dimension: str,
        period_a: Period,
        period_b: Period,
        filters: Mapping[str, str] | None = None,
        *,
        allow_unequal_periods: bool = False,
    ) -> DimensionDecomposition:
        """把可加指标的变化拆到某个维度的各个取值上。

        引擎会跑**两条**查询:一条不带 join 直接算总量,一条按维度分组。
        两者之差就是残差 —— 这样 join 造成的丢行/放大才可能被发现。
        若只跑分组查询再自行求和,残差恒为 0,信号也就消失了。

        Args:
            metric: **可加**指标(有 ``sql`` + ``source_table``)。
                比率指标会抛 :class:`MetricNotAdditiveError`。
            dimension: 维度名,须在该指标的 ``dimensions`` 白名单内。
            period_a: 基期。
            period_b: 当期。
            filters: 额外等值过滤,键为维度名。
            allow_unequal_periods: 允许两窗口天数不同(默认拒绝)。

        Returns:
            :class:`DimensionDecomposition`,``parts`` 按 ``|delta|`` 降序。

        Raises:
            MetricNotAdditiveError: 指标是比率型。
            DimensionNotFoundError: 维度不存在或不在该指标白名单内。
            PeriodError: 窗口重叠或长度不一致。
        """
        filters = dict(filters or {})
        _check_periods(period_a, period_b, allow_unequal_periods)

        definition = self.knowledge.metric(metric)
        if not definition.is_additive:
            raise MetricNotAdditiveError(
                f"指标 {metric} 是比率型(由 {definition.ratio} 定义),不可按维度做加法分解:"
                f"各分组的比率之和不等于总体比率。"
                f"请改为分解它的分子/分母,或等 V1+ 的 mix/rate 拆分。"
            )
        if dimension not in definition.dimensions:
            raise DimensionNotFoundError(
                f"维度 {dimension!r} 不在指标 {metric} 的白名单内;"
                f"允许的维度(按优先级):{self.knowledge.ordered_dimensions(metric)}"
            )
        table = definition.source_table
        assert table is not None

        total_sql = _build_aggregate_sql(
            self.knowledge, self.dialect, table, [definition], filters, period_a, period_b
        )
        total_rows, total_q = self._run(f"dimension_total:{table}", total_sql)

        parts_sql = _build_aggregate_sql(
            self.knowledge,
            self.dialect,
            table,
            [definition],
            filters,
            period_a,
            period_b,
            group_dimension=dimension,
        )
        part_rows, parts_q = self._run(f"dimension_parts:{table}.{dimension}", parts_sql)

        totals = {"A": 0.0, "B": 0.0}
        for row in total_rows:
            if row["period"] is not None:
                totals[row["period"]] = to_float(row[metric])

        per_value: dict[str, dict[str, float]] = {}
        for row in part_rows:
            if row["period"] is None:
                continue
            bucket = per_value.setdefault(str(row["dim_value"]), {"A": 0.0, "B": 0.0})
            bucket[row["period"]] = to_float(row[metric])

        value_a, value_b = totals["A"], totals["B"]
        delta = value_b - value_a
        pct_change = value_b / value_a - 1.0 if abs(value_a) > _EPS else None
        change_defined = abs(delta) > _EPS

        parts = [
            DimensionContribution(
                value=name,
                value_a=vals["A"],
                value_b=vals["B"],
                delta=vals["B"] - vals["A"],
                share_of_change=(vals["B"] - vals["A"]) / delta if change_defined else None,
                contribution_pct_points=(
                    (vals["B"] - vals["A"]) / value_a * 100.0 if abs(value_a) > _EPS else 0.0
                ),
            )
            for name, vals in per_value.items()
        ]
        parts.sort(key=lambda p: (-abs(p.delta), p.value))

        sum_a = sum(p.value_a for p in parts)
        sum_b = sum(p.value_b for p in parts)
        residual_abs = delta - sum(p.delta for p in parts)
        denominator = abs(delta) if change_defined else max(abs(value_a), _EPS)
        residual_pct = residual_abs / denominator * 100.0

        return DimensionDecomposition(
            metric=metric,
            dimension=dimension,
            period_a=period_a,
            period_b=period_b,
            filters=filters,
            value_a=value_a,
            value_b=value_b,
            delta=delta,
            pct_change=pct_change,
            parts=tuple(parts),
            residual_abs=residual_abs,
            residual_pct=residual_pct,
            level_residual_a_pct=_relative_gap(sum_a, value_a),
            level_residual_b_pct=_relative_gap(sum_b, value_b),
            queries=(total_q, parts_q),
        )


# ---------------------------------------------------------------------------
# 模块级函数(简报 V0 验收标准 3 要求的两个函数)
# ---------------------------------------------------------------------------
def decompose_factors(
    engine: DecompositionEngine,
    metric: str,
    period_a: Period,
    period_b: Period,
    filters: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> FactorDecomposition:
    """见 :meth:`DecompositionEngine.decompose_factors`。"""
    return engine.decompose_factors(metric, period_a, period_b, filters, **kwargs)


def decompose_by_dimension(
    engine: DecompositionEngine,
    metric: str,
    dimension: str,
    period_a: Period,
    period_b: Period,
    filters: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> DimensionDecomposition:
    """见 :meth:`DecompositionEngine.decompose_by_dimension`。"""
    return engine.decompose_by_dimension(
        metric, dimension, period_a, period_b, filters, **kwargs
    )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _require_positive(name: str, value_a: float, value_b: float) -> None:
    if value_a <= 0 or value_b <= 0:
        raise NonPositiveValueError(
            f"{name} 在窗口内取值非正(基期 {value_a}, 当期 {value_b}),对数分解无定义。"
            f"通常意味着该切片在某个窗口内没有数据 —— 这是数据问题,不是业务问题。"
        )


def _relative_gap(actual: float, expected: float) -> float:
    """``|actual − expected| / |expected|`` 的百分数。"""
    if abs(expected) < _EPS:
        return 0.0 if abs(actual) < _EPS else float("inf")
    return abs(actual - expected) / abs(expected) * 100.0


def residual_severity(residual_pct: float) -> str:
    """把残差映射到简报 §3.2 的三档判定。

    这是**确定性**判定(观察层 L2),不经过 LLM。

    简报给的三档是 ``< 1%`` / ``5-20%`` / ``> 20%``,中间的 1~5% 没有定义。
    这里把它并入 ``structural``(偏保守:宁可多查一个维度,
    也不要把一个真实的结构变化当成「解释干净了」而提前收敛)。
    若 V1 的评估显示这段区间误报过多,调整点就在这一个函数里。
    """
    magnitude = abs(residual_pct)
    if magnitude < 1.0:
        return "clean"        # 这一层解释干净,可继续向下钻取
    if magnitude <= 20.0:
        return "structural"   # 有维度未覆盖,或存在 mix effect
    return "data_quality"     # 大概率是数据问题,不是业务问题
