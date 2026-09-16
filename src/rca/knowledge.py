"""知识库加载与校验。

Git 中的 YAML 是唯一真源(简报 §6)。本模块负责:

1. 加载 ``knowledge/tables.yaml``(schema 登记簿)、``knowledge/dimensions.yaml``
   (维度注册表)、``knowledge/metrics/*.yaml``(指标树);
2. **校验**:缺字段、拼错字段、引用了不存在的指标/维度/表/列、
   指标维度顺序与全局优先级不一致、以及乘法分解恒等式不成立 —— 一律抛错;
3. 向 ``decompose`` 暴露只读查询接口。

校验的价值在于:分解引擎生成的 SQL 完全由这些 YAML 驱动。
若知识库能通过校验,SQL 里就不可能出现不存在的列名;
若知识库有错,错误停在加载阶段,而不是变成报告里一个错误的数字。
"""

from __future__ import annotations

from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import (
    DimensionNotFoundError,
    KnowledgeError,
    MetricNotFoundError,
)

# 项目内约定:所有 pydantic 模型禁止未知字段(拼错的 YAML key 必须报错)且不可变。
_STRICT = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# tables.yaml
# ---------------------------------------------------------------------------
class ColumnDef(BaseModel):
    model_config = _STRICT

    type: str
    nullable: bool = True


class TableDef(BaseModel):
    model_config = _STRICT

    name: str
    kind: Literal["fact", "dim"]
    grain: str
    columns: dict[str, ColumnDef]
    partition_column: str | None = None
    primary_key: str | None = None

    @model_validator(mode="after")
    def _check_refs(self) -> "TableDef":
        for label, col in (
            ("partition_column", self.partition_column),
            ("primary_key", self.primary_key),
        ):
            if col is not None and col not in self.columns:
                raise ValueError(f"表 {self.name} 的 {label}={col!r} 不在 columns 中")
        if self.kind == "fact" and self.partition_column is None:
            raise ValueError(
                f"事实表 {self.name} 必须声明 partition_column(L0 数据质量检查依赖它)"
            )
        if self.kind == "dim" and self.primary_key is None:
            raise ValueError(f"维表 {self.name} 必须声明 primary_key(join 唯一性依赖它)")
        return self


# ---------------------------------------------------------------------------
# dimensions.yaml
# ---------------------------------------------------------------------------
class JoinPath(BaseModel):
    model_config = _STRICT

    table: str
    left_key: str
    right_key: str
    unique: bool = True


class DimensionAvailability(BaseModel):
    """某个维度在某张表上如何取到。

    ``join is None`` 表示原生列 —— 这类维度的加法分解一定闭合(残差 = 0)。
    ``join`` 不为空表示需要 LEFT JOIN 维表,残差因此变成一个**有信息量**的信号:
    负残差意味着 join fan-out(右表主键不唯一,行被放大)。
    """

    model_config = _STRICT

    column: str
    join: JoinPath | None = None


class DimensionDef(BaseModel):
    model_config = _STRICT

    name: str
    display_name: str
    priority: int
    availability: dict[str, DimensionAvailability]
    parent_of: list[str] = Field(default_factory=list)

    def on(self, table: str) -> DimensionAvailability | None:
        return self.availability.get(table)

    def is_native_on(self, table: str) -> bool:
        """该维度是否是 ``table`` 上的原生列(无需 join)。"""
        avail = self.availability.get(table)
        return avail is not None and avail.join is None


# ---------------------------------------------------------------------------
# metrics/*.yaml
# ---------------------------------------------------------------------------
class RatioDef(BaseModel):
    model_config = _STRICT

    numerator: str
    denominator: str


class DecompositionDef(BaseModel):
    model_config = _STRICT

    type: Literal["multiplicative"]
    factors: list[str]

    @model_validator(mode="after")
    def _check_factors(self) -> "DecompositionDef":
        if len(self.factors) < 2:
            raise ValueError("乘法分解至少需要 2 个因子")
        if len(set(self.factors)) != len(self.factors):
            raise ValueError("乘法分解的因子不得重复")
        return self


class FreshnessCheck(BaseModel):
    model_config = _STRICT

    table: str
    max_lag_hours: int


class CompletenessCheck(BaseModel):
    model_config = _STRICT

    table: str
    check: Literal["partition_row_count_vs_7d_avg"]
    min_ratio: float


class DataQualityChecks(BaseModel):
    model_config = _STRICT

    freshness: list[FreshnessCheck] = Field(default_factory=list)
    completeness: list[CompletenessCheck] = Field(default_factory=list)


class MetricDef(BaseModel):
    """一个指标。

    指标只有两种形态,二选一(由 ``_check_shape`` 强制):

    * **可加基础指标** —— 给出 ``sql``(聚合表达式)与 ``source_table``。
      例:``gmv = SUM(order_amount) on fact_orders``。可按维度做加法分解。
    * **比率指标** —— 给出 ``ratio.numerator`` / ``ratio.denominator``,
      二者都必须最终落到可加基础指标上。例:``cvr = orders / uv``。
      比率指标**不可**按维度加法分解。

    ``decomposition`` 独立于上述形态,描述「变化如何被拆成因子」,
    目前只支持 multiplicative(取对数后可加)。
    """

    model_config = _STRICT

    name: str
    display_name: str
    scope: list[str]
    sql: str | None = None
    source_table: str | None = None
    ratio: RatioDef | None = None
    filters_default: dict[str, str] = Field(default_factory=dict)
    decomposition: DecompositionDef | None = None
    dimensions: list[str] = Field(default_factory=list)
    data_quality_checks: DataQualityChecks = Field(default_factory=DataQualityChecks)

    @model_validator(mode="after")
    def _check_shape(self) -> "MetricDef":
        is_base = self.sql is not None and self.source_table is not None
        is_ratio = self.ratio is not None
        if (self.sql is None) != (self.source_table is None):
            raise ValueError(f"指标 {self.name}:sql 与 source_table 必须成对出现")
        if is_base == is_ratio:
            raise ValueError(
                f"指标 {self.name}:必须二选一 —— 要么同时给出 sql + source_table"
                f"(可加基础指标),要么给出 ratio(比率指标)"
            )
        if is_ratio and self.filters_default:
            raise ValueError(
                f"指标 {self.name}:比率指标不得声明 filters_default,"
                f"口径必须由分子/分母各自的基础指标决定,否则会出现两套口径"
            )
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError(f"指标 {self.name}:dimensions 不得重复")
        return self

    @property
    def is_additive(self) -> bool:
        """是否是可加基础指标(可按维度做加法分解)。"""
        return self.sql is not None


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------
class Knowledge:
    """加载并校验后的知识库。构造成功即代表全部校验通过。"""

    def __init__(
        self,
        tables: Mapping[str, TableDef],
        dimensions: Mapping[str, DimensionDef],
        metrics: Mapping[str, MetricDef],
        root: Path | None = None,
    ) -> None:
        self.tables = dict(tables)
        self.dimensions = dict(dimensions)
        self.metrics = dict(metrics)
        self.root = root
        self._validate()

    # -- 访问器 ------------------------------------------------------------
    def metric(self, name: str) -> MetricDef:
        try:
            return self.metrics[name]
        except KeyError:
            raise MetricNotFoundError(
                f"未知指标 {name!r};已知指标:{sorted(self.metrics)}"
            ) from None

    def dimension(self, name: str) -> DimensionDef:
        try:
            return self.dimensions[name]
        except KeyError:
            raise DimensionNotFoundError(
                f"未知维度 {name!r};已知维度:{sorted(self.dimensions)}"
            ) from None

    def table(self, name: str) -> TableDef:
        try:
            return self.tables[name]
        except KeyError:
            raise KnowledgeError(f"未知表 {name!r};已知表:{sorted(self.tables)}") from None

    def ordered_dimensions(self, metric: str) -> list[str]:
        """返回该指标允许钻取的维度,按全局 priority 升序。

        这是 L1 循环「下一个查哪个维度」的**默认**顺序;
        V2 接入 LLM 后,LLM 只能在这个集合里选择,不能凭空造维度。
        """
        dims = self.metric(metric).dimensions
        return sorted(dims, key=lambda d: (self.dimension(d).priority, d))

    def base_metrics_of(self, name: str) -> list[str]:
        """展开成所依赖的全部可加基础指标(去重,稳定顺序)。"""
        num, den = self._expand(name, set())
        seen: dict[str, None] = {}
        for m in sorted(num.elements()) + sorted(den.elements()):
            seen.setdefault(m, None)
        return list(seen)

    def tables_of(self, metric_name: str) -> set[str]:
        """该指标(展开到基础指标后)涉及的全部事实表。"""
        return {
            self.metric(base).source_table  # type: ignore[misc]
            for base in self.base_metrics_of(metric_name)
        }

    # -- 校验 --------------------------------------------------------------
    def _validate(self) -> None:
        self._validate_dimensions()
        self._validate_metrics()

    def _validate_dimensions(self) -> None:
        priorities = Counter(d.priority for d in self.dimensions.values())
        dupes = sorted(p for p, n in priorities.items() if n > 1)
        if dupes:
            raise KnowledgeError(
                f"dimensions.yaml:priority 重复 {dupes},钻取顺序将不确定"
            )

        for dim in self.dimensions.values():
            if not dim.availability:
                raise KnowledgeError(f"维度 {dim.name}:availability 不得为空")
            for child in dim.parent_of:
                if child not in self.dimensions:
                    raise DimensionNotFoundError(
                        f"维度 {dim.name}.parent_of 引用了不存在的维度 {child!r}"
                    )
            for table_name, avail in dim.availability.items():
                table = self.table(table_name)
                if avail.join is None:
                    if avail.column not in table.columns:
                        raise KnowledgeError(
                            f"维度 {dim.name} 声明在 {table_name}.{avail.column} 上,"
                            f"但该表没有这个列"
                        )
                    continue
                right = self.table(avail.join.table)
                if avail.join.left_key not in table.columns:
                    raise KnowledgeError(
                        f"维度 {dim.name}:join 的 left_key {avail.join.left_key!r} "
                        f"不在 {table_name} 上"
                    )
                if avail.join.right_key not in right.columns:
                    raise KnowledgeError(
                        f"维度 {dim.name}:join 的 right_key {avail.join.right_key!r} "
                        f"不在 {right.name} 上"
                    )
                if avail.column not in right.columns:
                    raise KnowledgeError(
                        f"维度 {dim.name}:列 {avail.column!r} 不在被 join 的表 {right.name} 上"
                    )
                if avail.join.unique and right.primary_key != avail.join.right_key:
                    raise KnowledgeError(
                        f"维度 {dim.name}:join 声明 unique=true,但 {right.name} 的主键是 "
                        f"{right.primary_key!r} 而非 {avail.join.right_key!r} —— 会发生 fan-out"
                    )

    def _validate_metrics(self) -> None:
        # 先做环检测,后续展开才安全
        for name in self.metrics:
            self._expand(name, set())

        for metric in self.metrics.values():
            if metric.is_additive:
                table = self.table(metric.source_table)  # type: ignore[arg-type]
                for col in metric.filters_default:
                    if col not in table.columns:
                        raise KnowledgeError(
                            f"指标 {metric.name}:filters_default 引用了 "
                            f"{table.name} 上不存在的列 {col!r}"
                        )
            else:
                assert metric.ratio is not None
                for role, ref in (
                    ("numerator", metric.ratio.numerator),
                    ("denominator", metric.ratio.denominator),
                ):
                    if ref not in self.metrics:
                        raise MetricNotFoundError(
                            f"指标 {metric.name}.ratio.{role} 引用了不存在的指标 {ref!r}"
                        )

            # 维度存在性 + 在所有依赖表上可用 + 顺序与全局 priority 一致
            for dim_name in metric.dimensions:
                dim = self.dimension(dim_name)
                for table_name in self.tables_of(metric.name):
                    if dim.on(table_name) is None:
                        raise KnowledgeError(
                            f"指标 {metric.name}:维度 {dim_name!r} 在其依赖的表 "
                            f"{table_name} 上不可用"
                        )
            expected = sorted(
                metric.dimensions, key=lambda d: (self.dimension(d).priority, d)
            )
            if metric.dimensions != expected:
                raise KnowledgeError(
                    f"指标 {metric.name}:dimensions 顺序必须与 dimensions.yaml 的 "
                    f"priority 一致。期望 {expected},实际 {metric.dimensions}"
                )

            for check in (
                *metric.data_quality_checks.freshness,
                *metric.data_quality_checks.completeness,
            ):
                self.table(check.table)

            if metric.decomposition is not None:
                self._validate_multiplicative(metric)

    def _validate_multiplicative(self, metric: MetricDef) -> None:
        """符号化验证乘法分解恒等式确实成立。

        把每个因子展开成「可加基础指标的分子/分母多重集」,连乘后约分,
        结果必须与被分解指标自身的展开完全相同。

        以 GMV = UV × CVR × AOV 为例::

            uv  -> ({uv}, {})
            cvr -> ({orders}, {uv})
            aov -> ({gmv}, {orders})
            连乘 -> ({uv, orders, gmv}, {uv, orders}) --约分--> ({gmv}, {})
            gmv -> ({gmv}, {})                                    ✓ 相等

        任何一个因子写错(比如把 aov 误定义成 gmv/uv),约分后都对不上,加载即失败。
        这条校验保证了简报 §3.2「残差是唯一确定性信号」的前提:分解公式本身不可能错。
        """
        assert metric.decomposition is not None
        num: Counter[str] = Counter()
        den: Counter[str] = Counter()
        for factor in metric.decomposition.factors:
            if factor not in self.metrics:
                raise MetricNotFoundError(
                    f"指标 {metric.name}.decomposition.factors 引用了不存在的指标 {factor!r}"
                )
            f_num, f_den = self._expand(factor, set())
            num += f_num
            den += f_den
        num, den = _cancel(num, den)
        want_num, want_den = _cancel(*self._expand(metric.name, set()))
        if num != want_num or den != want_den:
            raise KnowledgeError(
                f"指标 {metric.name}:乘法分解恒等式不成立。因子连乘约分后得到 "
                f"{_fmt(num, den)},而 {metric.name} 本身是 {_fmt(want_num, want_den)}"
            )

        # 同一张表上的基础指标必须共用同一套默认过滤,否则因子口径互相打架。
        # 例:orders 只统计 COMPLETED 而 gmv 统计全部状态时,AOV = GMV/orders
        # 得到的是一个既非「完成单均价」也非「全单均价」的混合物。
        all_bases = set(self.base_metrics_of(metric.name))
        for factor in metric.decomposition.factors:
            all_bases |= set(self.base_metrics_of(factor))
        by_table: dict[str, dict[str, dict[str, str]]] = {}
        for base in sorted(all_bases):
            bm = self.metric(base)
            by_table.setdefault(bm.source_table, {})[base] = bm.filters_default  # type: ignore[index]
        for table_name, per_metric in by_table.items():
            distinct = {tuple(sorted(f.items())) for f in per_metric.values()}
            if len(distinct) > 1:
                raise KnowledgeError(
                    f"指标 {metric.name}:基础指标 {sorted(per_metric)} 同出自 "
                    f"{table_name} 却使用了不同的 filters_default {per_metric} —— "
                    f"因子口径不一致,分解结果无法解释"
                )

    def _expand(self, name: str, stack: set[str]) -> tuple[Counter[str], Counter[str]]:
        """把指标展开成 (分子多重集, 分母多重集),元素均为可加基础指标名。"""
        if name in stack:
            raise KnowledgeError(f"指标定义存在环:{sorted(stack)} -> {name}")
        metric = self.metric(name)
        if metric.is_additive:
            return Counter([name]), Counter()
        assert metric.ratio is not None
        inner = stack | {name}
        n_num, n_den = self._expand(metric.ratio.numerator, inner)
        d_num, d_den = self._expand(metric.ratio.denominator, inner)
        return n_num + d_den, n_den + d_num

    @cached_property
    def dimensions_by_priority(self) -> list[DimensionDef]:
        return sorted(self.dimensions.values(), key=lambda d: d.priority)


def _cancel(num: Counter[str], den: Counter[str]) -> tuple[Counter[str], Counter[str]]:
    """约去分子分母中相同的项。"""
    common = num & den
    return num - common, den - common


def _fmt(num: Counter[str], den: Counter[str]) -> str:
    def side(c: Counter[str]) -> str:
        return "·".join(sorted(c.elements())) or "1"

    return f"{side(num)} / {side(den)}"


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        raise KnowledgeError(f"知识库文件不存在:{path}") from None
    except yaml.YAMLError as exc:
        raise KnowledgeError(f"YAML 解析失败 {path}: {exc}") from None


def _build(model: type[BaseModel], payload: Any, where: str):
    if not isinstance(payload, dict):
        raise KnowledgeError(
            f"{where}:期望一个 mapping,实际是 {type(payload).__name__}"
        )
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise KnowledgeError(f"{where} 校验失败:\n{exc}") from None


def load_knowledge(root: str | Path = "knowledge") -> Knowledge:
    """从 ``root`` 目录加载并校验整个知识库。

    Args:
        root: ``knowledge/`` 目录路径,需包含 ``tables.yaml``、``dimensions.yaml``
            和 ``metrics/*.yaml``。

    Returns:
        校验通过的 :class:`Knowledge`。

    Raises:
        KnowledgeError: 任何一条校验失败;错误信息指明具体文件与具体项。
    """
    root = Path(root)

    tables_payload = _read_yaml(root / "tables.yaml")
    if not isinstance(tables_payload, dict) or "tables" not in tables_payload:
        raise KnowledgeError("tables.yaml 顶层必须是 {tables: [...]}")
    tables: dict[str, TableDef] = {}
    for item in tables_payload["tables"] or []:
        name = item.get("name", "?") if isinstance(item, dict) else "?"
        t = _build(TableDef, item, f"tables.yaml/{name}")
        if t.name in tables:
            raise KnowledgeError(f"tables.yaml:表名重复 {t.name!r}")
        tables[t.name] = t

    dims_payload = _read_yaml(root / "dimensions.yaml")
    if not isinstance(dims_payload, dict) or "dimensions" not in dims_payload:
        raise KnowledgeError("dimensions.yaml 顶层必须是 {dimensions: [...]}")
    dimensions: dict[str, DimensionDef] = {}
    for item in dims_payload["dimensions"] or []:
        name = item.get("name", "?") if isinstance(item, dict) else "?"
        d = _build(DimensionDef, item, f"dimensions.yaml/{name}")
        if d.name in dimensions:
            raise KnowledgeError(f"dimensions.yaml:维度名重复 {d.name!r}")
        dimensions[d.name] = d

    metrics_dir = root / "metrics"
    if not metrics_dir.is_dir():
        raise KnowledgeError(f"指标目录不存在:{metrics_dir}")
    metrics: dict[str, MetricDef] = {}
    for path in sorted(metrics_dir.glob("*.yaml")):
        m = _build(MetricDef, _read_yaml(path), f"metrics/{path.name}")
        if m.name != path.stem:
            raise KnowledgeError(
                f"metrics/{path.name}:文件名与 name={m.name!r} 不一致"
            )
        metrics[m.name] = m
    if not metrics:
        raise KnowledgeError(f"{metrics_dir} 下没有任何指标定义")

    return Knowledge(tables=tables, dimensions=dimensions, metrics=metrics, root=root)
