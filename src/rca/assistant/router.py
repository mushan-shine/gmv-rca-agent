"""问题分流:这道题该「查数」还是该「归因」。

两条路径的性质完全不同::

    查数   「上周美国站 GMV 多少」     → text-to-SQL 自修复循环(LLM 写 SQL,守卫把关)
    归因   「上周 GMV 为什么跌了」     → 确定性分解内核(不经 LLM,残差可验证)

归因刻意**不交给 LLM 写 SQL**:它要的是一组互相勾稽的数字(各因子贡献之和 = 总变化),
LLM 写出来的多条 SQL 口径稍有不齐,拼出来的解释就对不上账。分解内核保证闭合,
LLM 在这条路径上只需要做一件它擅长的事 —— 什么都不做。叙述也是模板生成的。

分流本身也是确定性的:关键词 + 业务词表(``knowledge/assistant/synonyms.yaml``)。
规则能判的不交给模型;判错的题会进待复核队列,由人把新说法补进词表。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..decompose import Period
from ..knowledge import Knowledge
from ..nl2sql.domains import ValueDomains

INTENT_EXPLAIN = "explain"
INTENT_QUERY = "query"
INTENT_HELP = "help"

GRANULARITIES = ("week", "day")


class Synonyms(BaseModel):
    """``knowledge/assistant/synonyms.yaml`` 的结构。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    explain_keywords: list[str]
    metrics: dict[str, list[str]]
    default_metric: str
    values: dict[str, dict[str, str]] = Field(default_factory=dict)
    periods: dict[str, list[str]] = Field(default_factory=dict)


def load_synonyms(path: str | Path) -> Synonyms:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Synonyms.model_validate(payload)


@dataclass(frozen=True)
class Route:
    """分流结果,连同本次采用的**全部假设** —— 回答里要原样展示给提问人。"""

    intent: str
    metric: str = ""
    granularity: str = "week"
    filters: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def _contains(text: str, phrase: str, *, case_sensitive: bool = False) -> bool:
    """中文按子串匹配;含拉丁字母的按词边界匹配(「us」不能命中「focus」)。"""
    if not re.search(r"[A-Za-z0-9]", phrase):
        return phrase in text
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(phrase)}(?![A-Za-z0-9_])", text, flags) is not None


@dataclass
class Router:
    knowledge: Knowledge
    synonyms: Synonyms
    domains: ValueDomains | None = None

    def route(self, question: str) -> Route:
        text = (question or "").strip()
        if not text:
            return Route(INTENT_HELP)
        if not any(_contains(text, keyword) for keyword in self.synonyms.explain_keywords):
            return Route(INTENT_QUERY)

        notes: list[str] = []
        metric = self._metric(text)
        if metric is None:
            metric = self.synonyms.default_metric
            notes.append(f"问题里没有点名指标,按默认指标 {metric.upper()} 归因")

        granularity = self._granularity(text)
        if granularity is None:
            granularity = "week"
            notes.append("问题里没有识别出时间范围(目前支持「上周」「昨天」),按「上周 vs 前一周」对比")
        filters, filter_notes = self._filters(text, metric)
        notes += filter_notes
        return Route(INTENT_EXPLAIN, metric, granularity, filters, tuple(notes))

    # -- 槽位 --------------------------------------------------------------
    def _metric(self, text: str) -> str | None:
        for metric, phrases in self.synonyms.metrics.items():
            if metric in self.knowledge.metrics and any(_contains(text, p) for p in phrases):
                return metric
        return None

    def _granularity(self, text: str) -> str | None:
        for granularity in GRANULARITIES:
            if any(_contains(text, p) for p in self.synonyms.periods.get(granularity, [])):
                return granularity
        return None

    def filterable_dimensions(self, metric: str) -> list[str]:
        """能用来过滤这个指标的维度:在它涉及的**每一张**事实表上都是原生列。

        GMV = UV × CVR × AOV 横跨 fact_orders 与 fact_sessions。只能在一边过滤的维度
        (比如要 join 商品表的品类)会让 UV 和 GMV 落在不同口径上,分解就不成立了。
        """
        tables = set(self.knowledge.tables_of(metric))
        definition = self.knowledge.metric(metric)
        if definition.decomposition is not None:
            for factor in definition.decomposition.factors:
                tables |= self.knowledge.tables_of(factor)
        return [
            name for name, dim in self.knowledge.dimensions.items()
            if all(dim.is_native_on(table) for table in tables)
        ]

    def _filters(self, text: str, metric: str) -> tuple[dict[str, str], list[str]]:
        found: dict[str, set[str]] = {}
        for dimension in self.filterable_dimensions(metric):
            allowed = self.domains.allowed(dimension) if self.domains is not None else frozenset()
            for spoken, value in self.synonyms.values.get(dimension, {}).items():
                if (not allowed or value in allowed) and _contains(text, spoken):
                    found.setdefault(dimension, set()).add(value)
            for value in allowed:
                # 短取值(US / DE / NA)区分大小写:「tell us why」里的 us 不是美国
                if _contains(text, value, case_sensitive=len(value) <= 3):
                    found.setdefault(dimension, set()).add(value)

        filters: dict[str, str] = {}
        notes: list[str] = []
        for dimension, values in sorted(found.items()):
            label = self.knowledge.dimension(dimension).display_name
            if len(values) == 1:
                filters[dimension] = next(iter(values))
            else:
                notes.append(
                    f"问题里出现了多个{label}({', '.join(sorted(values))}),"
                    f"本次不按{label}过滤,而是在拆分结果里分别列出"
                )
        return filters, notes


def comparison_periods(granularity: str, as_of: date) -> tuple[Period, Period]:
    """按报告日历给出(基期, 当期)。与 text-to-SQL 用的是同一套「上周」定义。"""
    if granularity == "day":
        current = as_of - timedelta(days=1)
        baseline = current - timedelta(days=1)
        return (
            Period(baseline, baseline, f"前一天 {baseline}"),
            Period(current, current, f"昨天 {current}"),
        )
    this_monday = as_of - timedelta(days=as_of.weekday())
    last_monday = this_monday - timedelta(days=7)
    prior_monday = last_monday - timedelta(days=7)
    return (
        Period(prior_monday, prior_monday + timedelta(days=6),
               f"前一周 {prior_monday}~{prior_monday + timedelta(days=6)}"),
        Period(last_monday, last_monday + timedelta(days=6),
               f"上周 {last_monday}~{last_monday + timedelta(days=6)}"),
    )
