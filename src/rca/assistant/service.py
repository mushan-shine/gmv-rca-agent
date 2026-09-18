"""问答助手:一个问题进来,一个可核对的回答出去。

这是把离线系统接到线上的那一层。它本身几乎不含新逻辑,只做装配::

    问题 → Router 分流
             ├─ 查数 → AnswerLoop(L1 自修复,用的是 04 采纳后合并进 prompting.yaml 的提示)
             └─ 归因 → DecompositionEngine(因子分解 + 各维度拆分,确定性,不经 LLM)
         → 模板叙述(数字全部来自查询结果,没有一个是模型写的)
         → 写进 assistant_requests;👍👎 写进 assistant_feedback

**线上日志就是下一轮评估的原料。** 失败的、被点了 👎 的问题进入待复核队列
(:meth:`StateStore.review_queue`),人确认正确口径后加进 ``eval/cases.yaml`` ——
评估集从线上真实问题里长大,04 的改进循环就有了更多、更真的题可以练。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from ..decompose import (
    NULL_BUCKET,
    UNKNOWN_BUCKET,
    DecompositionEngine,
    DimensionDecomposition,
    FactorDecomposition,
)
from ..dialects import SqlDialect
from ..errors import RcaError
from ..knowledge import Knowledge, load_knowledge
from ..nl2sql.domains import ValueDomains, load_value_domains
from ..nl2sql.generate import SqlGenerator
from ..nl2sql.knowledge_store import load_prompt_knowledge
from ..nl2sql.loop import AnswerLoop, Status
from ..nl2sql.state import StateStore
from ..warehouse import SqlExecutor
from .router import (
    INTENT_EXPLAIN,
    INTENT_HELP,
    INTENT_QUERY,
    Route,
    Router,
    comparison_periods,
    load_synonyms,
)

MAX_DRIVERS = 5
TOP_PER_DIMENSION = 3

EXAMPLES = (
    "上周 GMV 为什么变化了?",
    "上周美国站 GMV 为什么下降?",
    "昨天移动端订单量为什么变了?",
    "上周各渠道的 GMV 是多少?",
    "上周美国站的转化率是多少?",
)


@dataclass(frozen=True)
class Answer:
    request_id: str
    question: str
    intent: str
    status: str
    """``answered`` / ``refused`` / ``failed`` / ``unsupported`` / ``error`` / ``help``。"""

    text: str
    """给人看的结论(Markdown)。数字全部来自查询结果。"""

    tables: Mapping[str, list[dict[str, Any]]] = field(default_factory=dict)
    sql: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    """本次回答采用的假设(用的哪两周、按什么过滤)。必须展示给提问人。"""

    attempts: int = 0
    model: str = ""
    tokens: int = 0
    latency_ms: int = 0
    detail: str = ""
    """失败原因等排查信息。"""

    logged: bool = False


@dataclass
class Assistant:
    knowledge: Knowledge
    dialect: SqlDialect
    executor: SqlExecutor
    router: Router
    as_of: date
    loop: AnswerLoop | None = None
    """查数用的循环。为空表示没接 LLM:归因照常可用,查数会明确告知。"""

    store: StateStore | None = None
    knowledge_version: str = ""
    engine: DecompositionEngine | None = None

    def __post_init__(self) -> None:
        if self.engine is None:
            self.engine = DecompositionEngine(self.knowledge, self.executor, self.dialect)

    # -- 入口 --------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        route = self.router.route(question)
        try:
            if route.intent == INTENT_HELP:
                answer = self._help(request_id, question)
            elif route.intent == INTENT_EXPLAIN:
                answer = self._explain(request_id, question, route)
            else:
                answer = self._query(request_id, question)
        except RcaError as exc:
            answer = Answer(
                request_id, question, route.intent, "error",
                f"这个问题没能算出来:{exc}", notes=route.notes, detail=str(exc)[:500],
            )
        answer = _with(answer, latency_ms=int((time.perf_counter() - started) * 1000))
        return _with(answer, logged=self._log(answer))

    def feedback(self, request_id: str, rating: int, comment: str = "") -> bool:
        """记录 👍(1)/ 👎(-1)。👎 的问题会进入待复核队列。"""
        if self.store is None:
            return False
        self.store.record_feedback(request_id, 1 if rating > 0 else -1, comment)
        return True

    # -- 归因 --------------------------------------------------------------
    def _explain(self, request_id: str, question: str, route: Route) -> Answer:
        metric = self.knowledge.metric(route.metric)
        name = route.metric.upper()
        baseline, current = comparison_periods(route.granularity, self.as_of)
        notes = [
            f"对比窗口:{current.label} vs {baseline.label}(报告日历,数据截至 {self.as_of})",
            *(f"过滤:{self.knowledge.dimension(d).display_name} = {v}" for d, v in route.filters.items()),
            *route.notes,
        ]
        if not metric.is_additive and metric.decomposition is None:
            return Answer(
                request_id, question, INTENT_EXPLAIN, "unsupported",
                f"{name}({metric.display_name})是比率指标,不能按维度相加拆分 ——"
                f"各分组的比率加起来不等于总体比率(需要 mix/rate 拆分,尚未实现)。\n\n"
                f"可以改问「GMV 为什么变化」:{name} 会作为 GMV 的一个因子出现在结果里。",
                notes=tuple(notes),
            )

        engine = self.engine
        assert engine is not None
        factor: FactorDecomposition | None = None
        if metric.decomposition is not None:
            factor = engine.decompose_factors(route.metric, baseline, current, route.filters)

        dimensions: list[DimensionDecomposition] = []
        if metric.is_additive:
            for dimension in self.knowledge.ordered_dimensions(route.metric):
                if dimension in route.filters:
                    continue
                result = engine.decompose_by_dimension(
                    route.metric, dimension, baseline, current, route.filters
                )
                if sum(1 for p in result.parts if p.value not in (UNKNOWN_BUCKET, NULL_BUCKET)) > 1:
                    dimensions.append(result)   # 过滤后只剩一个取值的维度(如只看 US 时的 region)没有信息量

        value_a = factor.value_a if factor else dimensions[0].value_a if dimensions else 0.0
        value_b = factor.value_b if factor else dimensions[0].value_b if dimensions else 0.0
        pct = (value_b / value_a - 1.0) if value_a else 0.0
        direction = "上升" if pct > 0 else "下降" if pct < 0 else "持平"
        scope = "、".join(f"{v}" for v in route.filters.values())
        lines = [
            f"**{name}{f'({scope})' if scope else ''} {current.label} 为 {_num(value_b)},"
            f"较{baseline.label.split(' ')[0]}{direction} {abs(pct):.1%}({_signed(value_b - value_a)})。**",
            "",
        ]
        tables: dict[str, list[dict[str, Any]]] = {}
        sql: list[str] = []

        if abs(pct) < 1e-9:
            lines.append("两期几乎没有变化,不做归因。")
        if factor is not None and factor.factors and factor.factors[0].contribution_pct_points is not None:
            parts = " · ".join(
                f"{_factor_label(self.knowledge, f.factor)} {f.contribution_pct_points * 100:+.1f}"
                for f in factor.factors
            )
            lines.append(f"- **因子**(单位:百分点,相加 = {pct * 100:+.1f}):{parts}")
            tables["因子分解"] = [
                {
                    "因子": _factor_label(self.knowledge, f.factor),
                    baseline.label.split(" ")[0]: round(f.value_a, 6),
                    current.label.split(" ")[0]: round(f.value_b, 6),
                    "自身变化": f"{f.pct_change:+.2%}",
                    "贡献(百分点)": round((f.contribution_pct_points or 0.0) * 100, 2),
                }
                for f in factor.factors
            ]
            sql += [q.sql for q in factor.queries]

        drivers = _top_drivers(self.knowledge, dimensions)
        if drivers:
            lines.append(
                "- **维度**(各维度分别拆分,不同维度之间不能相加):贡献最大的是 "
                + "、".join(f"{d['维度']} = {d['取值']}({d['贡献(百分点)']:+.1f})" for d in drivers[:3])
            )
            tables["维度贡献 Top"] = drivers
            tables["各维度明细"] = [
                {
                    "维度": self.knowledge.dimension(result.dimension).display_name,
                    "取值": part.value,
                    baseline.label.split(" ")[0]: round(part.value_a, 2),
                    current.label.split(" ")[0]: round(part.value_b, 2),
                    "变化": round(part.delta, 2),
                    "贡献(百分点)": round(part.contribution_pct_points, 2),   # 维度分解已是百分点
                }
                for result in dimensions
                for part in result.parts[:TOP_PER_DIMENSION]
            ]
            sql += [q.sql for result in dimensions for q in result.queries]

        residuals = [abs(factor.residual_pct)] if factor else []
        residuals += [abs(result.residual_pct) for result in dimensions]
        if residuals:
            worst = max(residuals)
            verdict = "闭合" if worst < 0.1 else "**未闭合,结论不可信,请联系数据团队**"
            lines.append(f"- **可信度**:分解{verdict}(最大残差 {worst:.2e}%)")

        return Answer(
            request_id, question, INTENT_EXPLAIN, "answered", "\n".join(lines),
            tables=tables, sql=tuple(sql), notes=tuple(notes),
        )

    # -- 查数 --------------------------------------------------------------
    def _query(self, request_id: str, question: str) -> Answer:
        notes = (f"「上周」「昨天」按报告日历解释,数据截至 {self.as_of}",)
        if self.loop is None:
            return Answer(
                request_id, question, INTENT_QUERY, "unsupported",
                "查数需要接入 LLM,当前没有配置。归因类问题(「……为什么……」)不需要 LLM,可以直接问。",
                notes=notes,
            )
        outcome = self.loop.answer(question)
        common = dict(
            sql=(outcome.final_sql,) if outcome.final_sql else (),
            notes=notes, attempts=outcome.attempts_used,
            model=outcome.model, tokens=outcome.total_tokens,
        )
        if outcome.status is Status.REFUSED:
            return Answer(
                request_id, question, INTENT_QUERY, "refused",
                f"这个问题用现有数据答不了:{outcome.refusal_reason or '数据里没有相关字段'}",
                **common,
            )
        if outcome.status is Status.FAILED:
            last = outcome.attempts[-1] if outcome.attempts else None
            reason = ""
            if last is not None:
                parts = [*last.guard_messages, *last.context_messages,
                         last.execution_error, last.generation_error]
                reason = " | ".join(part for part in parts if part)
            return Answer(
                request_id, question, INTENT_QUERY, "failed",
                f"试了 {outcome.attempts_used} 次都没写出能执行的 SQL,已记入待复核队列。",
                detail=reason[:500], **common,
            )
        rows = [dict(row) for row in outcome.rows]
        if len(rows) == 1 and len(rows[0]) == 1:
            value = next(iter(rows[0].values()))
            numeric = isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
            text = f"**{_num(float(value)) if numeric else value}**"
        elif rows:
            text = f"共 {len(rows)} 行,见下表。"
        else:
            text = "查询成功,但没有符合条件的数据。"
        return Answer(
            request_id, question, INTENT_QUERY, "answered", text,
            tables={"结果": rows} if rows else {}, **common,
        )

    def _help(self, request_id: str, question: str) -> Answer:
        return Answer(
            request_id, question, INTENT_HELP, "help",
            "可以这样问:\n" + "\n".join(f"- {example}" for example in EXAMPLES),
        )

    # -- 日志 --------------------------------------------------------------
    def _log(self, answer: Answer) -> bool:
        if self.store is None or answer.intent == INTENT_HELP:
            return False
        try:
            self.store.record_request(
                {
                    "request_id": answer.request_id,
                    "created_at": datetime.now(timezone.utc),
                    "question": answer.question,
                    "intent": answer.intent,
                    "status": answer.status,
                    "answer": answer.text[:4000],
                    "sql": "\n;\n".join(answer.sql)[:20000],
                    "notes": " | ".join(answer.notes),
                    "attempts": answer.attempts,
                    "model": answer.model,
                    "tokens": answer.tokens,
                    "latency_ms": answer.latency_ms,
                    "knowledge_version": self.knowledge_version,
                    "detail": answer.detail,
                }
            )
        except RcaError:
            return False   # 日志写不进去不该让提问的人拿不到答案
        return True


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------
def build_assistant(
    knowledge_root: str | Path,
    executor: SqlExecutor,
    dialect: SqlDialect,
    *,
    as_of: date,
    generator: SqlGenerator | None = None,
    store: StateStore | None = None,
    load_domains: bool = True,
) -> Assistant:
    """从 ``knowledge/`` 装配一个助手。

    查数用的提示来自 ``knowledge/nl2sql/prompting.yaml`` —— 04 采纳、经 PR 合并的提示,
    就是在这里进入线上的。
    """
    root = Path(knowledge_root)
    knowledge = load_knowledge(root)
    domains: ValueDomains | None = (
        load_value_domains(knowledge, executor, dialect) if load_domains else None
    )
    prompting = load_prompt_knowledge(root / "nl2sql" / "prompting.yaml")
    loop = None
    if generator is not None:
        loop = AnswerLoop(
            knowledge=knowledge,
            dialect=dialect,
            executor=executor,
            generator=generator,
            hints=prompting.hint_texts(),
            examples=prompting.few_shots(),
            as_of=as_of,
            value_domains=domains,
        )
    if store is not None:
        store.ensure_tables()
    return Assistant(
        knowledge=knowledge,
        dialect=dialect,
        executor=executor,
        router=Router(knowledge, load_synonyms(root / "assistant" / "synonyms.yaml"), domains),
        as_of=as_of,
        loop=loop,
        store=store,
        knowledge_version=prompting.label,
    )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _with(answer: Answer, **changes: Any) -> Answer:
    from dataclasses import replace

    return replace(answer, **changes)


def _num(value: float) -> str:
    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.4f}"


def _signed(value: float) -> str:
    return ("+" if value >= 0 else "−") + _num(abs(value))


def _factor_label(knowledge: Knowledge, factor: str) -> str:
    labels = {"uv": "流量 UV", "cvr": "转化率 CVR", "aov": "客单价 AOV"}
    return labels.get(factor, knowledge.metric(factor).display_name)


def _top_drivers(knowledge: Knowledge, dimensions: list[DimensionDecomposition]) -> list[dict[str, Any]]:
    """各维度里贡献最大的取值,跨维度按 |贡献| 排序。"""
    rows = []
    for result in dimensions:
        parts = [p for p in result.parts if p.value not in (UNKNOWN_BUCKET, NULL_BUCKET)]
        if not parts:
            continue
        top = max(parts, key=lambda p: abs(p.contribution_pct_points))
        rows.append(
            {
                "维度": knowledge.dimension(result.dimension).display_name,
                "取值": top.value,
                "变化": round(top.delta, 2),
                "贡献(百分点)": round(top.contribution_pct_points, 2),
            }
        )
    rows.sort(key=lambda row: -abs(row["贡献(百分点)"]))
    return rows[:MAX_DRIVERS]
