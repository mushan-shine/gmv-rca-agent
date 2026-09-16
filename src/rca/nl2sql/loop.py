"""L1 自修复内循环:生成 → 守卫 → 执行 → 把**真实错误**喂回去重生成。

这是三个循环里最内层的那个(几秒之内)::

    问题
      ↓
    生成 SQL ──► 静态守卫 ──失败──► 「列 revenue 不存在,fact_orders 上有 order_amount」
      ▲              │                                    │
      │              ok                                   │
      │              ▼                                    │
      │           执行 ──失败──► 引擎的原始错误信息 ───────┤
      │              │                                    │
      │              ok                                   │
      │              ▼                                 反馈进下一轮
      └──────────  返回结果                                │
                                                          ▼
                                            超过 max_attempts → 放弃

**它之所以是循环,而不是重试**,在于反馈是**具体的、可照着改的**:
守卫会说清哪个标识符不存在、那张表上有哪些列;执行失败会把引擎原话带上。
如果只是把同一个 prompt 再发一遍(常见的「retry 3 次」),那是重试不是循环,
模型没有获得任何新信息,收敛纯属撞运气。

每一轮尝试都被完整记录(:class:`Attempt`),写进 ``nl2sql_attempts`` 状态表 ——
「平均修复轮数」「第几轮修好的」这些指标都来自它。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from ..dialects import SqlDialect
from ..errors import QueryExecutionError
from ..knowledge import Knowledge
from ..warehouse import SqlExecutor, sql_fingerprint
from .llm import LlmError
from .generate import (
    FewShotExample,
    PromptContext,
    SqlGenerator,
    render_metric_glossary,
    render_schema,
)
from .guard import GuardReport, SqlGuard

DEFAULT_MAX_ATTEMPTS = 3
"""默认最多三轮。

不是拍脑袋:守卫能给出的反馈类型有限(语法、未知表、未知列、执行错),
一轮改一类,三轮足以覆盖绝大多数可修复的情况。再多就是在烧 token 赌运气,
并且会把「模型其实不会」伪装成「多试几次就好」。
这条上限同时是**预算熔断** —— 对应原简报 §10 的停止条件。
"""


class Status(str, Enum):
    ANSWERED = "answered"
    REFUSED = "refused"
    FAILED = "failed"


class StopReason(str, Enum):
    SUCCESS = "success"
    REFUSED = "refused"
    MAX_ATTEMPTS = "max_attempts"
    GENERATOR_ERROR = "generator_error"
    """模型调用本身失败(重试用尽 / 超预算)。

    与「模型答得不好」是两回事:不该被算成一次自修复尝试,
    也不该让整批评估中断。
    """


@dataclass(frozen=True)
class Attempt:
    """一轮尝试的完整记录。"""

    attempt_no: int
    sql: str
    fingerprint: str
    guard_messages: tuple[str, ...] = ()
    hallucinated: bool = False
    execution_error: str | None = None
    generation_error: str | None = None
    """模型侧的问题:回复被截断、没给出 SQL、调用失败。与 SQL 本身的错是两回事。"""

    row_count: int | None = None
    latency_ms: int = 0
    refused: bool = False
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    """本轮是否命中 prompt 缓存(命中则没有真实计费)。"""

    @property
    def ok(self) -> bool:
        return (
            not self.refused
            and not self.guard_messages
            and self.execution_error is None
            and self.generation_error is None
        )

    @property
    def failure_kind(self) -> str:
        """失败归类。改进循环按它聚合失败模式。"""
        if self.refused:
            return "refused"
        if self.generation_error is not None:
            return "generator"
        if self.hallucinated:
            return "hallucination"
        if self.guard_messages:
            return "guard"
        if self.execution_error is not None:
            return "execution"
        return "none"


@dataclass(frozen=True)
class LoopOutcome:
    """一次提问的最终结果。"""

    question: str
    status: Status
    stop_reason: StopReason
    attempts: tuple[Attempt, ...]
    final_sql: str = ""
    rows: tuple[dict[str, Any], ...] = ()
    refusal_reason: str = ""
    generator: str = ""

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def solved_first_try(self) -> bool:
        return self.status is Status.ANSWERED and self.attempts_used == 1

    @property
    def any_hallucination(self) -> bool:
        return any(attempt.hallucinated for attempt in self.attempts)

    @property
    def model(self) -> str:
        """实际应答的模型。写进状态表,让指标变化能归因到模型还是知识版本。"""
        for attempt in reversed(self.attempts):
            if attempt.model:
                return attempt.model
        return ""

    @property
    def total_tokens(self) -> int:
        return sum(a.prompt_tokens + a.completion_tokens for a in self.attempts)


@dataclass
class AnswerLoop:
    """把生成器、守卫、执行器串成一个自修复循环。"""

    knowledge: Knowledge
    dialect: SqlDialect
    executor: SqlExecutor
    generator: SqlGenerator
    guard: SqlGuard | None = None
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    hints: tuple[str, ...] = ()
    examples: tuple[FewShotExample, ...] = ()
    max_rows: int = 500
    """单次查询返回行数上限。防止模型写出 ``SELECT *`` 把整张表拉回来。"""

    _schema_text: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.guard is None:
            self.guard = SqlGuard(self.knowledge, dialect=self.dialect.name)
        if not self.hints:
            self.hints = tuple(render_metric_glossary(self.knowledge))
        self._schema_text = render_schema(self.knowledge, self.dialect)

    def initial_context(self, question: str) -> PromptContext:
        return PromptContext(
            question=question,
            schema_text=self._schema_text,
            hints=self.hints,
            examples=self.examples,
        )

    def answer(self, question: str) -> LoopOutcome:
        """回答一个问题,必要时自我修复。

        Args:
            question: 自然语言问题。

        Returns:
            :class:`LoopOutcome`,含每一轮的完整记录。**不抛异常** ——
            失败是一种正常结果,评估需要统计它,而不是被中断。
        """
        assert self.guard is not None
        context = self.initial_context(question)
        attempts: list[Attempt] = []

        for attempt_no in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                generation = self.generator.generate(context)
            except LlmError as exc:
                # 模型调用本身失败(重试用尽 / 超预算)。客户端已经退避重发过
                # **同一个 prompt**,这里再转一轮不会有新信息 —— 记录并停止。
                # 但**不抛出**:一次限流不该把整批评估打断。
                attempts.append(
                    Attempt(
                        attempt_no=attempt_no,
                        sql="",
                        fingerprint="",
                        generation_error=_condense(str(exc)),
                        latency_ms=_elapsed_ms(started),
                    )
                )
                return LoopOutcome(
                    question=question,
                    status=Status.FAILED,
                    stop_reason=StopReason.GENERATOR_ERROR,
                    attempts=tuple(attempts),
                    generator=getattr(self.generator, "name", ""),
                )

            usage = {
                "model": generation.model,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "cached": generation.cached,
            }

            if generation.error:
                attempts.append(
                    Attempt(
                        attempt_no=attempt_no,
                        sql="",
                        fingerprint="",
                        generation_error=generation.error,
                        latency_ms=_elapsed_ms(started),
                        **usage,
                    )
                )
                context = context.with_feedback(generation.error)
                continue

            if generation.refused:
                attempts.append(
                    Attempt(
                        attempt_no=attempt_no,
                        sql="",
                        fingerprint="",
                        refused=True,
                        latency_ms=_elapsed_ms(started),
                        **usage,
                    )
                )
                return LoopOutcome(
                    question=question,
                    status=Status.REFUSED,
                    stop_reason=StopReason.REFUSED,
                    attempts=tuple(attempts),
                    refusal_reason=generation.refusal_reason or "",
                    generator=getattr(self.generator, "name", ""),
                )

            sql = generation.sql or ""
            report: GuardReport = self.guard.check(sql)
            if not report.ok:
                attempts.append(
                    Attempt(
                        attempt_no=attempt_no,
                        sql=sql,
                        fingerprint=sql_fingerprint(sql) if sql else "",
                        guard_messages=report.messages,
                        hallucinated=report.hallucinated,
                        latency_ms=_elapsed_ms(started),
                        **usage,
                    )
                )
                context = context.with_feedback(report.feedback())
                continue

            try:
                rows = self.executor.run(report.sql)
            except QueryExecutionError as exc:
                message = _condense(str(exc))
                attempts.append(
                    Attempt(
                        attempt_no=attempt_no,
                        sql=report.sql,
                        fingerprint=sql_fingerprint(report.sql),
                        execution_error=message,
                        latency_ms=_elapsed_ms(started),
                        **usage,
                    )
                )
                context = context.with_feedback(
                    f"执行失败,引擎报错:{message}。请修正这条 SQL。"
                )
                continue

            attempts.append(
                Attempt(
                    attempt_no=attempt_no,
                    sql=report.sql,
                    fingerprint=sql_fingerprint(report.sql),
                    row_count=len(rows),
                    latency_ms=_elapsed_ms(started),
                    **usage,
                )
            )
            return LoopOutcome(
                question=question,
                status=Status.ANSWERED,
                stop_reason=StopReason.SUCCESS,
                attempts=tuple(attempts),
                final_sql=report.sql,
                rows=tuple(rows[: self.max_rows]),
                generator=getattr(self.generator, "name", ""),
            )

        return LoopOutcome(
            question=question,
            status=Status.FAILED,
            stop_reason=StopReason.MAX_ATTEMPTS,
            attempts=tuple(attempts),
            generator=getattr(self.generator, "name", ""),
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _condense(message: str, limit: int = 400) -> str:
    """把引擎的长报错压成能塞进 prompt 的一段。

    保留**第一行**(错误类型与位置通常都在那里),丢掉后面的 SQL 回显 ——
    模型自己写的 SQL 不需要再喂一遍给它。
    """
    first = message.strip().split("\n--- SQL ---")[0].strip()
    lines = [line for line in first.splitlines() if line.strip()]
    condensed = " ".join(lines)
    return condensed[:limit] + ("…" if len(condensed) > limit else "")


def answer_questions(
    loop: AnswerLoop, questions: Sequence[str]
) -> list[LoopOutcome]:
    """顺序回答一批问题。刻意不并发 —— Free Edition 只有一个 2X-Small warehouse。"""
    return [loop.answer(question) for question in questions]


def attempt_rows(
    run_id: str, case_id: str, outcome: LoopOutcome, verdict: str = ""
) -> list[Mapping[str, Any]]:
    """把一次 outcome 摊平成可写进状态表的行。

    ``verdict`` 由**评估器**填(``correct`` / ``wrong_answer`` / ``false_refusal`` …)。
    循环自己不知道答案对不对 —— 它只知道 SQL 跑没跑通,
    所以 ``failure_kind`` 与 ``verdict`` 是两件事:
    一次「正确拒答」的 failure_kind 是 ``refused``,但 verdict 是 ``correct_refusal``。
    """
    return [
        {
            "run_id": run_id,
            "case_id": case_id,
            "verdict": verdict,
            "attempt_no": attempt.attempt_no,
            "generator": outcome.generator,
            "sql": attempt.sql,
            "fingerprint": attempt.fingerprint,
            "failure_kind": attempt.failure_kind,
            "model": attempt.model,
            "guard_errors": " | ".join(attempt.guard_messages),
            "execution_error": attempt.execution_error or "",
            "generation_error": attempt.generation_error or "",
            "row_count": attempt.row_count if attempt.row_count is not None else -1,
            "prompt_tokens": attempt.prompt_tokens,
            "completion_tokens": attempt.completion_tokens,
            "cached": attempt.cached,
            "latency_ms": attempt.latency_ms,
        }
        for attempt in outcome.attempts
    ]
