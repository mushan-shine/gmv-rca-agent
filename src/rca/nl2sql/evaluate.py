"""L2 评估循环:跑评估集、算指标、写 ``nl2sql_eval_runs``、把关回归。

一个必须说清楚的边界 —— **自修复循环修的是「跑不跑得通」,不是「答得对不对」**。

L1 内循环拿到的反馈只有守卫报错与引擎报错;它没有、也不应该有判断
「这个数字对不对」的能力(原简报 §5:LLM 不判断自己的结果对不对)。
一条语法正确、执行成功、但把 CANCELLED 订单也算进 GMV 的 SQL,
L1 永远发现不了。

所以正确性只能来自这一层:**拿程序化生成的标准答案做执行式比对**。
这也是为什么评估集不是「锦上添花的测试」,而是这类系统的地基。

指标
----

=========================  ==================================================
``pass_at_1``              可答问题中,第一轮就跑通**且答案正确**的比例
``pass_final``             可答问题中,最终答案正确的比例(执行式比对)
``avg_attempts``           平均尝试轮数 —— 自修复的成本
``hallucination_rate``     出现过「引用不存在的表/列」的 case 比例
``correct_refusal_rate``   不可答问题中被正确拒答的比例
``false_refusal_rate``     可答问题中被误拒的比例(拒答不能靠一律拒答刷分)
``holdout_pass_final``     **留出集**上的正确率 —— 回归闸门看的是它
=========================  ==================================================

最后两个是一对制衡:只看 ``correct_refusal_rate``,模型学会「全拒答」就能满分;
把 ``false_refusal_rate`` 一起看,这条捷径就堵死了。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..dialects import SqlDialect
from ..knowledge import Knowledge
from ..sampledata import SampleDataConfig
from ..warehouse import SqlExecutor
from .cases import EvalCase, Golden, GoldKind, build_golden
from .compare import MatchResult, compare_result_sets
from .llm import UsageMeter
from .loop import AnswerLoop, LoopOutcome, Status, attempt_rows
from .state import StateStore


@dataclass(frozen=True)
class CaseResult:
    """一条 case 的评估结果。"""

    case_id: str
    question: str
    answerable: bool
    holdout: bool
    status: Status
    correct: bool
    attempts_used: int
    hallucinated: bool
    detail: str = ""
    final_sql: str = ""
    reference_sql: str = ""
    tags: tuple[str, ...] = ()
    model: str = ""
    tokens: int = 0
    last_sql: str = ""
    """最后一轮尝试写出的 SQL(没跑通的也记)。改进循环拿它做失败证据。"""

    last_feedback: str = ""
    """最后一轮收到的反馈(守卫 / 上下文检查 / 引擎报错)。"""

    @property
    def verdict(self) -> str:
        if self.correct:
            return "correct_refusal" if not self.answerable else "correct"
        if not self.answerable:
            return "should_have_refused"
        if self.status is Status.REFUSED:
            return "false_refusal"
        if self.status is Status.FAILED:
            return "no_runnable_sql"
        return "wrong_answer"


@dataclass(frozen=True)
class EvalReport:
    """一次评估跑批的完整结果。"""

    run_id: str
    generator: str
    knowledge_version: str
    created_at: datetime
    results: tuple[CaseResult, ...]
    usage: UsageMeter | None = None
    """**本次跑批**的 LLM 用量(已对计量器做差)。
    为空表示这次没用模型(例如 ReferenceGenerator 基线)。"""

    prompt_fingerprint: str = ""
    """这次跑批所用上下文的指纹。见 :meth:`AnswerLoop.prompt_fingerprint`。"""

    # -- 指标 --------------------------------------------------------------
    @property
    def answerable(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if r.answerable)

    @property
    def unanswerable(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.answerable)

    @property
    def holdout(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if r.holdout)

    @property
    def pass_at_1(self) -> float:
        return _ratio(
            [r for r in self.answerable if r.correct and r.attempts_used == 1],
            self.answerable,
        )

    @property
    def pass_final(self) -> float:
        return _ratio([r for r in self.answerable if r.correct], self.answerable)

    @property
    def avg_attempts(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.attempts_used for r in self.results) / len(self.results)

    @property
    def hallucination_rate(self) -> float:
        return _ratio([r for r in self.results if r.hallucinated], self.results)

    @property
    def correct_refusal_rate(self) -> float:
        return _ratio([r for r in self.unanswerable if r.correct], self.unanswerable)

    @property
    def false_refusal_rate(self) -> float:
        return _ratio(
            [r for r in self.answerable if r.status is Status.REFUSED], self.answerable
        )

    @property
    def holdout_pass_final(self) -> float:
        holdout_answerable = [r for r in self.holdout if r.answerable]
        return _ratio([r for r in holdout_answerable if r.correct], holdout_answerable)

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.correct)

    # -- 训练集 / 留出集 ------------------------------------------------------
    # 「准确率」把可答题的执行比对与不可答题的正确拒答合在一起算:
    # 改进循环的目标是整体答对,单看其中一类都会被另一类的捷径钻空子。
    @property
    def train(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.holdout)

    @property
    def accuracy(self) -> float:
        return _ratio([r for r in self.results if r.correct], self.results)

    @property
    def train_correct(self) -> int:
        return sum(1 for r in self.train if r.correct)

    @property
    def train_accuracy(self) -> float:
        return _ratio([r for r in self.train if r.correct], self.train)

    @property
    def holdout_accuracy(self) -> float:
        return _ratio([r for r in self.holdout if r.correct], self.holdout)

    @property
    def model(self) -> str:
        """实际应答的模型。

        与 ``knowledge_version`` 一起写进 ``nl2sql_eval_runs``,
        这样「这一版好了 5 个点」才能归因到底是改了知识还是换了模型 ——
        两者一起变的那次跑批,是解释不清的。
        """
        for result in self.results:
            if result.model:
                return result.model
        return ""

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    def metrics_row(self, notes: str = "") -> dict[str, Any]:
        """摊平成可写进 ``nl2sql_eval_runs`` 的一行。"""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "generator": self.generator,
            "model": self.model,
            "knowledge_version": self.knowledge_version,
            "n_cases": len(self.results),
            "n_answerable": len(self.answerable),
            "n_unanswerable": len(self.unanswerable),
            "pass_at_1": self.pass_at_1,
            "pass_final": self.pass_final,
            "avg_attempts": self.avg_attempts,
            "hallucination_rate": self.hallucination_rate,
            "correct_refusal_rate": self.correct_refusal_rate,
            "false_refusal_rate": self.false_refusal_rate,
            "holdout_pass_final": self.holdout_pass_final,
            "llm_calls": self.usage.calls if self.usage else 0,
            "cached_calls": self.usage.cached_calls if self.usage else 0,
            "total_tokens": self.usage.total_tokens if self.usage else self.total_tokens,
            "notes": notes,
            "prompt_fingerprint": self.prompt_fingerprint,
            "train_accuracy": self.train_accuracy,
            "holdout_accuracy": self.holdout_accuracy,
        }

    def summary(self) -> str:
        return (
            f"run {self.run_id} · {self.generator} · knowledge {self.knowledge_version}"
            f" · prompt {self.prompt_fingerprint or '-'}\n"
            f"  accuracy (train)    {self.train_accuracy:6.1%}   ({self.train_correct}/{len(self.train)})\n"
            f"  accuracy (holdout)  {self.holdout_accuracy:6.1%}\n"
            f"  pass@1              {self.pass_at_1:6.1%}\n"
            f"  pass (final)        {self.pass_final:6.1%}\n"
            f"  holdout pass        {self.holdout_pass_final:6.1%}\n"
            f"  avg attempts        {self.avg_attempts:6.2f}\n"
            f"  hallucination rate  {self.hallucination_rate:6.1%}\n"
            f"  correct refusal     {self.correct_refusal_rate:6.1%}\n"
            f"  false refusal       {self.false_refusal_rate:6.1%}"
            + (f"\n  {self.usage.summary()}" if self.usage else "")
        )


def _last_feedback(outcome: LoopOutcome) -> str:
    if not outcome.attempts:
        return ""
    last = outcome.attempts[-1]
    parts = [*last.guard_messages, *last.context_messages]
    if last.execution_error:
        parts.append(last.execution_error)
    if last.generation_error:
        parts.append(last.generation_error)
    return " | ".join(parts)


def _ratio(subset: Sequence[Any], whole: Sequence[Any]) -> float:
    return len(subset) / len(whole) if whole else 0.0


def _judge(case: EvalCase, golden: Golden, outcome: LoopOutcome) -> tuple[bool, str]:
    """判定一条 case 是否答对。**完全确定性,不经 LLM。**"""
    if not golden.answerable:
        if outcome.status is Status.REFUSED:
            return True, f"正确拒答:{outcome.refusal_reason[:120]}"
        if outcome.status is Status.FAILED:
            return False, "没有拒答,也没写出能跑的 SQL —— 正确行为是明确拒答"
        return False, "本应拒答,却给出了 SQL(在数据里不存在的东西上编了答案)"

    if outcome.status is Status.REFUSED:
        return False, f"误拒:这个问题其实答得了。模型给的理由:{outcome.refusal_reason[:120]}"
    if outcome.status is Status.FAILED:
        # 取最后一轮的**全部**反馈。只取引擎报错和守卫信息时,被取值检查 / 日历检查
        # 拦下的失败原因是空的 —— 04 第一次真实运行里两道题的原因就这样丢了。
        reason = _last_feedback(outcome) or "无尝试记录"
        return False, f"{outcome.attempts_used} 轮都没写出能跑的 SQL:{reason}"

    match: MatchResult = compare_result_sets(
        golden.rows, outcome.rows, ordered=case.ordered
    )
    failure = match.failure.value if match.failure else ""
    return match.match, ("" if match.match else f"{failure}: {match.detail}")


@dataclass
class Evaluator:
    """把评估集、循环、状态表串起来。"""

    knowledge: Knowledge
    dialect: SqlDialect
    executor: SqlExecutor
    loop: AnswerLoop
    cases: Sequence[EvalCase]
    config: SampleDataConfig | None = None
    store: StateStore | None = None
    usage: UsageMeter | None = None
    """LLM 用量与预算。超限时 :class:`~rca.nl2sql.llm.LlmBudgetExceeded` 会被
    循环捕获成 ``generator_error``,整批评估不会中断 —— 但后续 case 会全部失败,
    报告里能一眼看出是熔断而不是模型退步。"""
    _goldens: dict[str, Golden] = field(default_factory=dict, repr=False)

    def with_loop(self, loop: AnswerLoop) -> "Evaluator":
        """换一个循环(例如候选知识版本),**复用已算好的标准答案**。

        标准答案只取决于数据和评估集,与模型/提示无关。每试一个候选就重算一遍,
        等于每轮多扫 22 次仓库 —— Free Edition 的配额经不起。
        """
        return replace(self, loop=loop, _goldens=self.goldens())

    def goldens(self) -> dict[str, Golden]:
        """程序化生成全部标准答案(带缓存)。"""
        if not self._goldens:
            self._goldens = {
                case.id: build_golden(
                    case, self.knowledge, self.dialect, self.executor, self.config
                )
                for case in self.cases
            }
        return self._goldens

    def run(
        self,
        *,
        knowledge_version: str = "v0",
        run_id: str | None = None,
        persist: bool = True,
        notes: str = "",
    ) -> EvalReport:
        """跑一遍评估集。

        Args:
            knowledge_version: 本次用的提示知识版本,写进 ``nl2sql_eval_runs``。
            run_id: 批次号;省略则自动生成。
            persist: 是否写状态表。改进循环做「候选版本试跑」时可以传 ``False``。
            notes: 备注,例如「加了 3 条 few-shot」。

        Returns:
            :class:`EvalReport`。
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        created_at = datetime.now(timezone.utc)
        goldens = self.goldens()
        # 计量器跨跑批累计(预算要管整个进程),这里记下起点,结束时做差 ——
        # 否则第二次跑批会把第一次的用量再报一遍。
        usage_before = self.usage.snapshot() if self.usage is not None else None

        results: list[CaseResult] = []
        attempts: list[Mapping[str, Any]] = []
        for case in self.cases:
            outcome = self.loop.answer(case.question)
            golden = goldens[case.id]
            correct, detail = _judge(case, golden, outcome)
            results.append(
                CaseResult(
                    case_id=case.id,
                    question=case.question,
                    answerable=case.gold.kind is not GoldKind.UNANSWERABLE,
                    holdout=case.holdout,
                    status=outcome.status,
                    correct=correct,
                    attempts_used=outcome.attempts_used,
                    hallucinated=outcome.any_hallucination,
                    detail=detail,
                    final_sql=outcome.final_sql,
                    reference_sql=golden.reference_sql,
                    tags=tuple(case.tags),
                    model=outcome.model,
                    tokens=outcome.total_tokens,
                    last_sql=outcome.attempts[-1].sql if outcome.attempts else "",
                    last_feedback=_last_feedback(outcome),
                )
            )
            attempts.extend(
                attempt_rows(run_id, case.id, outcome, verdict=results[-1].verdict)
            )

        report = EvalReport(
            run_id=run_id,
            generator=getattr(self.loop.generator, "name", "unknown"),
            knowledge_version=knowledge_version,
            created_at=created_at,
            results=tuple(results),
            prompt_fingerprint=self.loop.prompt_fingerprint(),
            usage=(
                self.usage.delta_from(usage_before)
                if self.usage is not None and usage_before is not None
                else None
            ),
        )

        if persist and self.store is not None:
            self.store.ensure_tables()
            self.store.record_attempts(attempts)
            self.store.record_eval_run(report.metrics_row(notes))
        return report


# ---------------------------------------------------------------------------
# 回归闸门
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateDecision:
    """回归闸门的判定。"""

    accepted: bool
    reasons: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.accepted


def regression_gate(
    candidate: EvalReport,
    baseline: Mapping[str, Any] | EvalReport | None,
    *,
    tolerance: float = 0.0,
) -> GateDecision:
    """候选知识版本能不能合并。

    规则刻意简单粗暴,因为它要能写进 CI:

    * **留出集**正确率不得低于基线(超过 ``tolerance``);
    * 全量正确率不得低于基线;
    * 幻觉率与误拒率不得升高。

    留出集那一条是关键:只看全量指标的话,把失败 case 的答案直接塞进 few-shot
    就能刷上去 —— 那是背答案,不是改进。

    Args:
        candidate: 新版本的评估结果。
        baseline: 上一版的指标行(``StateStore.previous_run`` 的输出)或 report;
            ``None`` 表示没有基线,直接通过。
        tolerance: 允许的退步幅度(绝对值,如 0.02 表示允许掉 2 个点)。

    Returns:
        :class:`GateDecision`,不通过时列出每一条原因。
    """
    if baseline is None:
        return GateDecision(True, ("没有基线,首次评估直接接受",))

    base = baseline.metrics_row() if isinstance(baseline, EvalReport) else baseline
    reasons: list[str] = []

    for label, key, current, higher_is_better in (
        ("留出集正确率", "holdout_pass_final", candidate.holdout_pass_final, True),
        ("全量正确率", "pass_final", candidate.pass_final, True),
        ("幻觉率", "hallucination_rate", candidate.hallucination_rate, False),
        ("误拒率", "false_refusal_rate", candidate.false_refusal_rate, False),
    ):
        previous = base.get(key)
        if previous is None:
            continue
        previous = float(previous)
        if higher_is_better and current < previous - tolerance:
            reasons.append(f"{label}退步:{previous:.1%} → {current:.1%}")
        elif not higher_is_better and current > previous + tolerance:
            reasons.append(f"{label}升高:{previous:.1%} → {current:.1%}")

    if reasons:
        return GateDecision(False, tuple(reasons))
    return GateDecision(True, ("各项指标均未退步",))


def reference_sql_map(evaluator: Evaluator) -> dict[str, str]:
    """case_id -> 参考 SQL。改进循环拿它做 few-shot 示例的正确答案。"""
    return {
        case_id: golden.reference_sql
        for case_id, golden in evaluator.goldens().items()
        if golden.answerable
    }


def question_map(cases: Sequence[EvalCase]) -> dict[str, str]:
    return {case.id: case.question for case in cases}


def default_cases_path() -> Path:
    return Path(__file__).resolve().parents[3] / "eval" / "cases.yaml"
