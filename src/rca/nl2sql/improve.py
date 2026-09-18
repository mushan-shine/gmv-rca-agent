"""L3 自主改进循环:**LLM 提议,确定性闸门裁决,台账记住每一次尝试。**

之前的 L3 是「人在闭环」:失败原因是人看出来的,修改是人写的,合不合并是人定的。
本模块把这三步交给系统::

    ┌─→ ① 读状态:当前知识版本的训练集失败 + 台账里这个模型以往被拒绝过的提示
    │        ↓
    │   ② 提议:LLM 看失败证据(问题、它写的 SQL、错在哪),写一条**通用**提示
    │        ↓
    │   ③ 试跑:带着候选提示重跑整个评估集(训练集 + 留出集)
    │        ↓
    │   ④ 裁决(确定性,不经 LLM):
    │        训练集多答对至少一道 且 留出集不退步 且 幻觉率/误拒率不升高
    │        ↓
    │   ⑤ 写状态:采纳 → 知识版本 +1,成为新基线;拒绝 → 连同原因记进台账
    │        ↓
    └── ⑥ 停止:达到目标 / 训练集没有错题 / 连续 N 轮没有采纳 / 预算用完 / 达到轮数上限

三条纪律,决定了这是 loop engineering 而不是「让 LLM 调 prompt」:

**提议者看不到留出集。** 失败证据只取训练集。留出集只在裁决时出现 ——
否则系统能把留出集的错题也「修」掉,闸门就成了摆设。

**提议者不给标准答案。** 它看到的是自己写错的 SQL 和错在哪,看不到参考 SQL。
给了参考 SQL,它最省事的做法就是把答案抄进提示,那是背答案。

**被拒绝的尝试要被记住。** 台账跨循环保存;下一次提议时,这个模型以往所有被拒绝的
提示连同拒绝原因都会出现在 prompt 里。没有这一条,确定性解码的模型会一轮又一轮地
提同一个无效修改 —— 那是复读机,不是循环。

一次只改一个变量:每轮只加一条提示,模型、评估集、循环配置都不变。
所以每一行台账里的指标差值,都可以归因到那一条提示上。

第一次在 Databricks 上真实运行(glm-4-flash,基线训练集 14/21)得到的教训,
决定了下面三条机制::

    第 1 轮  "Always verify column names and values against the actual data dictionary."  14 → 12  拒绝
    第 2 轮  "Always verify the correct usage of dimensions and metrics in queries."       14 → 12  拒绝

**一次只打一类错。** 把 7 道各不相同的错题一股脑给提议者,它只能写出「放之四海皆准」的空话。
现在先按失败形态分组(维度取值、该拒答没拒答、数值错、形状错……),每轮只给一组;
这一组试过没被采纳,下一轮换下一组。

**先做便宜的检查,再做昂贵的试跑。** 一次试跑要重跑整个评估集(约 37 次模型调用、4~5 分钟)。
一条没有点名任何表、列、取值或指标的提示,在 prompt 里只是噪声 —— 上面两条就是。
这类提示在试跑前就被拒掉(``too_generic``),照样记进台账、告诉下一轮的提议者。

**提升要落在它针对的题上。** 两条内容空洞的提示都让训练集变了 2 道题,
说明在 21 道题的规模上,「随便加一句话」本身就会带来 ±2 道的波动。
如果只看总分,一次恰好 +1 的噪声就会被当成改进采纳。所以闸门还要求:
这一轮针对的那组错题里,至少有一道被修好。
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..knowledge import Knowledge
from .compare import MatchFailure
from .domains import VIOLATION_MARKER, ValueDomains
from .evaluate import CaseResult, EvalReport, Evaluator, GateDecision
from .generate import REFUSAL_PREFIX
from .knowledge_store import Hint, PromptKnowledge
from .llm import ChatClient, LlmError, UsageMeter
from .loop import AnswerLoop
from .state import StateStore

MAX_HINT_CHARS = 400
MAX_FAILURES_SHOWN = 10


# ---------------------------------------------------------------------------
# 失败分组
# ---------------------------------------------------------------------------
PATTERNS: dict[str, str] = {
    "dimension_value": (
        "The SQL filtered a dimension column on a value that does not exist in the data "
        "(wrong code, wrong letter case, or the value lives in a different column)."
    ),
    "missed_refusal": (
        f"The question asks for data this warehouse does not contain. The correct reply is "
        f"{REFUSAL_PREFIX}, but the assistant wrote SQL or failed instead."
    ),
    "wrong_value": "The SQL ran, but the number it returned is wrong (wrong formula, filter or scale).",
    "wrong_shape": (
        "The SQL ran, but returned the wrong number of rows or columns, or nothing at all "
        "(wrong grouping, or a filter that removed rows it should keep)."
    ),
    "no_runnable_sql": "The assistant never produced SQL that passed the checks and executed.",
    "false_refusal": "The assistant refused a question that the data can answer.",
}
"""失败形态 -> 给提议者看的说明。字典顺序即数量相同时的优先顺序。"""


def failure_pattern(result: CaseResult) -> str:
    """把一道错题归到一种失败形态。完全确定性。"""
    verdict = result.verdict
    if verdict == "should_have_refused":
        return "missed_refusal"
    if verdict == "false_refusal":
        return "false_refusal"
    if verdict == "no_runnable_sql":
        return "dimension_value" if VIOLATION_MARKER in result.last_feedback else "no_runnable_sql"
    if result.detail.startswith(MatchFailure.VALUE_MISMATCH.value):
        return "wrong_value"
    return "wrong_shape"


def choose_focus(patterns: Iterable[str], tried: Iterable[str] = ()) -> str:
    """这一轮打哪一类错:没试过的里面数量最多的;全试过了就从头再来。"""
    counts = Counter(patterns)
    if not counts:
        return ""
    order = list(PATTERNS)
    untried = [p for p in counts if p not in set(tried)] or list(counts)
    return min(untried, key=lambda p: (-counts[p], order.index(p) if p in order else len(order)))


# ---------------------------------------------------------------------------
# 具体性检查:试跑之前的便宜闸门
# ---------------------------------------------------------------------------
_GENERIC_WORDS = frozenset({"orders"})
"""既是指标名又是普通英文单词的词。「check the orders」不算点名了任何东西。"""


@dataclass(frozen=True)
class Vocabulary:
    """提示可以点名的具体对象:表、列、指标、维度名(不分大小写)与维度取值(区分大小写)。"""

    identifiers: frozenset[str] = frozenset()
    values: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.identifiers or self.values)

    def named_in(self, text: str) -> tuple[str, ...]:
        """``text`` 点名了哪些对象。"""
        tokens = {token.lower() for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)}
        found = sorted(tokens & self.identifiers)
        found += sorted(
            value for value in self.values
            if re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", text)
        )
        return tuple(dict.fromkeys(found))


def build_vocabulary(knowledge: Knowledge, domains: ValueDomains | None = None) -> Vocabulary:
    names = set(knowledge.tables) | set(knowledge.metrics) | set(knowledge.dimensions)
    for table in knowledge.tables.values():
        names |= set(table.columns)
    names.add(REFUSAL_PREFIX)
    identifiers = frozenset(n.lower() for n in names) - _GENERIC_WORDS
    values: set[str] = set()
    if domains is not None:
        for allowed in domains.values.values():
            values |= set(allowed)
    return Vocabulary(identifiers, frozenset(values))


def case_flips(before: EvalReport, after: EvalReport) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """训练集里哪些题从错变对(fixed)、从对变错(broken)。"""
    was = {r.case_id: r.correct for r in before.train}
    fixed = tuple(r.case_id for r in after.train if r.correct and not was.get(r.case_id, False))
    broken = tuple(r.case_id for r in after.train if not r.correct and was.get(r.case_id, False))
    return fixed, broken


# ---------------------------------------------------------------------------
# 提议
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FailureEvidence:
    """提议者能看到的一条失败证据。**不含参考 SQL。**"""

    case_id: str
    question: str
    verdict: str
    detail: str
    sql: str
    feedback: str
    pattern: str = ""

    @classmethod
    def of(cls, result: CaseResult) -> "FailureEvidence":
        return cls(
            case_id=result.case_id,
            question=result.question,
            verdict=result.verdict,
            detail=result.detail,
            sql=result.final_sql or result.last_sql,
            feedback=result.last_feedback,
            pattern=failure_pattern(result),
        )


@dataclass(frozen=True)
class ProposalContext:
    failures: tuple[FailureEvidence, ...]
    """这一轮要打的那一类错题。``focus`` 为空表示不分组、全部给出。"""

    current_hints: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    """(被拒绝的提示, 拒绝原因)。"""

    focus: str = ""
    other_patterns: tuple[tuple[str, int], ...] = ()
    """其他尚未解决的失败形态及题数。只告诉提议者「还有这些」,不给细节,免得它又去面面俱到。"""


class HintProposal(BaseModel):
    """提议者的结构化输出。"""

    model_config = ConfigDict(extra="ignore", frozen=True)

    hint: str = Field(min_length=10, max_length=MAX_HINT_CHARS)
    rationale: str = ""
    targets: list[str] = Field(default_factory=list)

    @field_validator("hint")
    @classmethod
    def _strip(cls, value: str) -> str:
        return " ".join(value.split())


@runtime_checkable
class HintProposer(Protocol):
    name: str

    def propose(self, context: ProposalContext) -> HintProposal | None:
        ...


def build_proposer_prompt(context: ProposalContext) -> str:
    """提议者的 prompt。纯函数,可 diff、可复现。"""
    lines = [
        "You maintain the instruction set of a text-to-SQL assistant that answers business "
        "questions on a Databricks e-commerce warehouse. Below are questions it answered "
        "WRONG in the latest evaluation, with the SQL it wrote and what went wrong.",
        "",
        "Propose exactly ONE new guideline to add to its instructions.",
        "",
    ]
    if context.focus:
        lines += [
            f"## This round: fix ONE failure pattern - {context.focus}",
            PATTERNS.get(context.focus, ""),
            "Every failure listed below has this pattern. Read the SQL and the checker "
            "feedback, find the concrete mistake they share, and write the rule that "
            "prevents it.",
            "",
        ]
    lines += [
        "Requirements:",
        "- It must be SPECIFIC: name the exact tables, columns, values, metrics or the "
        f"{REFUSAL_PREFIX} reply it applies to, and say exactly what to do. Vague advice "
        "such as 'verify column names' or 'check the data dictionary' is rejected "
        "automatically without being tested.",
        "- It must be GENERAL enough to help on similar, unseen questions. Do not mention "
        "case ids or expected numbers, and do not restate a single answer.",
        "- Do not repeat or rephrase any existing guideline, and do not re-propose any "
        "previously rejected guideline.",
        f"- At most {MAX_HINT_CHARS} characters.",
        "",
        "Reply with ONLY a JSON object in a ```json code block:",
        '{"hint": "...", "rationale": "why this fixes the failures", '
        '"targets": ["case ids this should fix"]}',
        "",
        "## Existing guidelines",
    ]
    lines += [f"- {hint}" for hint in context.current_hints] or ["(none)"]
    lines += ["", "## Previously rejected guidelines (do NOT propose these again)"]
    if context.rejected:
        lines += [f"- {hint}  -- rejected because: {reason}" for hint, reason in context.rejected]
    else:
        lines += ["(none)"]
    lines += ["", f"## Failures{f' ({context.focus})' if context.focus else ''}"]
    for failure in context.failures[:MAX_FAILURES_SHOWN]:
        lines += [
            f"### {failure.case_id}  [{failure.verdict}]",
            f"Question: {failure.question}",
            f"SQL written:\n{failure.sql or '(no SQL produced)'}",
            f"What went wrong: {failure.detail}",
        ]
        if failure.feedback:
            lines.append(f"Checker feedback: {failure.feedback}")
        lines.append("")
    if context.other_patterns:
        lines += ["## Other open failure patterns (later rounds - ignore them now)"]
        lines += [f"- {pattern}: {count} question(s)" for pattern, count in context.other_patterns]
    return "\n".join(lines)


def parse_proposal(text: str) -> HintProposal | None:
    """从模型回复里取出 JSON 提议。取不出、字段不合法都返回 ``None``,不猜。"""
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else raw[raw.find("{"): raw.rfind("}") + 1]
    if not candidate:
        return None
    try:
        return HintProposal.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError):
        return None


@dataclass
class LlmHintProposer:
    """用 LLM 写候选提示。它只负责「提」,采不采纳由闸门决定。"""

    client: ChatClient
    meter: UsageMeter | None = None
    last_prompt: str = field(default="", repr=False)
    last_reply: str = field(default="", repr=False)

    @property
    def name(self) -> str:
        return f"llm-proposer:{self.client.model}"

    def propose(self, context: ProposalContext) -> HintProposal | None:
        prompt = build_proposer_prompt(context)
        self.last_prompt = prompt
        if self.meter is not None:
            self.meter.check_budget()
        response = self.client.complete(prompt)
        if self.meter is not None:
            self.meter.record(response)
        self.last_reply = response.text
        return parse_proposal(response.text)


@dataclass
class ScriptedProposer:
    """按顺序返回预设提议。测试用。"""

    proposals: Sequence[HintProposal | None]
    name: str = "scripted-proposer"
    seen: list[ProposalContext] = field(default_factory=list)

    def propose(self, context: ProposalContext) -> HintProposal | None:
        self.seen.append(context)
        index = len(self.seen) - 1
        return self.proposals[index] if index < len(self.proposals) else None


# ---------------------------------------------------------------------------
# 裁决
# ---------------------------------------------------------------------------
def improvement_gate(
    candidate: EvalReport,
    baseline: EvalReport,
    targeted: Sequence[str] = (),
) -> GateDecision:
    """候选提示能不能被采纳。**完全确定性。**

    - 训练集必须**多答对至少一道**。持平不采纳:没有证据的改动只会让 prompt 越来越长;
    - 给了 ``targeted``(这一轮针对的错题)时,其中**至少一道要被修好**。
      提升落在别的题上,多半是 prompt 变动带来的波动,不能归因到这条提示;
    - 留出集准确率不得下降。这是防「背答案」的那道闸 —— 提议者从没见过留出集;
    - 幻觉率、误拒率不得升高。否则「一律拒答」或「多编几个列」都可能刷高训练分。
    """
    reasons: list[str] = []
    if candidate.train_correct <= baseline.train_correct:
        reasons.append(
            f"训练集没有多答对:{baseline.train_correct} → {candidate.train_correct}"
            f"(共 {len(candidate.train)} 道)"
        )
    if targeted:
        fixed, _ = case_flips(baseline, candidate)
        hit = [case for case in targeted if case in fixed]
        if not hit:
            reasons.append(
                f"针对的 {len(targeted)} 道题一道都没修好,总分变化不能归因到这条提示"
            )
    if candidate.holdout_accuracy < baseline.holdout_accuracy:
        reasons.append(
            f"留出集退步:{baseline.holdout_accuracy:.1%} → {candidate.holdout_accuracy:.1%}"
        )
    if candidate.hallucination_rate > baseline.hallucination_rate:
        reasons.append(
            f"幻觉率升高:{baseline.hallucination_rate:.1%} → {candidate.hallucination_rate:.1%}"
        )
    if candidate.false_refusal_rate > baseline.false_refusal_rate:
        reasons.append(
            f"误拒率升高:{baseline.false_refusal_rate:.1%} → {candidate.false_refusal_rate:.1%}"
        )
    if reasons:
        return GateDecision(False, tuple(reasons))
    return GateDecision(
        True,
        (
            f"训练集 {baseline.train_correct} → {candidate.train_correct} 道,"
            f"留出集 {baseline.holdout_accuracy:.1%} → {candidate.holdout_accuracy:.1%}",
        ),
    )


# ---------------------------------------------------------------------------
# 循环
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoundRecord:
    round: int
    decision: str
    """``accepted`` / ``rejected`` / ``duplicate`` / ``no_proposal`` / ``proposer_error``。"""

    hint: str
    reasons: tuple[str, ...]
    knowledge_version: str
    train_accuracy: float
    holdout_accuracy: float
    train_correct: int
    trial_run_id: str = ""
    rationale: str = ""
    targets: tuple[str, ...] = ()
    focus: str = ""
    """这一轮针对的失败形态。"""

    fixed: tuple[str, ...] = ()
    """试跑中从错变对的训练题。"""

    broken: tuple[str, ...] = ()
    """试跑中从对变错的训练题。与 ``fixed`` 一起看,才知道分数变化是怎么来的。"""


@dataclass(frozen=True)
class ImprovementResult:
    loop_id: str
    stop_reason: str
    initial_knowledge: PromptKnowledge
    final_knowledge: PromptKnowledge
    baseline: EvalReport
    final: EvalReport
    rounds: tuple[RoundRecord, ...]

    @property
    def accepted(self) -> tuple[RoundRecord, ...]:
        return tuple(r for r in self.rounds if r.decision == "accepted")

    def curve(self) -> list[dict[str, Any]]:
        """逐轮曲线:第 0 行是基线,之后每轮一行(含被拒绝的)。"""
        rows = [
            {
                "round": 0,
                "decision": "baseline",
                "knowledge": self.initial_knowledge.label,
                "train_accuracy": round(self.baseline.train_accuracy, 4),
                "holdout_accuracy": round(self.baseline.holdout_accuracy, 4),
                "hint": "",
                "reasons": "",
                "focus": "",
                "fixed": "",
                "broken": "",
            }
        ]
        for record in self.rounds:
            rows.append(
                {
                    "round": record.round,
                    "decision": record.decision,
                    "knowledge": record.knowledge_version,
                    "train_accuracy": round(record.train_accuracy, 4),
                    "holdout_accuracy": round(record.holdout_accuracy, 4),
                    "hint": record.hint,
                    "reasons": " | ".join(record.reasons),
                    "focus": record.focus,
                    "fixed": ", ".join(record.fixed),
                    "broken": ", ".join(record.broken),
                }
            )
        return rows

    def summary(self) -> str:
        return (
            f"loop {self.loop_id} · 停止原因 {self.stop_reason} · "
            f"{len(self.rounds)} 轮,采纳 {len(self.accepted)} 条\n"
            f"  训练集  {self.baseline.train_accuracy:6.1%} → {self.final.train_accuracy:6.1%}\n"
            f"  留出集  {self.baseline.holdout_accuracy:6.1%} → {self.final.holdout_accuracy:6.1%}\n"
            f"  知识    {self.initial_knowledge.label} → {self.final_knowledge.label}"
        )


def _normalize(text: str) -> str:
    """去重用的归一化:忽略大小写、标点和空白。

    用 ``\\W`` 而不是 ``[^a-z0-9]`` —— 后者会把中文提示整条抹成空串,所有中文提示互判重复。
    """
    return re.sub(r"\W+", " ", text.casefold()).strip()


@dataclass
class AutoImprover:
    """自主改进循环。

    Args:
        evaluator: 用来跑评估的评估器(标准答案会在各轮之间复用)。
        loop_factory: 给定一个知识版本,造出对应的 :class:`AnswerLoop`。
            除了提示/示例之外,生成器、日历、取值检查都必须保持不变 —— 一次只改一个变量。
        proposer: 提议者。
        store: 状态表;给了就写台账、读取以往被拒绝的提示。
        max_rounds: 最多试几个候选。
        patience: 连续几轮没有采纳就停。
        target_accuracy: 训练集与留出集都达到这个准确率就停。
        vocabulary: 提示可以点名的表、列、指标与取值(见 :func:`build_vocabulary`)。
            给了就在试跑前做具体性检查;为空则跳过。
        focus_by_pattern: 每轮只给提议者一类错题,并要求提升落在这类题上。
    """

    evaluator: Evaluator
    loop_factory: Callable[[PromptKnowledge], AnswerLoop]
    proposer: HintProposer
    store: StateStore | None = None
    max_rounds: int = 4
    patience: int = 2
    target_accuracy: float = 0.9
    on_round: Callable[[RoundRecord], None] | None = None
    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    focus_by_pattern: bool = True

    def run(
        self,
        knowledge: PromptKnowledge,
        *,
        baseline: EvalReport | None = None,
        loop_id: str | None = None,
    ) -> ImprovementResult:
        loop_id = loop_id or f"loop_{uuid.uuid4().hex[:8]}"
        initial = knowledge
        generator_name = getattr(self.loop_factory(knowledge).generator, "name", "unknown")

        if baseline is None:
            baseline = self.evaluator.with_loop(self.loop_factory(knowledge)).run(
                knowledge_version=knowledge.label,
                run_id=f"{loop_id}_r0",
                notes=f"autoloop {loop_id} baseline",
            )
        start = baseline
        current, current_report = knowledge, baseline

        rejected: list[tuple[str, str]] = []
        tried: set[str] = set()   # 本次循环里试过、没被采纳的失败形态
        if self.store is not None:
            self.store.ensure_tables()
            rejected = [
                (str(row["hint"]), str(row["reasons"]))
                for row in self.store.rejected_hints(generator_name)
            ]

        rounds: list[RoundRecord] = []
        stale = 0
        stop_reason = "max_rounds"

        for number in range(1, self.max_rounds + 1):
            if (
                current_report.train_accuracy >= self.target_accuracy
                and current_report.holdout_accuracy >= self.target_accuracy
            ):
                stop_reason = "target_reached"
                break

            # ① 读状态。只给训练集的失败 —— 留出集对提议者不可见。
            failures = tuple(FailureEvidence.of(r) for r in current_report.train if not r.correct)
            if not failures:
                stop_reason = "no_train_failures"
                break
            focus = choose_focus((f.pattern for f in failures), tried) if self.focus_by_pattern else ""
            shown = tuple(f for f in failures if f.pattern == focus) if focus else failures
            counts = Counter(f.pattern for f in failures if f.pattern != focus)
            context = ProposalContext(
                failures=shown,
                current_hints=current.hint_texts(),
                rejected=tuple(rejected),
                focus=focus,
                other_patterns=tuple(counts.most_common()) if focus else (),
            )
            targeted = tuple(f.case_id for f in shown) if focus else ()

            # ② 提议
            try:
                proposal = self.proposer.propose(context)
            except LlmError as exc:
                self._record(loop_id, number, "proposer_error", current, current_report,
                             None, None, (str(exc)[:300],), generator_name, rounds, focus=focus)
                stop_reason = "proposer_error"
                break

            if proposal is None:
                self._record(loop_id, number, "no_proposal", current, current_report,
                             None, None, ("提议者没有给出合法的提议",), generator_name, rounds,
                             focus=focus)
                tried.add(focus)
                stale += 1
                if stale >= self.patience:
                    stop_reason = "patience"
                    break
                continue

            seen = {_normalize(h) for h in current.hint_texts()} | {_normalize(h) for h, _ in rejected}
            if _normalize(proposal.hint) and _normalize(proposal.hint) in seen:
                self._record(loop_id, number, "duplicate", current, current_report,
                             proposal, None, ("与已有提示或已被拒绝的提示重复",), generator_name, rounds,
                             focus=focus)
                tried.add(focus)
                stale += 1
                if stale >= self.patience:
                    stop_reason = "patience"
                    break
                continue

            # 便宜的检查先做:没点名任何具体对象的提示不值得花一次完整试跑
            if self.vocabulary and not self.vocabulary.named_in(proposal.hint):
                reason = "提示没有点名任何具体的表、列、取值或指标,试跑前拒绝(不花试跑成本)"
                self._record(loop_id, number, "too_generic", current, current_report,
                             proposal, None, (reason,), generator_name, rounds, focus=focus)
                rejected.insert(0, (proposal.hint, reason))
                tried.add(focus)
                stale += 1
                if stale >= self.patience:
                    stop_reason = "patience"
                    break
                continue

            # ③ 试跑:只加这一条提示,其余全部不变
            candidate = PromptKnowledge(
                version=current.version + 1,
                hints=[
                    *current.hints,
                    Hint(
                        text=proposal.hint,
                        source_run=current_report.run_id,
                        fixes_case=",".join(proposal.targets),
                    ),
                ],
                examples=list(current.examples),
            )
            trial = self.evaluator.with_loop(self.loop_factory(candidate)).run(
                knowledge_version=f"{candidate.label}-candidate",
                run_id=f"{loop_id}_r{number}",
                notes=f"autoloop {loop_id} round {number}: {proposal.hint[:120]}",
            )

            # ④ 裁决
            decision = improvement_gate(trial, current_report, targeted)

            # ⑤ 写状态
            if decision.accepted:
                self._record(loop_id, number, "accepted", current, current_report,
                             proposal, trial, decision.reasons, generator_name, rounds, candidate,
                             focus=focus)
                current, current_report = candidate, trial
                tried.clear()   # 知识变了,错题也变了,各类错都值得重新试
                stale = 0
            else:
                self._record(loop_id, number, "rejected", current, current_report,
                             proposal, trial, decision.reasons, generator_name, rounds, candidate,
                             focus=focus)
                rejected.insert(0, (proposal.hint, " | ".join(decision.reasons)))
                tried.add(focus)
                stale += 1
                if stale >= self.patience:
                    stop_reason = "patience"
                    break

        return ImprovementResult(
            loop_id=loop_id,
            stop_reason=stop_reason,
            initial_knowledge=initial,
            final_knowledge=current,
            baseline=start,
            final=current_report,
            rounds=tuple(rounds),
        )

    def _record(
        self,
        loop_id: str,
        number: int,
        decision: str,
        parent: PromptKnowledge,
        baseline: EvalReport,
        proposal: HintProposal | None,
        trial: EvalReport | None,
        reasons: tuple[str, ...],
        generator_name: str,
        rounds: list[RoundRecord],
        candidate: PromptKnowledge | None = None,
        *,
        focus: str = "",
    ) -> None:
        after = trial or baseline
        fixed, broken = case_flips(baseline, trial) if trial is not None else ((), ())
        record = RoundRecord(
            round=number,
            decision=decision,
            hint=proposal.hint if proposal else "",
            reasons=tuple(reasons),
            knowledge_version=(candidate.label if candidate and decision == "accepted" else parent.label),
            train_accuracy=after.train_accuracy,
            holdout_accuracy=after.holdout_accuracy,
            train_correct=after.train_correct,
            trial_run_id=trial.run_id if trial else "",
            rationale=proposal.rationale if proposal else "",
            targets=tuple(proposal.targets) if proposal else (),
            focus=focus,
            fixed=fixed,
            broken=broken,
        )
        rounds.append(record)
        if self.on_round is not None:
            self.on_round(record)
        if self.store is None:
            return
        self.store.record_ledger(
            {
                "loop_id": loop_id,
                "round": number,
                "decision": decision,
                "hint": record.hint,
                "rationale": record.rationale,
                "targets": ",".join(record.targets),
                "reasons": " | ".join(record.reasons),
                "parent_version": parent.label,
                "candidate_version": candidate.label if candidate else "",
                "baseline_run_id": baseline.run_id,
                "trial_run_id": record.trial_run_id,
                "train_before": baseline.train_accuracy,
                "train_after": after.train_accuracy,
                "holdout_before": baseline.holdout_accuracy,
                "holdout_after": after.holdout_accuracy,
                "hallucination_after": after.hallucination_rate,
                "false_refusal_after": after.false_refusal_rate,
                "prompt_fingerprint": after.prompt_fingerprint,
                "proposer": getattr(self.proposer, "name", ""),
                "generator": generator_name,
                "knowledge_json": (candidate or parent).model_dump_json(),
                "focus": focus,
                "fixed": ",".join(fixed),
                "broken": ",".join(broken),
            }
        )
