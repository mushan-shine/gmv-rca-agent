"""L3 改进循环唯一被允许修改的东西:**提示与示例,不是代码**。

沿用原简报 §5 的边界 —— 改进循环改知识,不改逻辑。这里的「知识」是
``knowledge/nl2sql/prompting.yaml``:一组口径提示 + 一个 few-shot 示例库。

闭环长这样::

    评估跑批 ──► nl2sql_attempts(失败记录)
                      │
                      ▼
              按失败类型归因 ──► 给失败 case 补一条 few-shot 示例 / 一条提示
                      │              (带上 source_run 与 fixes_case:哪次评估、为了哪条 case)
                      ▼
              version += 1,写回 YAML ──► Git PR
                      │
                      ▼
              重跑评估 + **留出集** ──► 指标没退步才允许合并
                      │
                      ▼
              下一轮生成时 build_prompt() 读到新示例   ← 「下一轮读上一轮写下的状态」

``source_run`` / ``fixes_case`` 不是装饰:没有它们,知识库过几周就会变成
一堆没人说得清来历的提示,而「这条提示当初解决了什么问题、现在还需不需要」
正是评估驱动改进与凭感觉调 prompt 的分界线。

留出集(``holdout: true`` 的 case)**不参与**归因 —— 它专门用来回答
「这次改进是真的泛化了,还是只是把已知失败背下来了」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..errors import RcaError
from .generate import FewShotExample

_STRICT = ConfigDict(extra="forbid")

DEFAULT_PATH = Path("knowledge/nl2sql/prompting.yaml")


class PromptKnowledgeError(RcaError):
    """prompting.yaml 不合法。"""


class Hint(BaseModel):
    model_config = _STRICT

    text: str
    source_run: str = ""
    fixes_case: str = ""


class Example(BaseModel):
    model_config = _STRICT

    question: str
    sql: str
    source_run: str = ""
    fixes_case: str = ""


class PromptKnowledge(BaseModel):
    """一个**有版本**的提示与示例集合。

    版本号是回归闸门的锚点:``nl2sql_eval_runs.knowledge_version`` 记的就是它,
    「v3 比 v2 差」这句话才有意义。
    """

    model_config = _STRICT

    version: int = 1
    hints: list[Hint] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)

    def few_shots(self) -> tuple[FewShotExample, ...]:
        return tuple(
            FewShotExample(
                question=example.question,
                sql=example.sql,
                source_run=example.source_run,
                fixes_case=example.fixes_case,
            )
            for example in self.examples
        )

    def hint_texts(self) -> tuple[str, ...]:
        return tuple(hint.text for hint in self.hints)

    @property
    def label(self) -> str:
        return f"v{self.version}"


def load_prompt_knowledge(path: str | Path = DEFAULT_PATH) -> PromptKnowledge:
    """加载提示知识。文件不存在时返回空的 v1 —— 冷启动是合法状态。"""
    path = Path(path)
    if not path.is_file():
        return PromptKnowledge()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PromptKnowledgeError(f"YAML 解析失败 {path}: {exc}") from None
    try:
        return PromptKnowledge.model_validate(payload)
    except ValidationError as exc:
        raise PromptKnowledgeError(f"{path} 校验失败:\n{exc}") from None


def save_prompt_knowledge(knowledge: PromptKnowledge, path: str | Path = DEFAULT_PATH) -> Path:
    """写回 YAML。改动应当走 Git PR,而不是直接改线上。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": knowledge.version,
        "hints": [hint.model_dump(exclude_defaults=True) for hint in knowledge.hints],
        "examples": [example.model_dump(exclude_defaults=True) for example in knowledge.examples],
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


@dataclass(frozen=True)
class Improvement:
    """一次候选改进。还没被接受 —— 要先过回归闸门。"""

    knowledge: PromptKnowledge
    added_hints: tuple[str, ...]
    added_examples: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not self.added_hints and not self.added_examples


def propose_improvement(
    current: PromptKnowledge,
    *,
    run_id: str,
    failing_cases: Iterable[dict[str, Any]],
    reference_sql: dict[str, str],
    questions: dict[str, str],
    max_examples: int = 3,
) -> Improvement:
    """根据一次评估的失败,提出一个候选知识版本。

    做法刻意保守:**给失败的 case 补一条 few-shot 示例**(用它的参考 SQL),
    并按失败类型补一条提示。不改代码、不改指标定义、不动留出集。

    Args:
        current: 当前知识版本。
        run_id: 产生这些失败的评估批次号 —— 会写进每条新增项的 ``source_run``。
        failing_cases: ``StateStore.cases_needing_help`` 的输出。
        reference_sql: case_id -> 参考 SQL(标准答案那条)。
        questions: case_id -> 问题原文。
        max_examples: 一次最多加几条示例。加太多会把 prompt 撑爆,
            也会让「是哪一条起了作用」无从判断。

    Returns:
        :class:`Improvement`;``is_empty`` 为真表示这一轮没什么可补的。
    """
    known_questions = {example.question for example in current.examples}
    known_hints = {hint.text for hint in current.hints}

    examples = list(current.examples)
    hints = list(current.hints)
    added_examples: list[str] = []
    added_hints: list[str] = []

    for failure in failing_cases:
        if len(added_examples) >= max_examples:
            break
        case_id = str(failure.get("case_id", ""))
        question = questions.get(case_id)
        sql = reference_sql.get(case_id)
        if not question or not sql or question in known_questions:
            continue
        examples.append(
            Example(question=question, sql=sql, source_run=run_id, fixes_case=case_id)
        )
        known_questions.add(question)
        added_examples.append(case_id)

        text = _hint_for(str(failure.get("last_failure_kind", "")))
        if text and text not in known_hints:
            hints.append(Hint(text=text, source_run=run_id, fixes_case=case_id))
            known_hints.add(text)
            added_hints.append(text)

    if not added_examples and not added_hints:
        return Improvement(current, (), ())

    candidate = PromptKnowledge(
        version=current.version + 1, hints=hints, examples=examples
    )
    return Improvement(candidate, tuple(added_hints), tuple(added_examples))


def _hint_for(failure_kind: str) -> str:
    """把失败类型翻译成一条可执行的提示。

    刻意写死成查表而不是让 LLM 自由发挥 —— 提示库必须可审查、可回滚。
    """
    return {
        "hallucination": (
            "Only use tables and columns that appear verbatim in the schema block. "
            "If the data you need is not there, refuse with CANNOT_ANSWER."
        ),
        "guard": (
            "Return a single read-only statement. Do not add trailing statements, "
            "comments with semicolons, or DDL."
        ),
        "execution": (
            "Qualify every table with catalog.schema as shown, and make sure every "
            "column you reference belongs to a table in the FROM clause."
        ),
        "refused": (
            "Refuse only when the schema genuinely lacks the data. If the question "
            "can be answered from the listed tables, write the SQL."
        ),
    }.get(failure_kind, "")
