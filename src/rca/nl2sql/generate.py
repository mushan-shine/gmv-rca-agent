"""SQL 生成器:唯一用到 LLM 的地方。

设计上刻意把「生成」压缩成一个**可替换的小接口**::

    class SqlGenerator(Protocol):
        def generate(self, context: PromptContext) -> Generation: ...

好处有三:

1. **循环、守卫、评估全都能在没有 LLM 的情况下跑通并被测试** ——
   用 :class:`ReferenceGenerator` / :class:`ScriptedGenerator` 就行。
   这直接决定了项目能不能在 LLM 到位之前先把工程做完。
2. 换模型、换供应商、换 prompt 策略都只动这一个文件;
3. :class:`ReferenceGenerator` 还是**评估器自身的校验**:
   一个「永远给出正确 SQL」的生成器必须拿到满分,
   否则说明是评估逻辑有问题,而不是模型不行。

prompt 里给模型的东西全部来自 ``knowledge/`` —— schema、口径说明、few-shot 示例。
其中 few-shot 示例由**改进循环**从失败 case 沉淀而来(见 :mod:`rca.nl2sql.knowledge_store`),
所以「下一轮生成」读的是「上一轮评估」写下的东西。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

from ..dialects import SqlDialect
from ..knowledge import Knowledge
from .llm import ChatClient, UsageMeter

REFUSAL_PREFIX = "CANNOT_ANSWER"
"""模型判断问题答不了时必须原样输出的前缀。

用一个**结构化的标记**而不是自由文本,是为了让「拒答」成为可被确定性识别的动作 ——
评估里的「正确拒答率」依赖它。
"""


# ---------------------------------------------------------------------------
# prompt 的输入
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FewShotExample:
    """一个 few-shot 示例。

    ``source_run`` / ``fixes_case`` 是**物证**:这条示例是哪一次评估、
    因为哪条 case 失败而被加进来的。没有它,知识库就只是一堆凭感觉写的提示。
    """

    question: str
    sql: str
    source_run: str = ""
    fixes_case: str = ""


@dataclass(frozen=True)
class PromptContext:
    """生成一次 SQL 所需的全部输入。"""

    question: str
    schema_text: str
    hints: tuple[str, ...] = ()
    examples: tuple[FewShotExample, ...] = ()
    feedback: tuple[str, ...] = ()
    """上一轮失败的**具体**原因。自修复循环靠它收敛。"""

    attempt_no: int = 1
    calendar: tuple[str, ...] = ()
    """报告日历:「今天」是哪天、「上周」是哪几天。见 :func:`render_reporting_calendar`。"""

    def with_feedback(self, message: str) -> "PromptContext":
        return PromptContext(
            question=self.question,
            schema_text=self.schema_text,
            hints=self.hints,
            examples=self.examples,
            feedback=(*self.feedback, message),
            attempt_no=self.attempt_no + 1,
            calendar=self.calendar,
        )


@dataclass(frozen=True)
class Generation:
    """生成器的一次输出。"""

    sql: str | None = None
    refusal_reason: str | None = None
    raw: str = ""
    model: str = ""

    error: str = ""
    """模型侧的问题(回复被截断、返回空)。**不是** SQL 本身的错 ——
    那是守卫的事。置了这个字段,循环会把它当作本轮失败并原样反馈回去。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cached: bool = False

    @property
    def refused(self) -> bool:
        return self.sql is None and not self.error


@runtime_checkable
class SqlGenerator(Protocol):
    name: str

    def generate(self, context: PromptContext) -> Generation:
        ...


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------
def render_schema(knowledge: Knowledge, dialect: SqlDialect) -> str:
    """把 ``tables.yaml`` 渲染成给模型看的 schema 说明。

    用完全限定名,免得模型写出能解析、但指向别的 schema 的表名。
    """
    blocks: list[str] = []
    for table in knowledge.tables.values():
        columns = ", ".join(
            f"{name} {column.type.upper()}" for name, column in table.columns.items()
        )
        note = f"  -- {table.grain}"
        blocks.append(f"{dialect.qualify(table.name)}({columns}){note}")
    return "\n".join(blocks)


def render_metric_glossary(knowledge: Knowledge) -> list[str]:
    """把指标口径变成提示行。

    口径是 text-to-SQL 最容易错、也最难从 schema 里看出来的东西:
    「GMV」到底算不算取消单、「转化率」的分母是会话还是访客 ——
    这些只有知识库知道,必须显式喂给模型。
    """
    lines: list[str] = []
    for name in sorted(knowledge.metrics):
        metric = knowledge.metric(name)
        if metric.is_additive:
            where = ""
            if metric.filters_default:
                where = " WHERE " + " AND ".join(
                    f"{column} = '{value}'"
                    for column, value in sorted(metric.filters_default.items())
                )
            lines.append(
                f"{name} ({metric.display_name}) = {metric.sql} "
                f"FROM {metric.source_table}{where}"
            )
        else:
            assert metric.ratio is not None
            lines.append(
                f"{name} ({metric.display_name}) = {metric.ratio.numerator} / "
                f"{metric.ratio.denominator}"
            )
    return lines


def render_reporting_calendar(as_of: date) -> tuple[str, ...]:
    """把「今天」翻译成相对时间词对应的具体日期。

    为什么需要它 —— 第一次接真实模型评估时,26 条 case 里 22 条答错,
    模型写的是 ``dt BETWEEN DATEADD(day, -7, CURRENT_DATE()) AND CURRENT_DATE()``:
    把「上周」理解成了真实世界的上周。数据却是一段历史快照,于是查到空集,
    SQL 照样跑通、返回 NULL。换一个人类分析师,拿到同样的上下文也会这么写 ——
    这是上下文缺失,不是模型能力问题。

    真实系统里「上周」「上个月」由语义层 / 报表日历统一定义,所以这不算泄题:
    给的是日历规则,不是任何一条问题的答案。周从周一开始、到周日结束。
    """
    this_monday = as_of - timedelta(days=as_of.weekday())
    last_monday = this_monday - timedelta(days=7)
    prior_monday = last_monday - timedelta(days=7)
    fmt = lambda d: d.isoformat()  # noqa: E731
    return (
        f"Today is {fmt(as_of)} ({as_of.strftime('%A')}). The data is a historical snapshot "
        f"that ends on {fmt(as_of - timedelta(days=1))}.",
        "Weeks run Monday to Sunday.",
        f'"last week" means {fmt(last_monday)} to {fmt(last_monday + timedelta(days=6))} inclusive.',
        f'"the week before (last week)" and "two weeks ago" mean '
        f"{fmt(prior_monday)} to {fmt(prior_monday + timedelta(days=6))} inclusive.",
        "Filter dates with literal bounds, e.g. dt BETWEEN DATE 'YYYY-MM-DD' AND DATE 'YYYY-MM-DD'. "
        "NEVER use CURRENT_DATE(), CURRENT_TIMESTAMP(), NOW() or GETDATE() - "
        "they point at the real-world date, where there is no data.",
    )


def build_prompt(context: PromptContext, dialect_name: str = "Databricks SQL") -> str:
    """把 :class:`PromptContext` 拼成完整 prompt。

    刻意写成纯函数:prompt 的每一个字都可复现、可 diff、可进 Git ——
    与简报约束 3「每个数字都能回溯到确定性生成的 SQL」是同一条纪律。
    """
    parts = [
        f"You are a careful data analyst. Write ONE read-only {dialect_name} query "
        f"that answers the user's question.",
        "",
        "## Schema",
        context.schema_text,
    ]
    if context.hints:
        parts += ["", "## Metric definitions and gotchas"]
        parts += [f"- {hint}" for hint in context.hints]

    if context.calendar:
        parts += ["", "## Reporting calendar"]
        parts += [f"- {line}" for line in context.calendar]

    parts += [
        "",
        "## Rules",
        "1. Output ONLY the SQL, wrapped in a ```sql code fence. No explanation.",
        "2. Use fully qualified table names exactly as given in the schema.",
        "3. Return exactly the columns the question asks for - no extra columns.",
        "4. Never write INSERT / UPDATE / DELETE / DROP / CREATE.",
        f"5. If the schema genuinely cannot answer the question, reply with a single "
        f"line `{REFUSAL_PREFIX}: <short reason>` and no SQL. "
        f"Do NOT invent columns or tables.",
    ]

    if context.examples:
        parts += ["", "## Examples"]
        for example in context.examples:
            parts += [f"Q: {example.question}", f"A:\n```sql\n{example.sql}\n```", ""]

    if context.feedback:
        parts += [
            "",
            "## Your previous attempt failed",
            "Fix exactly these problems. Do not change anything else.",
        ]
        parts += [f"- {item}" for item in context.feedback]

    parts += ["", "## Question", context.question]
    return "\n".join(parts)


def parse_response(text: str, model: str = "") -> Generation:
    """把模型的自由文本回复解析成结构化的 :class:`Generation`。

    容忍三种常见格式:```sql 围栏、裸 SQL、以及拒答行。
    解析失败时返回空 SQL —— 交给守卫报「生成结果为空」,而不是在这里猜。
    """
    raw = (text or "").strip()
    if not raw:
        return Generation(sql=None, refusal_reason=None, raw=raw, model=model)

    refusal = re.search(rf"{REFUSAL_PREFIX}\s*[::]?\s*(.*)", raw, re.IGNORECASE)
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", raw, re.DOTALL | re.IGNORECASE)
    if refusal and not fenced:
        return Generation(
            sql=None,
            refusal_reason=refusal.group(1).strip() or "(未给出理由)",
            raw=raw,
            model=model,
        )
    if fenced:
        return Generation(sql=fenced.group(1).strip(), raw=raw, model=model)
    if re.match(r"^\s*(SELECT|WITH)\b", raw, re.IGNORECASE):
        return Generation(sql=raw, raw=raw, model=model)
    return Generation(sql="", raw=raw, model=model)


# ---------------------------------------------------------------------------
# 不需要 LLM 的生成器
# ---------------------------------------------------------------------------
@dataclass
class ReferenceGenerator:
    """永远返回正确答案的「作弊」生成器。

    它的用途不是刷分,而是**校验评估器本身**:
    如果一个永远正确的生成器拿不到满分,那问题出在比对逻辑、标准答案或守卫上,
    而不是模型上。任何评估流水线都应该有这么一条基线。
    """

    answers: Mapping[str, str]
    """问题 -> 正确 SQL。"""

    refusals: Mapping[str, str] = field(default_factory=dict)
    """问题 -> 拒答理由。"""

    name: str = "reference"

    def generate(self, context: PromptContext) -> Generation:
        if context.question in self.refusals:
            return Generation(
                sql=None,
                refusal_reason=self.refusals[context.question],
                raw=f"{REFUSAL_PREFIX}: {self.refusals[context.question]}",
                model=self.name,
            )
        sql = self.answers.get(context.question)
        if sql is None:
            return Generation(sql="", raw="", model=self.name)
        return Generation(sql=sql, raw=f"```sql\n{sql}\n```", model=self.name)


@dataclass
class ScriptedGenerator:
    """按剧本逐轮返回预设回复,用来测试自修复循环。

    第 N 次调用返回剧本里的第 N 条 —— 于是可以精确构造
    「第一轮写了不存在的列、第二轮改对」这类场景,并断言循环确实收敛了。
    """

    script: Mapping[str, Sequence[str]]
    """问题 -> 依次返回的原始回复。"""

    name: str = "scripted"
    calls: dict[str, int] = field(default_factory=dict)

    def generate(self, context: PromptContext) -> Generation:
        index = self.calls.get(context.question, 0)
        self.calls[context.question] = index + 1
        responses = self.script.get(context.question, [])
        if index >= len(responses):
            return Generation(sql="", raw="", model=self.name)
        return parse_response(responses[index], model=self.name)


@dataclass
class CallableGenerator:
    """把任意 ``str -> str`` 的函数包成生成器。

    接 LLM 时最省事的路径:``CallableGenerator(lambda prompt: my_llm(prompt))``。
    """

    call: Callable[[str], str]
    name: str = "callable"
    dialect_name: str = "Databricks SQL"
    prompts: list[str] = field(default_factory=list)

    def generate(self, context: PromptContext) -> Generation:
        prompt = build_prompt(context, self.dialect_name)
        self.prompts.append(prompt)
        return parse_response(self.call(prompt), model=self.name)


@dataclass
class LlmSqlGenerator:
    """真实 LLM 生成器。

    它做的事很薄 —— 拼 prompt、查预算、调模型、解析回复。
    重试、缓存、超时都在 :mod:`rca.nl2sql.llm` 里,循环层看不见,
    因为**传输层重发同一个 prompt 不该被记成一次「尝试」**。

    Args:
        client: 任何讲 OpenAI chat-completions 的客户端(见 :func:`rca.nl2sql.llm.build_chat_client`)。
        meter: 用量与预算。传了就会在每次调用**之前**检查上限。
        dialect_name: 写进 prompt 的方言名。
    """

    client: ChatClient
    meter: UsageMeter | None = None
    dialect_name: str = "Databricks SQL"
    last_prompt: str = field(default="", repr=False)

    @property
    def name(self) -> str:
        """写进状态表的生成器标识 —— **必须带模型名**。

        否则指标变化无法归因:是换了知识版本,还是换了模型?
        """
        return f"llm:{self.client.model}"

    def generate(self, context: PromptContext) -> Generation:
        prompt = build_prompt(context, self.dialect_name)
        self.last_prompt = prompt

        if self.meter is not None:
            self.meter.check_budget()   # 先检查再花钱
        response = self.client.complete(prompt)
        if self.meter is not None:
            self.meter.record(response)

        usage = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_ms": response.latency_ms,
            "cached": response.cached,
        }

        if response.filtered:
            # 被供应商的内容安全策略拦下。告诉模型「放进代码块」没有用,
            # 只能原样说明,由人去看是哪个问题触发了拦截。
            return Generation(
                sql="",
                raw=response.text,
                model=response.model,
                error=(
                    "上一轮回复被模型供应商的内容安全策略拦截,没有返回内容。"
                    "请只输出一条只读 SQL。"
                ),
                **usage,
            )

        if response.truncated:
            # 截断的回复里往往是半条 SQL。让它进守卫会得到一个误导性的语法错,
            # 模型下一轮会去「修」一条本来没写完的 SQL。直接说清楚是被截断了。
            return Generation(
                sql="",
                raw=response.text,
                model=response.model,
                error=(
                    "上一轮回复被 max_tokens 截断,SQL 不完整。"
                    "请只输出 SQL、不要任何解释文字。"
                ),
                **usage,
            )

        parsed = parse_response(response.text, model=response.model)
        if parsed.sql == "" and not parsed.refused:
            return Generation(
                sql="",
                raw=response.text,
                model=response.model,
                error="回复里没有找到 SQL。请把 SQL 放在 ```sql 代码块里。",
                **usage,
            )
        return Generation(
            sql=parsed.sql,
            refusal_reason=parsed.refusal_reason,
            raw=parsed.raw,
            model=parsed.model,
            **usage,
        )
