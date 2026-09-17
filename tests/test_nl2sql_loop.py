"""循环本身:自修复、评估、状态表、改进闭环。

这组测试跑在真实数据上,并且和其它测试一样可以切换运行目标::

    pytest -q tests/test_nl2sql_loop.py --rca-target=spark

最要紧的两条是:

* :func:`test_reference_generator_scores_perfectly` —— **校验评估器本身**。
  一个永远正确的生成器必须拿满分,否则问题出在比对/标准答案/守卫上,不在模型上。
* :func:`test_valid_but_wrong_sql_is_caught_only_by_the_evaluator` —— 划清
  L1 与 L2 的职责:自修复修的是「跑不跑得通」,正确性只能靠评估集。
"""

from __future__ import annotations

import pytest

from rca.nl2sql.evaluate import (
    Evaluator,
    question_map,
    reference_sql_map,
    regression_gate,
)
from rca.nl2sql.generate import (
    REFUSAL_PREFIX,
    Generation,
    PromptContext,
    ScriptedGenerator,
)
from rca.nl2sql.knowledge_store import PromptKnowledge, propose_improvement
from rca.nl2sql.llm import (
    LlmResponse,
    LlmTransportError,
    StaticChatClient,
    UsageMeter,
)
from rca.nl2sql.generate import LlmSqlGenerator
from rca.nl2sql.loop import DEFAULT_MAX_ATTEMPTS, Status, StopReason


# ---------------------------------------------------------------------------
# 自修复内循环
# ---------------------------------------------------------------------------
def test_self_repair_converges_using_the_guard_feedback(make_loop, target):
    """第一轮写了不存在的列,第二轮照着守卫的反馈改对。

    这才叫循环:反馈是**具体的**(哪个列不存在、这张表上有什么),
    而不是把同一个 prompt 再发一遍。
    """
    table = target.dialect.qualify("fact_orders")
    question = "What was total GMV?"
    generator = ScriptedGenerator(
        {
            question: [
                f"```sql\nSELECT SUM(revenue) AS gmv FROM {table}\n```",
                f"```sql\nSELECT SUM(order_amount) AS gmv FROM {table}"
                f" WHERE order_status = 'COMPLETED'\n```",
            ]
        }
    )
    outcome = make_loop(generator).answer(question)

    assert outcome.status is Status.ANSWERED
    assert outcome.attempts_used == 2
    assert not outcome.solved_first_try
    assert outcome.attempts[0].failure_kind == "hallucination"
    assert outcome.attempts[1].ok
    assert outcome.rows


def test_repair_feedback_names_the_offending_identifier(make_loop, target, knowledge):
    """反馈必须具体到能照着改 —— 这是循环与重试的分界。"""
    from rca.nl2sql.guard import SqlGuard

    guard = SqlGuard(knowledge, dialect=target.dialect.name)
    feedback = guard.check("SELECT revenue FROM fact_orders").feedback()
    assert "revenue" in feedback and "order_amount" in feedback


def test_execution_error_is_fed_back_to_the_generator(make_loop, target):
    """守卫放行但引擎报错时,引擎的原话也要进下一轮 prompt。"""
    table = target.dialect.qualify("fact_orders")
    question = "q"
    # 列都存在,但 GROUP BY 缺失 —— 静态守卫抓不到,只有引擎会报错
    generator = ScriptedGenerator(
        {
            question: [
                f"```sql\nSELECT channel, SUM(order_amount) FROM {table}\n```",
                f"```sql\nSELECT channel, SUM(order_amount) FROM {table} GROUP BY channel\n```",
            ]
        }
    )
    outcome = make_loop(generator).answer(question)
    assert outcome.status is Status.ANSWERED
    assert outcome.attempts[0].failure_kind == "execution"
    assert outcome.attempts[0].execution_error


def test_loop_stops_at_the_attempt_budget(make_loop):
    """预算熔断:不能无限重试,否则「模型其实不会」会被伪装成「多试几次就好」。"""
    question = "q"
    generator = ScriptedGenerator(
        {question: ["```sql\nSELECT nope FROM fact_orders\n```"] * 10}
    )
    outcome = make_loop(generator).answer(question)

    assert outcome.status is Status.FAILED
    assert outcome.stop_reason is StopReason.MAX_ATTEMPTS
    assert outcome.attempts_used == DEFAULT_MAX_ATTEMPTS
    assert outcome.any_hallucination


def test_refusal_stops_immediately(make_loop):
    """拒答是终态,不该再修三轮。"""
    question = "q"
    generator = ScriptedGenerator({question: [f"{REFUSAL_PREFIX}: no campaign_id anywhere"]})
    outcome = make_loop(generator).answer(question)

    assert outcome.status is Status.REFUSED
    assert outcome.stop_reason is StopReason.REFUSED
    assert outcome.attempts_used == 1
    assert "campaign_id" in outcome.refusal_reason


def test_write_statement_never_reaches_the_warehouse(make_loop, target, scalar):
    """生成器写了 DROP,必须在执行前被拦下 —— 表还得在。"""
    table = target.dialect.qualify("fact_orders")
    question = "q"
    generator = ScriptedGenerator({question: [f"```sql\nDROP TABLE {table}\n```"] * 5})
    outcome = make_loop(generator).answer(question)

    assert outcome.status is Status.FAILED
    assert scalar("SELECT COUNT(*) FROM {fact_orders}") > 0


def test_loop_never_raises_on_failure(make_loop):
    """失败是正常结果,评估要统计它 —— 不能把整批评估打断。"""
    question = "q"
    generator = ScriptedGenerator({question: []})
    outcome = make_loop(generator).answer(question)
    assert outcome.status is Status.FAILED


def test_max_rows_caps_the_payload(make_loop, target):
    """防止模型一句 SELECT * 把整张表拉回驱动进程。"""
    table = target.dialect.qualify("fact_orders")
    question = "q"
    generator = ScriptedGenerator({question: [f"```sql\nSELECT order_id FROM {table}\n```"]})
    outcome = make_loop(generator, max_rows=5).answer(question)
    assert outcome.status is Status.ANSWERED
    assert len(outcome.rows) == 5
    assert outcome.attempts[-1].row_count > 5, "记录的是真实行数,截断只发生在返回值上"


# ---------------------------------------------------------------------------
# 真实 LLM 接入:传输层失败不能污染循环指标
# ---------------------------------------------------------------------------
def test_transport_retry_is_not_a_self_repair_attempt(make_loop, target):
    """**最关键的一条分界。**

    ``429`` / 超时重发的是**同一个 prompt** —— 模型没有获得任何新信息,
    那是传输层的事,封在客户端里。循环只看到一次成功的生成,
    因此 ``attempts_used`` 必须是 1。

    混同的话,一次网络抖动会被读成「模型改了两轮才对」,
    ``avg_attempts`` 这个指标立刻失去意义。
    """
    table = target.dialect.qualify("fact_orders")
    sql = f"SELECT SUM(order_amount) AS v FROM {table} WHERE order_status = 'COMPLETED'"

    class RetryingClient:
        """前两次抛错、第三次成功 —— 模拟客户端内部已经退避重发过。"""

        model = "m"
        params: dict = {}

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str) -> LlmResponse:
            self.calls += 1
            # 客户端内部重试完毕后才返回,循环看到的就是一次调用
            return LlmResponse(text=f"```sql\n{sql}\n```", model="m", attempts=3)

    client = RetryingClient()
    outcome = make_loop(LlmSqlGenerator(client)).answer("q")

    assert outcome.status is Status.ANSWERED
    assert outcome.attempts_used == 1, "传输层重发不得计入自修复轮数"
    assert outcome.solved_first_try
    assert client.calls == 1


def test_model_call_failure_stops_cleanly_without_breaking_the_batch(make_loop):
    """模型调用失败(重试用尽)必须被记录、停止,但**不抛出** ——
    一次限流不该把整批评估打断。"""

    class DeadClient:
        model = "m"
        params: dict = {}

        def complete(self, prompt: str) -> LlmResponse:
            raise LlmTransportError("HTTP 503: upstream unavailable")

    outcome = make_loop(LlmSqlGenerator(DeadClient())).answer("q")

    assert outcome.status is Status.FAILED
    assert outcome.stop_reason is StopReason.GENERATOR_ERROR
    assert outcome.attempts_used == 1, "调用失败不该再空转两轮"
    assert outcome.attempts[0].failure_kind == "generator"
    assert "503" in (outcome.attempts[0].generation_error or "")


def test_budget_exhaustion_is_reported_as_a_generator_error(make_loop):
    """熔断要能在报告里一眼认出来,而不是被误读成「模型突然变差了」。"""
    client = StaticChatClient({"": "```sql\nSELECT 1\n```"}, model="m")
    generator = LlmSqlGenerator(client, meter=UsageMeter(max_calls=0))

    outcome = make_loop(generator).answer("q")

    assert outcome.stop_reason is StopReason.GENERATOR_ERROR
    assert client.calls == [], "熔断必须发生在花钱之前"


def test_truncated_reply_is_fed_back_and_can_be_repaired(make_loop, target):
    """回复被截断 -> 反馈「只输出 SQL」-> 下一轮写对。这一轮**算**一次自修复。"""
    table = target.dialect.qualify("fact_orders")
    sql = f"SELECT SUM(order_amount) AS v FROM {table} WHERE order_status = 'COMPLETED'"

    class TruncateThenFix:
        model = "m"
        params: dict = {}

        def __init__(self) -> None:
            self.n = 0

        def complete(self, prompt: str) -> LlmResponse:
            self.n += 1
            if self.n == 1:
                return LlmResponse(text="SELECT SUM(order_amo", model="m", finish_reason="length")
            assert "截断" in prompt, "第二轮的 prompt 里必须带上具体反馈"
            return LlmResponse(text=f"```sql\n{sql}\n```", model="m")

    outcome = make_loop(LlmSqlGenerator(TruncateThenFix())).answer("q")

    assert outcome.status is Status.ANSWERED
    assert outcome.attempts_used == 2
    assert outcome.attempts[0].failure_kind == "generator"


def test_model_and_tokens_land_in_the_attempt_record(make_loop, target):
    """状态表要能回答「这次跑批用的哪个模型、花了多少 token」。"""
    table = target.dialect.qualify("fact_orders")

    class CountingClient:
        model = "llama-3.3-70b"
        params: dict = {}

        def complete(self, prompt: str) -> LlmResponse:
            return LlmResponse(
                text=f"```sql\nSELECT COUNT(*) AS n FROM {table}\n```",
                model="llama-3.3-70b",
                prompt_tokens=800,
                completion_tokens=40,
            )

    outcome = make_loop(LlmSqlGenerator(CountingClient())).answer("q")

    assert outcome.model == "llama-3.3-70b"
    assert outcome.total_tokens == 840
    assert outcome.attempts[0].prompt_tokens == 800


def test_usage_meter_totals_reach_the_eval_report(knowledge, target, make_loop, cases, store):
    """``nl2sql_eval_runs`` 要记下模型与用量 ——
    否则「这一版好了 5 个点」分不清是改了知识还是换了模型。"""
    from rca.nl2sql.cases import build_reference_sql

    case = next(c for c in cases if c.id == "gmv_total_w8")
    sql = build_reference_sql(case, knowledge, target.dialect)

    class FixedClient:
        model = "llama-3.3-70b"
        params: dict = {}

        def complete(self, prompt: str) -> LlmResponse:
            return LlmResponse(
                text=f"```sql\n{sql}\n```",
                model="llama-3.3-70b",
                prompt_tokens=500,
                completion_tokens=60,
            )

    meter = UsageMeter()
    evaluator = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(LlmSqlGenerator(FixedClient(), meter=meter)),
        cases=[case],
        store=store,
        usage=meter,
    )
    report = evaluator.run(knowledge_version="v1", run_id="run_usage_01")

    assert report.pass_final == 1.0
    assert report.model == "llama-3.3-70b"
    row = next(r for r in store.recent_runs(20) if r["run_id"] == "run_usage_01")
    assert row["model"] == "llama-3.3-70b"
    assert int(row["llm_calls"]) == 1
    assert int(row["total_tokens"]) == 560


# ---------------------------------------------------------------------------
# 评估器自身的校验
# ---------------------------------------------------------------------------
@pytest.fixture()
def evaluator(knowledge, target, make_loop, reference_generator, cases, store):
    return Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(reference_generator),
        cases=cases,
        store=store,
    )


def test_reference_generator_scores_perfectly(evaluator):
    """**这是评估器的自检。**

    一个永远给出正确 SQL、并对不可答问题正确拒答的生成器,必须拿满分。
    拿不到就说明是比对逻辑、标准答案或守卫有问题 —— 而不是模型不行。
    任何评估流水线都应该有这么一条基线,否则你无法区分
    「模型差」和「尺子坏了」。
    """
    report = evaluator.run(knowledge_version="v1", run_id="run_baseline", persist=False)

    assert report.pass_final == 1.0, [
        (r.case_id, r.verdict, r.detail) for r in report.failures
    ]
    assert report.pass_at_1 == 1.0
    assert report.correct_refusal_rate == 1.0
    assert report.false_refusal_rate == 0.0
    assert report.hallucination_rate == 0.0
    assert report.holdout_pass_final == 1.0
    assert report.avg_attempts == 1.0


def test_goldens_are_generated_for_every_case(evaluator, cases):
    goldens = evaluator.goldens()
    assert set(goldens) == {case.id for case in cases}
    for case in cases:
        golden = goldens[case.id]
        if case.gold.kind.value == "unanswerable":
            assert not golden.answerable and golden.reason
        else:
            assert golden.answerable and golden.rows and golden.reference_sql


def test_valid_but_wrong_sql_is_caught_only_by_the_evaluator(
    knowledge, target, make_loop, cases, store
):
    """一条语法正确、跑得通、但**口径错了**的 SQL。

    忘了 ``order_status = 'COMPLETED'`` —— 守卫放行、引擎不报错、
    自修复循环完全没机会发现。只有拿程序化标准答案做执行式比对才抓得住。
    这就是 L1 与 L2 的分工。
    """
    case = next(c for c in cases if c.id == "gmv_total_w8")
    table = target.dialect.qualify("fact_orders")
    period = "dt BETWEEN DATE '2025-06-23' AND DATE '2025-06-29'"
    wrong_sql = f"SELECT SUM(order_amount) AS value FROM {table} WHERE {period}"

    class WrongGenerator:
        name = "forgets-the-status-filter"

        def generate(self, context: PromptContext) -> Generation:
            return Generation(sql=wrong_sql, raw=wrong_sql, model=self.name)

    evaluator = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(WrongGenerator()),
        cases=[case],
        store=store,
    )
    report = evaluator.run(run_id="run_wrong")

    assert report.pass_final == 0.0
    result = report.results[0]
    assert result.status is Status.ANSWERED, "SQL 确实跑通了 —— 循环无从察觉"
    assert result.verdict == "wrong_answer"
    assert "value_mismatch" in result.detail

    # 而且它必须出现在失败画像里 —— 循环看不见的失败,改进循环更要看得见
    profile = store.failure_profile(report.run_id)
    assert ("wrong_answer", "none") in {(r["verdict"], r["failure_kind"]) for r in profile}


def test_answering_an_unanswerable_question_is_counted_as_a_failure(
    knowledge, target, make_loop, cases, store
):
    """对着不存在的数据编一个能跑的 SQL,是最危险的失败模式。"""
    case = next(c for c in cases if c.id == "unanswerable_campaign_gmv")
    table = target.dialect.qualify("fact_orders")

    class EagerGenerator:
        name = "eager"

        def generate(self, context: PromptContext) -> Generation:
            sql = f"SELECT channel, SUM(order_amount) FROM {table} GROUP BY channel"
            return Generation(sql=sql, raw=sql, model=self.name)

    evaluator = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(EagerGenerator()),
        cases=[case],
        store=store,
    )
    report = evaluator.run(run_id="run_eager", persist=False)
    assert report.correct_refusal_rate == 0.0
    assert report.results[0].verdict == "should_have_refused"


# ---------------------------------------------------------------------------
# 状态表:下一轮读上一轮写下的东西
# ---------------------------------------------------------------------------
def test_attempts_and_metrics_are_persisted(evaluator, store, cases):
    report = evaluator.run(knowledge_version="v1", run_id="run_persist_01")

    attempts = store.failures_of_run("run_persist_01")
    assert attempts == [], "完美生成器不该留下失败记录"

    runs = store.recent_runs(limit=5)
    recorded = next(row for row in runs if row["run_id"] == "run_persist_01")
    assert float(recorded["pass_final"]) == pytest.approx(1.0)
    assert recorded["knowledge_version"] == "v1"
    assert int(recorded["n_cases"]) == len(cases)
    assert report.run_id == "run_persist_01"


def test_failures_are_queryable_for_the_improvement_loop(
    knowledge, target, make_loop, cases, store
):
    """改进循环的输入就是这张表 —— 它必须能回答「哪些 case 一次都没成功」。"""

    class BlankGenerator:
        name = "blank"

        def generate(self, context: PromptContext) -> Generation:
            return Generation(sql="", raw="", model=self.name)

    subset = [c for c in cases if c.gold.kind.value != "unanswerable"][:3]
    evaluator = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(BlankGenerator()),
        cases=subset,
        store=store,
    )
    evaluator.run(run_id="run_blank_01")

    needing_help = store.cases_needing_help("run_blank_01")
    assert {row["case_id"] for row in needing_help} == {c.id for c in subset}

    profile = store.failure_profile("run_blank_01")
    assert profile and sum(int(row["n"]) for row in profile) == len(subset) * DEFAULT_MAX_ATTEMPTS
    assert {row["verdict"] for row in profile} == {"no_runnable_sql"}


def test_previous_run_is_readable_for_the_regression_gate(evaluator, store):
    """回归闸门比的是「这一版 vs 上一版」,所以必须读得到上一版。"""
    evaluator.run(knowledge_version="v1", run_id="run_gate_a")
    evaluator.run(knowledge_version="v2", run_id="run_gate_b")

    previous = store.previous_run("run_gate_b")
    assert previous is not None
    assert previous["run_id"] != "run_gate_b"


# ---------------------------------------------------------------------------
# 完整的改进闭环
# ---------------------------------------------------------------------------
def test_improvement_loop_closes(knowledge, target, make_loop, cases, store):
    """一整圈:评估 → 失败归因 → 补知识(带来历)→ 版本号 +1 → 回归闸门。

    这就是原简报 §3.1 那句判据的落点:
    下一轮生成读到的 few-shot,是上一轮评估失败写下来的。
    """

    class BlankGenerator:
        name = "model-improve"

        def generate(self, context: PromptContext) -> Generation:
            return Generation(sql="", raw="", model=self.name)

    # 刻意把一条留出集 case 放进来 —— 否则 holdout 指标恒为 0,闸门那一条形同虚设
    answerable = [c for c in cases if c.gold.kind.value != "unanswerable"]
    subset = [c for c in answerable if not c.holdout][:3] + [
        c for c in answerable if c.holdout
    ][:1]
    weak = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(BlankGenerator()),
        cases=subset,
        store=store,
    )
    baseline = weak.run(knowledge_version="v1", run_id="run_improve_v1")
    assert baseline.pass_final == 0.0

    # 归因 -> 候选知识
    improvement = propose_improvement(
        PromptKnowledge(version=1),
        run_id=baseline.run_id,
        failing_cases=store.cases_needing_help(baseline.run_id),
        reference_sql=reference_sql_map(weak),
        questions=question_map(subset),
        max_examples=3,
    )
    assert not improvement.is_empty
    assert improvement.knowledge.version == 2
    for example in improvement.knowledge.examples:
        assert example.source_run == baseline.run_id
        assert example.fixes_case, "每条新增示例都要说清是为哪条 case 加的"

    # 候选版本重跑(这里用完美生成器代表「改进生效了」)
    improved = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(_reference_for(subset, knowledge, target, name="model-improve")),
        cases=subset,
        store=store,
    )
    candidate = improved.run(
        knowledge_version=improvement.knowledge.label, run_id="run_improve_v2"
    )
    assert candidate.pass_final == 1.0
    assert candidate.holdout_pass_final == 1.0

    baseline = store.previous_run("run_improve_v2")
    assert baseline is not None and baseline["run_id"] == "run_improve_v1", (
        "闸门必须拿到紧挨着的上一版,而不是随便一条历史记录"
    )
    decision = regression_gate(candidate, baseline)
    assert decision.accepted, decision.reasons


def test_gate_would_block_a_regression(knowledge, target, make_loop, cases, store):
    """反向:新版本更差时,闸门必须拦住。"""

    class BlankGenerator:
        name = "model-regress"

        def generate(self, context: PromptContext) -> Generation:
            return Generation(sql="", raw="", model=self.name)

    answerable = [c for c in cases if c.gold.kind.value != "unanswerable"]
    subset = [c for c in answerable if not c.holdout][:3] + [
        c for c in answerable if c.holdout
    ][:1]
    good = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(_reference_for(subset, knowledge, target, name="model-regress")),
        cases=subset,
        store=store,
    )
    good.run(knowledge_version="v1", run_id="run_regress_v1")

    bad = Evaluator(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        loop=make_loop(BlankGenerator()),
        cases=subset,
        store=store,
    )
    candidate = bad.run(knowledge_version="v2", run_id="run_regress_v2")

    baseline = store.previous_run("run_regress_v2")
    assert baseline is not None and baseline["run_id"] == "run_regress_v1"
    decision = regression_gate(candidate, baseline)
    assert not decision.accepted
    assert decision.reasons


def _reference_for(subset, knowledge, target, name="reference"):
    from rca.nl2sql.cases import build_reference_sql
    from rca.nl2sql.generate import ReferenceGenerator

    answers = {
        case.question: build_reference_sql(case, knowledge, target.dialect)
        for case in subset
        if case.gold.kind.value != "unanswerable"
    }
    return ReferenceGenerator(answers, name=name)



# ---------------------------------------------------------------------------
# 报告日历 + 回归闸门的基线隔离(第一次接真实模型时踩到的两个坑)
# ---------------------------------------------------------------------------
def test_wall_clock_sql_is_repaired_using_the_reporting_calendar(make_loop, target):
    """模型用 CURRENT_DATE() 查「上周」—— 语法正确、执行成功、返回 NULL。

    第一次接智谱时 26 条里 22 条就是这样答错的,而且 L1 完全没发现。
    设了报告日期之后,这必须成为一次**可修复的**失败,反馈里要给出具体日期写法。
    """
    from datetime import date

    table = target.dialect.qualify("fact_orders")
    question = "What was the total GMV last week?"
    generator = ScriptedGenerator(
        {
            question: [
                f"```sql\nSELECT SUM(order_amount) AS v FROM {table} WHERE order_status = 'COMPLETED' "
                f"AND dt BETWEEN DATEADD(day, -7, CURRENT_DATE()) AND CURRENT_DATE()\n```",
                f"```sql\nSELECT SUM(order_amount) AS v FROM {table} WHERE order_status = 'COMPLETED' "
                f"AND dt BETWEEN DATE '2025-06-23' AND DATE '2025-06-29'\n```",
            ]
        }
    )
    outcome = make_loop(generator, as_of=date(2025, 6, 30)).answer(question)

    assert outcome.status is Status.ANSWERED
    assert outcome.attempts_used == 2
    assert outcome.attempts[0].failure_kind == "guard"
    assert "CURRENT_DATE" in " ".join(outcome.attempts[0].guard_messages)
    assert outcome.rows and outcome.rows[0]["v"] is not None


def test_wall_clock_is_allowed_when_no_report_date_is_set(make_loop, target):
    """没设报告日期(真实的线上数据)时,CURRENT_DATE 是合法写法,不能误伤。"""
    table = target.dialect.qualify("fact_orders")
    generator = ScriptedGenerator(
        {"q": [f"```sql\nSELECT COUNT(*) AS n FROM {table} WHERE dt <= CURRENT_DATE()\n```"]}
    )
    outcome = make_loop(generator).answer("q")
    assert outcome.status is Status.ANSWERED
    assert outcome.attempts_used == 1


def test_prompt_carries_the_reporting_calendar(make_loop):
    from datetime import date

    loop = make_loop(ScriptedGenerator({}), as_of=date(2025, 6, 30))
    context = loop.initial_context("What was GMV last week?")
    joined = " ".join(context.calendar)
    assert "2025-06-23 to 2025-06-29" in joined
    assert "CURRENT_DATE" in joined
    assert context.with_feedback("x").calendar == context.calendar, "修复轮次不能把日历丢掉"


def test_gate_baseline_is_isolated_per_generator(store):
    """真实踩过的坑:接上智谱后第一次评估,被拿去和 ReferenceGenerator 的 100% 比,
    闸门报「全量正确率退步 100% → 0%」。基线必须是**同一个生成器**的上一次。"""
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2030, 1, 1, tzinfo=timezone.utc)
    rows = [
        ("iso_ref_1", "reference", t0),
        ("iso_llm_1", "llm:iso-model", t0 + timedelta(seconds=1)),
        ("iso_ref_2", "reference", t0 + timedelta(seconds=2)),
        ("iso_llm_2", "llm:iso-model", t0 + timedelta(seconds=3)),
    ]
    for run_id, generator, created_at in rows:
        store.record_eval_run(
            {"run_id": run_id, "generator": generator, "created_at": created_at,
             "knowledge_version": "v1", "n_cases": 1, "pass_final": 0.5}
        )

    assert store.previous_run("iso_llm_2")["run_id"] == "iso_llm_1"
    assert store.previous_run("iso_llm_1") is None, "同一个模型之前没跑过,就没有基线"
    assert store.previous_run("iso_llm_2", same_generator=False)["run_id"] == "iso_ref_2"
