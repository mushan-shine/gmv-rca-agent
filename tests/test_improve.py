"""自主改进循环与维度取值检查。

这里的生成器是假的,但**它的对错取决于 prompt 里有没有某条提示** ——
于是「提议 → 试跑 → 裁决 → 台账」的每一步都跑在真实的评估器、真实的状态表上,
而采纳/拒绝的结果是确定的,可以逐条断言。
"""

from __future__ import annotations

import pytest

from rca.nl2sql.cases import build_reference_sql
from rca.nl2sql.domains import ValueDomains, load_value_domains
from rca.nl2sql.evaluate import CaseResult, EvalReport, Evaluator
from rca.nl2sql.generate import Generation, PromptContext, ScriptedGenerator
from rca.nl2sql.improve import (
    AutoImprover,
    HintProposal,
    LlmHintProposer,
    ProposalContext,
    FailureEvidence,
    ScriptedProposer,
    Vocabulary,
    build_proposer_prompt,
    build_vocabulary,
    choose_focus,
    failure_pattern,
    improvement_gate,
    parse_proposal,
)
from rca.nl2sql.knowledge_store import Hint, PromptKnowledge
from rca.nl2sql.llm import StaticChatClient
from rca.nl2sql.loop import AnswerLoop, Status

TRAIN_A = "gmv_total_w8"
TRAIN_B = "sessions_w8"
HOLDOUT = "holdout_orders_by_region_w8"


# ---------------------------------------------------------------------------
# 维度取值检查
# ---------------------------------------------------------------------------
@pytest.fixture()
def domains(knowledge, target) -> ValueDomains:
    return load_value_domains(knowledge, target.executor, target.dialect)


def test_domains_are_read_from_the_warehouse(domains):
    assert domains.allowed("market") == {"US", "UK", "DE"}
    assert domains.allowed("region") == {"NA", "EMEA"}
    assert domains.allowed("order_status") == {"COMPLETED", "CANCELLED", "REFUNDED"}
    assert "order_id" not in domains, "高基数的 ID 列不参与检查"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        # 第二轮真实评估里 glm-4-flash 写出的错误
        ("SELECT 1 FROM fact_orders WHERE region = 'US'", "'US' 实际出现在列:market"),
        ("SELECT 1 FROM fact_orders WHERE market = 'Germany'", "DE, UK, US"),
        ("SELECT 1 FROM fact_orders WHERE fact_orders.device = 'DESKTOP'", "注意大小写"),
        ("SELECT 1 FROM fact_orders o WHERE o.order_status = 'RETURNED'", "CANCELLED, COMPLETED, REFUNDED"),
        ("SELECT 1 FROM fact_orders WHERE market IN ('US', 'Deutschland')", "Deutschland"),
    ],
)
def test_real_model_mistakes_get_actionable_feedback(domains, sql, expected):
    messages = domains.check(sql, dialect="duckdb")
    assert messages, sql
    assert expected in " ".join(messages)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1 FROM fact_orders WHERE market = 'US' AND device = 'mobile'",
        "SELECT 1 FROM fact_orders WHERE LOWER(device) = 'Desktop'",   # 包了函数,语义不确定,放过
        "SELECT 1 FROM fact_orders WHERE order_id = 'o_whatever'",      # 不在取值表里的列
        "SELECT 1 FROM fact_orders WHERE channel LIKE '%Search%'",
    ],
)
def test_valid_or_uncheckable_filters_are_left_alone(domains, sql):
    assert domains.check(sql, dialect="duckdb") == ()


def test_domain_violation_is_repaired_inside_the_loop(make_loop, target, domains):
    """market = 'Germany' 能跑通、返回空 —— 现在它在 L1 里就被拦下并改对。"""
    table = target.dialect.qualify("fact_orders")
    generator = ScriptedGenerator(
        {
            "q": [
                f"```sql\nSELECT SUM(order_amount) AS v FROM {table} WHERE market = 'Germany'\n```",
                f"```sql\nSELECT SUM(order_amount) AS v FROM {table} WHERE market = 'DE'\n```",
            ]
        }
    )
    outcome = make_loop(generator, value_domains=domains).answer("q")

    assert outcome.status is Status.ANSWERED
    assert outcome.attempts_used == 2
    assert outcome.attempts[0].failure_kind == "context"
    assert outcome.rows[0]["v"] is not None


def test_prompt_fingerprint_tracks_hints_and_loop_settings(make_loop, domains):
    """补报告日历前后两次评估都记成 v1,只看状态表看不出为什么 0% → 63.6%。指纹要能看出来。"""
    base = make_loop(ScriptedGenerator({}), hints=("hint one",))
    same = make_loop(ScriptedGenerator({}), hints=("hint one",))
    more_hints = make_loop(ScriptedGenerator({}), hints=("hint one", "hint two"))
    with_domains = make_loop(ScriptedGenerator({}), hints=("hint one",), value_domains=domains)

    assert base.prompt_fingerprint() == same.prompt_fingerprint()
    assert base.prompt_fingerprint() != more_hints.prompt_fingerprint()
    assert base.prompt_fingerprint() != with_domains.prompt_fingerprint()


# ---------------------------------------------------------------------------
# 提议者
# ---------------------------------------------------------------------------
def test_proposal_parses_from_a_fenced_json_block():
    proposal = parse_proposal(
        'Here:\n```json\n{"hint": "Use market for country codes such as US, UK, DE.", '
        '"rationale": "region holds NA/EMEA", "targets": ["gmv_us_w8"]}\n```'
    )
    assert proposal is not None
    assert proposal.hint.startswith("Use market")
    assert proposal.targets == ["gmv_us_w8"]


@pytest.mark.parametrize(
    "reply",
    ["no json here", '```json\n{"hint": "short"}\n```', '```json\n{"rationale": "x"}\n```', "```json\n{bad json}\n```"],
)
def test_invalid_proposals_are_rejected_rather_than_guessed(reply):
    assert parse_proposal(reply) is None


def test_proposer_prompt_carries_rejections_and_never_the_reference_sql(knowledge, target, cases):
    case = next(c for c in cases if c.id == TRAIN_A)
    reference_sql = build_reference_sql(case, knowledge, target.dialect)
    context = ProposalContext(
        failures=(
            FailureEvidence(TRAIN_A, case.question, "wrong_answer", "value_mismatch", "SELECT 0", ""),
        ),
        current_hints=("existing rule",),
        rejected=(("a rule that did not help", "训练集没有多答对"),),
    )
    prompt = build_proposer_prompt(context)
    assert "a rule that did not help" in prompt
    assert "训练集没有多答对" in prompt
    assert reference_sql not in prompt, "给了参考 SQL,提议者最省事的做法就是把答案抄进提示"


def test_llm_proposer_uses_the_client():
    client = StaticChatClient(
        {"": '```json\n{"hint": "Compute rates as fractions between 0 and 1.", "targets": []}\n```'},
        model="glm-4-flash",
    )
    proposer = LlmHintProposer(client)
    proposal = proposer.propose(ProposalContext(failures=(), current_hints=(), rejected=()))
    assert proposal is not None and "fractions" in proposal.hint
    assert proposer.name == "llm-proposer:glm-4-flash"


# ---------------------------------------------------------------------------
# 裁决
# ---------------------------------------------------------------------------
def _report(results) -> EvalReport:
    from datetime import datetime, timezone

    return EvalReport("r", "g", "v1", datetime.now(timezone.utc), tuple(results))


def _result(case_id, *, correct, holdout=False, status=Status.ANSWERED):
    return CaseResult(case_id, case_id, True, holdout, status, correct, 1, False)


def test_gate_requires_a_strict_train_gain():
    """持平不采纳:没有证据的改动只会让 prompt 越来越长。"""
    base = _report([_result("a", correct=True), _result("b", correct=False)])
    same = _report([_result("a", correct=True), _result("b", correct=False)])
    better = _report([_result("a", correct=True), _result("b", correct=True)])
    assert not improvement_gate(same, base).accepted
    assert improvement_gate(better, base).accepted


def test_gate_blocks_train_gain_that_costs_holdout():
    """训练集涨了、留出集掉了 —— 背答案,必须拒绝。"""
    base = _report([_result("a", correct=False), _result("h", correct=True, holdout=True)])
    overfit = _report([_result("a", correct=True), _result("h", correct=False, holdout=True)])
    decision = improvement_gate(overfit, base)
    assert not decision.accepted
    assert any("留出集" in r for r in decision.reasons)


def test_gate_blocks_gains_bought_with_false_refusals():
    base = _report([_result("a", correct=False), _result("b", correct=True)])
    refuse = _report([_result("a", correct=True), _result("b", correct=False, status=Status.REFUSED)])
    assert not improvement_gate(refuse, base).accepted


# ---------------------------------------------------------------------------
# 完整循环
# ---------------------------------------------------------------------------
class HintSensitiveGenerator:
    """对错取决于 prompt 里的提示:这就是改进循环要影响的那个变量。"""

    def __init__(self, name, answers, rules):
        self.name = name
        self.answers = answers          # question -> 正确 SQL
        self.rules = rules              # question -> callable(hints) -> bool

    def generate(self, context: PromptContext) -> Generation:
        hints = " ".join(context.hints)
        if self.rules[context.question](hints):
            sql = self.answers[context.question]
        else:
            sql = "SELECT 0 AS wrong"
        return Generation(sql=sql, raw=sql, model=self.name)


@pytest.fixture()
def loop_setup(knowledge, target, cases, store, make_loop):
    subset = [c for c in cases if c.id in (TRAIN_A, TRAIN_B, HOLDOUT)]
    by_id = {c.id: c for c in subset}
    answers = {c.question: build_reference_sql(c, knowledge, target.dialect) for c in subset}

    def build(name, rules_by_id):
        rules = {by_id[k].question: v for k, v in rules_by_id.items()}
        generator = HintSensitiveGenerator(name, answers, rules)
        evaluator = Evaluator(
            knowledge=knowledge,
            dialect=target.dialect,
            executor=target.executor,
            loop=make_loop(generator),
            cases=subset,
            store=store,
        )

        def factory(prompt_knowledge: PromptKnowledge) -> AnswerLoop:
            return make_loop(generator, hints=prompt_knowledge.hint_texts() or ("seed",))

        return evaluator, factory

    return build


def _proposal(text, *targets):
    return HintProposal(hint=text, rationale="because", targets=list(targets))


def test_the_loop_accepts_what_helps_rejects_what_does_not_and_remembers(loop_setup, store):
    evaluator, factory = loop_setup(
        "gen-closes",
        {
            TRAIN_A: lambda h: "RULE-A" in h,
            TRAIN_B: lambda h: "RULE-B" in h,
            HOLDOUT: lambda h: "RULE-A" in h,     # RULE-A 能泛化到没见过的题
        },
    )
    proposer = ScriptedProposer(
        [
            _proposal("RULE-X a guideline that changes nothing"),
            _proposal("RULE-A always filter completed orders", TRAIN_A),
            _proposal("RULE-B count sessions from the sessions table", TRAIN_B),
        ]
    )
    improver = AutoImprover(evaluator, factory, proposer, store=store, max_rounds=6, patience=3, target_accuracy=1.0)
    result = improver.run(PromptKnowledge(version=1), loop_id="loop_closes")

    assert [r.decision for r in result.rounds] == ["rejected", "accepted", "accepted"]
    assert result.stop_reason == "target_reached"
    assert result.baseline.train_accuracy == 0.0
    assert result.final.train_accuracy == 1.0 and result.final.holdout_accuracy == 1.0
    assert result.final_knowledge.version == 3
    assert result.final_knowledge.hint_texts() == (
        "RULE-A always filter completed orders",
        "RULE-B count sessions from the sessions table",
    ), "被拒绝的提示不能混进最终知识"

    # 采纳的提示带来历
    assert result.final_knowledge.hints[0].source_run.startswith("loop_closes_r")

    # 台账:每个候选一行,含被拒绝的
    ledger = store.ledger("loop_closes")
    assert [row["decision"] for row in ledger] == ["rejected", "accepted", "accepted"]
    assert ledger[0]["hint"].startswith("RULE-X")
    assert float(ledger[1]["train_after"]) > float(ledger[1]["train_before"])

    # 第 2 轮提议时,已经知道第 1 轮被拒绝了什么
    assert any("RULE-X" in hint for hint, _ in proposer.seen[1].rejected)


def test_the_proposer_never_sees_holdout_failures(loop_setup, store):
    evaluator, factory = loop_setup(
        "gen-holdout-blind",
        {TRAIN_A: lambda h: False, TRAIN_B: lambda h: False, HOLDOUT: lambda h: False},
    )
    proposer = ScriptedProposer([None, None])
    AutoImprover(evaluator, factory, proposer, store=store, patience=2).run(
        PromptKnowledge(version=1), loop_id="loop_blind"
    )
    shown = {failure.case_id for context in proposer.seen for failure in context.failures}
    assert HOLDOUT not in shown
    assert shown == {TRAIN_A, TRAIN_B}


def test_overfitting_candidate_is_rejected_by_the_holdout(loop_setup, store):
    """一条让训练题变对、却让留出题变错的提示 —— 背答案,闸门必须拒。"""
    evaluator, factory = loop_setup(
        "gen-overfit",
        {
            TRAIN_A: lambda h: "OVERFIT" in h,
            TRAIN_B: lambda h: False,
            HOLDOUT: lambda h: "OVERFIT" not in h,
        },
    )
    proposer = ScriptedProposer([_proposal("OVERFIT memorise this exact question")])
    result = AutoImprover(evaluator, factory, proposer, store=store, patience=1).run(
        PromptKnowledge(version=1), loop_id="loop_overfit"
    )
    assert result.rounds[0].decision == "rejected"
    assert any("留出集" in reason for reason in result.rounds[0].reasons)
    assert result.final_knowledge.version == 1


def test_rejections_carry_over_to_the_next_loop(loop_setup, store):
    """跨循环记忆:上一次循环里被拒绝的提示,这一次一开始就要知道。"""
    evaluator, factory = loop_setup(
        "gen-memory",
        {TRAIN_A: lambda h: False, TRAIN_B: lambda h: False, HOLDOUT: lambda h: False},
    )
    first = ScriptedProposer([_proposal("MEMORY a rule that did not work")])
    AutoImprover(evaluator, factory, first, store=store, patience=1).run(
        PromptKnowledge(version=1), loop_id="loop_memory_1"
    )

    second = ScriptedProposer([_proposal("MEMORY a rule that did not work")])
    result = AutoImprover(evaluator, factory, second, store=store, patience=1).run(
        PromptKnowledge(version=1), loop_id="loop_memory_2"
    )
    assert any("MEMORY" in hint for hint, _ in second.seen[0].rejected)
    assert result.rounds[0].decision == "duplicate", "不能把已知无效的修改再试一遍"
    assert result.rounds[0].trial_run_id == "", "重复的候选不该花钱试跑"


def test_patience_stops_the_loop(loop_setup, store):
    evaluator, factory = loop_setup(
        "gen-patience",
        {TRAIN_A: lambda h: False, TRAIN_B: lambda h: False, HOLDOUT: lambda h: False},
    )
    result = AutoImprover(evaluator, factory, ScriptedProposer([None] * 10), store=store, max_rounds=10, patience=2).run(
        PromptKnowledge(version=1), loop_id="loop_patience"
    )
    assert result.stop_reason == "patience"
    assert len(result.rounds) == 2


def test_goldens_are_computed_once_across_rounds(loop_setup, store, monkeypatch):
    """标准答案与提示无关;每轮重算等于每轮多扫一遍仓库。"""
    import rca.nl2sql.evaluate as evaluate_module

    calls = {"n": 0}
    original = evaluate_module.build_golden

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "build_golden", counting)
    evaluator, factory = loop_setup(
        "gen-goldens",
        {TRAIN_A: lambda h: "G" in h, TRAIN_B: lambda h: False, HOLDOUT: lambda h: False},
    )
    AutoImprover(evaluator, factory, ScriptedProposer([_proposal("G one rule to test goldens"), None]), store=store, patience=1).run(
        PromptKnowledge(version=1), loop_id="loop_goldens"
    )
    assert calls["n"] == 3, "3 道题只该各算一次标准答案"


def test_curve_starts_with_the_baseline(loop_setup, store):
    evaluator, factory = loop_setup(
        "gen-curve",
        {TRAIN_A: lambda h: "C" in h, TRAIN_B: lambda h: False, HOLDOUT: lambda h: False},
    )
    result = AutoImprover(evaluator, factory, ScriptedProposer([_proposal("C a helpful rule for the curve")]), store=store, patience=1, max_rounds=1).run(
        PromptKnowledge(version=1), loop_id="loop_curve"
    )
    curve = result.curve()
    assert curve[0]["decision"] == "baseline"
    assert curve[1]["decision"] == "accepted"
    assert curve[1]["train_accuracy"] > curve[0]["train_accuracy"]


def test_hint_used_in_existing_knowledge_is_a_duplicate(loop_setup, store):
    evaluator, factory = loop_setup(
        "gen-dup",
        {TRAIN_A: lambda h: False, TRAIN_B: lambda h: False, HOLDOUT: lambda h: False},
    )
    knowledge = PromptKnowledge(version=4, hints=[Hint(text="Always use DATE literals.")])
    result = AutoImprover(evaluator, factory, ScriptedProposer([_proposal("always use date literals")]), store=store, patience=1).run(
        knowledge, loop_id="loop_dup"
    )
    assert result.rounds[0].decision == "duplicate"


# ---------------------------------------------------------------------------
# 04 第一次真实运行之后补上的三条机制
# ---------------------------------------------------------------------------
# glm-4-flash 在 Databricks 上真实提出、并被闸门拒绝的两条提示(各让训练集 14 → 12)
REAL_VAGUE_HINTS = (
    "Always verify column names and values against the actual data dictionary.",
    "Always verify the correct usage of dimensions and metrics in queries.",
)


@pytest.fixture()
def vocabulary(knowledge, domains) -> Vocabulary:
    return build_vocabulary(knowledge, domains)


@pytest.mark.parametrize("hint", REAL_VAGUE_HINTS)
def test_the_real_vague_hints_name_nothing(vocabulary, hint):
    assert vocabulary.named_in(hint) == ()


@pytest.mark.parametrize(
    ("hint", "named"),
    [
        ("Countries are stored in market as US, UK and DE; region only holds NA and EMEA.", "market"),
        ("CVR is a fraction between 0 and 1. Never multiply it by 100.", "cvr"),
        ("device values are lowercase, e.g. mobile.", "mobile"),
        ("If the question needs data no table has, reply CANNOT_ANSWER.", "cannot_answer"),
    ],
)
def test_specific_hints_are_recognised(vocabulary, hint, named):
    assert named in vocabulary.named_in(hint)


def test_short_values_match_case_sensitively(vocabulary):
    """「US」是取值,「us」是英文代词。后者不能让一句空话蒙混过关。"""
    assert vocabulary.named_in("This rule helps us write better SQL.") == ()


def _failed(case_id, verdict_status, *, answerable=True, detail="", feedback="", correct=False):
    return CaseResult(case_id, case_id, answerable, False, verdict_status, correct, 3, False,
                      detail=detail, last_feedback=feedback)


def test_failures_are_grouped_by_pattern():
    assert failure_pattern(_failed("a", Status.FAILED, feedback="列 market 没有取值 'Germany'")) == "dimension_value"
    assert failure_pattern(_failed("b", Status.FAILED, feedback="执行失败")) == "no_runnable_sql"
    assert failure_pattern(_failed("c", Status.ANSWERED, answerable=False)) == "missed_refusal"
    assert failure_pattern(_failed("d", Status.ANSWERED, detail="value_mismatch: 期望 0.04,实际 100.0")) == "wrong_value"
    assert failure_pattern(_failed("e", Status.ANSWERED, detail="row_count: 标准答案 2 行")) == "wrong_shape"
    assert failure_pattern(_failed("f", Status.REFUSED)) == "false_refusal"


def test_focus_takes_the_largest_untried_pattern_and_rotates():
    patterns = ["wrong_value", "wrong_value", "dimension_value", "missed_refusal", "missed_refusal"]
    first = choose_focus(patterns)
    assert first == "missed_refusal", "数量相同时按 PATTERNS 的顺序"
    second = choose_focus(patterns, tried={first})
    assert second == "wrong_value", "试过没被采纳的一类,下一轮换一类"
    assert choose_focus(patterns, tried=set(patterns)) == first, "全试过了就从头再来"
    assert choose_focus([]) == ""


def test_gate_rejects_a_gain_that_misses_the_targeted_cases():
    """总分 +1,但涨在别的题上 —— 在 21 道题的规模上,这和噪声分不开。"""
    base = _report([_result("t", correct=False), _result("other", correct=False)])
    lucky = _report([_result("t", correct=False), _result("other", correct=True)])
    assert improvement_gate(lucky, base).accepted, "不指定针对的题时,只看总分"
    decision = improvement_gate(lucky, base, targeted=("t",))
    assert not decision.accepted
    assert any("针对的" in reason for reason in decision.reasons)
    on_target = _report([_result("t", correct=True), _result("other", correct=False)])
    assert improvement_gate(on_target, base, targeted=("t",)).accepted


def test_context_failures_keep_their_reason(knowledge, target, cases, make_loop, domains):
    """被取值检查拦下 3 轮的题,失败原因不能是空的 —— 04 第一次运行时就是空的。"""
    case = next(c for c in cases if c.id == TRAIN_A)
    table = target.dialect.qualify("fact_orders")
    bad = f"```sql\nSELECT SUM(order_amount) AS v FROM {table} WHERE market = 'Germany'\n```"
    loop = make_loop(ScriptedGenerator({case.question: [bad] * 3}), value_domains=domains)
    report = Evaluator(knowledge=knowledge, dialect=target.dialect, executor=target.executor,
                       loop=loop, cases=[case]).run(persist=False)

    result = report.results[0]
    assert result.verdict == "no_runnable_sql"
    assert "Germany" in result.detail
    assert failure_pattern(result) == "dimension_value"


def test_vague_hint_is_rejected_before_any_trial(loop_setup, store, vocabulary):
    evaluator, factory = loop_setup(
        "gen-vague",
        {TRAIN_A: lambda h: False, TRAIN_B: lambda h: False, HOLDOUT: lambda h: False},
    )
    runs = {"n": 0}
    original = Evaluator.run

    def counting(self, **kwargs):
        runs["n"] += 1
        return original(self, **kwargs)

    Evaluator.run = counting
    try:
        result = AutoImprover(
            evaluator, factory, ScriptedProposer([_proposal(REAL_VAGUE_HINTS[0])]),
            store=store, patience=1, vocabulary=vocabulary,
        ).run(PromptKnowledge(version=1), loop_id="loop_vague")
    finally:
        Evaluator.run = original

    record = result.rounds[0]
    assert record.decision == "too_generic"
    assert record.trial_run_id == ""
    assert runs["n"] == 1, "只跑了基线,没有为一条空话花一次完整试跑"
    assert record.focus, "这一轮针对哪一类错,要记下来"

    # 进了台账,下一次循环的提议者看得到
    assert store.ledger("loop_vague")[0]["decision"] == "too_generic"
    assert any(REAL_VAGUE_HINTS[0] == row["hint"] for row in store.rejected_hints("gen-vague"))


def test_round_records_which_cases_flipped(loop_setup, store):
    evaluator, factory = loop_setup(
        "gen-flips",
        {TRAIN_A: lambda h: "FLIP" in h, TRAIN_B: lambda h: "FLIP" not in h, HOLDOUT: lambda h: False},
    )
    result = AutoImprover(
        evaluator, factory, ScriptedProposer([_proposal("FLIP a rule that trades one case for another")]),
        store=store, patience=1,
    ).run(PromptKnowledge(version=1), loop_id="loop_flips")

    record = result.rounds[0]
    assert record.fixed == (TRAIN_A,)
    assert record.broken == (TRAIN_B,)
    assert record.decision == "rejected", "一换一,总分没涨"
    ledger = store.ledger("loop_flips")[0]
    assert ledger["fixed"] == TRAIN_A and ledger["broken"] == TRAIN_B


def test_proposer_prompt_shows_only_the_focused_pattern():
    context = ProposalContext(
        failures=(FailureEvidence("cvr_a", "CVR last week?", "wrong_answer",
                                  "value_mismatch: 期望 0.04,实际 100.0", "SELECT 100.0", "",
                                  "wrong_value"),),
        current_hints=(),
        rejected=((REAL_VAGUE_HINTS[0], "提示没有点名任何具体的表、列、取值或指标"),),
        focus="wrong_value",
        other_patterns=(("missed_refusal", 2),),
    )
    prompt = build_proposer_prompt(context)
    assert "fix ONE failure pattern - wrong_value" in prompt
    assert "SPECIFIC" in prompt
    assert "missed_refusal: 2 question(s)" in prompt, "别的错只给数量,不给细节"
    assert REAL_VAGUE_HINTS[0] in prompt, "被判为空话的提示也要告诉提议者"


def test_old_state_tables_are_migrated_not_dropped(target, store):
    """workspace 里的台账是上一版代码建的,缺新加的列。补列,不删表 —— 里面是循环的记忆。"""
    from rca.nl2sql.state import LEDGER_SCHEMA, LEDGER_TABLE, _create_table_sql, _insert_sql

    new_columns = {"focus", "fixed", "broken"}
    old_schema = {k: v for k, v in LEDGER_SCHEMA.items() if k not in new_columns}
    table = target.dialect.qualify(LEDGER_TABLE)
    target.executor.run(f"DROP TABLE IF EXISTS {table}")
    target.executor.run(_create_table_sql(LEDGER_TABLE, old_schema, target.dialect))
    target.executor.run(_insert_sql(
        LEDGER_TABLE, old_schema,
        [{"loop_id": "loop_legacy", "round": 1, "decision": "rejected", "hint": "an old rejected rule",
          "generator": "gen-legacy", "created_at": "2026-09-01 00:00:00"}],
        target.dialect,
    ))

    store.ensure_tables()

    assert new_columns <= store._existing_columns(LEDGER_TABLE)
    assert [row["hint"] for row in store.rejected_hints("gen-legacy")] == ["an old rejected rule"], \
        "旧的拒绝记录必须还在"
    store.record_ledger({"loop_id": "loop_after_migration", "round": 1, "decision": "rejected",
                         "hint": "x", "focus": "wrong_value", "generator": "gen-legacy"})
    assert store.ledger("loop_after_migration")[0]["focus"] == "wrong_value"
