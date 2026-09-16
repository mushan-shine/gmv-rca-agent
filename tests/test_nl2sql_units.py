"""不需要数据库的单元测试:守卫、比对器、评估集定义、改进循环、回归闸门。

这一层刻意做厚。text-to-SQL 项目里最容易出问题的不是「模型答不对」——
那是可以改进的;而是**评估逻辑本身有毛病**,于是所有改进都在追一个错误的靶子。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rca.errors import DimensionNotFoundError
from rca.nl2sql.cases import (
    EvalCase,
    EvalCaseError,
    GoldKind,
    build_reference_sql,
    load_cases,
    resolve_period,
    validate_cases,
)
from rca.nl2sql.compare import MatchFailure, compare_result_sets
from rca.nl2sql.evaluate import CaseResult, EvalReport, GateDecision, regression_gate
from rca.nl2sql.loop import Status
from rca.nl2sql.generate import (
    REFUSAL_PREFIX,
    FewShotExample,
    PromptContext,
    build_prompt,
    parse_response,
    render_metric_glossary,
)
from rca.nl2sql.guard import SqlGuard, Violation
from rca.nl2sql.knowledge_store import (
    Example,
    Hint,
    PromptKnowledge,
    load_prompt_knowledge,
    propose_improvement,
    save_prompt_knowledge,
)

from conftest import CASES_PATH


# ---------------------------------------------------------------------------
# 守卫
# ---------------------------------------------------------------------------
@pytest.fixture()
def offline_guard(knowledge):
    return SqlGuard(knowledge, dialect="databricks")


def test_valid_query_passes(offline_guard):
    report = offline_guard.check(
        "SELECT channel, SUM(order_amount) AS gmv FROM fact_orders GROUP BY channel"
    )
    assert report.ok
    assert not report.hallucinated
    assert "fact_orders" in report.tables


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE fact_orders",
        "DELETE FROM fact_orders WHERE 1=1",
        "INSERT INTO fact_orders SELECT * FROM fact_orders",
        "CREATE TABLE x AS SELECT 1",
    ],
)
def test_write_statements_are_blocked(offline_guard, sql):
    """LLM 生成的文本会被送进仓库执行 —— 只读之外的一切都必须在执行前拦下。"""
    report = offline_guard.check(sql)
    assert not report.ok
    assert Violation.NOT_READ_ONLY in report.violations or Violation.PARSE_ERROR in report.violations


def test_trailing_statement_is_blocked(offline_guard):
    report = offline_guard.check("SELECT 1 FROM fact_orders; DROP TABLE fact_orders")
    assert not report.ok


def test_unknown_table_is_reported_with_the_allowed_list(offline_guard):
    report = offline_guard.check("SELECT * FROM fact_revenue")
    assert Violation.UNKNOWN_TABLE in report.violations
    assert report.hallucinated
    assert "fact_revenue" in report.feedback()
    assert "fact_orders" in report.feedback(), "反馈必须列出可用的表,否则模型改不动"


def test_unknown_column_feedback_names_real_columns(offline_guard):
    report = offline_guard.check("SELECT revenue FROM fact_orders")
    assert Violation.UNKNOWN_COLUMN in report.violations
    assert report.hallucinated
    feedback = report.feedback()
    assert "revenue" in feedback
    assert "order_amount" in feedback, "反馈必须具体到可以照着改"


def test_select_alias_can_be_reused(offline_guard):
    """``SELECT SUM(x) AS gmv ... ORDER BY gmv`` 是合法的,不能误判成未知列。"""
    report = offline_guard.check(
        "SELECT channel, SUM(order_amount) AS gmv FROM fact_orders"
        " GROUP BY channel ORDER BY gmv DESC"
    )
    assert report.ok, report.messages


def test_cte_name_is_not_an_unknown_table(offline_guard):
    report = offline_guard.check(
        "WITH t AS (SELECT order_amount FROM fact_orders) SELECT SUM(order_amount) FROM t"
    )
    assert report.ok, report.messages


def test_string_literal_containing_a_keyword_is_not_a_write(offline_guard):
    """``WHERE channel = 'Paid Search'`` 里没有关键字,但字面量里可能有。"""
    report = offline_guard.check(
        "SELECT SUM(order_amount) FROM fact_orders WHERE order_status = 'UPDATE'"
    )
    assert report.ok, report.messages


def test_empty_generation_is_reported(offline_guard):
    report = offline_guard.check("")
    assert Violation.EMPTY in report.violations


def test_unparseable_sql_is_reported(offline_guard):
    report = offline_guard.check("SELECT FROM WHERE GROUP")
    assert not report.ok


# ---------------------------------------------------------------------------
# 比对器
# ---------------------------------------------------------------------------
def test_column_names_are_ignored():
    """同一个答案换个别名仍然是对的 —— 否则评估变成「像不像我写的那条」。"""
    assert compare_result_sets([{"value": 10}], [{"total_gmv": 10}])


def test_row_order_is_ignored_by_default():
    expected = [{"c": "A", "v": 1}, {"c": "B", "v": 2}]
    actual = [{"c": "B", "v": 2}, {"c": "A", "v": 1}]
    assert compare_result_sets(expected, actual)


def test_row_order_matters_when_the_question_asks_for_a_ranking():
    expected = [{"c": "A", "v": 2}, {"c": "B", "v": 1}]
    actual = [{"c": "B", "v": 1}, {"c": "A", "v": 2}]
    assert not compare_result_sets(expected, actual, ordered=True)


def test_numeric_tolerance_absorbs_float_noise():
    assert compare_result_sets([{"v": 1234.5678}], [{"v": 1234.56780001}])


def test_decimal_and_float_are_comparable():
    from decimal import Decimal

    assert compare_result_sets([{"v": Decimal("29312.60")}], [{"v": 29312.6}])


def test_extra_column_is_a_mismatch():
    """多选一列算错 —— 答案的形状也是答案的一部分。"""
    result = compare_result_sets([{"v": 10}], [{"v": 10, "dt": "2025-06-23"}])
    assert not result
    assert result.failure is MatchFailure.COLUMN_COUNT


def test_row_count_mismatch_is_classified():
    result = compare_result_sets([{"v": 1}, {"v": 2}], [{"v": 1}])
    assert result.failure is MatchFailure.ROW_COUNT


def test_value_mismatch_is_classified_with_detail():
    result = compare_result_sets([{"v": 100.0}], [{"v": 110.0}])
    assert result.failure is MatchFailure.VALUE_MISMATCH
    assert "100" in result.detail and "110" in result.detail


def test_null_is_not_zero():
    """NULL 与 0 是不同的答案 —— 混同会掩盖「这个切片根本没数据」。"""
    assert not compare_result_sets([{"v": None}], [{"v": 0}])


def test_empty_results_are_classified_separately():
    assert compare_result_sets([], []).match
    assert compare_result_sets([{"v": 1}], []).failure is MatchFailure.GOT_EMPTY
    assert compare_result_sets([], [{"v": 1}]).failure is MatchFailure.EXPECTED_EMPTY


# ---------------------------------------------------------------------------
# 评估集
# ---------------------------------------------------------------------------
def test_repository_cases_load_and_validate(cases, knowledge):
    validate_cases(cases, knowledge)
    assert len(cases) >= 20
    assert any(case.holdout for case in cases), "必须有留出集"
    assert any(
        case.gold.kind is GoldKind.UNANSWERABLE for case in cases
    ), "必须有不可答对照组"


def test_unanswerable_cases_carry_a_reason(cases):
    for case in cases:
        if case.gold.kind is GoldKind.UNANSWERABLE:
            assert case.gold.reason, case.id


def test_holdout_covers_both_answerable_and_unanswerable(cases):
    """留出集只放可答题的话,就测不出「改进有没有把模型变得爱编答案」。"""
    holdout = [case for case in cases if case.holdout]
    kinds = {case.gold.kind is GoldKind.UNANSWERABLE for case in holdout}
    assert kinds == {True, False}


def test_case_referencing_unknown_metric_is_rejected(knowledge):
    from rca.errors import MetricNotFoundError

    case = EvalCase.model_validate(
        {
            "id": "bad",
            "question": "?",
            "gold": {"kind": "aggregate", "metric": "revenue", "period": "W8"},
        }
    )
    with pytest.raises(MetricNotFoundError):
        validate_cases([case], knowledge)


def test_case_grouping_a_ratio_metric_is_rejected(knowledge):
    """各渠道的 CVR 之和不等于总体 CVR —— 这样的标准答案本身就是错的。"""
    case = EvalCase.model_validate(
        {
            "id": "bad",
            "question": "?",
            "gold": {
                "kind": "group_by",
                "metric": "cvr",
                "dimension": "channel",
                "period": "W8",
            },
        }
    )
    with pytest.raises(EvalCaseError, match="比率"):
        validate_cases([case], knowledge)


def test_case_with_dimension_outside_whitelist_is_rejected(knowledge):
    case = EvalCase.model_validate(
        {
            "id": "bad",
            "question": "?",
            "gold": {
                "kind": "group_by",
                "metric": "uv",
                "dimension": "category",
                "period": "W8",
            },
        }
    )
    with pytest.raises(EvalCaseError, match="白名单"):
        validate_cases([case], knowledge)


def test_case_with_unknown_filter_dimension_is_rejected(knowledge):
    case = EvalCase.model_validate(
        {
            "id": "bad",
            "question": "?",
            "gold": {
                "kind": "aggregate",
                "metric": "gmv",
                "period": "W8",
                "filters": {"weather": "rain"},
            },
        }
    )
    with pytest.raises(DimensionNotFoundError):
        validate_cases([case], knowledge)


def test_unanswerable_case_must_not_declare_a_metric():
    with pytest.raises(Exception):
        EvalCase.model_validate(
            {
                "id": "bad",
                "question": "?",
                "gold": {"kind": "unanswerable", "reason": "x", "metric": "gmv"},
            }
        )


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "cases:\n"
        "  - {id: a, question: q, gold: {kind: aggregate, metric: gmv, period: W8}}\n"
        "  - {id: a, question: q2, gold: {kind: aggregate, metric: gmv, period: W8}}\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalCaseError, match="重复"):
        load_cases(path)


def test_period_labels_resolve_to_aligned_weeks():
    w7, w8 = resolve_period("W7"), resolve_period("W8")
    assert w7.days == w8.days == 7
    assert not w7.overlaps(w8)
    assert w8.start.toordinal() - w7.start.toordinal() == 7


def test_out_of_range_period_is_rejected():
    with pytest.raises(EvalCaseError):
        resolve_period("W99")


def test_reference_sql_for_unanswerable_case_is_refused(knowledge, cases):
    from rca.dialects import DatabricksDialect

    case = next(c for c in cases if c.gold.kind is GoldKind.UNANSWERABLE)
    with pytest.raises(EvalCaseError, match="拒答"):
        build_reference_sql(case, knowledge, DatabricksDialect())


def test_every_reference_sql_is_valid_databricks(knowledge, cases, offline_guard):
    """标准答案自己必须先过守卫 —— 否则评估的地基就是歪的。"""
    from rca.dialects import DatabricksDialect

    dialect = DatabricksDialect()
    for case in cases:
        if case.gold.kind is GoldKind.UNANSWERABLE:
            continue
        sql = build_reference_sql(case, knowledge, dialect)
        report = offline_guard.check(sql)
        assert report.ok, f"{case.id}: {report.messages}"


def test_reference_sql_applies_the_completed_filter(knowledge, cases):
    """GMV 的口径必须进参考 SQL —— 否则标准答案会把取消单也算进去。"""
    from rca.dialects import DatabricksDialect

    case = next(c for c in cases if c.id == "gmv_total_w8")
    sql = build_reference_sql(case, knowledge, DatabricksDialect())
    assert "order_status = 'COMPLETED'" in sql


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------
def test_prompt_contains_schema_rules_and_refusal_contract(knowledge):
    from rca.dialects import DatabricksDialect
    from rca.nl2sql.generate import render_schema

    context = PromptContext(
        question="What was GMV last week?",
        schema_text=render_schema(knowledge, DatabricksDialect()),
        hints=tuple(render_metric_glossary(knowledge)),
    )
    prompt = build_prompt(context)
    assert "fact_orders" in prompt
    assert REFUSAL_PREFIX in prompt, "必须把「拒答」写成结构化契约,否则无法确定性识别"
    assert "What was GMV last week?" in prompt


def test_feedback_is_appended_and_attempt_number_advances():
    context = PromptContext(question="q", schema_text="s")
    second = context.with_feedback("列 revenue 不存在")
    assert second.attempt_no == 2
    assert "列 revenue 不存在" in build_prompt(second)
    assert context.attempt_no == 1, "原 context 必须不可变"


@pytest.mark.parametrize(
    ("raw", "expect_sql"),
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("SELECT 1", "SELECT 1"),
        ("Here you go:\n```sql\nSELECT 1\n```\nHope that helps", "SELECT 1"),
    ],
)
def test_response_parsing_handles_common_formats(raw, expect_sql):
    assert parse_response(raw).sql == expect_sql


def test_refusal_is_parsed_as_a_refusal():
    generation = parse_response(f"{REFUSAL_PREFIX}: there is no campaign_id anywhere")
    assert generation.refused
    assert "campaign_id" in (generation.refusal_reason or "")


def test_prose_without_sql_is_not_mistaken_for_sql():
    generation = parse_response("I think you should ask the data team.")
    assert generation.sql == ""
    assert not generation.refused


# ---------------------------------------------------------------------------
# 改进循环
# ---------------------------------------------------------------------------
def test_improvement_adds_examples_with_provenance():
    current = PromptKnowledge(version=3)
    improvement = propose_improvement(
        current,
        run_id="run_abc",
        failing_cases=[{"case_id": "c1", "last_failure_kind": "hallucination"}],
        reference_sql={"c1": "SELECT 1"},
        questions={"c1": "how many?"},
    )
    assert improvement.knowledge.version == 4
    example = improvement.knowledge.examples[-1]
    assert example.source_run == "run_abc"
    assert example.fixes_case == "c1", "没有来历的知识几周后就没人敢删"
    assert improvement.knowledge.hints[-1].source_run == "run_abc"


def test_improvement_is_capped():
    """一次加太多示例,就说不清是哪一条起了作用。"""
    failures = [{"case_id": f"c{i}", "last_failure_kind": "guard"} for i in range(10)]
    improvement = propose_improvement(
        PromptKnowledge(),
        run_id="r",
        failing_cases=failures,
        reference_sql={f"c{i}": "SELECT 1" for i in range(10)},
        questions={f"c{i}": f"q{i}" for i in range(10)},
        max_examples=2,
    )
    assert len(improvement.added_examples) == 2


def test_improvement_is_empty_when_nothing_to_learn():
    improvement = propose_improvement(
        PromptKnowledge(), run_id="r", failing_cases=[], reference_sql={}, questions={}
    )
    assert improvement.is_empty
    assert improvement.knowledge.version == 1, "没有改动就不该跳版本号"


def test_existing_example_is_not_duplicated():
    current = PromptKnowledge(
        version=1, examples=[Example(question="q", sql="SELECT 1")]
    )
    improvement = propose_improvement(
        current,
        run_id="r",
        failing_cases=[{"case_id": "c", "last_failure_kind": "guard"}],
        reference_sql={"c": "SELECT 2"},
        questions={"c": "q"},
    )
    assert improvement.is_empty


def test_prompt_knowledge_round_trips(tmp_path):
    knowledge = PromptKnowledge(
        version=7,
        hints=[Hint(text="h", source_run="r1", fixes_case="c1")],
        examples=[Example(question="q", sql="SELECT 1", source_run="r1")],
    )
    path = save_prompt_knowledge(knowledge, tmp_path / "prompting.yaml")
    reloaded = load_prompt_knowledge(path)
    assert reloaded.version == 7
    assert reloaded.examples[0].source_run == "r1"
    assert reloaded.few_shots()[0] == FewShotExample("q", "SELECT 1", "r1", "")


def test_missing_prompt_knowledge_is_a_valid_cold_start(tmp_path):
    knowledge = load_prompt_knowledge(tmp_path / "nope.yaml")
    assert knowledge.version == 1 and not knowledge.examples


def test_repository_prompt_knowledge_loads():
    knowledge = load_prompt_knowledge(CASES_PATH.parent.parent / "knowledge" / "nl2sql" / "prompting.yaml")
    assert knowledge.hints, "应当有种子提示(口径说明)"
    assert any("COMPLETED" in hint.text for hint in knowledge.hints)


# ---------------------------------------------------------------------------
# 回归闸门
# ---------------------------------------------------------------------------
def _case(
    case_id: str,
    *,
    answerable: bool = True,
    holdout: bool = False,
    correct: bool = True,
    status: Status = Status.ANSWERED,
    hallucinated: bool = False,
    attempts: int = 1,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        question=case_id,
        answerable=answerable,
        holdout=holdout,
        status=status,
        correct=correct,
        attempts_used=attempts,
        hallucinated=hallucinated,
    )


def _report(results) -> EvalReport:
    return EvalReport(
        run_id="candidate",
        generator="g",
        knowledge_version="v2",
        created_at=datetime.now(timezone.utc),
        results=tuple(results),
    )


# 指标本身也要测 —— 闸门比的是它们,算错了闸门就形同虚设。
def test_metrics_are_computed_over_the_right_denominators():
    report = _report(
        [
            _case("a1", correct=True),
            _case("a2", correct=False, attempts=3),
            _case("u1", answerable=False, correct=True, status=Status.REFUSED),
            _case("u2", answerable=False, correct=False),
            _case("h1", holdout=True, correct=True),
            _case("h2", holdout=True, correct=False),
        ]
    )
    # 可答题共 4 条(a1 a2 h1 h2,留出集也算在全量里),对 2 条
    assert report.pass_final == pytest.approx(2 / 4)
    assert report.correct_refusal_rate == pytest.approx(1 / 2)
    assert report.holdout_pass_final == pytest.approx(1 / 2)
    assert report.avg_attempts == pytest.approx((1 + 3 + 1 + 1 + 1 + 1) / 6)


def test_pass_at_1_requires_both_correctness_and_a_single_attempt():
    """修了两轮才对,不算 pass@1 —— 自修复是有成本的,指标必须体现出来。"""
    report = _report([_case("a", correct=True, attempts=2)])
    assert report.pass_final == 1.0
    assert report.pass_at_1 == 0.0


def test_false_refusal_counts_against_the_model():
    report = _report([_case("a", correct=False, status=Status.REFUSED)])
    assert report.false_refusal_rate == 1.0
    assert report.results[0].verdict == "false_refusal"


def test_answering_an_unanswerable_question_is_named_for_what_it_is():
    report = _report([_case("u", answerable=False, correct=False, status=Status.ANSWERED)])
    assert report.results[0].verdict == "should_have_refused"


def test_gate_accepts_when_there_is_no_baseline():
    assert regression_gate(_report([_case("a")]), None).accepted


def test_gate_blocks_holdout_regression():
    """全量涨了、留出集掉了 —— 这是背答案,必须拦住。"""
    candidate = _report(
        [_case(f"h{i}", holdout=True, correct=i < 3) for i in range(5)]
        + [_case(f"a{i}") for i in range(15)]
    )
    assert candidate.holdout_pass_final == pytest.approx(0.6)
    assert candidate.pass_final == pytest.approx(0.9)

    baseline = {
        "holdout_pass_final": 0.80,
        "pass_final": 0.70,
        "hallucination_rate": 0.0,
        "false_refusal_rate": 0.0,
    }
    decision = regression_gate(candidate, baseline)
    assert not decision.accepted
    assert any("留出集" in reason for reason in decision.reasons)


def test_gate_blocks_rising_hallucination_rate():
    candidate = _report(
        [_case(f"a{i}", hallucinated=i < 2) for i in range(10)]
        + [_case("h", holdout=True)]
    )
    baseline = {
        "holdout_pass_final": 1.0,
        "pass_final": 1.0,
        "hallucination_rate": 0.05,
        "false_refusal_rate": 0.0,
    }
    assert not regression_gate(candidate, baseline)


def test_gate_blocks_rising_false_refusal():
    """否则模型学会「拿不准就拒答」就能把正确率刷上去。"""
    candidate = _report(
        [
            _case(f"a{i}", correct=i >= 3, status=Status.REFUSED if i < 3 else Status.ANSWERED)
            for i in range(10)
        ]
    )
    assert candidate.false_refusal_rate == pytest.approx(0.3)
    baseline = {
        "holdout_pass_final": 0.0,
        "pass_final": 0.0,
        "hallucination_rate": 0.0,
        "false_refusal_rate": 0.05,
    }
    assert not regression_gate(candidate, baseline)


def test_gate_accepts_a_genuine_improvement():
    candidate = _report(
        [_case(f"h{i}", holdout=True, correct=i < 9) for i in range(10)]
        + [_case(f"a{i}") for i in range(10)]
    )
    baseline = {
        "holdout_pass_final": 0.80,
        "pass_final": 0.75,
        "hallucination_rate": 0.05,
        "false_refusal_rate": 0.05,
    }
    decision: GateDecision = regression_gate(candidate, baseline)
    assert decision.accepted, decision.reasons
