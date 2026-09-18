"""问题分流:不需要数据库,规则判定可以逐条钉死。"""

from __future__ import annotations

from datetime import date

import pytest

from rca.assistant.router import (
    INTENT_EXPLAIN,
    INTENT_HELP,
    INTENT_QUERY,
    Router,
    comparison_periods,
    load_synonyms,
)
from rca.nl2sql.domains import ValueDomains
from rca.nl2sql.generate import render_reporting_calendar

from conftest import KNOWLEDGE_DIR

DOMAINS = ValueDomains(
    {
        "market": frozenset({"US", "UK", "DE"}),
        "region": frozenset({"NA", "EMEA"}),
        "device": frozenset({"mobile", "desktop"}),
        "channel": frozenset({"Paid Search", "Organic", "Social", "Email", "Direct"}),
        "category": frozenset({"Electronics", "Apparel"}),
    }
)


@pytest.fixture(scope="module")
def router(knowledge) -> Router:
    return Router(knowledge, load_synonyms(KNOWLEDGE_DIR / "assistant" / "synonyms.yaml"), DOMAINS)


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("上周 GMV 为什么下降了?", INTENT_EXPLAIN),
        ("GMV 下跌的原因是什么", INTENT_EXPLAIN),
        ("Why did GMV drop last week?", INTENT_EXPLAIN),
        ("上周 GMV 是多少", INTENT_QUERY),
        ("上周 GMV 环比变化多少", INTENT_QUERY),       # 问的是「多少」,不是「为什么」
        ("", INTENT_HELP),
        ("   ", INTENT_HELP),
    ],
)
def test_intent(router, question, intent):
    assert router.route(question).intent == intent


@pytest.mark.parametrize(
    ("question", "metric"),
    [
        ("上周 GMV 为什么跌了", "gmv"),
        ("销售额为什么下降", "gmv"),
        ("订单量为什么变少了", "orders"),
        ("为什么流量涨了", "uv"),
        ("订单转化率为什么下降", "cvr"),     # 更具体的说法优先:不是订单量
        ("客单价为什么变了", "aov"),
        ("why did sessions drop", "uv"),
    ],
)
def test_metric(router, question, metric):
    assert router.route(question).metric == metric


def test_missing_metric_falls_back_to_default_and_says_so(router):
    route = router.route("上周为什么跌了")
    assert route.metric == "gmv"
    assert any("默认指标" in note for note in route.notes), "假设必须说出来"


@pytest.mark.parametrize(
    ("question", "filters"),
    [
        ("上周美国站 GMV 为什么下降", {"market": "US"}),
        ("why did US GMV drop", {"market": "US"}),
        ("为什么移动端 GMV 变了", {"device": "mobile"}),
        ("为什么 mobile 的 GMV 变了", {"device": "mobile"}),
        ("为什么 paid search 的 GMV 跌了", {"channel": "Paid Search"}),
        ("为什么德国手机端的 GMV 跌了", {"market": "DE", "device": "mobile"}),
        ("tell us why GMV dropped", {}),     # us 是代词,不是美国
    ],
)
def test_filters(router, question, filters):
    assert dict(router.route(question).filters) == filters


def test_several_values_of_one_dimension_are_not_guessed(router):
    route = router.route("为什么美国和英国的 GMV 都跌了")
    assert "market" not in route.filters
    assert any("多个" in note for note in route.notes)


def test_filters_that_would_break_the_identity_are_not_offered(router):
    """品类要 join 商品表,只在订单侧可用。按它过滤 GMV,UV × CVR × AOV 就对不上了。"""
    assert "category" not in router.filterable_dimensions("gmv")
    assert dict(router.route("为什么 Electronics 的 GMV 跌了").filters) == {}
    assert {"market", "device", "channel", "region"} <= set(router.filterable_dimensions("gmv"))


def test_granularity(router):
    assert router.route("昨天 GMV 为什么跌了").granularity == "day"
    assert router.route("上周 GMV 为什么跌了").granularity == "week"
    assert router.route("GMV 为什么跌了").granularity == "week"


def test_unrecognised_time_range_is_stated_not_hidden(router):
    """「上个月」目前不识别。按上周算可以,但必须在回答里说出来。"""
    route = router.route("上个月 GMV 为什么跌了")
    assert route.granularity == "week"
    assert any("时间范围" in note for note in route.notes)
    assert not any("时间范围" in note for note in router.route("上周 GMV 为什么跌了").notes)


def test_weeks_match_the_reporting_calendar_used_by_text_to_sql():
    """归因和查数对「上周」必须是同一个定义,否则同一个人问的两个问题口径不一致。"""
    as_of = date(2025, 6, 30)
    baseline, current = comparison_periods("week", as_of)
    calendar = " ".join(render_reporting_calendar(as_of))
    assert (current.start, current.end) == (date(2025, 6, 23), date(2025, 6, 29))
    assert (baseline.start, baseline.end) == (date(2025, 6, 16), date(2025, 6, 22))
    assert f"{current.start} to {current.end}" in calendar
    assert f"{baseline.start} to {baseline.end}" in calendar


def test_day_comparison():
    baseline, current = comparison_periods("day", date(2025, 6, 30))
    assert current.start == current.end == date(2025, 6, 29)
    assert baseline.start == baseline.end == date(2025, 6, 28)
