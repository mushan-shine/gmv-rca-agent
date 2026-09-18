"""问答助手端到端:分流 → 查数 / 归因 → 叙述 → 日志 → 待复核队列。

归因的数字要和直接写 SQL 算出来的一致;查数走的是和评估完全相同的自修复循环;
线上失败和被点 👎 的问题要进入待复核队列 —— 那是评估集从线上长大的入口。
"""

from __future__ import annotations

import pytest

from rca.assistant import build_assistant
from rca.nl2sql.generate import ScriptedGenerator
from rca.nl2sql.knowledge_store import load_prompt_knowledge

from conftest import KNOWLEDGE_DIR, TEST_CONFIG

AS_OF = TEST_CONFIG.report_date          # 2025-06-30,上周 = 06-23 ~ 06-29
LAST_WEEK = "DATE '2025-06-23' AND DATE '2025-06-29'"
WEEK_BEFORE = "DATE '2025-06-16' AND DATE '2025-06-22'"


@pytest.fixture()
def assistant(target, store):
    return build_assistant(KNOWLEDGE_DIR, target.executor, target.dialect, as_of=AS_OF, store=store)


def _gmv(scalar, window: str, extra: str = "") -> float:
    return float(scalar(
        "SELECT SUM(order_amount) FROM {fact_orders} "
        f"WHERE order_status = 'COMPLETED' AND dt BETWEEN {window} {extra}"
    ))


def test_explain_gmv_matches_direct_sql_and_closes(assistant, scalar, store):
    answer = assistant.ask("上周 GMV 为什么变化了?")

    assert answer.intent == "explain" and answer.status == "answered"
    current, baseline = _gmv(scalar, LAST_WEEK), _gmv(scalar, WEEK_BEFORE)
    assert f"{current:,.0f}" in answer.text, "标题里的数字必须和直接查出来的一致"

    factors = answer.tables["因子分解"]
    assert [row["因子"] for row in factors] == ["流量 UV", "转化率 CVR", "客单价 AOV"]
    total_pp = (current / baseline - 1) * 100
    assert sum(row["贡献(百分点)"] for row in factors) == pytest.approx(total_pp, abs=0.02), \
        "三个因子的贡献相加必须等于总变化"
    assert "闭合" in answer.text
    assert answer.tables["维度贡献 Top"], "要指出贡献最大的维度取值"
    assert answer.sql, "每个数字都能追溯到执行过的 SQL"
    assert any("2025-06-23~2025-06-29" in note for note in answer.notes), "对比窗口必须说出来"

    assert answer.logged
    logged = store.request(answer.request_id)
    assert logged is not None and logged["status"] == "answered" and logged["intent"] == "explain"


def test_explain_with_a_filter(assistant, scalar):
    answer = assistant.ask("上周美国站 GMV 为什么下降?")
    assert answer.status == "answered"
    current = _gmv(scalar, LAST_WEEK, "AND market = 'US'")
    assert f"{current:,.0f}" in answer.text
    assert any("US" in note for note in answer.notes)
    drilled = {row["维度"] for row in answer.tables["各维度明细"]}
    assert "市场(国家)" not in drilled, "已经按市场过滤了,再按市场拆没有意义"


def test_explain_an_additive_metric_by_day(assistant, scalar):
    answer = assistant.ask("昨天移动端订单量为什么变了?")
    assert answer.status == "answered"
    assert "因子分解" not in answer.tables, "订单量没有声明乘法分解,只做维度拆分"
    orders = scalar(
        "SELECT COUNT(*) FROM {fact_orders} WHERE order_status = 'COMPLETED' "
        "AND device = 'mobile' AND dt = DATE '2025-06-29'"
    )
    assert f"为 {int(orders):,}" in answer.text


def test_ratio_metrics_are_declined_with_a_way_forward(assistant):
    answer = assistant.ask("转化率为什么跌了")
    assert answer.status == "unsupported"
    assert "GMV" in answer.text, "要告诉提问人换个问法能得到什么"


def test_query_without_llm_says_so(assistant):
    answer = assistant.ask("上周 GMV 是多少")
    assert answer.intent == "query" and answer.status == "unsupported"
    assert "归因" in answer.text


def test_query_goes_through_the_self_repair_loop(target, store, scalar):
    question = "上周 GMV 是多少"
    table = target.dialect.qualify("fact_orders")
    generator = ScriptedGenerator({
        question: [
            f"```sql\nSELECT SUM(revenue) AS gmv FROM {table}\n```",   # 不存在的列 → 守卫拦下
            f"```sql\nSELECT SUM(order_amount) AS gmv FROM {table} "
            f"WHERE order_status = 'COMPLETED' AND dt BETWEEN {LAST_WEEK}\n```",
        ]
    })
    assistant = build_assistant(
        KNOWLEDGE_DIR, target.executor, target.dialect, as_of=AS_OF, store=store, generator=generator
    )
    answer = assistant.ask(question)

    assert answer.status == "answered"
    assert answer.attempts == 2, "第一次写错列,自修复后答对"
    assert f"{_gmv(scalar, LAST_WEEK):,.0f}" in answer.text
    assert "order_amount" in answer.sql[0]


def test_online_hints_come_from_the_merged_knowledge(target, store):
    """04 采纳、经 PR 合并进 prompting.yaml 的提示,就是从这里进入线上的。"""
    assistant = build_assistant(
        KNOWLEDGE_DIR, target.executor, target.dialect, as_of=AS_OF, store=store,
        generator=ScriptedGenerator({}),
    )
    merged = load_prompt_knowledge(KNOWLEDGE_DIR / "nl2sql" / "prompting.yaml")
    assert assistant.loop is not None
    assert assistant.loop.hints == merged.hint_texts()
    assert assistant.knowledge_version == merged.label


def test_failures_and_thumbs_down_land_in_the_review_queue(target, store):
    bad = "上周 GMV 的利润是多少"
    good = "上周 GMV 是多少"
    table = target.dialect.qualify("fact_orders")
    generator = ScriptedGenerator({
        bad: ["```sql\nSELECT SUM(profit) FROM " + table + "\n```"] * 3,
        good: [f"```sql\nSELECT SUM(order_amount) FROM {table}\n```"],
    })
    assistant = build_assistant(
        KNOWLEDGE_DIR, target.executor, target.dialect, as_of=AS_OF, store=store, generator=generator
    )

    failed = assistant.ask(bad)
    assert failed.status == "failed"
    assert "profit" in failed.detail, "失败原因要能直接看到"

    disliked = assistant.ask(good)
    liked = assistant.ask("上周 GMV 为什么变化了?")
    assert disliked.status == "answered"
    assert assistant.feedback(disliked.request_id, -1, "没按 COMPLETED 过滤")
    assert assistant.feedback(liked.request_id, 1)

    queue = {row["request_id"]: row for row in store.review_queue()}
    assert failed.request_id in queue
    assert disliked.request_id in queue, "答出来了但被点 👎,是评估器发现不了的错"
    assert queue[disliked.request_id]["comment"] == "没按 COMPLETED 过滤"
    assert liked.request_id not in queue


def test_help_is_not_logged(assistant):
    answer = assistant.ask("")
    assert answer.status == "help" and not answer.logged
    assert "为什么" in answer.text
