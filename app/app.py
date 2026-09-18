"""电商指标问答 —— 网页入口(Streamlit)。

三种跑法,代码完全相同:

* **Databricks Apps**(线上):``app.yaml`` 声明 SQL warehouse 和 LLM 密钥两个资源,
  用 App 自己的 service principal 连仓库,公司账号登录后即可使用。
* **本机演示**:``RCA_TARGET=duckdb streamlit run app/app.py`` —— 内存库 + 样例数据,
  不连任何远端;不配 LLM 也能用归因。
* **本机连远端**:``RCA_TARGET=databricks``,凭据从 ``.env`` 读。

页面只负责展示。分流、查数、归因、日志全部在 :mod:`rca.assistant` 里,
notebook ``05_assistant`` 调用的是同一份代码。
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from rca.assistant import EXAMPLES, Answer, Assistant, build_assistant  # noqa: E402
from rca.config import ENV_WAREHOUSE_ID, build_app_target, build_target, load_dotenv  # noqa: E402
from rca.nl2sql.generate import LlmSqlGenerator  # noqa: E402
from rca.nl2sql.llm import LlmError, UsageMeter, build_chat_client  # noqa: E402
from rca.nl2sql.state import StateStore  # noqa: E402
from rca.sampledata import SampleDataConfig, build_setup_statements  # noqa: E402

st.set_page_config(page_title="电商指标问答", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# 装配(整个进程只做一次)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="正在连接数据仓库…")
def get_assistant() -> tuple[Assistant, dict[str, str], threading.Lock]:
    load_dotenv()
    status: dict[str, str] = {}

    if os.environ.get(ENV_WAREHOUSE_ID):
        target = build_app_target()
        status["数据"] = "SQL Warehouse(App 身份)"
    else:
        name = os.environ.get("RCA_TARGET", "duckdb")
        target = build_target(name)
        status["数据"] = target.description
        if name == "duckdb":   # 本机演示:内存库里灌一份样例数据
            knowledge_root = ROOT / "knowledge"
            from rca.knowledge import load_knowledge

            for statement in build_setup_statements(
                load_knowledge(knowledge_root), target.dialect, SampleDataConfig()
            ):
                target.executor.run(statement.sql)

    # 「上周」相对于哪一天。样例数据是历史快照,默认取它的报告日;接真实数据时设成今天。
    as_of_env = os.environ.get("RCA_AS_OF", "").strip()
    as_of = (
        date.today() if as_of_env == "today"
        else date.fromisoformat(as_of_env) if as_of_env
        else SampleDataConfig().report_date
    )

    generator = None
    try:
        client = build_chat_client(cache=True)
        generator = LlmSqlGenerator(client, meter=UsageMeter(max_calls=int(os.environ.get("RCA_MAX_CALLS", "5000"))))
        status["查数模型"] = client.model
    except LlmError:
        status["查数模型"] = "未接入 —— 归因可用,查数不可用(配置见 app.yaml)"

    assistant = build_assistant(
        ROOT / "knowledge", target.executor, target.dialect,
        as_of=as_of, generator=generator,
        store=StateStore(target.executor, target.dialect),
    )
    status["数据截至"] = str(as_of)
    status["知识版本"] = assistant.knowledge_version
    # 一个进程共用一条仓库连接;Streamlit 每个会话一个线程,串行化最省事也最安全。
    return assistant, status, threading.Lock()


# ---------------------------------------------------------------------------
# 展示
# ---------------------------------------------------------------------------
STATUS_BADGE = {
    "answered": "", "refused": "🚫 ", "failed": "⚠️ ", "unsupported": "ℹ️ ", "error": "⚠️ ", "help": "",
}


def md(text: str) -> str:
    """Markdown 里两个 ~ 之间会被渲染成删除线,「06-23~06-29 vs 06-16~06-22」就中招了。"""
    return text.replace("~", r"\~")


def render_answer(assistant: Assistant, answer: Answer, *, key: str) -> None:
    st.markdown(md(STATUS_BADGE.get(answer.status, "") + answer.text))
    for note in answer.notes:
        st.caption(md(f"假设:{note}"))
    for name, rows in answer.tables.items():
        if rows:
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if answer.sql or answer.detail:
        with st.expander(f"SQL 与过程 · {answer.latency_ms} ms"
                         + (f" · 尝试 {answer.attempts} 次" if answer.attempts else "")):
            if answer.detail:
                st.text(answer.detail)
            for sql in answer.sql:
                st.code(sql, language="sql")
    if answer.status == "help":
        return

    rated = st.session_state.setdefault("rated", {})
    if answer.request_id in rated:
        st.caption("已收到反馈,谢谢" + ("。这个问题会进入待复核队列。" if rated[answer.request_id] < 0 else ""))
        return
    up, down, _ = st.columns([1, 1, 10])
    if up.button("👍", key=f"up-{key}"):
        assistant.feedback(answer.request_id, 1)
        rated[answer.request_id] = 1
        st.rerun()
    if down.button("👎", key=f"down-{key}"):
        assistant.feedback(answer.request_id, -1)
        rated[answer.request_id] = -1
        st.rerun()


def main() -> None:
    try:
        assistant, status, lock = get_assistant()
    except Exception as exc:  # noqa: BLE001 - 启动失败要在页面上说清楚,而不是白屏
        st.error(f"启动失败:{exc}")
        st.stop()

    with st.sidebar:
        st.header("📊 电商指标问答")
        st.caption("查数走 text-to-SQL 自修复循环;「为什么」走确定性归因,数字全部来自查询结果。")
        for label, value in status.items():
            st.markdown(f"**{label}**:{value}")
        st.divider()
        st.markdown("**可以这样问**")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state["pending"] = example
        st.divider()
        if st.button("清空对话", use_container_width=True):
            st.session_state["history"] = []
        if assistant.store is not None:
            with st.expander("待复核队列"):
                try:
                    queue = assistant.store.review_queue(limit=20)
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"读取失败:{type(exc).__name__}")
                else:
                    st.caption("没答出来的、被点了 👎 的问题。确认口径后加进 eval/cases.yaml。")
                    if queue:
                        st.dataframe(
                            pd.DataFrame(queue)[["question", "status", "rating", "comment"]],
                            hide_index=True,
                        )
                    else:
                        st.caption("(空)")

    history: list[Answer] = st.session_state.setdefault("history", [])
    if not history:
        st.info("在下方输入问题,或点左侧的示例。")
    for index, answer in enumerate(history):
        with st.chat_message("user"):
            st.markdown(answer.question)
        with st.chat_message("assistant"):
            render_answer(assistant, answer, key=str(index))

    question = st.chat_input("例如:上周美国站 GMV 为什么下降?") or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("正在查询…"), lock:
                answer = assistant.ask(question)
            history.append(answer)
            render_answer(assistant, answer, key=str(len(history) - 1))


main()
