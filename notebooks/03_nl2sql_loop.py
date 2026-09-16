# Databricks notebook source
# MAGIC %md
# MAGIC # text-to-SQL 循环:生成 → 守卫 → 执行 → 评估 → 改进
# MAGIC
# MAGIC 先跑 `00_setup.py`(建表灌数)。
# MAGIC
# MAGIC 本 notebook 是项目的**循环主线**。三层:
# MAGIC
# MAGIC ```
# MAGIC L1 自修复（秒级）    生成 SQL → 静态守卫 → 执行 → 把真实错误喂回去重生成，最多 3 轮
# MAGIC L2 评估（每次改动）  26 条 case，执行式比对，标准答案由知识库程序化生成
# MAGIC L3 改进（跨版本）    失败归因 → 补 few-shot/提示 → 版本 +1 → 回归闸门 → 合并
# MAGIC ```
# MAGIC
# MAGIC **第 2 格会自动探测你的 workspace 有没有可用的 LLM 端点。**
# MAGIC 没有也不影响:除生成环节之外的一切都能跑,并且用「永远正确的生成器」
# MAGIC 校验评估器本身。
# MAGIC
# MAGIC 接入真实模型时,有两件事封在 `rca.nl2sql.llm` 里、循环层看不见:
# MAGIC
# MAGIC * **传输层重试**(429 / 超时重发同一个 prompt)—— 它没有新信息,
# MAGIC   不该被记成一次「自修复尝试」,否则一次网络抖动会污染 `avg_attempts`;
# MAGIC * **prompt 缓存** —— 评估会把逐字相同的 prompt 打很多遍,
# MAGIC   缓存让迭代不必反复付费,也让同一版本的结果可复现。

# COMMAND ----------

# MAGIC %pip install -q pydantic PyYAML sqlglot

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

dbutils.widgets.text("catalog", "main", "Unity Catalog 目录")
dbutils.widgets.text("schema", "gmv_rca", "Schema")
dbutils.widgets.text("endpoint", "databricks-meta-llama-3-3-70b-instruct", "LLM 端点(可留空)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT = dbutils.widgets.get("endpoint").strip()

from rca.config import build_target
from rca.knowledge import load_knowledge
from rca.nl2sql.cases import build_reference_sql, load_cases, validate_cases
from rca.nl2sql.evaluate import Evaluator, question_map, reference_sql_map, regression_gate
from rca.nl2sql.generate import LlmSqlGenerator, ReferenceGenerator
from rca.nl2sql.llm import (
    CachingChatClient,
    OpenAIChatClient,
    UsageMeter,
)
from rca.nl2sql.knowledge_store import (
    load_prompt_knowledge,
    propose_improvement,
    save_prompt_knowledge,
)
from rca.nl2sql.loop import AnswerLoop
from rca.nl2sql.state import StateStore

knowledge = load_knowledge(REPO_ROOT / "knowledge")
cases = load_cases(REPO_ROOT / "eval" / "cases.yaml")
validate_cases(cases, knowledge)
target = build_target("spark", catalog=CATALOG, schema=SCHEMA, spark=spark)
prompt_knowledge = load_prompt_knowledge(REPO_ROOT / "knowledge" / "nl2sql" / "prompting.yaml")

print(f"{len(cases)} 条 case  |  留出集 {sum(c.holdout for c in cases)}  |  "
      f"不可答对照组 {sum(c.gold.kind.value == 'unanswerable' for c in cases)}")
print(f"提示知识 {prompt_knowledge.label}:{len(prompt_knowledge.hints)} 条提示,"
      f"{len(prompt_knowledge.examples)} 条示例")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 这个 workspace 有 LLM 可用吗?
# MAGIC
# MAGIC 列出所有 serving 端点。看到 `databricks-meta-llama-*`、`databricks-gpt-*`
# MAGIC 之类的名字,就说明 Foundation Model APIs 可用,把它填进上面的 `endpoint` 组件。

# COMMAND ----------

LLM_AVAILABLE = False
WORKSPACE_HOST = ""
WORKSPACE_TOKEN = ""

try:
    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient()
    # notebook 内的凭据由 Runtime 注入,不需要手工配 token
    WORKSPACE_HOST = (workspace.config.host or "").rstrip("/")
    WORKSPACE_TOKEN = workspace.config.token or ""

    endpoints = list(workspace.serving_endpoints.list())
    if endpoints:
        print("可用的 serving 端点:")
        for item in endpoints:
            print(f"  - {item.name}")
        LLM_AVAILABLE = any(item.name == ENDPOINT for item in endpoints)
        if not LLM_AVAILABLE and ENDPOINT:
            print(f"\n⚠ 列表里没有 {ENDPOINT!r} —— 把 endpoint 组件改成上面某个名字再重跑本格。")
    else:
        print("这个 workspace 没有任何 serving 端点。")
except Exception as exc:  # noqa: BLE001
    print(f"无法列出 serving 端点:{exc}")

print(f"\nLLM 可用:{LLM_AVAILABLE}")
if not LLM_AVAILABLE:
    print("→ 下面会用 ReferenceGenerator(永远正确)跑通整条流水线并校验评估器本身。")
    print("  要接外部 API,把下一格的 client 换成:")
    print("  OpenAIChatClient(base_url='https://api.openai.com/v1',")
    print("                   api_key=dbutils.secrets.get('scope', 'openai'), model='gpt-4o-mini')")
    print("  ⚠ 外部 API 意味着请求出站 + schema 离开 workspace,方案里要写清楚。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 装配循环
# MAGIC
# MAGIC 生成器是唯一可替换的部件,其余(守卫、执行、比对、状态表)完全一样。

# COMMAND ----------

usage = UsageMeter(max_calls=200, max_tokens=500_000)   # 预算熔断

if LLM_AVAILABLE:
    # 走 workspace 内的 serving 端点 —— 请求不出 Databricks,符合出站受限的约束。
    # 外面包一层缓存:评估会把逐字相同的 prompt 打很多遍,第二遍不该再付费,
    # 同一版本的评估结果也因此完全可复现。
    client = CachingChatClient(
        OpenAIChatClient(
            base_url=f"{WORKSPACE_HOST}/serving-endpoints",
            api_key=WORKSPACE_TOKEN,
            model=ENDPOINT,
            temperature=0.0,      # 采样噪声会让指标涨跌无法归因
            max_tokens=1024,
        )
    )
    generator = LlmSqlGenerator(client, meter=usage)
else:
    # 基线:永远给出正确答案。它的作用不是刷分,而是**校验评估器本身** ——
    # 一个永远正确的生成器必须拿满分,否则问题出在比对/标准答案/守卫上。
    answers, refusals = {}, {}
    for case in cases:
        if case.gold.kind.value == "unanswerable":
            refusals[case.question] = case.gold.reason or ""
        else:
            answers[case.question] = build_reference_sql(case, knowledge, target.dialect)
    generator = ReferenceGenerator(answers, refusals)

loop = AnswerLoop(
    knowledge=knowledge,
    dialect=target.dialect,
    executor=target.executor,
    generator=generator,
    hints=prompt_knowledge.hint_texts(),
    examples=prompt_knowledge.few_shots(),
)
store = StateStore(target.executor, target.dialect)
store.ensure_tables()
evaluator = Evaluator(
    knowledge=knowledge,
    dialect=target.dialect,
    executor=target.executor,
    loop=loop,
    cases=cases,
    store=store,
)
print("生成器:", getattr(generator, "name", type(generator).__name__))
print("预算上限:", f"{usage.max_calls} 次调用 / {usage.max_tokens:,} token")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 看一次自修复循环
# MAGIC
# MAGIC 每一轮尝试都留下记录:守卫报了什么、引擎报了什么、第几轮修好的。

# COMMAND ----------

demo_question = cases[0].question
outcome = loop.answer(demo_question)

print(f"Q: {demo_question}\n状态: {outcome.status.value}   尝试轮数: {outcome.attempts_used}\n")
for attempt in outcome.attempts:
    print(f"--- 第 {attempt.attempt_no} 轮  [{attempt.failure_kind}]  {attempt.latency_ms} ms")
    if attempt.sql:
        print(attempt.sql)
    for message in attempt.guard_messages:
        print(f"  守卫: {message}")
    if attempt.execution_error:
        print(f"  引擎: {attempt.execution_error}")
print("\n结果:", outcome.rows[:5])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 跑整个评估集
# MAGIC
# MAGIC 标准答案由 `knowledge/` 里的指标定义**程序化生成**,不是手工标注 ——
# MAGIC 样例数据重建之后答案自动跟着变,口径也不会与系统不一致。
# MAGIC
# MAGIC `false_refusal_rate` 与 `correct_refusal_rate` 是一对制衡:
# MAGIC 只看后者的话,模型学会「一律拒答」就能满分。

# COMMAND ----------

evaluator.usage = usage if LLM_AVAILABLE else None
report = evaluator.run(
    knowledge_version=prompt_knowledge.label,
    notes=f"generator={getattr(generator, 'name', '?')}",
)
print(report.summary())

if report.failures:
    print("\n失败明细:")
    for result in report.failures:
        print(f"  [{result.verdict:<20}] {result.case_id}")
        print(f"      {result.detail[:160]}")
else:
    print("\n全部正确。用 ReferenceGenerator 得到这个结果,说明评估器本身是可信的。")

# COMMAND ----------

# knowledge_version 与 model 并排 —— 指标变了要能分清是改了知识还是换了模型。
# 两者同时变的那一次跑批,是解释不清的。
display(spark.sql(f"""
    SELECT run_id, knowledge_version, model, generator,
           pass_at_1, pass_final, holdout_pass_final,
           hallucination_rate, correct_refusal_rate, false_refusal_rate,
           avg_attempts, llm_calls, cached_calls, total_tokens, created_at
    FROM {CATALOG}.{SCHEMA}.nl2sql_eval_runs
    ORDER BY created_at DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 改进循环:失败 → 知识 → 回归闸门
# MAGIC
# MAGIC 这一格是整个项目的**循环判据**所在(原简报 §3.1):
# MAGIC
# MAGIC > 一个循环之所以成立,在于「下一轮会读取上一轮写下的状态」。
# MAGIC
# MAGIC 下面产生的 few-shot 示例会被写进 `knowledge/nl2sql/prompting.yaml`,
# MAGIC **下一次生成时 `build_prompt()` 会读到它**。每条都带 `source_run` 与
# MAGIC `fixes_case`,说清是哪一次评估、为了修哪条 case 才加进来的。

# COMMAND ----------

print("失败画像(评估判定 × 循环失败类型):")
for row in store.failure_profile(report.run_id):
    kind = row["failure_kind"]
    note = "  ← SQL 跑通了但答错,循环自己发现不了" if kind == "none" else ""
    print(f"  {row['verdict']:<20} {kind:<14} {row['n']} 次,{row['cases']} 条 case{note}")

needing_help = store.cases_needing_help(report.run_id)
print(f"\n答错的 case:{[row['case_id'] for row in needing_help] or '无'}")

improvement = propose_improvement(
    prompt_knowledge,
    run_id=report.run_id,
    failing_cases=needing_help,
    reference_sql=reference_sql_map(evaluator),
    questions=question_map(cases),
)
if improvement.is_empty:
    print("\n没有可补的知识 —— 这一轮没有新的失败模式。")
else:
    print(f"\n候选版本 {improvement.knowledge.label}:")
    print(f"  新增示例(修复 case):{list(improvement.added_examples)}")
    for hint in improvement.added_hints:
        print(f"  新增提示:{hint[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 回归闸门
# MAGIC
# MAGIC 候选版本必须**重跑评估 + 留出集**,指标不退步才允许合并。
# MAGIC
# MAGIC 留出集那一条是关键:只看全量指标的话,把失败 case 的答案直接塞进 few-shot
# MAGIC 就能刷上去 —— 那是背答案,不是改进。

# COMMAND ----------

baseline = store.previous_run(report.run_id)
decision = regression_gate(report, baseline)
print(f"基线:{baseline['run_id'] if baseline else '(无,首次评估)'}")
print(f"判定:{'通过 ✓' if decision.accepted else '拒绝 ✗'}")
for reason in decision.reasons:
    print(f"  - {reason}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 合并
# MAGIC
# MAGIC 通过闸门后才写回 YAML,然后**走 Git PR 提交** —— 知识的变更要能被 review、被回滚。
# MAGIC 取消下面的注释来执行。

# COMMAND ----------

# if decision.accepted and not improvement.is_empty:
#     path = save_prompt_knowledge(
#         improvement.knowledge, REPO_ROOT / "knowledge" / "nl2sql" / "prompting.yaml"
#     )
#     print(f"已写出 {path}。接下来在 Git folder 界面 commit & push,开 PR。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 逐次尝试的明细
# MAGIC
# MAGIC `nl2sql_attempts` 是 append-only 的 Delta 表,跨运行累积。
# MAGIC 「平均修复轮数」「哪类问题总在第几轮才修好」都来自它。

# COMMAND ----------

display(spark.sql(f"""
    SELECT run_id, case_id, verdict, attempt_no, failure_kind, model,
           prompt_tokens, completion_tokens, cached, latency_ms, row_count,
           fingerprint, generation_error, sql
    FROM {CATALOG}.{SCHEMA}.nl2sql_attempts
    ORDER BY created_at DESC, case_id, attempt_no
    LIMIT 200
"""))
