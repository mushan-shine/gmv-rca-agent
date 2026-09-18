# Databricks notebook source
# MAGIC %md
# MAGIC # 自主改进循环:LLM 提议 → 试跑 → 闸门裁决 → 台账
# MAGIC
# MAGIC 先跑过 `00_setup.py`(建表灌数)。`03` 不是前提,但 `03` 里的 LLM 接入步骤要先走通。
# MAGIC
# MAGIC `03` 里的 L3 是**人在闭环**:失败原因是人看出来的,修改是人写的,合不合并是人定的。
# MAGIC 本 notebook 把这三步交给系统:
# MAGIC
# MAGIC ```
# MAGIC ┌─→ ① 读状态:训练集的失败 + 台账里这个模型以往被拒绝过的提示
# MAGIC │   ② 提议:LLM 看失败证据,写一条通用提示(看不到留出集,也看不到参考 SQL)
# MAGIC │   ③ 试跑:带着这条提示重跑整个评估集
# MAGIC │   ④ 裁决(确定性):训练集多答对 ≥1 道 且 留出集不退步 且 幻觉率/误拒率不升高
# MAGIC │   ⑤ 写状态:采纳 → 知识版本 +1;拒绝 → 连同原因记进台账
# MAGIC └── ⑥ 停止:达到目标 / 连续 N 轮没有采纳 / 预算用完 / 轮数上限
# MAGIC ```
# MAGIC
# MAGIC 一次只改一个变量:每轮只加一条提示,模型、评估集、循环配置都不变。
# MAGIC 所以台账里每一行的指标差值,都能归因到那一条提示上。
# MAGIC
# MAGIC **第一次真实运行之后加的三条机制**(glm-4-flash 两轮都提了「Always verify column names…」
# MAGIC 这种空话,各让训练集 14 → 12,被闸门拒绝):
# MAGIC
# MAGIC | 机制 | 做什么 | 为什么 |
# MAGIC |---|---|---|
# MAGIC | 按失败形态分组 | 每轮只给提议者**一类**错题(维度取值 / 该拒答没拒答 / 数值错 …),没采纳就换下一类 | 7 道各不相同的错题一起给,只能得到放之四海皆准的空话 |
# MAGIC | 具体性检查 | 提示没点名任何表、列、取值或指标 → **试跑前**拒绝,记为 `too_generic` | 一次试跑约 37 次调用、4~5 分钟,不该花在空话上 |
# MAGIC | 归因检查 | 总分 +1 还不够,这一轮**针对的那类错题**里至少修好一道 | 空话也让分数变了 2 道 —— 21 道题上,±2 是噪声量级 |
# MAGIC
# MAGIC **耗时:** 基线一次评估 + 每轮一次评估,每次约 3~5 分钟。默认最多 4 轮,总计约 15~25 分钟。

# COMMAND ----------

# MAGIC %pip install -q "pydantic>=2.7" "PyYAML>=6.0" "sqlglot>=25.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

dbutils.widgets.text("catalog", "workspace", "Unity Catalog 目录")
dbutils.widgets.text("schema", "gmv_rca", "Schema")
dbutils.widgets.dropdown("llm_provider", "zhipu", ["zhipu", "databricks"], "LLM 供应商")
dbutils.widgets.text("endpoint", "databricks-meta-llama-3-3-70b-instruct", "Databricks 端点名")
dbutils.widgets.text("zhipu_model", "glm-4-flash", "智谱模型")
dbutils.widgets.text("proposer_model", "", "提议者模型(留空=同上)")
dbutils.widgets.text("secret_scope", "llm", "Secret scope")
dbutils.widgets.text("secret_key", "zhipu_api_key", "Secret key")
dbutils.widgets.text("max_rounds", "4", "最多几轮")
dbutils.widgets.text("patience", "2", "连续几轮不采纳就停")
dbutils.widgets.text("target_accuracy", "0.9", "目标准确率")
dbutils.widgets.text("max_calls", "400", "模型调用上限(预算熔断)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
PROVIDER = dbutils.widgets.get("llm_provider")
ENDPOINT = dbutils.widgets.get("endpoint").strip()
ZHIPU_MODEL = dbutils.widgets.get("zhipu_model").strip()
PROPOSER_MODEL = dbutils.widgets.get("proposer_model").strip()
SECRET_SCOPE = dbutils.widgets.get("secret_scope").strip()
SECRET_KEY = dbutils.widgets.get("secret_key").strip()
MAX_ROUNDS = int(dbutils.widgets.get("max_rounds"))
PATIENCE = int(dbutils.widgets.get("patience"))
TARGET = float(dbutils.widgets.get("target_accuracy"))
MAX_CALLS = int(dbutils.widgets.get("max_calls"))

from rca.config import build_target
from rca.knowledge import load_knowledge
from rca.nl2sql.cases import load_cases, validate_cases
from rca.nl2sql.domains import load_value_domains
from rca.nl2sql.evaluate import Evaluator
from rca.nl2sql.generate import LlmSqlGenerator
from rca.nl2sql.improve import AutoImprover, LlmHintProposer, build_vocabulary, failure_pattern
from rca.nl2sql.knowledge_store import load_prompt_knowledge, save_prompt_knowledge
from rca.nl2sql.llm import (
    ZHIPU_BASE_URL,
    CachingChatClient,
    LlmError,
    OpenAIChatClient,
    UsageMeter,
    build_chat_client,
    check_connectivity,
)
from rca.nl2sql.loop import AnswerLoop
from rca.nl2sql.state import StateStore
from rca.sampledata import SampleDataConfig

knowledge = load_knowledge(REPO_ROOT / "knowledge")
cases = load_cases(REPO_ROOT / "eval" / "cases.yaml")
validate_cases(cases, knowledge)
target = build_target("spark", catalog=CATALOG, schema=SCHEMA, spark=spark)
prompt_knowledge = load_prompt_knowledge(REPO_ROOT / "knowledge" / "nl2sql" / "prompting.yaml")
REPORT_DATE = SampleDataConfig().report_date

print(f"评估集 {len(cases)} 条:训练集 {sum(not c.holdout for c in cases)} · 留出集 {sum(c.holdout for c in cases)}")
print(f"起始知识 {prompt_knowledge.label}:{len(prompt_knowledge.hints)} 条提示")
print(f"循环参数:最多 {MAX_ROUNDS} 轮 · 连续 {PATIENCE} 轮不采纳即停 · 目标准确率 {TARGET:.0%} · 调用上限 {MAX_CALLS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 连接 LLM 并做冒烟测试
# MAGIC
# MAGIC 与 `03` 相同。连不上就在这里停下 —— 自主循环没有模型是跑不起来的。

# COMMAND ----------

chat_client = None
proposer_client = None   # 留空 proposer_model 时与 chat_client 相同

if PROVIDER == "databricks":
    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient()
    host = (workspace.config.host or "").rstrip("/")
    auth_header = workspace.config.authenticate().get("Authorization", "")
    token = auth_header[len("Bearer "):] if auth_header.startswith("Bearer ") else ""
    endpoints = [item.name for item in workspace.serving_endpoints.list()]
    if ENDPOINT in endpoints:
        chat_client = OpenAIChatClient(
            base_url=f"{host}/serving-endpoints", api_key=token, model=ENDPOINT,
            temperature=0.0, max_tokens=1024,
        )
        if PROPOSER_MODEL and PROPOSER_MODEL in endpoints:
            proposer_client = OpenAIChatClient(
                base_url=f"{host}/serving-endpoints", api_key=token, model=PROPOSER_MODEL,
                temperature=0.0, max_tokens=1024,
            )
        elif PROPOSER_MODEL:
            print(f"✗ 提议者端点 {PROPOSER_MODEL!r} 不存在,提议者改用 {ENDPOINT}")
    else:
        print(f"✗ 端点 {ENDPOINT!r} 不存在。可用端点:{endpoints}")
else:
    reachable, message = check_connectivity(ZHIPU_BASE_URL)
    print(f"open.bigmodel.cn:{message}")
    if reachable:
        try:
            chat_client = build_chat_client(
                {}, provider="zhipu", api_key=dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY),
                model=ZHIPU_MODEL, cache=False,
            )
            if PROPOSER_MODEL:
                proposer_client = build_chat_client(
                    {}, provider="zhipu", api_key=dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY),
                    model=PROPOSER_MODEL, cache=False,
                )
        except LlmError as exc:
            print(f"✗ {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ 读不到 secret {SECRET_SCOPE}/{SECRET_KEY}:{type(exc).__name__}")

assert chat_client is not None, "LLM 没有接通,自主循环无法运行。先按 03 / DATABRICKS_SETUP.md 第 7.5 节排查。"

probe = chat_client.complete("Reply with exactly: SELECT 1")
print(f"冒烟测试通过:模型 {probe.model} · {probe.latency_ms} ms")

proposer_client = proposer_client or chat_client
if proposer_client is not chat_client:
    probe = proposer_client.complete("Reply with exactly: OK")
    print(f"提议者冒烟测试通过:模型 {probe.model} · {probe.latency_ms} ms")
print(f"生成 SQL 的模型:{chat_client.model} · 提议改进的模型:{proposer_client.model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 读取维度取值
# MAGIC
# MAGIC 从仓库实时查出每个维度列有哪些值。L1 会用它拦下 `market = 'Germany'` 这类
# MAGIC 「跑得通但必然查空」的 SQL,并反馈实际取值。
# MAGIC
# MAGIC 取值**不写进 prompt** —— 模型只在写错时才从反馈里看到。
# MAGIC 能不能把这类错误沉淀成一条提示(比如「国家用 market 列,取值是 US/UK/DE」),
# MAGIC 正是要让改进循环自己完成的事。

# COMMAND ----------

domains = load_value_domains(knowledge, target.executor, target.dialect)
for column, values in sorted(domains.values.items()):
    shown = ", ".join(sorted(values)) if len(values) <= 10 else f"{len(values)} 个取值"
    print(f"  {column:<14} {shown}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 装配

# COMMAND ----------

usage = UsageMeter(max_calls=MAX_CALLS)                    # 生成器与提议者共用一个预算
cached_client = CachingChatClient(chat_client)
generator = LlmSqlGenerator(cached_client, meter=usage)
proposer = LlmHintProposer(CachingChatClient(proposer_client), meter=usage)
store = StateStore(target.executor, target.dialect)
store.ensure_tables()                                      # 旧表缺的列会在这里自动补上

# 具体性检查用的词表:表名、列名、指标、维度,以及上一步查出来的维度取值
vocabulary = build_vocabulary(knowledge, domains)
print(f"具体性检查词表:{len(vocabulary.identifiers)} 个表/列/指标名 · {len(vocabulary.values)} 个维度取值")


def loop_factory(candidate_knowledge):
    """除了提示之外,每一轮的循环配置都完全相同 —— 一次只改一个变量。"""
    return AnswerLoop(
        knowledge=knowledge,
        dialect=target.dialect,
        executor=target.executor,
        generator=generator,
        hints=candidate_knowledge.hint_texts(),
        examples=candidate_knowledge.few_shots(),
        as_of=REPORT_DATE,
        value_domains=domains,
    )


evaluator = Evaluator(
    knowledge=knowledge,
    dialect=target.dialect,
    executor=target.executor,
    loop=loop_factory(prompt_knowledge),
    cases=cases,
    store=store,
    usage=usage,
)


def show_round(record):
    mark = {
        "accepted": "✓ 采纳", "rejected": "✗ 拒绝", "duplicate": "↺ 重复",
        "no_proposal": "∅ 无提议", "too_generic": "⊘ 太笼统(未试跑)",
    }.get(record.decision, record.decision)
    tried = record.trial_run_id != ""
    score = f"训练集 {record.train_accuracy:.1%} · 留出集 {record.holdout_accuracy:.1%}" if tried else "未试跑"
    print(f"\n── 第 {record.round} 轮  {mark}  针对 {record.focus or '-'}  {score}")
    if record.hint:
        print(f"   提示:{record.hint}")
    if tried:
        print(f"   修好:{', '.join(record.fixed) or '-'}   改坏:{', '.join(record.broken) or '-'}")
    for reason in record.reasons:
        print(f"   原因:{reason}")
    print(f"   累计用量:{usage.summary()}")


improver = AutoImprover(
    evaluator=evaluator,
    loop_factory=loop_factory,
    proposer=proposer,
    store=store,
    max_rounds=MAX_ROUNDS,
    patience=PATIENCE,
    target_accuracy=TARGET,
    on_round=show_round,
    vocabulary=vocabulary,
)

rejected_before = store.rejected_hints(generator.name)
print(f"生成器:{generator.name}")
print(f"台账里这个模型以往被拒绝过的提示:{len(rejected_before)} 条(本次提议时会告诉提议者)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 运行自主改进循环
# MAGIC
# MAGIC 先跑一次基线评估,然后逐轮提议、试跑、裁决。每轮结束会打印一行进度。
# MAGIC **这一格需要 15~25 分钟,期间不要点停止。**

# COMMAND ----------

result = improver.run(prompt_knowledge)

print("\n" + "=" * 70)
print(result.summary())
print(f"总用量:{usage.summary()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 迭代曲线
# MAGIC
# MAGIC 第 0 行是基线,之后每轮一行 —— **被拒绝的也在里面**。
# MAGIC 这张表是「系统自己发现问题、自己验证、自己取舍」的直接证据。

# COMMAND ----------

display(spark.createDataFrame(result.curve()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 台账(Delta 表,跨循环保存)
# MAGIC
# MAGIC 下一次运行本 notebook 时,这里被拒绝的提示会出现在提议者的 prompt 里 ——
# MAGIC **下一轮读取上一轮写下的状态**。

# COMMAND ----------

display(spark.sql(f"""
    SELECT loop_id, round, decision, focus, hint, reasons, fixed, broken,
           train_before, train_after, holdout_before, holdout_after,
           parent_version, candidate_version, trial_run_id, prompt_fingerprint, created_at
    FROM {CATALOG}.{SCHEMA}.nl2sql_improvement_ledger
    ORDER BY created_at DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 被采纳的提示,以及还剩哪些错题

# COMMAND ----------

if result.accepted:
    for record in result.accepted:
        print(f"[第 {record.round} 轮] {record.hint}")
        print(f"    理由:{record.rationale}")
        print(f"    针对:{', '.join(record.targets) or '-'}\n")
else:
    print("这次没有采纳任何提示。")

print("最终仍然答错的题(按失败形态):")
for failure in sorted(result.final.failures, key=failure_pattern):
    scope = "留出" if failure.holdout else "训练"
    print(f"  [{scope}] [{failure_pattern(failure):<16}] {failure.case_id}")
    print(f"        {failure.detail[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. 提议者最后一轮看到的 prompt(透明度)
# MAGIC
# MAGIC 可以确认三件事:里面**没有留出集的题**,**没有参考 SQL**,
# MAGIC 以及以往被拒绝的提示(含 `too_generic`)都列在「Previously rejected」里。

# COMMAND ----------

print(proposer.last_prompt[:6000] if proposer.last_prompt else "(本次没有调用提议者)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. 合并新知识(可选)
# MAGIC
# MAGIC 被采纳的提示写回 `knowledge/nl2sql/prompting.yaml`,然后在 Git folder 界面 commit & push。
# MAGIC 知识的变更要能被 review、能回滚。取消注释执行。

# COMMAND ----------

# if result.accepted:
#     path = save_prompt_knowledge(result.final_knowledge, REPO_ROOT / "knowledge" / "nl2sql" / "prompting.yaml")
#     print(f"已写出 {path}({result.initial_knowledge.label} → {result.final_knowledge.label})")
