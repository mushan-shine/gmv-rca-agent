# Databricks notebook source
# MAGIC %md
# MAGIC # 问答助手:在 notebook 里提问
# MAGIC
# MAGIC 这是助手的 **notebook 入口**,和网页入口(`app/app.py`,Databricks Apps)调用的是同一份代码。
# MAGIC workspace 没有 Apps 时,用这里;挂成 Job 时,也用这里。
# MAGIC
# MAGIC **怎么用:** 改顶部的 `question` 组件 → 运行第 3 格。
# MAGIC
# MAGIC | 问法 | 走哪条路 | 要不要 LLM |
# MAGIC |---|---|---|
# MAGIC | 「上周 GMV **为什么**变化了?」「上周美国站 GMV **为什么**下降?」 | **归因**:确定性分解,数字全部来自查询结果 | 不要 |
# MAGIC | 「上周各渠道的 GMV **是多少**?」 | **查数**:text-to-SQL 自修复循环 | 要 |
# MAGIC
# MAGIC 每个问题都记进 `assistant_requests`;没答出来的、被点了 👎 的进入**待复核队列**(第 5 格),
# MAGIC 人确认口径后加进 `eval/cases.yaml` —— 评估集从线上真实问题里长大。
# MAGIC
# MAGIC 前提:`00_setup` 跑过。查数要接 LLM,步骤同 `03`。

# COMMAND ----------

# MAGIC %pip install -q "pydantic>=2.7" "PyYAML>=6.0" "sqlglot>=25.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

dbutils.widgets.text("question", "上周 GMV 为什么变化了?", "问题")
dbutils.widgets.text("catalog", "workspace", "Unity Catalog 目录")
dbutils.widgets.text("schema", "gmv_rca", "Schema")
dbutils.widgets.dropdown("llm_provider", "zhipu", ["zhipu", "databricks", "none"], "LLM 供应商(查数用)")
dbutils.widgets.text("endpoint", "databricks-meta-llama-3-3-70b-instruct", "Databricks 端点名")
dbutils.widgets.text("zhipu_model", "glm-4-flash", "智谱模型")
dbutils.widgets.text("secret_scope", "llm", "Secret scope")
dbutils.widgets.text("secret_key", "zhipu_api_key", "Secret key")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
PROVIDER = dbutils.widgets.get("llm_provider")

from rca.assistant import EXAMPLES, build_assistant
from rca.config import build_target
from rca.nl2sql.generate import LlmSqlGenerator
from rca.nl2sql.llm import ZHIPU_BASE_URL, LlmError, OpenAIChatClient, UsageMeter, build_chat_client, check_connectivity
from rca.nl2sql.state import StateStore
from rca.sampledata import SampleDataConfig

target = build_target("spark", catalog=CATALOG, schema=SCHEMA, spark=spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 接 LLM(只有查数需要;归因不需要)
# MAGIC
# MAGIC 连不上也没关系:归因照常可用,查数会明确告诉你「没有接 LLM」。

# COMMAND ----------

chat_client = None
if PROVIDER == "zhipu":
    reachable, message = check_connectivity(ZHIPU_BASE_URL)
    print(f"open.bigmodel.cn:{message}")
    if reachable:
        try:
            chat_client = build_chat_client(
                {}, provider="zhipu", api_key=dbutils.secrets.get(
                    dbutils.widgets.get("secret_scope"), dbutils.widgets.get("secret_key")),
                model=dbutils.widgets.get("zhipu_model").strip(), cache=True,
            )
        except LlmError as exc:
            print(f"✗ {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ 读不到 secret:{type(exc).__name__}")
elif PROVIDER == "databricks":
    from databricks.sdk import WorkspaceClient

    workspace = WorkspaceClient()
    header = workspace.config.authenticate().get("Authorization", "")
    endpoint = dbutils.widgets.get("endpoint").strip()
    if endpoint in [item.name for item in workspace.serving_endpoints.list()]:
        chat_client = OpenAIChatClient(
            base_url=f"{(workspace.config.host or '').rstrip('/')}/serving-endpoints",
            api_key=header.removeprefix("Bearer "), model=endpoint, temperature=0.0, max_tokens=1024,
        )
    else:
        print(f"✗ 端点 {endpoint!r} 不存在")

generator = LlmSqlGenerator(chat_client, meter=UsageMeter(max_calls=200)) if chat_client else None
print(f"查数:{'可用 · ' + chat_client.model if chat_client else '不可用(只能归因)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 装配助手
# MAGIC
# MAGIC 查数用的提示来自 `knowledge/nl2sql/prompting.yaml` —— `04` 采纳、经 Git 合并的提示,就是从这里进入线上的。

# COMMAND ----------

store = StateStore(target.executor, target.dialect)
assistant = build_assistant(
    REPO_ROOT / "knowledge", target.executor, target.dialect,
    as_of=SampleDataConfig().report_date,   # 样例数据是历史快照;接真实数据时改成 date.today()
    generator=generator, store=store,
)
print(f"知识版本 {assistant.knowledge_version} · 数据截至 {assistant.as_of}")
print("可以这样问:")
for example in EXAMPLES:
    print(f"  - {example}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 提问
# MAGIC
# MAGIC 改顶部的 `question`,然后**只运行这一格**(不用 Run all)。

# COMMAND ----------

import pandas as pd

answer = assistant.ask(dbutils.widgets.get("question"))

print(f"[{answer.intent} · {answer.status} · {answer.latency_ms} ms · request_id {answer.request_id}]\n")
print(answer.text.replace("**", ""))
if answer.detail:
    print(f"\n原因:{answer.detail}")
print()
for note in answer.notes:
    print(f"假设:{note}")

for name, rows in answer.tables.items():
    if rows:
        print(f"\n{name}")
        display(pd.DataFrame(rows))

# COMMAND ----------

# 展开看每一个数字背后执行过的 SQL
for sql in answer.sql:
    print(sql, "\n;\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 反馈(可选)
# MAGIC
# MAGIC 觉得答错了:把 `RATING` 改成 `-1`、写上原因,运行本格。这道题会进入待复核队列。

# COMMAND ----------

RATING = 0          # 1 = 👍,-1 = 👎,0 = 不评价
COMMENT = ""        # 例如:"应该只算已完成订单"

if RATING:
    assistant.feedback(answer.request_id, RATING, COMMENT)
    print("已记录。" + ("这道题会进入待复核队列。" if RATING < 0 else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 待复核队列与最近的问题
# MAGIC
# MAGIC 待复核 = 没答出来的 + 被点了 👎 的。逐条确认正确口径,值得收的写进 `eval/cases.yaml`。

# COMMAND ----------

queue = store.review_queue()
# 用 pandas 而不是 spark.createDataFrame:rating 全为空时 Spark 推断不出列类型
display(pd.DataFrame(queue)) if queue else print("待复核队列是空的。")

# COMMAND ----------

display(spark.sql(f"""
    SELECT created_at, question, intent, status, latency_ms, attempts, model
    FROM {CATALOG}.{SCHEMA}.assistant_requests
    ORDER BY created_at DESC
    LIMIT 50
"""))
