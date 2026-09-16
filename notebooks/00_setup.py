# Databricks notebook source
# MAGIC %md
# MAGIC # V0 · 建表与样例数据
# MAGIC
# MAGIC 把本仓库以 **Git folder**(Repos)方式克隆进 workspace,然后运行本 notebook。
# MAGIC
# MAGIC 它执行的 SQL 与 `sql/setup/databricks_setup.sql` **完全相同** ——
# MAGIC 两者都由 `src/rca/sampledata.py` 生成,不存在第二份逻辑。
# MAGIC
# MAGIC 全部语句幂等(`CREATE OR REPLACE` + 确定性 `INSERT`),重跑得到逐行相同的数据。

# COMMAND ----------

import sys
from pathlib import Path

# Git folder 里,notebook 的上级目录就是仓库根
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

from rca.config import build_target
from rca.knowledge import load_knowledge
from rca.sampledata import SampleDataConfig, build_setup_statements, run_verification

print("repo root:", REPO_ROOT)

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Unity Catalog 目录")
dbutils.widgets.text("schema", "gmv_rca", "Schema")
dbutils.widgets.text("daily_sessions", "6000", "日均会话数")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
DAILY_SESSIONS = int(dbutils.widgets.get("daily_sessions"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 加载并校验知识库
# MAGIC
# MAGIC 知识库过不了校验就不该继续 —— 分解引擎生成的 SQL 完全由这些 YAML 驱动。

# COMMAND ----------

knowledge = load_knowledge(REPO_ROOT / "knowledge")
print("指标:", sorted(knowledge.metrics))
print("维度(按钻取优先级):", [d.name for d in knowledge.dimensions_by_priority])
print("GMV 的乘法因子:", knowledge.metric("gmv").decomposition.factors)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 建表 + 灌数
# MAGIC
# MAGIC 每张表一条集合式 `INSERT ... SELECT`,全部在仓库侧计算 ——
# MAGIC Free Edition 只有一个 2X-Small,逐行插入 33 万行会直接烧掉配额。

# COMMAND ----------

import time

target = build_target("spark", catalog=CATALOG, schema=SCHEMA, spark=spark)
config = SampleDataConfig(daily_sessions=DAILY_SESSIONS)

for index, statement in enumerate(build_setup_statements(knowledge, target.dialect, config), 1):
    started = time.time()
    target.executor.run(statement.sql)
    print(f"[{index:>2}] {statement.label:<26} {time.time() - started:>7.2f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 自检
# MAGIC
# MAGIC **`sessions_converted` 必须等于 `completed_orders`。**
# MAGIC 不相等说明「一个转化会话恰好一个订单」这条前提被破坏,
# MAGIC `UV × CVR × AOV = GMV` 在数据层就不再成立,后续所有分解都不可信。

# COMMAND ----------

checks = run_verification(target.executor, target.dialect)
for name, value in checks.items():
    print(f"{name:<20} {value}")

assert checks["sessions_converted"] == checks["completed_orders"], (
    f"自检失败:sessions_converted={checks['sessions_converted']} "
    f"≠ completed_orders={checks['completed_orders']}"
)
print(f"\n✓ sessions_converted == completed_orders == {checks['completed_orders']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 看一眼数据

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT dt, SUM(order_amount) AS gmv, COUNT(*) AS orders
        FROM {CATALOG}.{SCHEMA}.fact_orders
        WHERE order_status = 'COMPLETED'
        GROUP BY dt
        ORDER BY dt
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC 下一步:打开 `01_decompose.py`。
