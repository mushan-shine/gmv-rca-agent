# Databricks notebook source
# MAGIC %md
# MAGIC # V0 · 在 Databricks 上跑验收测试
# MAGIC
# MAGIC 简报的 V0 验收标准是「**有测试证明**各分项贡献之和与总变化的差值 < 0.1%」。
# MAGIC 目标平台是 Databricks,所以这份证明必须能在 Databricks 上出具 ——
# MAGIC 只在本地 DuckDB 上跑通,证明的只是本地的算术。
# MAGIC
# MAGIC 本 notebook 用 `--rca-target=spark` 把**同一套测试**跑在当前 SparkSession 上:
# MAGIC 建库、灌数、分解、闭合性断言,全部发生在你的 workspace 里。
# MAGIC
# MAGIC 写入的是独立的 `gmv_rca_test` schema,不碰 `00_setup.py` 建出来的正式 schema。

# COMMAND ----------

# MAGIC %pip install -q pytest "sqlglot>=25.0" "pydantic>=2.7" "PyYAML>=6.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(REPO_ROOT)          # pytest 要靠 pyproject.toml 定位 rootdir
sys.path.insert(0, str(REPO_ROOT / "src"))

# /Workspace 文件系统不支持创建 __pycache__(Errno 95)。
# 普通 import 会静默忽略这个错误,但 pytest 改写断言时写缓存会直接崩,
# 表现为「ImportError while loading conftest」。关掉字节码写入,缓存引到 /tmp。
import tempfile
sys.dont_write_bytecode = True
sys.pycache_prefix = os.path.join(tempfile.gettempdir(), "pycache")

print("repo root:", REPO_ROOT)
print("存在 pyproject.toml:", (REPO_ROOT / "pyproject.toml").is_file())

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Unity Catalog 目录")
dbutils.widgets.text("test_schema", "gmv_rca_test", "测试用 schema(会被覆盖重建)")

os.environ["DATABRICKS_CATALOG"] = dbutils.widgets.get("catalog")
os.environ["RCA_TEST_SCHEMA"] = dbutils.widgets.get("test_schema")

print("测试目标:", os.environ["DATABRICKS_CATALOG"], ".", os.environ["RCA_TEST_SCHEMA"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 不需要数据库的那部分
# MAGIC
# MAGIC 知识库校验、手算对照、SQL 方言校验、凭据处理、SQL 守卫与比对器、LLM 接入层(用假 HTTP)、问题分流 ——
# MAGIC 先跑这些,快且能提前发现低级错误。
# MAGIC
# MAGIC **期望:`198 passed, 1 skipped`**(跳过的那条需要本地 duckdb)

# COMMAND ----------

import pytest

offline = pytest.main([
    "-q",
    "-p", "no:cacheprovider",   # 不往 Git folder 里写 .pytest_cache
    "tests/test_knowledge.py",
    "tests/test_math_analytic.py",
    "tests/test_databricks_dialect.py",
    "tests/test_config.py",
    "tests/test_nl2sql_units.py",
    "tests/test_llm.py",
    "tests/test_assistant_router.py",
])
print("\n退出码:", offline)
assert offline == 0, "离线测试未通过 —— 先修这些,不要急着连库"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 在 Spark 上跑数据相关的测试
# MAGIC
# MAGIC 这一步会在 `gmv_rca_test` 里建表灌数(约 7 万行会话),然后:
# MAGIC
# MAGIC * 样例数据的不变量(`test_sample_data.py`)
# MAGIC * 分解的算术闭合性(`test_closure.py`)
# MAGIC * 职责边界:越线必须报错(`test_decompose_guards.py`)
# MAGIC * **循环本身** —— 自修复、评估器自检、状态表、改进闭环(`test_nl2sql_loop.py`)
# MAGIC * **自主改进循环** —— 提议、裁决、台账、跨循环记忆、维度取值检查(`test_improve.py`)
# MAGIC * **问答助手** —— 归因数字与直接 SQL 一致、查数走自修复、失败与 👎 进待复核队列(`test_assistant.py`)
# MAGIC
# MAGIC **期望:`142 passed, 1 skipped`**(跳过的那条是 duckdb 专用)
# MAGIC
# MAGIC ⚠ 第一次跑要等 serverless 计算启动,可能一两分钟。

# COMMAND ----------

online = pytest.main([
    "-q",
    "-p", "no:cacheprovider",
    "--rca-target=spark",
    "tests/test_sample_data.py",
    "tests/test_closure.py",
    "tests/test_decompose_guards.py",
    "tests/test_nl2sql_loop.py",
    "tests/test_improve.py",
    "tests/test_assistant.py",
])
print("\n退出码:", online)
assert online == 0, "在 Databricks 上未通过 —— 这才是真正要紧的失败"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 把闭合性的实测数字打出来
# MAGIC
# MAGIC 测试只给通过/失败。简历和汇报里要引用的是**具体数量级**,所以单独打一遍。

# COMMAND ----------

from datetime import timedelta

from rca.config import build_target
from rca.decompose import DecompositionEngine, Period
from rca.knowledge import load_knowledge
from rca.sampledata import SampleDataConfig

knowledge = load_knowledge(REPO_ROOT / "knowledge")
target = build_target("spark", schema=os.environ["RCA_TEST_SCHEMA"], spark=spark)
engine = DecompositionEngine(knowledge, target.executor, target.dialect)

config = SampleDataConfig(daily_sessions=1200)
current = Period(config.end_date - timedelta(days=6), config.end_date, "本周")
baseline = Period(current.start - timedelta(days=7), current.start - timedelta(days=1), "上周")

factor = engine.decompose_factors("gmv", baseline, current)
print(f"因子分解残差      {factor.residual_pct:.3e} %   (简报要求 < 0.1%)")
print(f"恒等式偏差        {factor.identity_gap_pct:.3e} %")

worst = 0.0
for dimension in knowledge.ordered_dimensions("gmv"):
    result = engine.decompose_by_dimension("gmv", dimension, baseline, current)
    worst = max(worst, abs(result.residual_pct))
    print(f"  {dimension:<14} 残差 {result.residual_pct:>11.3e} %")
print(f"\n维度分解最大残差  {worst:.3e} %   (简报要求 < 0.1%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 收尾
# MAGIC
# MAGIC 测试 schema 留着可以重复跑(下次加 `--rca-skip-setup` 跳过建库更快)。
# MAGIC 要清掉就执行下面这格。

# COMMAND ----------

# spark.sql(f"DROP SCHEMA IF EXISTS {os.environ['DATABRICKS_CATALOG']}.{os.environ['RCA_TEST_SCHEMA']} CASCADE")
