# Databricks notebook source
# MAGIC %md
# MAGIC # V0 · 分解引擎
# MAGIC
# MAGIC 先跑 `00_setup.py`。
# MAGIC
# MAGIC 本 notebook 演示 V0 的两个函数,并**在真实 warehouse 上验证算术闭合性** ——
# MAGIC 这是简报 V0 验收标准 4 的现场复核版本。
# MAGIC
# MAGIC 这里没有 LLM,没有循环,没有状态表。维度的遍历顺序是 `dimensions.yaml`
# MAGIC 里写死的 priority,**不是**「根据上一轮结果决定下一轮」—— 后者是 V2。

# COMMAND ----------

# MAGIC %pip install -q "pydantic>=2.7" "PyYAML>=6.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

from rca.config import build_target
from rca.decompose import DecompositionEngine, Period, residual_severity
from rca.knowledge import load_knowledge
from rca.sampledata import SampleDataConfig

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace", "Unity Catalog 目录")
dbutils.widgets.text("schema", "gmv_rca", "Schema")

knowledge = load_knowledge(REPO_ROOT / "knowledge")
target = build_target(
    "spark",
    catalog=dbutils.widgets.get("catalog"),
    schema=dbutils.widgets.get("schema"),
    spark=spark,
)
engine = DecompositionEngine(knowledge, target.executor, target.dialect)

config = SampleDataConfig()
current = Period(config.end_date - timedelta(days=6), config.end_date, "本周")
baseline = Period(current.start - timedelta(days=7), current.start - timedelta(days=1), "上周")
print(f"基期 {baseline}    当期 {current}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. 乘法分解:GMV = UV × CVR × AOV
# MAGIC
# MAGIC ```
# MAGIC Δln GMV = Δln UV + Δln CVR + Δln AOV      ← 精确恒等,无近似
# MAGIC ```
# MAGIC
# MAGIC 注意 `占变化份额` 与 `自身变化` 是两个不同的量:
# MAGIC 前者是把总变化按对数份额**分配**给该因子,后者是该因子自己变了多少。
# MAGIC 报告里混用会得出错误结论。

# COMMAND ----------

factor = engine.decompose_factors("gmv", baseline, current)

print(f"GMV {factor.value_a:,.2f} → {factor.value_b:,.2f}   {factor.pct_change:+.2%}\n")
for c in factor.factors:
    share = "n/a" if c.share_of_change is None else f"{c.share_of_change:+.1%}"
    points = "n/a" if c.contribution_pct_points is None else f"{c.contribution_pct_points:+.2%}"
    print(
        f"  {c.factor:<6}{c.value_a:>14,.4f} → {c.value_b:>14,.4f}"
        f"   自身 {c.pct_change:>+8.2%}   份额 {share:>8}   分配 {points:>8}"
    )

print(f"\n残差        {factor.residual_pct:.3e} %")
print(f"恒等式偏差  {factor.identity_gap_pct:.3e} %")
print(f"闭合(<0.1%){factor.is_closed}")

assert factor.is_closed, "闭合性在真实 warehouse 上不成立 —— 停下来查口径"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 加法分解:按维度
# MAGIC
# MAGIC 引擎跑**两条**查询:一条不带 join 独立算总量,一条按维度分组,两者之差即残差。
# MAGIC 只跑分组查询再自行求和的话,残差恒为 0,信号就消失了。

# COMMAND ----------

rows = []
for dimension in knowledge.ordered_dimensions("gmv"):
    result = engine.decompose_by_dimension("gmv", dimension, baseline, current)
    assert result.is_closed, f"{dimension} 未闭合:残差 {result.residual_pct}%"
    top = result.parts[0]
    rows.append(
        {
            "dimension": dimension,
            "delta": round(result.delta, 2),
            "residual_pct": result.residual_pct,
            "severity": residual_severity(result.residual_pct),
            "unknown_share_pct": result.unknown_share_pct,
            "top_value": top.value,
            "top_delta": round(top.delta, 2),
            "top_share": None if top.share_of_change is None else round(top.share_of_change, 4),
        }
    )

display(spark.createDataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 可复现性(简报约束 3)
# MAGIC
# MAGIC 每个数字都对应一条确定性生成的 SQL,指纹会在 V2 写进
# MAGIC `decomposition_steps.sql_fingerprint`,任何结论都能被原样重放。

# COMMAND ----------

for record in factor.queries:
    print(f"── {record.purpose}   指纹 {record.fingerprint}   {record.row_count} 行")
    print(record.sql)
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 边界:越线必须报错,而不是给出一个像样但错误的数字

# COMMAND ----------

from rca.errors import FilterNotSupportedError, MetricNotAdditiveError

try:
    engine.decompose_by_dimension("cvr", "channel", baseline, current)
    raise AssertionError("应当拒绝比率指标的加法分解")
except MetricNotAdditiveError as exc:
    print("✓ 拒绝比率指标的维度分解:", exc)

try:
    engine.decompose_factors("gmv", baseline, current, {"category": "Apparel"})
    raise AssertionError("应当拒绝跨表不对齐的过滤")
except FilterNotSupportedError as exc:
    print("\n✓ 拒绝口径不对齐的过滤:", exc)
