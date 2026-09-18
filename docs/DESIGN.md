# 实现设计文档(按模块)

> 配套:[REQUIREMENTS.md](REQUIREMENTS.md)(需求,每条带编号)· [DEPLOYMENT.md](DEPLOYMENT.md)(上线)
>
> **怎么读:** 先看第 1 节的三张图,弄清系统由哪几层组成、三个循环怎么套在一起、每个 notebook 在干什么。
> 第 2 节按模块展开,每个模块都回答同样六个问题:**职责、输入输出、运行逻辑、关键设计、边界、测试**。
> 第 3 节用四个真实例子把模块串起来走一遍 —— 如果只想弄懂「它到底怎么跑的」,可以直接跳到第 3 节。

---

## 0. 一段话讲清楚

一个问题进来,系统先判断它是「查数」还是「归因」。

- **归因**(「上周 GMV 为什么跌了」)交给**确定性分解内核**:按 YAML 里定义的口径生成 SQL,把变化拆成流量 × 转化率 × 客单价,再按渠道、设备、品类等逐个拆开,并检查各部分加起来是否等于总变化。全程不经 LLM。
- **查数**(「上周各渠道 GMV 是多少」)交给 **LLM 写 SQL**,但每条 SQL 执行前都要过守卫和上下文检查。写错了,就把**具体错在哪**反馈给模型重写,最多 3 轮(**L1 自修复**)。

「LLM 写的 SQL 对不对」靠一套**评估集**来衡量:26 道题,标准答案由同一个分解内核**程序化生成**,不靠人工标注(**L2 评估**)。
评估发现的错题交给另一个 LLM 提出改进提示,改进必须通过**确定性闸门**才会被采纳,被拒绝的尝试记进台账,下一轮读到(**L3 自主改进**)。
所有过程都写进 Delta 状态表 —— **下一轮读取上一轮写下的状态**,这是它成为「循环」而不是「脚本」的原因。

---

## 1. 全局视图

### 1.1 分层架构

```
┌─────────────────────────────── 入口层 ───────────────────────────────┐
│ notebooks/00~05     app/app.py(网页)     databricks.yml(定时 Job)   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────── 应用层 ──────────────────────────────────┐
│  M14 问答助手 assistant/        router(分流)  service(装配、叙述、日志) │
└───────────┬───────────────────────────────────────────┬──────────────┘
            │ 查数                                       │ 归因
┌───────────▼──────────── 循环层 ──────────┐            │
│ M9  L1 自修复  loop.py                    │            │
│ M10 L2 评估    evaluate.py + compare.py   │            │
│ M11 提示知识   knowledge_store.py         │            │
│ M12 L3 自主改进 improve.py                │            │
└───┬─────────────┬──────────────┬─────────┘            │
    │             │              │                       │
┌───▼──────┐ ┌────▼─────────┐ ┌──▼─────────────┐ ┌──────▼─────────────┐
│ M6 LLM   │ │ M8 守卫       │ │ M5 评估集与    │ │ M4 分解引擎         │
│ llm.py   │ │ guard.py     │ │ 标准答案       │ │ decompose.py        │
│ M7 生成器 │ │ domains.py   │ │ cases.py ──────┼─► FromBuilder(共用)  │
│ generate │ │              │ │                │ │                     │
└──────────┘ └──────────────┘ └────────────────┘ └─────────────────────┘
                                   │
┌──────────────────────── 数据与执行层 ────────────────────────────────┐
│ M2 方言 dialects.py · 执行器 warehouse.py · 运行目标 config.py         │
│ M3 样例数据 sampledata.py            M13 状态表 state.py              │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────── 知识层(YAML,Git 唯一真源)──────────────┐
│ M1 tables.yaml · dimensions.yaml · metrics/*.yaml                      │
│    nl2sql/prompting.yaml(L3 唯一允许修改的文件)                       │
│    assistant/synonyms.yaml(业务口语词表)   eval/cases.yaml(评估集)   │
└──────────────────────────────────────────────────────────────────────┘
```

**依赖方向只向下。** 知识层不依赖任何代码;分解引擎不知道 LLM 的存在;LLM 客户端不知道 SQL 是什么。
这让每一层都能单独测试:分解引擎用 DuckDB 秒级验证,LLM 客户端用假 HTTP 验证。

### 1.2 三个循环怎么套在一起

```
                         ┌──────── L3 自主改进(跨版本,约 15~25 分钟一次)────────┐
                         │  读错题 + 台账 → LLM 提一条提示 → 试跑整个评估集        │
                         │  → 确定性闸门裁决 → 采纳 / 拒绝写进台账                 │
                         │                         ▲                            │
                         │                         │ 每次试跑 = 一次完整 L2        │
                         │   ┌───── L2 评估(一次跑批,约 4~5 分钟)─────┐          │
                         │   │  对 26 道题逐题调用 L1,结果与标准答案比对  │          │
                         │   │  → 指标写进 nl2sql_eval_runs              │          │
                         │   │            ▲                              │          │
                         │   │            │ 每道题 = 一次 L1             │          │
                         │   │  ┌── L1 自修复(一次提问,几秒)──┐        │          │
                         │   │  │ 生成 → 守卫 → 执行 → 反馈重写 │        │          │
                         │   │  └──────────────────────────────┘        │          │
                         │   └──────────────────────────────────────────┘          │
                         └────────────────────────────────────────────────────────┘

每一层循环写下的状态,都是外一层循环的输入:
  L1 → nl2sql_attempts(每一次尝试)
  L2 → nl2sql_eval_runs(每一次跑批的指标)
  L3 → nl2sql_improvement_ledger(每一个候选,含被拒绝的)→ 下一次 L3 读它
线上 → assistant_requests / assistant_feedback → 待复核队列 → 人确认后进 eval/cases.yaml → L2/L3
```

### 1.3 每个 notebook 在做什么

| notebook | 调用的模块 | 做什么 | 写哪些表 |
|---|---|---|---|
| `00_setup` | M1 → M3 → M2 | 加载并校验知识库,建 5 张业务表,确定性生成 8 周数据,跑自检 | 5 张业务表 |
| `01_decompose` | M4 | 演示因子分解、维度分解、可复现性、越界报错 | — |
| `02_run_tests` | 全部 | 在 Databricks 的 Spark 上跑整套测试(写入独立的 `gmv_rca_test`) | 测试 schema |
| `03_nl2sql_loop` | M6 M7 M8 M9 M10 M11 | 接 LLM → 演示一次自修复 → 跑评估集 → 半自动改进 → 回归闸门 | attempts、eval_runs |
| `04_improvement_loop` | 上面全部 + M12 | 自主改进循环:无人值守地提议、试跑、裁决、记账 | attempts、eval_runs、ledger |
| `05_assistant` | M14(及其依赖) | 问答入口:改输入框提问 | requests、feedback |

---

## 2. 模块设计

### M1 知识库 —— `knowledge/*.yaml` + `src/rca/knowledge.py`

**职责:** 定义「数据长什么样、指标怎么算、能按什么拆」。这是整个系统唯一的业务口径来源,代码里没有任何写死的业务规则。

**三个文件:**

| 文件 | 内容 | 例子 |
|---|---|---|
| `tables.yaml` | 表结构登记簿:每张表的列、类型、主键 | `fact_orders.order_amount: double` |
| `dimensions.yaml` | 维度注册表:名称、中文名、钻取优先级、**在哪张表上怎么取到** | `category` 在 `fact_orders` 上要 join `dim_product`(主键唯一) |
| `metrics/*.yaml` | 指标树 | 见下 |

指标有两种形态:

```yaml
# 可加基础指标:有 SQL 和来源表
name: gmv
sql: SUM(order_amount)
source_table: fact_orders
filters_default: {order_status: "COMPLETED"}      # 口径:只算已完成订单
decomposition: {type: multiplicative, factors: [uv, cvr, aov]}
dimensions: [channel, device, market, region, category, sub_category, price_tier, user_tier]

# 比率指标:由分子分母定义,没有自己的 SQL
name: cvr
ratio: {numerator: orders, denominator: uv}
```

**运行逻辑(`load_knowledge`):** 读三类 YAML → pydantic 严格模式解析(多一个字段都报错)→ 交叉校验:

1. 引用的表、列、维度都存在;维度的 join 路径指向存在的表和键
2. 维度优先级不重复(否则钻取顺序不确定)
3. 指标的维度白名单顺序与全局优先级一致
4. **乘法恒等式做符号验证**:把 GMV 和 UV × CVR × AOV 都展开成基础指标的分子/分母多重集合,
   UV × (orders / UV) × (GMV / orders) 化简后必须恰好等于 GMV。写错一个因子,加载就失败

**关键设计:** 恒等式在**加载时**验证,而不是在算出奇怪数字之后。分解的数学正确性建立在这个恒等式上,它错了后面全错。

**边界:** 知识库只描述,不执行。L3 改进循环**不允许**修改这三个文件(它只能改 `prompting.yaml`)。

**测试:** `test_knowledge.py`(正向加载 + 13 类错误配置都被拒绝)

---

### M2 方言、执行器与运行目标 —— `dialects.py` · `warehouse.py` · `config.py`

**职责:** 让同一份 SQL 生成逻辑跑在三个地方:Databricks notebook(Spark)、本机连 SQL Warehouse、本机 DuckDB。

**方言(`SqlDialect`):** 抹平两边写法真正不同的那几个原语,其余 SQL 完全相同。

| 原语 | Databricks | DuckDB |
|---|---|---|
| 表名 | `workspace.gmv_rca.fact_orders` | `main.fact_orders` |
| 数组展开 | `explode` | `unnest` |
| 哈希取模 | `pmod(hash(x), n)` | `hash(x) % n` |
| 建表后缀 | `USING DELTA` | — |
| 加列 | `ALTER TABLE t ADD COLUMNS (c T)` | `ALTER TABLE t ADD COLUMN c T` |

**执行器(`SqlExecutor` 协议):** 只有一个方法 `run(sql) -> list[dict]`。

| 执行器 | 什么时候用 | 说明 |
|---|---|---|
| `SparkExecutor` | notebook / Job 里(主路径) | 用当前 SparkSession,不需要 token |
| `DatabricksExecutor` | 本机连 warehouse;Databricks Apps | 连接复用;支持个人 token 或 **OAuth**(Apps 里用 App 自己的身份) |
| `DuckDBExecutor` | 本地测试 | 内存库,**不是部署目标** |

每个执行器记录最近 500 条执行过的 SQL(常驻的网页服务不能无限增长)。`sql_fingerprint` 把 SQL 折叠空白后取哈希,用于「这个数字是哪条 SQL 算的」。

**运行目标(`build_target`):** 按名字装配「方言 + 执行器」。凭据只从环境变量或 `.env` 读;`DatabricksSettings` 的 `repr` 把 token 打码,防止它出现在异常栈里。

**测试:** `test_databricks_dialect.py`(生成的 SQL 用 sqlglot 按 Databricks 方言解析,确认合法)、`test_config.py`(token 不出现在任何可打印的地方;OAuth 路径不使用个人 token)

---

### M3 样例数据 —— `sampledata.py`

**职责:** 在没有真实数据的情况下,造一个「像真的」且**可复现**的电商数仓。

**运行逻辑:** `build_setup_statements()` 生成一串 SQL:建表 DDL → 每张表一条 `INSERT ... SELECT`。
数据全部在仓库里计算,不从 Python 逐行插入。

| 设计 | 原因 |
|---|---|
| **集合式插入**,每张表一条 SQL | Free Edition 只有一个最小规格的 warehouse,逐行插入 33 万行会耗尽配额 |
| **不用 `rand()`**,所有随机量都来自 `hash(种子字符串)` | 同一方言下重跑逐行一致;订单可以从会话**重新推导**,不需要中间表 |
| **一个转化会话恰好产生一笔完成订单** | 让 GMV = UV × CVR × AOV 在数据层严格成立,CVR 可以从两张表交叉验证 |
| 渠道 × 设备 × 市场各有不同的基础转化率,叠加按日期的自然波动 | 让分解结果有东西可看 |

默认 2025-05-05 ~ 2025-06-29(8 周),每天 6000 个会话(测试用 1200)。**报告日**是数据结束的次日 2025-06-30 —— 「上周」就是 06-23 ~ 06-29。

`sql/setup/*.sql` 是这个模块渲染出的**产物**(贴进 SQL Editor 即可建库),`test_sql_rendering.py` 防止它和代码不一致。

**测试:** `test_sample_data.py`(行数、取值范围、确定性、会话与订单对齐)

---

### M4 分解引擎 —— `decompose.py`

**职责:** 把一个指标在两个时间窗口之间的变化,**精确地**拆开。这是归因的全部数学,也是评估集标准答案的来源。

#### 乘法(因子)分解 `decompose_factors(metric, 基期, 当期, filters)`

```
GMV = UV × CVR × AOV
ln GMV = ln UV + ln CVR + ln AOV
Δln GMV = Δln UV + Δln CVR + Δln AOV          ← 精确恒等,没有近似
```

运行逻辑:

1. 校验两个窗口不重叠、天数相同;校验过滤维度在**所有**涉及的事实表上都可用
2. 把 GMV 和三个因子展开成基础指标(gmv、orders、uv),**每张事实表一条 SQL**,同时算出两个窗口的值
3. 用基础指标算出每个因子两期的值,取对数差 `log_delta`
4. 每个因子的份额 = 它的 `log_delta` / 总 `log_delta`;分到的百分点 = 份额 × 总变化率
5. 残差 = 总 `log_delta` − 各因子 `log_delta` 之和(理论为 0,实测约 1e-12%)
6. 恒等式偏差 = 各因子乘积与 GMV 本身的差距 —— 这是对知识库口径的**运行期复核**

#### 加法(维度)分解 `decompose_by_dimension(metric, dimension, 基期, 当期, filters)`

```
ΔGMV = Σ ΔGMV(每个渠道)
```

运行逻辑:**跑两条 SQL** —— 一条不带 join 直接算总量,一条带 join 按维度分组。
两者之差就是残差。如果只跑分组查询再自己求和,残差恒为 0,join 把行放大或丢掉就再也发现不了。
join 路径由 `FromBuilder` 按 `dimensions.yaml` 生成:维表包在子查询里、列统一改名,不会和事实表同名列冲突。

| 残差 | 诊断 |
|---|---|
| ≈ 0 | 划分完整 |
| < 0 | join 放大(维表主键不唯一) |
| > 0 | join 丢行(维表缺记录);查看 `__UNKNOWN__` 桶的占比 |

**边界(越线就报错,而不是给一个像样但错误的数字):**
比率指标不能按维度加法拆分(各分组的转化率加起来不等于总转化率)· 窗口重叠 · 窗口天数不同(除非显式允许)·
过滤维度只在部分表可用(UV 和 GMV 会落在不同口径上)· 因子非正(对数无定义)。

**测试:** `test_closure.py`(残差 < 0.1%)· `test_math_analytic.py`(与手算结果逐位对照)· `test_decompose_guards.py`(每一种越界都报错)

---

### M5 评估集与标准答案 —— `eval/cases.yaml` + `nl2sql/cases.py`

**职责:** 给「LLM 写的 SQL 对不对」提供一把尺子。

**题目只声明问什么,不写 SQL:**

```yaml
- id: gmv_us_w8
  question: "What was the total GMV in the US market last week?"
  gold: {kind: aggregate, metric: gmv, period: W8, filters: {market: US}}
```

**运行逻辑:** `build_reference_sql(case)` 按知识库生成参考 SQL(复用 M4 的 `FromBuilder`,join 路径只有一份实现)→
`build_golden()` 执行它,得到标准答案结果集。窗口标签 W1~W8 解析成具体日期,W8 就是「上周」。

| 题型 | 数量 | 说明 |
|---|---|---|
| `aggregate` 标量 | 9 | 可加指标直接聚合;比率指标用 CTE 拼分子分母 |
| `group_by` 分组 | 9 | 只限可加指标 |
| `change` 两期之差 | 4 | |
| `unanswerable` 不可答 | 4 | 客户邮箱、广告计划的 GMV、毛利率、退货原因 —— 数据里没有,**正确答案是拒答** |

26 题分成**训练集 21 题**(改进循环能看到它们的失败)和**留出集 5 题**(改进循环看不到,专门检查是不是在背答案)。

**关键设计:** 程序化生成带来三个好处 —— 数据重建后答案自动更新;不存在「标注者以为的口径」和「系统的口径」不一致;
新增一道题只需要写三行声明。代价是题目只能覆盖知识库能表达的问题形态。

**测试:** `test_nl2sql_units.py`(题目校验、参考 SQL 生成);`ReferenceGenerator` 拿满分(M10)

---

### M6 LLM 接入 —— `nl2sql/llm.py`

**职责:** 把「调用大模型」这件事做成一个可靠、可计量、不泄密的部件。它不知道 SQL 是什么。

| 部件 | 做什么 |
|---|---|
| `ChatClient` 协议 | `complete(prompt) -> LlmResponse`(文本、token 数、耗时、是否被截断) |
| `OpenAIChatClient` | 走 OpenAI chat-completions 格式(用标准库 `urllib`,不依赖 SDK)。Databricks 端点、智谱、OpenAI 都讲这个格式,换供应商只改 `base_url` 和 `model` |
| `CachingChatClient` | 缓存键 = 模型 + 参数 + prompt。评估会把完全相同的 prompt 发很多次,第二次不再付费 |
| `UsageMeter` | 累计调用次数和 token;超过上限就在**发请求之前**熔断;支持按跑批做差 |
| `build_chat_client` | 按环境变量装配;智谱预设 `glm-4-flash` + `do_sample=false` |

**重试规则:** 429 / 5xx / 超时 → 退避后重发**同一个 prompt**,最多 3 次;400 / 401 → 立即失败(重发多少次都一样)。

**关键设计:传输重试 ≠ 自修复。** 重试封装在这一层,循环层看不见。否则一次网络抖动会被记成「模型改了两轮才答对」,
平均轮数这个指标就失去意义。

**凭据:** key 放在请求头,不进 URL;`repr` 打码;错误信息里不出现 key;发请求前检查 key 里有没有控制字符(真实踩过:Ctrl+V 把一个不可见字符存进了 key)。

**测试:** `test_llm.py`(用假的 `urlopen` 把 HTTP 行为钉死,不需要网络)

---

### M7 生成器与 prompt —— `nl2sql/generate.py`

**职责:** 把问题变成一段 prompt,把模型回复变成 SQL 或拒答。

**prompt 的结构(`build_prompt`,纯函数,可复现、可 diff):**

```
You are a careful data analyst. Write ONE read-only Databricks SQL query ...
## Schema                          ← 由 tables.yaml 生成:完全限定表名 + 列 + 类型
## Metric definitions and gotchas  ← prompting.yaml 里的提示(L3 改的就是这里)
## Reporting calendar              ← 「今天」是哪天,「上周」是哪几天
## Rules                           ← 只输出 SQL;不多选列;答不了就回 CANNOT_ANSWER
## Examples                        ← few-shot 示例(03 的半自动改进会加)
## Your previous attempt failed    ← 只在自修复时出现:上一轮的具体错误
```

**报告日历为什么重要:** 第 1 轮真实评估 0%,原因是模型写了 `CURRENT_DATE()`,查的是 2026 年,而数据在 2025 年。
加上报告日历后,**只改这一处**,准确率到了 63.6%。

**解析回复(`parse_response`):** 取 ```` ```sql ```` 代码块;回复以 `CANNOT_ANSWER` 开头(哪怕被包在代码块里)视为拒答;空回复视为生成错误,而不是拒答。

**四种生成器,同一个接口:**

| 生成器 | 用途 |
|---|---|
| `LlmSqlGenerator` | 真模型 |
| `ReferenceGenerator` | 永远给出参考 SQL —— 用来**校验评估器本身** |
| `ScriptedGenerator` | 按剧本逐轮返回 —— 用来构造「第一轮写错、第二轮改对」这类测试 |
| `CallableGenerator` | 把任意函数包成生成器 |

---

### M8 守卫与取值检查 —— `nl2sql/guard.py` · `nl2sql/domains.py`

**职责:** 在 SQL 执行**之前**做确定性检查,并给出**可以照着改的反馈**。不经 LLM。

**守卫(`SqlGuard.check`)按顺序检查:**

| 检查 | 失败时的反馈举例 |
|---|---|
| 空 SQL | 「没有写出 SQL」 |
| 能否解析(sqlglot) | 解析器报错原文 |
| 只读 | 出现 `INSERT` / `DROP` / `GRANT` 等关键字就拒绝(按词法判断,宁可误杀) |
| 单条语句 | 「只能写一条语句」 |
| 表存在 | 「表 orders 不存在,可用的表有 …」 |
| 列存在 | 「列 revenue 不存在,fact_orders 上有 order_amount, …」→ 同时标记为**幻觉** |

日期函数里的 `day`、`week` 这类时间单位不算列(真实踩过:DuckDB 把 `DATEADD(day, …)` 的 `day` 解析成了列,被误判成幻觉)。

**上下文检查(属于 M9,依赖运行上下文):**

- **真实时钟**:SQL 里出现 `CURRENT_DATE` / `NOW()` → 拦下,提示改用报告日历里的具体日期
- **维度取值(`ValueDomains`)**:启动时**从仓库实时查出**每个低基数字符串列的实际取值(≤ 50 个)。
  SQL 里写 `market = 'Germany'` → 拦下并反馈「market 没有取值 'Germany',全部取值:DE, UK, US」;
  写 `region = 'US'` → 「'US' 实际出现在列:market」

**关键设计:** 这类 SQL 语法正确、执行成功、返回空集 —— 守卫和引擎都不会报错,不在执行前拦下,L1 永远发现不了。
取值**不写进 prompt**:模型只在写错时才看到。能不能把这类错误沉淀成一条提示,交给 L3 去做。

**测试:** `test_nl2sql_units.py`(守卫各项)· `test_improve.py`(取值检查,用第 2 轮真实评估里模型写错的 SQL 作为用例)

---

### M9 L1 自修复循环 —— `nl2sql/loop.py`

**职责:** 回答一个问题,必要时自己改错。

**运行逻辑(`AnswerLoop.answer`),每一轮:**

```
1. 生成   generator.generate(context)
          ├─ 模型调用本身失败(重试用尽 / 超预算)→ 记录,停止,状态 FAILED
          ├─ 回复里取不出 SQL → 把「没有写出 SQL」作为反馈,进入下一轮
          └─ 拒答 → 状态 REFUSED,结束
2. 守卫   guard.check(sql)
3. 上下文 真实时钟检查 + 维度取值检查
          └─ 2 或 3 有问题 → 把全部问题作为反馈,进入下一轮
4. 执行   executor.run(sql)
          └─ 引擎报错 → 把报错第一行作为反馈,进入下一轮
5. 成功   状态 ANSWERED,返回结果
超过 3 轮 → 状态 FAILED
```

每一轮都记成一个 `Attempt`(SQL、指纹、守卫信息、上下文信息、引擎报错、token、耗时),失败类型按优先级归为
`refused / generator / hallucination / guard / context / execution / none` 之一。

**关键设计:它是循环,不是重试。** 区别在于反馈是具体的、可照着改的。把同一个 prompt 再发一遍,模型没有获得任何新信息,改对全凭运气。
真实数据:第 2 轮评估首轮准确率 50.0%,最终 63.6%,**+13.6 个百分点来自这里**。

**`prompt_fingerprint()`:** 把除问题以外的全部上下文(schema、提示、示例、日历、最大轮数、是否启用取值检查)取哈希。
真实踩过:加报告日历前后两次评估的知识版本都是 v1,光看版本号解释不了 0% → 63.6%。指纹变了,就说明上下文变了。

**边界:** L1 只能发现「跑不跑得通」,**发现不了「答得对不对」**。一条语法正确、执行成功、但忘了过滤已完成订单的 SQL,L1 会当成成功返回 —— 只有 L2 抓得住。

**测试:** `test_nl2sql_loop.py`(写错列 → 反馈 → 改对;超过轮数放弃;网络重试不计轮数;「能跑通但答错」只有评估器能发现)

---

### M10 L2 评估 —— `nl2sql/compare.py` · `nl2sql/evaluate.py`

**职责:** 给一个「模型 + 知识版本」打分,并判断它比上一次好还是差。

**比对规则(`compare_result_sets`)—— 比结果,不比 SQL 文本:**

| 规则 | 原因 |
|---|---|
| 忽略列名 | `AS total` 和 `AS value` 是同一个答案 |
| 忽略行序(题目声明有序的除外) | 只有「前三名」这类题才讲顺序 |
| 数值统一成 float,相对容差 1e-6 | `DECIMAL` / `DOUBLE` / `int` 混用 |
| **列数必须一致** | 顺手多选一列也算错:答案的形状是答案的一部分 |

失败时给出分类:列数不对 / 行数不对 / 数值不对 / 应为空 / 不应为空。

**运行逻辑(`Evaluator.run`):**

```
标准答案(第一次调用时生成,之后复用)
  ↓
逐题:loop.answer(问题) → 判定 → CaseResult
  ↓
判定(完全确定性):
  不可答题:拒答 → correct_refusal;写了 SQL → should_have_refused
  可答题:  拒答 → false_refusal;3 轮没跑通 → no_runnable_sql;
           比对不一致 → wrong_answer;一致 → correct
  ↓
EvalReport:指标 + 本次用量(对计量器做差)+ prompt 指纹
  ↓
写状态:每次尝试 → nl2sql_attempts;指标一行 → nl2sql_eval_runs
```

**两个闸门:**

| 闸门 | 用在哪 | 规则 |
|---|---|---|
| `regression_gate` | `03`:这次评估比上一次退步了吗 | 留出集准确率、全量准确率不降;幻觉率、误拒率不升。**只和同一模型的上一次比** |
| `improvement_gate` | `04`:这条候选提示能不能采纳 | 见 M12 |

**评估器自检:** `ReferenceGenerator`(永远给出参考 SQL)必须拿 100%。拿不到,说明是比对逻辑、标准答案或守卫坏了,而不是模型不行。

**测试:** `test_nl2sql_units.py`(比对规则、闸门)· `test_nl2sql_loop.py`(评估器自检、状态表)

---

### M11 提示知识 —— `knowledge/nl2sql/prompting.yaml` + `nl2sql/knowledge_store.py`

**职责:** 保存「L3 被允许修改的那部分知识」,带版本、带来历。

```yaml
version: 1
hints:
  - text: "GMV, order counts and AOV must be filtered to order_status = 'COMPLETED'. ..."
    source_run: run_xxx      # 哪次评估发现的
    fixes_case: gmv_us_w8    # 为了修哪道题
examples: []                 # few-shot 示例,初始为空,由失败驱动产生
```

**两种改进方式:**

| | `03` 的半自动改进(`propose_improvement`) | `04` 的自主改进(M12) |
|---|---|---|
| 谁提改进 | 规则:给答错的**训练题**补一条 few-shot(用它的参考 SQL),并按失败类型查表补一条固定提示 | LLM 读错题后提一条提示 |
| 谁决定 | 人看闸门结果决定是否合并 | 确定性闸门 |
| 留出集 | 参考 SQL 映射里**排除留出集**,留出集答错也不会变成示例 | 提议者看不到留出集 |

**边界:** 只改提示和示例,不改代码、不改指标定义、不动留出集。合并进 Git 要人 review。

---

### M12 L3 自主改进循环 —— `nl2sql/improve.py`

**职责:** 无人值守地:找错题 → 提改进 → 验证 → 取舍 → 记住。

**运行逻辑(`AutoImprover.run`):**

```
0. 基线:用当前知识跑一次完整评估
   读台账:这个模型以往所有被拒绝的提示(跨循环记忆)

每一轮:
1. 读状态    训练集错题(不含留出集)
2. 分组      每道错题归到一种失败形态:
             dimension_value(维度取值写错)· missed_refusal(该拒答没拒答)·
             wrong_value(数值错)· wrong_shape(行数/列数错)· no_runnable_sql · false_refusal
             选「本次循环还没试过的、数量最多的」一类作为本轮目标
3. 提议      LLM 只看这一类错题(问题、它写的 SQL、错在哪),以及以往被拒绝的提示 → 一条提示(JSON)
4. 便宜检查  重复?(和已有 / 已拒绝的提示一样)→ duplicate,不试跑
             太空泛?(没有点名任何表、列、取值、指标)→ too_generic,不试跑
5. 试跑      只加这一条提示,其余全部不变,重跑整个评估集(约 37 次模型调用、4~5 分钟)
6. 裁决      improvement_gate(完全确定性):
               训练集至少多对 1 题
               且 本轮针对的那类错题里至少修好 1 题
               且 留出集准确率不降
               且 幻觉率、误拒率不升
7. 写状态    采纳 → 知识版本 +1,成为新基线,清空「已试过的类别」
             拒绝 → 连同原因记进台账,这一类标为已试过
             每一轮都记录修好了哪些题、改坏了哪些题

停止:训练集和留出集都达标 / 训练集没有错题 / 连续 N 轮没有采纳 / 提议者出错 / 预算用完 / 轮数上限
```

**三条纪律(决定了这是 loop engineering,而不是「让 LLM 调 prompt」):**

1. **提议者看不到留出集** —— 否则留出集的错题也能被「修」掉,闸门形同虚设
2. **提议者看不到参考 SQL** —— 否则最省事的做法是把答案抄进提示
3. **被拒绝的尝试要被记住** —— 否则确定性解码的模型会一轮又一轮地提同一个无效修改

**第 2、4、6 步的分组、具体性检查和归因检查,来自第一次真实运行的教训:**
两轮都提了「核对列名和取值」这类空话,各让训练集 14 → 12,都被拒绝。
空话本身就让分数变了 2 题,说明在 21 题的规模上 ±2 是噪声量级 —— 只看总分的闸门会把运气当成改进。

**测试:** `test_improve.py`(用一个「对错取决于 prompt 里有没有某条提示」的假生成器,让采纳 / 拒绝的结果可以逐条断言;用真实被拒的两条原话验证具体性检查)

---

### M13 状态层 —— `nl2sql/state.py`

**职责:** 所有循环的记忆。全部是 Delta 表,只追加,不修改。

| 表 | 谁写 | 谁读 | 一行是什么 |
|---|---|---|---|
| `nl2sql_attempts` | L2(每题每轮) | `03` 的失败画像;排查 | 一次尝试:SQL、失败类型、判定、token |
| `nl2sql_eval_runs` | L2 | 回归闸门(找同一模型的上一次) | 一次跑批的全部指标 + 模型 + 知识版本 + prompt 指纹 |
| `nl2sql_improvement_ledger` | L3 | **下一次 L3**(以往被拒绝的提示) | 一个候选:提示、针对的类别、裁决、原因、前后指标、修好 / 改坏的题 |
| `assistant_requests` | 问答助手 | 待复核队列 | 一个线上问题:路径、状态、回答、SQL、耗时 |
| `assistant_feedback` | 问答助手 | 待复核队列 | 一次 👍 / 👎 |

**结构迁移(`ensure_tables`):** `CREATE TABLE IF NOT EXISTS` 不会更新已有表。代码加了列,workspace 里的旧表不变,下一次写入就失败。
所以启动时逐表比对列,缺的用 `ALTER TABLE … ADD` 补上。**绝不删表** —— 里面是循环的记忆。

---

### M14 问答助手 —— `src/rca/assistant/`

**职责:** 把上面这些部件装配成一个业务能用的问答服务。它本身几乎没有新的计算逻辑。

**分流(`router.py`),完全确定性:**

```
问题为空                          → 帮助
问题里有「为什么 / 原因 / 归因 / why …」 → 归因,并识别:
    指标    按 synonyms.yaml 的顺序匹配(「订单转化率」→ cvr,不是 orders);没有就用 GMV,并写明
    窗口    「昨天」→ 按天;「上周」→ 按周;都没有 → 按上周,并写明
    过滤    只识别**在全部相关事实表上都是原生列**的维度(渠道、设备、市场、大区)
            「美国站」→ market=US(词表);「mobile」→ device=mobile(仓库实时取值)
            短取值区分大小写:「tell us why」里的 us 不是美国
            同一维度出现多个值 → 不过滤,并写明
其他                              → 查数
```

**归因路径(`service._explain`):**

```
窗口(与 text-to-SQL 用同一套报告日历)
  → 指标有乘法分解?→ decompose_factors → 三个因子的贡献
  → 指标可加?     → 按优先级对每个维度 decompose_by_dimension
                     (跳过已过滤的维度;跳过过滤后只剩一个取值的维度)
  → 比率指标?     → 说明不能按维度相加拆分,建议改问 GMV
  → 模板叙述:结论、因子、贡献最大的维度取值、可信度(最大残差)
```

**查数路径(`service._query`):** 调用 M9 的 `AnswerLoop`,提示来自 `prompting.yaml`(04 采纳、经 Git 合并的提示就是在这里进入线上的)。
没有接 LLM 时明确告知,归因照常可用。

**日志与反馈:** 每个问题写进 `assistant_requests`;👍👎 写进 `assistant_feedback`;
`review_queue()` = 没答出来的 + 被点 👎 的。日志写失败不影响用户拿到答案。

**关键设计:** 回答里**没有一个数字是模型写的**。归因的数字来自分解内核,叙述由模板生成;每个回答附带它采用的全部假设。

**测试:** `test_assistant_router.py`(分流,不需要数据库)· `test_assistant.py`(归因数字与直接写 SQL 算出的一致;查数经过自修复;失败和 👎 进入待复核队列)

---

### M15 入口 —— notebooks · `app/app.py` · `databricks.yml`

| 入口 | 装配方式 | 说明 |
|---|---|---|
| notebook `05_assistant` | `build_target("spark")` + `build_assistant` | 改 `question` 组件后运行第 3 格 |
| 网页 `app/app.py` | Apps 里用 `build_app_target()`(OAuth);本机默认 DuckDB + 样例数据 | Streamlit;整个进程共用一个助手实例,问答串行执行 |
| `databricks.yml` | 三个 notebook 任务 | 每日评估(`03`)· 每周改进(`04`)· 每周归因(`05`);默认暂停 |

部署细节见 [DEPLOYMENT.md](DEPLOYMENT.md)。

---

## 3. 四个例子:把模块串起来走一遍

### 例 1:一道查数题在 L1 里被修好

问题:`What was the total GMV in the US market last week?`(评估集 `gmv_us_w8`)

```
第 1 轮  M7 拼 prompt(schema + 4 条提示 + 报告日历)→ M6 调智谱
         模型写:SELECT SUM(order_amount) FROM ... WHERE region = 'US' AND dt BETWEEN DATE '2025-06-23' AND DATE '2025-06-29'
         M8 守卫:只读 ✓ 表列都存在 ✓
         M9 上下文检查:region 的实际取值是 EMEA, NA → 拦下
         反馈:「列 region 没有取值 'US'。region 的全部取值:EMEA, NA。'US' 实际出现在列:market。」
第 2 轮  prompt 末尾多了「Your previous attempt failed」和上面这句
         模型改成 market = 'US' → 守卫 ✓ 上下文 ✓ → 执行 → 返回 1 行
M10 比对  与标准答案(M5 用知识库生成的参考 SQL 的结果)一致 → correct,用了 2 轮
          → 计入最终准确率,不计入首轮准确率
M13      两次尝试都写进 nl2sql_attempts
```

(示意。`region = 'US'` 是第 2 轮真实评估里模型确实犯过的错,这里把它放到这道题上演示。)
没有取值检查的话,第 1 轮的 SQL 会执行成功、返回 NULL,L1 当成功返回,只有 L2 能发现它答错了。

### 例 2:一道归因题

问题:`上周美国站 GMV 为什么下降?`

```
M14 分流   有「为什么」→ 归因;指标 gmv;窗口 week;「美国站」→ market=US
窗口       as_of 2025-06-30 → 上周 06-23~06-29,前一周 06-16~06-22
M4 因子    fact_orders 一条 SQL(gmv、orders 两期)+ fact_sessions 一条(uv 两期)
           → UV +1.4 · CVR −4.2 · AOV +0.8 个百分点,相加 = −2.1%,残差 6e-13%
M4 维度    按优先级:渠道、设备、(跳过市场:已过滤)、(跳过大区:只剩 NA)、品类、子品类、价格带、用户分层
           每个维度两条 SQL(总量 + 分组)
叙述       「GMV(US) 上周为 17,214,较前一周下降 2.1%(−363)。
            因子:流量 UV +1.4 · 转化率 CVR −4.2 · 客单价 AOV +0.8
            维度:贡献最大的是 用户分层 = RETURNING(−13.7)、渠道 = Direct(+7.2)……
            可信度:分解闭合」
           + 假设:对比窗口、过滤条件
M13        写进 assistant_requests
```

(数字来自本地 DuckDB 的测试数据,Databricks 上的样例数据规模不同,数字会不一样。)

### 例 3:一次评估跑批(`03` 第 4 格)

```
M5 生成 26 个标准答案(每题执行一次参考 SQL)
逐题调用例 1 的流程 → 26 个 CaseResult
M10 汇总:最终准确率 63.6% · 首轮 50.0% · 平均 1.35 轮 · 留出集 50.0% · 34 次调用 · 2.7 万 token
M13 写 nl2sql_eval_runs 一行(含 prompt 指纹)
回归闸门:找同一模型的上一次跑批(0%)→ 各项都没有退步 → 通过
```

### 例 4:一轮自主改进(`04`)

```
基线      训练集 14/21 · 留出集 4/5
台账      读到这个模型以往被拒绝的 2 条提示
分组      训练集 7 道错题:missed_refusal 2 · wrong_value 2 · no_runnable_sql 2 · wrong_shape 1
          → 本轮目标 missed_refusal(数量并列,按 PATTERNS 的顺序)
          (跑不通的 2 道如果是被取值检查拦下的,会归为 dimension_value,并排在最前)
提议      LLM 只看这 2 道题 → 例如「If a question needs data no table has (campaign-level revenue,
          cost or margin), reply CANNOT_ANSWER instead of guessing.」
便宜检查  不重复;点名了 CANNOT_ANSWER → 通过,进入试跑
试跑      加上这条提示重跑 26 题
裁决      训练集 14 → 15,修好 unanswerable_profit_margin(在本轮目标内),留出集 4/5 不变
          → 采纳,知识 v1 → v2;下一轮换一类错题
          (如果训练集是 14 → 15 但修好的是另一类题 → 按噪声处理,拒绝)
台账      这一轮写一行:提示、目标类别、裁决、原因、前后指标、修好 / 改坏的题
```

(这一例的提示和结果是示意。第一次真实运行 0 条采纳,见 [EXECUTION_LOG.md](EXECUTION_LOG.md) 阶段 H。)

---

## 4. 贯穿全部模块的设计取舍

| 取舍 | 选择 | 放弃了什么 | 理由 |
|---|---|---|---|
| 谁判断对错 | 确定性代码(守卫、比对器、闸门) | LLM-as-judge 的灵活性 | 尺子本身不能有随机性,否则分数变化无法归因 |
| 标准答案 | 程序化生成 | 题型的开放性 | 数据变了答案自动跟着变;口径与系统一致 |
| 归因 | 分解内核 + 模板叙述 | 更自然的语言 | 数字必须对得上账,且可回溯到 SQL |
| 改进的粒度 | 每轮只加一条提示 | 迭代速度 | 每一行台账的指标变化都能归因到那一条提示 |
| 改进的范围 | 只改提示,不改代码和口径 | 改进的上限 | 可审查、可回滚;口径变更必须有人签字 |
| 解码 | `do_sample=false` | 多样性 | 同一个问题两次得到同一个答案,指标涨跌才能归因 |
| 框架 | 不用 LangChain / 向量库 | 现成组件 | 重试、检索、反馈这些关键逻辑要摊开,可测试 |
| 测试 | 同一套测试跑在 DuckDB 和 Spark 上 | — | 本地秒级回归;Databricks 上出具真机证据 |

---

## 5. 测试策略

| 层 | 手段 | 证明什么 | 不能证明什么 |
|---|---|---|---|
| 算术 | DuckDB 内存库,`pytest -q`(约 15 秒) | 分解闭合、知识库校验、数据不变量、循环与闸门的逻辑 | SQL 在 Spark 上能不能跑 |
| 方言 | sqlglot 按 Databricks 方言解析全部生成的 SQL | 语法合法;DuckDB 写法没有混进 Databricks 产物 | 语义约束 |
| 真机 | `02_run_tests`(`--rca-target=spark`) | 在目标引擎上确实跑得通、算得对 | — |
| 真模型 | `03` / `04` 接智谱 | 真实准确率、真实失败模式 | 不进 CI(花钱、非确定) |

本地 352 条;Databricks 上 `02` 两段期望 `198 passed, 1 skipped` 和 `143 passed, 1 skipped`。

**测试里怎么模拟 LLM:** 从不调用真模型。`ScriptedGenerator` 按剧本逐轮返回;`HintSensitiveGenerator`
的对错取决于 prompt 里有没有某条提示;`StaticChatClient` 和假 `urlopen` 钉死 HTTP 行为。
所以循环、闸门、台账的每一步都能确定性地断言。
