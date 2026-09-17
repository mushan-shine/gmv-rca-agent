# 电商指标问答与归因 Agent(Databricks)

一个跑在 Databricks 上的 text-to-SQL 系统。重点不在「能把自然语言变成 SQL」——
那件事现在谁都做得到 —— 而在**它如何知道自己答得对不对,以及如何持续变得更准**。

三个循环,层层嵌套:

```
L1 自修复（一次提问内，秒级）
      生成 SQL → 静态守卫 → 执行 → 把真实错误喂回去重生成 → 最多 3 轮

L2 评估（每次改动后）                                    ★ 项目核心
      26 条 case → 执行式比对 → pass@1 / 幻觉率 / 拒答准确率
      标准答案由知识库**程序化生成**，不手工标注

L3 改进（跨版本）                                        ★ 项目核心
      失败归因 → 补 few-shot / 提示（带来历）→ 版本 +1
      → 重跑评估 + 留出集 → 指标不退步才允许合并 → 下一轮生成读到新知识
```

判据很硬:**一个循环之所以成立,在于「下一轮会读取上一轮写下的状态」。**
只写不读的定时任务是复读机,不是循环。在这里,这句话的落点是两张 append-only 的
Delta 表(`nl2sql_attempts` / `nl2sql_eval_runs`)和一个有版本的提示知识库。

---

## 这个项目与常见 text-to-SQL demo 的三个区别

**1. 标准答案是程序化生成的,不是手写的。**

常见做法是手写二三十条「问题 + 我认为对的 SQL」。规模上不去、人写的答案本身可能错、
数据一变答案全废。这里的 case 只声明「问的是哪个指标 / 维度 / 窗口」,
参考 SQL 由 `knowledge/` 里的指标与维度定义生成(复用同一套 join 路径实现),
执行后得到标准答案。样例数据重建后答案自动跟着变,口径也不会与系统不一致。

支撑这件事的,是仓库里那套**确定性分解内核**(`src/rca/decompose.py`,
对数可加的 GMV = UV × CVR × AOV 分解,残差 ~1e-13%)——
它本来是为归因写的,现在同时充当评估集的答案发生器。

**2. 有不可答对照组。**

26 条 case 里有 4 条**正确答案是「拒答」**:问广告计划的 GMV(没有 campaign_id)、
问客户邮箱(没有这个字段)、问毛利率(没有成本数据)。
一个 pass@1 很高、但对答不了的问题照样编 SQL 的模型,在生产里是危险的。
`correct_refusal_rate` 与 `false_refusal_rate` 一起看,堵死「一律拒答」这条捷径。

**3. 评估器自己被校验过。**

`ReferenceGenerator` 是一个永远给出正确答案的「作弊」生成器。
它必须拿满分 —— 拿不到就说明是比对逻辑、标准答案或守卫有问题,而不是模型不行。
没有这条基线,你分不清「模型差」和「尺子坏了」。

---

## 快速开始

**目标平台是 Databricks,开发也发生在 Databricks 上。**
逐步操作(注册后点哪里、跑哪个 notebook、看什么输出、出错怎么查)见
**[docs/DATABRICKS_SETUP.md](docs/DATABRICKS_SETUP.md)**。

想先了解这个项目在解决什么问题、为什么这么设计,读
**[docs/CASE_STUDY.md](docs/CASE_STUDY.md)** —— 它把整套代码反推成一道系统设计题:
题面、考点、实现思路、技术栈(含**不选**什么及原因)、架构、关键取舍。

1. 代码推到 GitHub,在 Databricks 里 **Workspace → Create → Git folder** 克隆进来;
2. SQL Editor 里跑 `SHOW CATALOGS` 确认你的 catalog 叫什么
   (Free Edition 常见是 `workspace`,不一定是 `main`);
3. 依次 Run all:

| notebook | 做什么 | 关键产出 |
|---|---|---|
| `notebooks/00_setup.py` | 建 5 张表 + 灌 8 周样例数据 + 自检 | `sessions_converted == completed_orders` |
| `notebooks/01_decompose.py` | 确定性分解 + 闭合性断言 | 残差 ~`1e-12 %` |
| `notebooks/02_run_tests.py` | **在 Databricks 上**跑整套验收测试 | `157 passed` + `85 passed`(各 1 条 duckdb 专用被跳过) |
| `notebooks/03_nl2sql_loop.py` | **循环主线**:自修复 → 评估 → 改进 → 回归闸门 | `nl2sql_eval_runs` 里的指标序列 |

`03` 的第 2 格会**自动探测你的 workspace 有没有可用的 LLM 端点**,有就直接用
(走 workspace 内的 serving 端点,请求不出 Databricks)。没有也能跑完整条流水线 ——
生成器换成 `ReferenceGenerator`,其余完全一样。

换供应商只是换 `base_url` + `model`:客户端只认 OpenAI chat-completions 线格式,
Databricks / 智谱 GLM / OpenAI / 自建代理都讲这个。配置见 [.env.example](.env.example);notebook 里用顶部的 `llm_provider` 下拉框切换,接智谱的步骤见 [DATABRICKS_SETUP.md 第 7.5 节](docs/DATABRICKS_SETUP.md)。

### 三种运行形态,SQL 完全相同

| target | 何时用 | 怎么连 |
|---|---|---|
| `spark` | **Databricks Notebook / Job 里**(主路径) | 当前 SparkSession,不需要 token |
| `databricks` | 从本机 / CI 连 SQL Warehouse | `databricks-sql-connector` + 环境变量 / `.env` |
| `duckdb` | 改代码时的秒级回归 | 内存库,**不是部署目标** |

同一套验收测试可以跑在任意一个上:

```bash
pytest -q                          # duckdb,约 7 秒
pytest -q --rca-target=spark       # 在 Databricks notebook 里(见 02_run_tests.py)
pytest -q --rca-target=databricks  # 从本机连 warehouse
```

---

## 循环的四个部件

| 部件 | 职责 | **绝对禁止** |
|---|---|---|
| **生成器** `nl2sql/generate.py` | 自然语言 → SQL;判断问题是否可答 | 判断自己的结果对不对 |
| **LLM 客户端** `nl2sql/llm.py` | 退避重试、prompt 缓存、预算熔断、用量计量 | 让传输层重试被记成一次「自修复尝试」 |
| **守卫** `nl2sql/guard.py` | 确定性静态校验:只读、表/列白名单、幻觉检测 | 使用 LLM |
| **比对器** `nl2sql/compare.py` | 执行式比对(比结果集,不比 SQL 文本) | 使用 LLM |
| **改进循环** `nl2sql/knowledge_store.py` | 修改**知识**(提示、few-shot) | 修改代码逻辑 |

**守卫为什么不能省。** LLM 生成的文本会被送进仓库执行,`DROP` 必须在执行前拦掉。
但它更重要的作用是**给出可照着改的反馈**:
「列 `revenue` 不存在,fact_orders 上有 order_amount」
比引擎返回的 `[UNRESOLVED_COLUMN]` 有用得多 —— 自修复循环的收敛靠的就是这个。
把同一个 prompt 再发一遍那是重试,不是循环。

**传输层重试 ≠ 自修复。** `429` / 超时重发的是**同一个 prompt**,没有任何新信息,
它封在 `llm.py` 里、循环层看不见 —— 否则一次网络抖动会被记成「模型改了两轮才对」,
`avg_attempts` 立刻失去意义。同一层还负责 prompt 缓存
(评估会把逐字相同的 prompt 打很多遍)和预算熔断(在花钱**之前**检查上限)。

**L1 修不了什么。** 自修复循环拿到的反馈只有守卫报错与引擎报错,
所以它修的是「跑不跑得通」,**不是「答得对不对」**。
一条语法正确、执行成功、但忘了 `order_status = 'COMPLETED'` 的 SQL,
L1 永远发现不了 —— 只有 L2 的执行式比对抓得住
(`tests/test_nl2sql_loop.py::test_valid_but_wrong_sql_is_caught_only_by_the_evaluator`)。
这条分工是整个设计的关键。

---

## 三层验证,各管一段

目标平台跑不到本地,所以验证被拆成三层,**任何一层都不能替代另一层**:

| 层 | 手段 | 证明什么 | 不能证明什么 |
|---|---|---|---|
| 1 · 算术 | DuckDB 内存库,`pytest -q` | 分解的**算术**闭合(残差 ~1e-13%)、知识库校验、数据不变量 | SQL 在 Spark 上跑不跑得通 |
| 2 · 方言 | `sqlglot` 按 Databricks 方言解析全部生成的 SQL | **语法**合法;DuckDB 写法没有漏进 Databricks 产物 | 语义约束(generator 位置、Delta DDL) |
| 3 · 真机 | `pytest --rca-target=spark`(即 `notebooks/02_run_tests.py`) | 在**目标引擎上确实跑得通且算得对** | —— |

**为什么本地那一层仍然有意义:** 所有 SQL 都由 `src/rca/` 里的**同一份生成器**构造,
`src/rca/dialects.py` 只抹平两边写法真正不同的那几个原语
(`explode` vs `unnest`、`pmod(hash())` vs `hash() %`、`datediff` 的参数顺序等)。
算术闭合性与具体数值无关,所以在 DuckDB 上毫秒级验证是充分的 ——
它让改代码的回归循环是 6 秒,而不是每次都去排队等 serverless 冷启动。
两个方言的 `hash()` 实现不同,因此**两边的数据并不相同** ——
确定性保证的是「同一目标下重跑逐行一致」,这已满足简报约束 3。

---

## 分解的数学

### 乘法(因子)分解

```
GMV = UV · CVR · AOV
ln GMV = ln UV + ln CVR + ln AOV
Δln GMV = Δln UV + Δln CVR + Δln AOV          ← 精确恒等,无近似
```

记 `L = Δln GMV = ln(GMV_b / GMV_a)`,`r = GMV_b / GMV_a − 1`,则 `L = ln(1+r)`。

| 输出字段 | 定义 | 求和性质 |
|---|---|---|
| `log_delta` | `ln(f_b) − ln(f_a)` | Σ **恒等于** `L` |
| `share_of_change` | `log_delta / L` | Σ **恒等于** 1 |
| `contribution_pct_points` | `share × r` | Σ **恒等于** `r` |
| `isolated_pct_change` | `exp(log_delta) − 1` | 不可加 |

**`contribution_pct_points` 与 `isolated_pct_change` 是两个不同的量,报告里不得混用。**
前者是把总变化 `r` 按对数份额**分配**给各因子;后者是「只有该因子变化时指标会变多少」。

举例(见 `tests/test_math_analytic.py`):UV 1000→1100、orders 50→44、GMV 5000→4400 时,
CVR 自身跌 20.00%,而它分配到的是 −20.95 个百分点,相差 0.95 个点(相对 4.7%)。
`|Δln f| ≤ 0.10` 时相对差约 5%,`≤ 0.20` 时约 10%。

总变化为 0 时份额**没有定义**,`share_of_change` 与 `contribution_pct_points` 返回 `None`,
而不是 0 —— 这是 V1「无异常对照组」的底座:系统不得在没有异常时编造贡献。

### 加法(维度)分解

```
ΔGMV = Σ_d [GMV_b(d) − GMV_a(d)]
```

引擎跑**两条**查询:一条不带 join 独立算总量,一条按维度分组。两者之差即残差。
若只跑分组查询再自行求和,残差恒为 0,信号就消失了。

| 观察到的现象 | 诊断 |
|---|---|
| 残差 ≈ 0 | 划分完整 |
| 残差 < 0 | join fan-out(维表主键不唯一),行被放大 |
| `__UNKNOWN__` 桶不为空 | join 丢行:事实表的键在维表里不存在 |
| `__NULL__` 桶不为空 | 原生列本身有 NULL |
| `level_residual_a/b_pct` ≠ 0 | 某一期的水位错位(即使 Δ 上恰好抵消) |

`residual_severity()` 实现简报 §3.2 的三档判定(`clean` / `structural` / `data_quality`),
**完全确定性,不经 LLM**。

比率指标(CVR、AOV)按维度做加法分解会被**直接拒绝** —— 各分组的比率之和不等于
总体比率,强行相加会得到一个算术上错误、却很像结论的数字。mix/rate 拆分留到 V1+。

---

## 口径决策(最容易出错的地方)

| 决策 | 取值 | 理由 |
|---|---|---|
| **UV** | **会话数**,不是去重访客数 | 只有 CVR 的分母与 UV 同为会话数,`UV × CVR × AOV = GMV` 才精确成立。改成去重访客数,恒等式立刻破裂,残差就失去「唯一确定性信号」的地位。 |
| **CVR** | 已完成订单数 / 会话数 | 同上 |
| **AOV** | GMV / 已完成订单数 | 同上 |
| **订单口径** | `order_status = 'COMPLETED'` | `gmv` 与 `orders` 必须共用同一套 `filters_default`,否则 AOV 是个混合物。`knowledge.py` 会强制检查这一点。 |
| **`fact_sessions.converted`** | 该会话产生了一笔 COMPLETED 订单 | 生成器保证「一个转化会话恰好一个订单」,于是 `COUNT(converted)` 必然等于 `orders` 指标,CVR 可以两侧交叉验证(`sql/setup` 末尾的自检查询)。 |
| **时间窗口** | 两窗口必须**不重叠**且**等长** | 长度不同会把「天数差」混进变化里,变成一个伪因子。确需 MTD 对比整月时须显式传 `allow_unequal_periods=True`。 |
| **维度优先级** | `dimensions.yaml` 的 `priority` 是唯一真源 | 简报里「dimensions.yaml 存优先级」与「指标 YAML 里顺序即优先级」两处说法会产生两个真源。这里合一:指标的 `dimensions` 是**允许集合**,顺序必须与全局 `priority` 一致(不一致则加载失败),L3 将来只改 `dimensions.yaml`。 |
| **scope** | 全部指标标记 `[ops]` | 财务口径(退款回冲、跨期确认)另有定义,不可与运营口径混用。见「开放问题」。 |

---

## 相对简报的偏离

都是必要的补充,没有一处缩减了 V0 的范围。

| # | 偏离 | 原因 |
|---|---|---|
| 1 | 新增 `knowledge/tables.yaml` | 简报要求校验「引用了不存在的维度/字段」,但原结构里没有 schema 的真源。现在 DDL 由它渲染、校验由它兜底,建表脚本与知识库不可能漂移。 |
| 2 | 新增 `knowledge/metrics/orders.yaml` | 乘法恒等式必须落到**可加基础指标**上:`uv × (orders/uv) × (gmv/orders) = gmv`。没有 `orders`,CVR 和 AOV 无法在 SQL 层闭合。 |
| 3 | 比率指标用 `ratio: {numerator, denominator}` 而非 `sql` + `source_table` | CVR 的分子分母跨两张表,写不成单表聚合表达式。指标因此只有两种形态:可加基础指标 / 比率指标,由 pydantic 强制二选一。 |
| 4 | `fact_orders` 增加 `session_id` 列 | 让 CVR 能从会话侧与订单侧双向核对。简报第 8 节列的是「关键列」,非穷举。 |
| 5 | `gmv.dimensions` **不含** `campaign` | `fact_orders` 上没有 `campaign_id`,广告计划只存在于 `fact_ad_spend`。这是一个真实的数据缺口,属于报告格式里的「待查项」,而不是可以硬写进 YAML 的东西。加载器会主动拒绝把它写进去。 |
| 6 | 新增 `src/rca/{dialects,warehouse,sampledata}.py` | 简报的仓库结构没列执行层与数据生成层,但两者是 `decompose.py` 能同时跑在两个引擎上的前提。 |
| 7 | 新增 `notebooks/`、`sql/setup/render.py` | 前者让 V0 在目标平台上可被肉眼验收,后者把 `.sql` 变成产物而非手写件。 |
| 8 | 两个必需函数同时提供**方法**与**模块级函数** | `DecompositionEngine.decompose_factors` 持有知识库与执行器;`decompose.decompose_factors(engine, ...)` 是它的薄封装,以字面满足验收标准 3。 |
| 9 | 新增 `src/rca/config.py` + `cli.py` + `notebooks/` + `docs/DATABRICKS_SETUP.md` | 目标平台是 Databricks,所以「怎么在上面跑起来」必须是一等公民而不是一句说明。凭据只从环境变量 / `.env` 读,不进代码、不进 YAML、不进日志(`DatabricksSettings.__repr__` 会打码 token)。 |
| 11 | 验收测试可切换运行目标(`pytest --rca-target=`) | 验收标准要求「有测试证明闭合性」;若测试只能跑在本地 DuckDB 上,它证明的就只是本地的算术,而不是这套 SQL 在**目标引擎**上算得对。 |
| 10 | `DatabricksExecutor` 复用连接 | Free Edition 只有一个 2X-Small warehouse;一次分解只有 2 条查询,每条重连一次的话握手开销会盖过查询本身。 |

---

## 仓库结构

```
gmv-rca-agent/
├── knowledge/                  # Git 是唯一真源
│   ├── tables.yaml             #   schema 登记簿(DDL 与校验都由它驱动)
│   ├── dimensions.yaml         #   维度注册表 + 全局钻取优先级(L3 将来只改这里)
│   └── metrics/                #   gmv / orders / uv / cvr / aov
├── eval/cases.yaml             # ★ 26 条评估 case(含 4 条不可答对照组、5 条留出集)
├── src/rca/
│   ├── nl2sql/                 # ★ 循环主线
│   │   ├── generate.py         #     生成器接口 + Reference/Scripted/LLM 实现 + prompt 构造
│   │   ├── llm.py              #     LLM 客户端:多供应商 / 重试 / 缓存 / 预算熔断
│   │   ├── guard.py            #     静态守卫:只读、表/列白名单、幻觉检测
│   │   ├── cases.py            #     评估 case + **程序化生成的标准答案**
│   │   ├── compare.py          #     执行式比对(比结果集,不比 SQL 文本)
│   │   ├── loop.py             #     L1 自修复内循环
│   │   ├── evaluate.py         #     L2 指标 + 回归闸门
│   │   ├── knowledge_store.py  #     L3 提示知识(有版本、有来历)
│   │   └── state.py            #     Delta 状态表:下一轮读上一轮写的
│   ├── knowledge.py            # 加载 + 校验(含乘法恒等式的符号验证)
│   ├── decompose.py            # 确定性分解 —— 同时是评估集的答案发生器
│   ├── dialects.py             # Databricks / DuckDB 方言
│   ├── warehouse.py            # 三个执行器 + SQL 指纹 + 可复现凭证
│   ├── config.py               # 运行目标装配;凭据只从环境变量 / .env 读
│   ├── cli.py                  # setup / verify / demo / sql
│   ├── sampledata.py           # 确定性样例数据生成器
│   └── errors.py
├── knowledge/nl2sql/prompting.yaml   # ★ L3 唯一被允许修改的东西
├── notebooks/                  # Databricks Git folder 里直接运行(主路径)
│   ├── 00_setup.py             #   建表 + 灌数 + 自检
│   ├── 01_decompose.py         #   确定性分解 + 闭合性断言
│   ├── 02_run_tests.py         #   在 Databricks 上跑整套验收测试
│   └── 03_nl2sql_loop.py       # ★ 循环主线
├── docs/
│   ├── CASE_STUDY.md           # ★ 场景题 + 设计思路 + 技术栈取舍 + 架构
│   └── DATABRICKS_SETUP.md     # ★ 平台操作手册(点哪里、跑什么、怎么排查)
├── sql/setup/
│   ├── render.py               # 生成器 -> .sql(产物,勿手改)
│   ├── databricks_setup.sql    #   贴进 SQL Editor 即可
│   └── duckdb_setup.sql
└── tests/
    ├── test_closure.py            # ★ V0 验收标准 4(DuckDB)
    ├── test_math_analytic.py      #   手算对照,不经数据库
    ├── test_knowledge.py          #   V0 验收标准 2(正向 + 13 类反向)
    ├── test_sample_data.py        #   V0 验收标准 1
    ├── test_decompose_guards.py   #   越过职责边界时必须报错
    ├── test_sql_rendering.py      #   产物防漂移
    ├── test_config.py              #   凭据不得泄漏进任何可打印的地方
    ├── test_databricks_dialect.py  #   生成的 SQL 必须是合法的 Databricks 方言
    ├── test_llm.py                 # ★ 重试 / 缓存 / 预算 / 凭据不泄漏
    ├── test_nl2sql_units.py        # ★ 守卫 / 比对器 / 评估集 / 改进 / 回归闸门
    ├── test_nl2sql_loop.py         # ★ 自修复、评估器自检、状态表、改进闭环
    └── conftest.py                 #   --rca-target 决定整套测试跑在哪个引擎上
```

---

## V0 验收标准对照

| # | 验收标准 | 落点 | 状态 |
|---|---|---|---|
| 1 | `sql/setup/` 能建出 5 张业务表并填入 8 周样例数据 | `notebooks/00_setup.py` / `sql/setup/databricks_setup.sql`;`tests/test_sample_data.py` | ✅ 56 天 × 5 渠道 × 2 设备 × 3 市场 × 7 品类,默认 ~33.6 万行会话 |
| 2 | `knowledge.py` 能加载并校验(缺字段、引用不存在的维度时报错) | `tests/test_knowledge.py` | ✅ 13 类错误全部在加载阶段拦下,含乘法恒等式的符号验证 |
| 3 | `decompose.py` 提供 `decompose_factors` 与 `decompose_by_dimension` | `src/rca/decompose.py` | ✅ 方法 + 模块级函数 |
| 4 | **测试证明各分项贡献之和与总变化的差值 < 0.1%** | `tests/test_closure.py`,可用 `--rca-target=spark` 在 Databricks 上复核 | ✅ 离线实测 ~1e-13%,断言同时卡 `1e-6%` 与简报的 `0.1%` |
| 5 | 全程不使用 LLM | 全仓无任何模型调用 | ✅ |

```
253 passed        # pytest -q(duckdb);--rca-target=spark 跑同一批,少 1 条 duckdb_only
```

⚠ **尚未在真实 workspace 上执行过。** 第 3 层验证的代码已就位
(`notebooks/02_run_tests.py`),但需要一个 Databricks workspace 才能跑。
在跑通它之前,「能在 Databricks 上运行」只有第 1、2 层的证据支持。

配额方面(简报 §13.1):一次因子分解 = **2 条查询**(两个窗口合并进一条 SQL,
每张事实表各一次);一次维度分解 = **2 条查询**(独立总量 + 分组求和)。

---

## V0 **不**包含什么

刻意不做,避免给 V1~V4 留下错误的脚手架:

* **没有 LLM**,因此没有 `decide.py`、没有 pydantic 约束的动作对象;
* **没有 L1 循环**,因此没有 `loop.py`。`scripts/local_demo.py` 里逐个维度的遍历
  是写死的优先级顺序,**不是**「根据上一轮结果决定下一轮」——
  两者的区别正是简报 §3.1 的判据;
* **没有状态表**,因此没有 `hypothesis_ledger` / `anomaly_ledger`,也就没有 L2、L3;
* **没有观察层 L0**(数据质量优先检查)、没有停止条件、没有三段式报告、没有评估集。

`residual_severity()` 是唯一提前落地的观察层能力(L2),因为闭合性测试需要它。

---

## 开放问题(需要业务侧确认,会影响 V1+)

1. **财务口径 vs 运营口径。** 当前全部指标标 `scope: [ops]`,GMV 按下单日、
   只算 COMPLETED。财务口径通常按确认收入日、含退款回冲。两套口径必须分开建模,
   否则报告会在两套数之间来回跳。V1 之前需要拿到财务口径的准确定义。
2. **campaign 级数据缺失。** 简报 §11 的示例报告里,「待查项」正是
   「需要 campaign 级别的关键词与落地页数据,当前 UC 中不存在」。
   目前 `fact_orders` / `fact_sessions` 上都没有 `campaign_id`,
   因此「投放扩量导致流量质量下降」这类假设在 V0~V3 内**无法被验证**。
   需要确认:是打点侧可以补,还是接受它永远停留在「推测」段?
3. **比率指标的维度分解。** CVR 按渠道下跌时,业务方真正想知道的是
   「是各渠道自己的 CVR 掉了(rate effect),还是流量结构变了(mix effect)」。
   V0 直接拒绝了这类请求。V1 需要确认优先做哪一种拆分。
4. **时间窗口的默认对齐方式。** 当前默认要求两窗口等长且不重叠(周一起的自然周)。
   环比、同比、节假日对齐(如双十一对双十一)各需要不同的基期选择规则,
   这会直接影响 L2 每日扫描的误报率。

---

## 里程碑

| 版本 | 内容 | 状态 |
|---|---|---|
| **V0** | 建表与样例数据、指标树加载校验、参数化分解模板、残差计算、算术闭合性测试 | **✅ 本次交付** |
| V1 | 合成异常注入器;15 个评估 case(含 3 个无异常对照组);评估脚本 | 未开始 |
| V2 | LLM 维度选择(pydantic 约束)、状态表、L1 循环、MLflow Tracing | 未开始 |
| V3 | 停止条件、预算熔断、三段式报告、完整评估集指标 | 未开始 |
| V4 | **L2 监控循环**:Jobs 定时 + `anomaly_ledger` 跨天台账 + 已排除假设延续 | 未开始 |
