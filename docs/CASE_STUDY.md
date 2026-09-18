# 场景题:自助指标问答系统

> 这份文档把本仓库的代码反推成一道**系统设计题**:题面、读题、实现思路、技术栈、
> 架构、关键取舍、以及面试常见追问。
>
> 代码是答案的一个实现。读这份文档可以不看代码;看代码时可以拿它当地图。

---

## 一、题面

你加入一家跨境电商的数据平台团队(美/英/德三个市场,GMV 量级中等)。

**现状。** 业务方(投放、运营、品类经理)每天在 Slack 里问数据团队大量指标问题:

- "上周美国站 Paid Search 的转化率多少?"
- "上周 GMV 按品类拆一下"
- "这周 GMV 比上周跌了多少?"
- "上周哪个广告计划带来的 GMV 最高?"

三个数据工程师每天约 40% 的时间花在这类一次性取数上。公司希望做一个**自助问答系统**:
业务方直接提问,系统生成 SQL、执行、返回结果。

**约束。**

1. **答错代价高。** 这些数字会直接用于调整投放预算。给出一个错误但看起来合理的数字,
   比拒绝回答糟糕得多。
2. **口径复杂且团队内部常搞错。** GMV 算不算取消单和退款单?"UV"是会话数还是去重访客?
   "转化率"的分母是哪个?这些在现有 BI 报表里就存在不一致。
3. **有些问题数据根本答不了。** 订单表上没有 `campaign_id`,广告计划级归因做不了;
   没有成本数据,毛利率算不出来。系统必须能说"答不了,因为缺 X",而不是编一个。
4. **必须能证明它在变好。** 老板会问"上个月改了一堆 prompt,到底有没有用"。
   "感觉准了不少"不是答案。
5. **预算受限。** 平台是 Databricks,只有 serverless 与一个小 SQL warehouse,有每日额度;
   出站网络受限,优先用平台内的模型服务。

**交付。** 实现思路、技术栈选型(含**不选**什么及原因)、架构设计、以及你如何度量并持续改进它。

<details>
<summary>English version (面试时直接用)</summary>

You are joining the data platform team of a cross-border e-commerce company operating in
the US, UK and DE. Business stakeholders ask the data team dozens of ad-hoc metric
questions per day in Slack; three data engineers spend roughly 40% of their time on them.
Build a self-service system that turns those questions into SQL.

Constraints: (1) wrong answers are costly — they drive ad budget decisions, so a confident
wrong number is worse than a refusal; (2) metric definitions are subtle and already
inconsistent across existing dashboards; (3) some questions are unanswerable from the
current schema and the system must say so rather than invent an answer; (4) you must be
able to prove the system is improving, not just claim it; (5) the platform is Databricks
with a single small serverless SQL warehouse, daily quota, and restricted egress.

Deliverables: approach, tech stack (including what you deliberately do **not** use and
why), architecture, and how you measure and continuously improve it.
</details>

---

## 二、读题:这道题真正在考什么

把它当成"写个 NL2SQL"会答得很差。约束 1、3、4 决定了考点不在生成,而在**验证与改进**。

| 题面里的约束 | 真正的考点 |
|---|---|
| 答错代价高 | 你怎么**知道**它答对了?归因/取数没有天然裁判 |
| 口径常搞错 | 语义层放在哪、谁是唯一真源、怎么防止漂移 |
| 有些问题答不了 | 幻觉。而且要防"一律拒答"这种作弊解 |
| 要证明在变好 | 评估集、留出集、回归闸门、版本化 |
| 预算受限 | 停止条件、熔断、查询次数 |
| 生成的文本要被执行 | 安全边界:只读、白名单 |

**一句话:这道题考的是「你如何给一个没有标准答案的任务造出标准答案,并围绕它建立循环」。**

还有一个隐含考点:**区分"循环"与"重试"**。
把同一个 prompt 重发三次是重试 —— 模型没有获得任何新信息,收敛靠撞运气。
循环的判据是:**下一轮读取上一轮写下的状态**。这个判据在下面会反复出现。

---

## 三、实现思路

按"先把裁判造出来,再造选手"的顺序。这个顺序本身就是答案的一部分。

### 第 0 步:先划边界 —— LLM 只做它无可替代的那一步

| 组件 | 允许做 | **绝对禁止** |
|---|---|---|
| 生成器(LLM) | 自然语言 → SQL;判断问题是否可答 | 判断自己的结果对不对 |
| 守卫 | 确定性静态校验 | 使用 LLM |
| 比对器 | 执行式比对 | 使用 LLM |
| 改进循环 | 修改**知识**(口径提示、few-shot) | 修改代码逻辑 |

"LLM 不判断自己的结果对不对"是最关键的一条。让模型自评会得到一个和它自身错误高度相关的
裁判 —— 它答错的题,往往也判不出错。裁判必须是确定性的、外部的。

### 第 1 步:把口径从代码里抽出来,变成知识库

这一步同时解决约束 2 和后面所有事情。Git 里的 YAML 是唯一真源:

```
knowledge/
├── tables.yaml           schema 登记簿(列、类型、主键、分区列)
├── dimensions.yaml       维度注册表 + join 路径 + 钻取优先级
├── metrics/*.yaml        指标口径:gmv / orders / uv / cvr / aov
└── nl2sql/prompting.yaml 提示与 few-shot 示例(唯一允许被改进循环修改)
```

一份定义,四处消费:

1. 渲染建表 DDL(建表脚本与 schema 不可能漂移);
2. 渲染 prompt 里的 schema 与口径说明;
3. 生成守卫的表/列白名单;
4. **生成评估集的标准答案**(见下一步)。

加载时做强校验:缺字段、拼错的 key、引用不存在的维度、指标顺序与全局优先级不一致 ——
一律在加载阶段失败。理由很简单:**错误停在加载阶段,总好过变成报告里一个错误的数字。**

其中最值钱的一条校验是**符号化验证乘法恒等式**。`gmv.yaml` 声明
`GMV = UV × CVR × AOV`,加载器把每个因子展开成"可加基础指标的分子/分母多重集",
连乘约分后必须等于被分解指标本身:

```
uv  → ({uv}, {})          cvr → ({orders}, {uv})      aov → ({gmv}, {orders})
连乘 → ({uv,orders,gmv}, {uv,orders}) --约分--> ({gmv}, {})  ✓ 与 gmv 自身相同
```

把 `aov` 误写成 `gmv/uv`、或漏掉一个因子,加载立刻失败。

### 第 2 步:造标准答案 —— 这是整道题的分水岭

常见做法:手写 20~30 条"问题 + 我认为对的 SQL"。三个问题:规模上不去、
人写的答案本身可能错、数据一变全废。

**本方案:case 只声明「问的是哪个指标 / 哪个维度 / 哪个窗口」,SQL 由知识库生成。**

```yaml
- id: gmv_by_category_w8
  question: "What was last week's GMV by product category?"
  gold: {kind: group_by, metric: gmv, dimension: category, period: W8}
```

系统按这份声明生成参考 SQL(复用与分析路径**同一套** join 实现),执行得到结果集,
那就是标准答案。于是:

- 评估集规模不受人力限制;
- 样例数据重建后答案自动跟着变,不需要重新标注;
- 口径由知识库统一保证,不会出现"标注者以为的口径"和"系统以为的口径"不一致。

四类 gold:`aggregate`(标量,支持比率指标跨表拼分子分母)、`group_by`、
`change`(两窗口之差)、以及最重要的 **`unanswerable`**。

**不可答对照组。** 26 条 case 里有 4 条正确答案是"拒答":广告计划 GMV(没有 `campaign_id`)、
客户邮箱(没有这个字段)、毛利率(没有成本数据)。这直接对应约束 3。

**留出集。** 5 条标记 `holdout: true`,改进循环**不得**用它们调知识。
它们专门回答一个问题:这次改进是真的泛化了,还是只是把已知失败背下来了。

### 第 3 步:比结果集,不比 SQL 文本

同一个问题有无数种正确写法。按字符串比 SQL,评估指标就变成了"像不像我写的那条"。

执行式比对(execution match)的规则:

- **忽略列名** —— `AS total` 与 `AS value` 是同一个答案;
- **忽略行序** —— 除非问题问的是排名(case 显式声明 `order`);
- **数值带容差** —— `DECIMAL` / `DOUBLE` / `int` 统一成 float 后比相对误差;
- **列数必须一致** —— 顺手多选一列 `dt` 算错。答案的形状也是答案的一部分。
- **`NULL ≠ 0`** —— 混同会掩盖"这个切片根本没数据"。

失败时给**分类**而不只是 True/False(`value_mismatch` / `row_count` / `column_count` / …),
改进循环靠这个分类做失败归因。

### 第 4 步:守卫 —— 执行之前的确定性检查

LLM 生成的文本会被送进仓库执行。用 `sqlglot` 解析后检查:

1. **只读** —— `DROP` / `INSERT` / `MERGE` 一律拦下;
2. **单语句** —— 防尾随语句注入;
3. **表/列白名单** —— 与 `tables.yaml` 对照。

第 3 条同时给出两样东西:

- **幻觉率**这个指标(只有静态解析才能把"引用了不存在的列"与"语法错""答案错"区分开);
- **可照着改的反馈**。这一点决定了循环能不能收敛:

```
❌  "SQL 有误,请重试"                        ← 这是重试
✓   "引用了不存在的列:['revenue']。
     fact_orders 上有:channel, device, order_amount, order_id, order_status, …"
```

### 第 5 步:L1 自修复内循环

```
问题 → 生成 SQL → 守卫 ──失败──► 具体反馈 ──┐
                    │                        │
                    ok                       │
                    ▼                        │
                  执行 ──失败──► 引擎原始报错 ─┤
                    │                        │
                    ok                       ▼
                    ▼                   进下一轮 prompt
                 返回结果          (超过 3 轮 → 放弃,记录在案)
```

**3 轮上限不是拍脑袋**:守卫能给出的反馈类型有限(语法、未知表、未知列、执行错),
一轮改一类。再多就是烧 token 赌运气,并且会把"模型其实不会"伪装成"多试几次就好"。
这条上限同时是**预算熔断**(约束 5)。

**这一层修不了什么 —— 想清楚这点比实现它更重要。** 循环拿到的反馈只有守卫报错和引擎报错,
所以它修的是"跑不跑得通",**不是"答得对不对"**。一条语法正确、执行成功、
但忘了 `order_status = 'COMPLETED'` 的 SQL,L1 永远发现不了。正确性只能来自 L2。

### 第 6 步:L2 评估 —— 指标与它们之间的制衡

| 指标 | 含义 |
|---|---|
| `pass_at_1` | 第一轮就跑通**且答对**的比例 |
| `pass_final` | 最终答对的比例(执行式比对) |
| `avg_attempts` | 平均尝试轮数 —— 自修复的成本 |
| `hallucination_rate` | 出现过"引用不存在的表/列"的 case 比例 |
| `correct_refusal_rate` | 不可答问题中被正确拒答的比例 |
| `false_refusal_rate` | 可答问题中被误拒的比例 |
| `holdout_pass_final` | **留出集**正确率 —— 回归闸门看的是它 |

最后两个是一对**制衡**:只看 `correct_refusal_rate`,模型学会"一律拒答"就能满分;
把 `false_refusal_rate` 一起看,这条捷径就堵死了。

还有一条基线容易被忽略:**`ReferenceGenerator`** —— 一个永远给出正确答案的"作弊"生成器。
它必须拿满分。拿不到就说明是比对逻辑、标准答案或守卫有问题,**而不是模型不行**。
没有这条基线,你分不清"模型差"和"尺子坏了"。

### 第 7 步:L3 改进循环 —— 让"下一轮读上一轮"成立

> **实现演进。** 最初的 L3 是人在闭环:失败原因人看、修改人写、合并人定。
> 按 loop engineering 的标准这不算闭环。现在由 `nl2sql/improve.py` 自主运行:
> **LLM 提议一条通用提示 → 带着它重跑评估 → 确定性闸门裁决 → 台账记住结果**。
> 三条纪律:提议者看不到留出集;看不到参考 SQL(否则会抄答案);
> 被拒绝的提示跨循环保存并出现在下一次提议的 prompt 里,不会被反复尝试。
> 闸门要求训练集**严格多答对一道**(持平不采纳,prompt 不会无证据地变长)、
> 留出集不退步、幻觉率与误拒率不升高。下面是它的数据流。

```
跑评估 ──► nl2sql_attempts(每轮尝试:SQL、失败类型、守卫报错、引擎报错)
                │
                ▼
        按 (评估判定 × 失败类型) 归因
                │
                ▼
        给失败 case 补一条 few-shot 示例 / 一条口径提示
        每条带 source_run(哪次评估) + fixes_case(为了修哪条)
                │
                ▼
        version += 1 → 写回 prompting.yaml → Git PR
                │
                ▼
        重跑评估 + 留出集 ──► 回归闸门
                │
                ▼
        下一轮生成时 build_prompt() 读到新示例   ← 判据在此成立
```

**回归闸门的规则**(简单到能进 CI):

- 留出集正确率不得低于基线;
- 全量正确率不得低于基线;
- 幻觉率与误拒率不得升高。

留出集那一条是关键。只看全量指标的话,把失败 case 的答案直接塞进 few-shot 就能刷上去 ——
**那是背答案,不是改进。**

`source_run` / `fixes_case` 不是装饰:没有它们,知识库过几周就会变成一堆没人说得清来历的提示,
而"这条提示当初解决了什么问题、现在还需不需要"正是**评估驱动改进**与**凭感觉调 prompt**
的分界线。

### 第 8 步:接真实 LLM —— stub 遇不到的四件事

前七步都可以用一个假的生成器跑通并测试。真接上模型之后会多出四个问题,
它们都**不属于**循环逻辑,所以全部封在客户端层,循环层看不见。

**1. 传输层重试 ≠ 自修复。** 这是最容易做错、也最值得讲的一点。

```
429 / 503 / 超时  →  退避重发【同一个 prompt】   ← 没有新信息,不是一次「尝试」
守卫报错 / 引擎报错 →  发出【带错误反馈的新 prompt】 ← 这才是一次尝试
```

把两者混同,一次网络抖动就会被记成「模型改了两轮才对」,`avg_attempts` 立刻失去意义。
所以退避重试封在客户端里,循环层只看到一次成功或一次彻底失败。

**2. 评估会把同一个 prompt 打很多遍。** 26 条 case × 每次改动重跑 × 多个知识版本,
其中大量 prompt 逐字相同。按 (模型 + 参数 + prompt) 指纹缓存,
既省钱,也让**同一版本的评估结果完全可复现** ——
否则即便 `temperature=0`,供应商侧的变动也会让两次跑批对不上。

**3. 预算要能熔断。** 在**发起调用之前**检查上限,超了直接抛错。
熔断在报告里表现为 `generator_error`,与"模型突然变差了"能一眼区分开。

**4. 指标变了,是知识变了还是模型变了?**
`nl2sql_eval_runs` 必须同时记 `knowledge_version` 和 `model`。
两者一起变的那一次跑批,是解释不清的 —— 这条纪律比任何 prompt 技巧都重要。

另外两个小而实的点:**`temperature=0`**(采样噪声会让指标涨跌无法归因);
**回复被截断要单独处理** —— 截断的是半条 SQL,直接交给守卫会得到一个误导性的语法错,
模型下一轮会去"修"一条本来就没写完的 SQL。正确做法是明说"被截断了,只输出 SQL"。

---

---

## 四、技术栈

### 选了什么

| 层 | 选型 | 为什么 |
|---|---|---|
| 数据 | Unity Catalog + Delta Lake | 平台既定;Delta 的 append-only 表正好用来做跨运行的状态账本 |
| 计算 | Databricks serverless SQL warehouse | 约束 5;一次问答只有 1~2 条查询,小 warehouse 够用 |
| 模型 | 任何讲 **OpenAI chat-completions** 线格式的端点 | 默认走 Databricks Foundation Model APIs —— **请求不出 workspace**,符合约束 5。同一个客户端换个 `base_url` 就能接 OpenAI 或自建代理,不锁供应商 |
| HTTP | 标准库 `urllib` | 不引 SDK:notebook / CI / 本机行为一致,也不用为了一个 POST 背上一条依赖链 |
| 结构化约束 | pydantic | 知识库与评估集的 schema 校验,`extra="forbid"` 让拼错的 key 直接报错 |
| SQL 静态分析 | **sqlglot** | 守卫的核心。多方言解析,能提取表/列引用 —— 幻觉检测与只读校验都靠它 |
| 知识库 | Git 里的 YAML | 唯一真源;L3 的改动走 PR,可 review 可回滚 |
| 编排 | Databricks Jobs | 定时跑评估,攒出跨天的指标时间序列 |
| 测试 | pytest | 验收测试可切换运行目标(见下) |
| 本地回归 | DuckDB | **不是部署目标**,是把改代码的回归循环从"等 serverless 冷启动"压到 7 秒 |

**同一份 SQL 生成器,两个方言。** 方言层只抹平真正不同的那几个原语
(`explode` vs `unnest`、`pmod(hash())` vs `hash() %`、`datediff` 参数顺序)。
于是同一套验收测试可以 `pytest --rca-target=duckdb`(7 秒)或
`--rca-target=spark`(在 Databricks notebook 里跑)。

这不是过度设计,是对一个真实痛点的回应:**如果验收测试只能连上 warehouse 才能跑,
它就不会被经常跑。** 而如果它只在本地跑,它证明的又只是本地的算术。所以两个都要。

### 刻意不选什么

| 不用 | 原因 |
|---|---|
| **LangChain / LangGraph** | 它会把检索与重试逻辑藏起来。而这道题的全部价值就在"重试时到底喂回去了什么" —— 这个必须显式、可测、可写进状态表。框架省的那点代码,换来的是调试时看不见循环内部 |
| **向量库 / schema RAG** | 只有 5 张表。整个 schema 塞进 prompt 绰绰有余。引入检索等于多一个失败模式(检错表)却没有收益。表数量上到几百张再说 |
| **让 LLM 做 self-critique / LLM-as-judge 判对错** | 裁判会与选手的错误高度相关。有确定性裁判(执行式比对 + 程序化标准答案)时,用 LLM 判对错是主动放弃精度 |
| **字符串比对 SQL / EM(exact match)** | 同一问题无数种正确写法,按文本比是在度量"像不像我" |
| **手工标注评估集** | 见第 2 步 |
| **Databricks Genie** | 现成的问答产品。这道题要的是自建循环与可度量的改进过程,用现成品就没有可讲的东西了 |
| **微调模型** | 评估集还没稳定就微调,等于把噪声固化进权重。先把尺子造好、把 prompt/few-shot 的上限摸到,再谈微调 |

---

## 五、架构设计

### 组件与数据流

```
                    ┌──────────────────────────────────────────────┐
                    │  knowledge/  (Git = 唯一真源)                 │
                    │    tables.yaml   dimensions.yaml             │
                    │    metrics/*.yaml   nl2sql/prompting.yaml    │
                    └───┬──────────┬──────────┬──────────┬─────────┘
             建表 DDL ──┘   prompt ─┘  守卫白名单─┘  参考 SQL ─┘
                            │          │            │
  用户问题                   ▼          ▼            │
     │      ┌───────────────────────────────────┐   │
     └─────►│  L1 AnswerLoop  (≤3 轮)            │   │
            │                                   │   │
            │   生成器 ──► 守卫 ──► 执行器        │   │
            │      ▲         │        │          │   │
            │      └── 具体反馈 ◄──────┘          │   │
            └───────┬───────────────────┬───────┘   │
                    │ 每轮尝试           │ 结果集     │
                    ▼                   ▼           ▼
          ╔═══════════════════╗    ┌──────────────────────┐
          ║ nl2sql_attempts   ║    │ L2 Evaluator         │
          ║  (Delta, append)  ║◄───┤  执行式比对 vs 标准答案│
          ╚═════════╤═════════╝    └──────────┬───────────┘
                    │                          ▼
                    │              ╔═══════════════════════╗
                    │              ║ nl2sql_eval_runs      ║
                    │              ║  (Delta, append)      ║
                    │              ╚═══════════╤═══════════╝
                    │                          │
                    └──────────┬───────────────┘
                               ▼
                    ┌─────────────────────────┐
                    │ L3 改进                  │
                    │  失败归因 → 候选知识版本  │
                    │  → 回归闸门 → Git PR     │
                    └────────────┬────────────┘
                                 │
                                 └──► prompting.yaml (version+1) ──┐
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    ▼
              回到 L1 的 prompt      ← **循环在此闭合**
```

### 两张状态表

刻意用 `CREATE TABLE IF NOT EXISTS`(而非 `OR REPLACE`)—— 它们的价值全在**跨运行累积**,
重建一次就等于把循环的记忆抹掉。

**`nl2sql_attempts`** — 每一次生成尝试

`run_id, case_id, verdict, attempt_no, generator, sql, fingerprint, failure_kind,
guard_errors, execution_error, row_count, latency_ms, created_at`

注意 `failure_kind`(循环视角:守卫错/执行错/幻觉/无)与 `verdict`(评估视角:
答对/答错/误拒/该拒未拒)是**两件事**。一次正确拒答的 `failure_kind` 是 `refused`,
但 `verdict` 是 `correct_refusal`。

失败画像按 `(verdict × failure_kind)` 聚合,**不过滤 `failure_kind = 'none'`** ——
一条跑得通但答错的 SQL 在循环看来毫无异常,而它恰恰是最危险、最需要补知识的失败模式。
把它过滤掉,改进循环就对"静悄悄答错"完全失明。

**`nl2sql_eval_runs`** — 每次评估的指标快照

`run_id, created_at, generator, knowledge_version, n_cases, pass_at_1, pass_final,
avg_attempts, hallucination_rate, correct_refusal_rate, false_refusal_rate,
holdout_pass_final, notes`

`knowledge_version` 是回归闸门的锚点 —— 有了它,"v3 比 v2 差"这句话才有意义。

### 部署形态

同一套代码三个运行目标,SQL 完全相同:

| target | 场景 | 连接方式 |
|---|---|---|
| `spark` | Databricks Notebook / Job 内(主路径) | 当前 SparkSession,不需要 token |
| `databricks` | 本机 / CI 连 warehouse | `databricks-sql-connector` + 环境变量 |
| `duckdb` | 改代码时的秒级回归 | 内存库 |

凭据只从环境变量 / `.env`(已 gitignore)读,`repr` 打码 token,并有测试专门断言
token 不出现在任何可打印的地方。

---

## 六、关键取舍(面试官会追问的地方)

**1. 为什么不让 LLM 直接写 SQL 之外还做别的?**
因为归因/取数没有天然裁判。LLM 做的每一件"无法被确定性验证"的事,都会变成一个
无法度量的风险点。所以边界是:生成交给它,验证一律不交。

**2. 标准答案由知识库生成,那知识库错了怎么办?**
这是这个方案唯一的单点。三道防线:
(a) 加载时的符号化恒等式校验(`GMV = UV × CVR × AOV` 约分后必须自洽);
(b) 分解引擎的**残差**作为运行期交叉验证 —— 各分项之和与总变化的偏差实测 ~1e-13%;
(c) 数据层自检:`fact_sessions.converted` 的计数必须等于 `orders` 指标,两套独立生成的
判定必须严丝合缝。
诚实地说:知识库错了,评估集就错了。所以它必须进 Git、必须有校验、必须能 review。

**3. `pass@1` 从 60% 提到 85%,怎么证明不是运气/不是背答案?**
留出集 + 回归闸门。改进循环**不得**接触 `holdout: true` 的 case;
新版本的留出集指标不得退步才允许合并。另外看 `false_refusal_rate` 有没有同步上升 ——
很多"提升"其实是模型变保守了。

**4. 为什么 3 轮,不是 5 轮或 1 轮?**
守卫能给出的反馈类型有限,一轮改一类。更重要的是:`avg_attempts` 本身是要上报的指标,
调高上限会让 `pass_final` 好看而把成本藏起来。上限必须显式,并且与预算挂钩。

**5. 如果 schema 从 5 张表变成 500 张?**
这时才需要引入检索:先做表级召回(表名/列名/注释的语义检索 + 业务域过滤),再把
召回的子 schema 塞进 prompt。但**评估集的构造方式不变** —— 那才是这个方案的可迁移部分。
同时要新增一个指标:表召回的 recall@k,因为此时会多出"检错表"这个失败模式。

**6. 怎么上生产?**
先影子模式:业务方照常问数据团队,系统在后台同时回答,只记录不返回,攒真实问题分布;
用真实问题扩充评估集(标注成本比冷启动低得多,因为已经有参考答案发生器);
指标达标后灰度放给低风险场景(探索性查数),高风险场景(直接驱动预算)始终保留人工复核。

**7. 成本怎么控?**
查询侧:一次问答 1~2 条 SQL,结果行数上限 500(防 `SELECT *` 把整表拉回来);
模型侧:3 轮上限 + prompt 里只放必要 schema + **prompt 缓存**(评估的重复请求不再付费)
+ **预算熔断**(调用数/token 上限,在发起调用之前检查);
评估侧:评估集跑批挂 Job 定时,而不是每次改动都全量跑。

**8. 为什么不用外部 LLM API?会锁死在 Databricks 上吗?**
默认走 workspace 内的 Foundation Model APIs,理由是约束 5 的出站限制,
以及**业务 schema 不出 workspace**这条合规上更好讲的性质。
但实现上并没有锁死:客户端只认 OpenAI chat-completions 线格式,
换 `base_url` + `model` 即可接 OpenAI / 自建代理 / 任何兼容端点。
如果确实要走外部 API,方案里必须写清两点:请求要出站(Free Edition 可能直接不通),
以及 schema 会离开 workspace。这是取舍,不是默认项。

**9. 换模型之后指标涨了,能说是模型更好吗?**
只有在 `knowledge_version` 没变的前提下才能。所以 `nl2sql_eval_runs` 同时记两者,
并且有一条纪律:**一次跑批只改一个变量。** 两者一起变的那次,数据是废的。

---

## 七、迭代路线

| 阶段 | 内容 | 判据 |
|---|---|---|
| **V0** | 知识库 + 校验;确定性参考 SQL 生成器;守卫;比对器 | `ReferenceGenerator` 拿满分(评估器自检通过) |
| **V1** | 评估集(含不可答对照组与留出集);L1 自修复循环;状态表 | 能跑出一条基线指标,失败可归类 |
| **V2** | 接 LLM(含重试/缓存/熔断/用量入表);把 prompt 与 few-shot 调到基线以上 | `pass_final` 与 `hallucination_rate` 同时改善,且 `knowledge_version` 与 `model` 分开归因 |
| **V3** | L3 改进循环 + 回归闸门 + Job 定时 | 连续 3 个知识版本,留出集不退步 |
| **V4** | 影子模式接真实问题;表级检索(若 schema 扩张) | 真实问题分布上的指标 |

**注意 V0 的顺序:先造裁判,再造选手。** 这个顺序不是洁癖 ——
没有裁判时调 prompt,你唯一的反馈是"我看着顺眼",那不是工程。

---

## 八、本仓库实现到哪一步

| 组件 | 状态 |
|---|---|
| 知识库 + 强校验(含恒等式符号验证) | ✅ |
| 确定性分解引擎(兼任标准答案发生器) | ✅ 残差实测 ~1e-13% |
| 评估集 26 条(4 条不可答 + 5 条留出) | ✅ 标准答案程序化生成 |
| 守卫(只读 / 白名单 / 幻觉检测) | ✅ |
| 执行式比对器 | ✅ |
| L1 自修复循环 | ✅ |
| L2 评估 + 状态表 | ✅ |
| L3 自主改进循环(LLM 提议 / 确定性裁决 / 台账跨循环记忆) | ✅ |
| 维度取值检查(L1 上下文反馈) | ✅ |
| LLM 接入层(多供应商含智谱 / 重试 / 缓存 / 预算熔断 / 用量入表) | ✅ |
| 在真实 Databricks workspace 上跑通 00~03 | ✅ Free Edition + glm-4-flash |
| 真实评估:0% → 63.6%(只补报告日历) | ✅ |
| 自主改进循环在真实 workspace 上的迭代曲线 | ⏳ 待 `04` 运行 |

`312 passed`。测试可切换运行目标,`--rca-target=spark` 即在 Databricks 上复核同一批断言。

代码入口:[README.md](../README.md) · 平台操作:[DATABRICKS_SETUP.md](DATABRICKS_SETUP.md)
