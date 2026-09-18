# 上线指南:从「一步步执行」到「打开就能问」

`00`~`04` 是研发和评估系统:证明 SQL 写得对、并且能持续变准。
这份文档讲怎么把它变成业务每天在用的问答助手:**入口在哪、怎么部署、上线后循环怎么转**。

---

## 一、上线后的样子

```
 业务人员:「上周美国站 GMV 为什么下降?」
   │
   ├─ ① 网页聊天       app/app.py(Databricks Apps,公司账号登录)
   ├─ ② notebook       notebooks/05_assistant.py(没有 Apps 时用它)
   └─ ③ 定时推送       Job 每周一自动回答「上周 GMV 为什么变化」
   │
   ▼  三个入口调用同一份代码:src/rca/assistant/
 问题分流(确定性:关键词 + knowledge/assistant/synonyms.yaml)
   ├─ 「为什么……」 → 归因:因子分解 + 各维度拆分    不经 LLM,残差可验证
   └─ 其他         → 查数:text-to-SQL 自修复循环    用 04 采纳、经 Git 合并的提示
   │
   ▼
 回答:结论 + 所用假设(哪两周、按什么过滤)+ 明细表 + 每个数字背后的 SQL + 👍👎
   │
   ▼  写进 assistant_requests / assistant_feedback(Delta)
 待复核队列:没答出来的 + 被点 👎 的 → 人确认口径 → 加进 eval/cases.yaml
   │
   ▼
 Job:每日评估(03)· 每周自主改进(04)→ 采纳的提示走 PR → 合并后线上生效
```

**两条路径为什么分开:** 归因要的是一组互相勾稽的数字(各因子贡献之和 = 总变化)。
让 LLM 写多条 SQL 再拼起来,口径稍有不齐就对不上账。分解内核保证闭合,叙述用模板生成,
**回答里没有一个数字是模型写的**。LLM 只用在它真正需要的地方:把自由问法翻译成 SQL。

---

## 二、三个入口

| 入口 | 适合 | 前提 | 状态 |
|---|---|---|---|
| **notebook `05_assistant`** | 自己用、演示、没有 Apps 的 workspace | `00` 跑过 | ✅ 随时可用 |
| **网页聊天(Databricks Apps)** | 业务同事日常使用 | workspace 开通了 Apps | 代码就绪,看 workspace 是否支持 |
| **定时推送(Job)** | 每周自动出归因结论 | — | ✅ 配置就绪,默认暂停 |

以后要接飞书 / 钉钉 / Slack 机器人,调用的也是同一个 `Assistant.ask()` —— 入口只负责收发消息。

---

## 三、入口 ①:notebook `05_assistant`

1. Git folder 里 **Pull**
2. 打开 `notebooks/05_assistant.py`,右上角选 **Serverless**(不是 SQL Warehouse)
3. **Run all** 一次(装依赖、接 LLM、装配助手)
4. 之后每次提问:改顶部 **`question`** 组件 → **只运行第 3 格「提问」**

| 试试这些 | 走哪条路 |
|---|---|
| `上周 GMV 为什么变化了?` | 归因(不需要 LLM) |
| `上周美国站 GMV 为什么下降?` | 归因 + 按市场过滤 |
| `昨天移动端订单量为什么变了?` | 归因,按天对比 |
| `上周各渠道的 GMV 是多少?` | 查数(需要 LLM) |
| `转化率为什么跌了` | 说明比率指标不能按维度相加拆分,建议改问 GMV |

觉得答错了:第 4 格把 `RATING` 改成 `-1`、写上原因再运行,这道题就进了待复核队列(第 5 格)。

---

## 四、入口 ②:网页聊天(Databricks Apps)

### 4.1 先确认 workspace 有没有 Apps

左侧导航栏 **Compute** → 顶部有没有 **Apps** 标签;或左上角 **+ New** 下拉里有没有 **App**。
都没有就是不支持,用入口 ① 即可。

### 4.2 本机先看效果(可选,不连任何远端)

```bash
pip install streamlit pandas
```

```bash
streamlit run app/app.py
```

默认 `RCA_TARGET=duckdb`:内存库 + 样例数据。不配 LLM 也能问所有「为什么」类问题。

### 4.3 部署到 Databricks Apps

1. **Compute → Apps → Create app** → 选 **Custom**,名字填 `gmv-rca-assistant`
2. **Resources** 里添加:

   | 资源键(必须一致) | 类型 | 选择 | 权限 |
   |---|---|---|---|
   | `sql-warehouse` | SQL warehouse | Serverless Starter Warehouse | Can use |
   | `llm-api-key` | Secret | scope `llm`,key `zhipu_api_key` | Can read |

   不接 LLM 就不加第二个,同时删掉 `app.yaml` 里 `RCA_LLM_API_KEY` 那一段(归因照常可用)。
3. 创建完成后,App 详情页会显示它的 **service principal**(一串名字或 ID)。在 SQL Editor 给它授权:

   ```sql
   -- <sp> 换成 App 详情页上显示的 service principal 名称或 application ID
   GRANT USE CATALOG ON CATALOG workspace TO `<sp>`;
   GRANT USE SCHEMA, SELECT ON SCHEMA workspace.gmv_rca TO `<sp>`;
   -- 写问答日志与反馈(两张表第一次会自动创建)
   GRANT CREATE TABLE, MODIFY ON SCHEMA workspace.gmv_rca TO `<sp>`;
   ```

   **为什么是单独的身份:** 线上服务不该拿某个人的个人 token 连仓库。
   它只有这个 schema 的读权限(加上写日志),SQL 守卫之外,权限层再挡一次。
4. **Deploy** → Source code path 选 Git folder 的**仓库根目录**(`.../gmv-rca-agent`,不是 `app/`)
   —— 部署需要 `src/` 和 `knowledge/`,依赖从根目录的 `requirements.txt` 安装
5. 等状态变成 **Running**,点页面上的 URL。同 workspace 的同事用公司账号登录即可访问

### 4.4 页面上有什么

- 中间:对话。每个回答下面有「假设」(用的哪两周、按什么过滤)、明细表、可展开的 SQL、👍👎
- 左侧:数据截至哪天、查数用的模型、知识版本、示例问题、**待复核队列**

### 4.5 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 页面显示「启动失败:缺少 DATABRICKS_WAREHOUSE_ID」 | 没加 `sql-warehouse` 资源,或资源键写错 | Resources 里检查资源键 |
| 「PERMISSION_DENIED」/「TABLE_OR_VIEW_NOT_FOUND」 | service principal 没有授权 | 执行 4.3 第 3 步的 GRANT |
| 查数模型显示「未接入」 | 没加 `llm-api-key` 资源 | 加资源,或只用归因 |
| Apps 连不上智谱 | Apps 的出站网络受限 | 改用 workspace 内的 serving 端点(`RCA_LLM_PROVIDER=databricks`) |

---

## 五、入口 ③:定时任务

三个 Job 定义在仓库根的 `databricks.yml`,**默认全部暂停**。

| Job | 频率 | 运行 | 作用 |
|---|---|---|---|
| 每日评估 | 每天 02:00 | `03` | 用当前知识版本把评估集跑一遍,指标写进 `nl2sql_eval_runs`。模型供应商悄悄换版本、数据口径变化,第一时间可见 |
| 每周自主改进 | 周日 03:00 | `04`(6 轮、耐心 4) | 提议 → 试跑 → 裁决 → 台账。**不会自动改线上知识**,采纳的提示要人在 Git 里合并 |
| 每周 GMV 归因 | 周一 09:00 | `05`(`llm_provider=none`) | 自动回答「上周 GMV 为什么变化」,不需要 LLM |

### 方式 A:命令行部署(你本机已经配好 Databricks CLI)

```bash
databricks bundle validate
```

```bash
databricks bundle deploy
```

```bash
databricks bundle run weekly_gmv_report
```

最后一条会手动触发一次「每周 GMV 归因」,确认能跑通。然后到 **Jobs & Pipelines** 页面,
把需要的 Job 的调度从 **Paused** 改成 **Active**。

### 方式 B:在界面上建(不用 CLI)

**Jobs & Pipelines → Create → Job**,每个 Job 一个 task:

| 字段 | 每周 GMV 归因 | 每日评估 | 每周自主改进 |
|---|---|---|---|
| Type | Notebook | Notebook | Notebook |
| Path | `notebooks/05_assistant.py` | `notebooks/03_nl2sql_loop.py` | `notebooks/04_improvement_loop.py` |
| Compute | Serverless | Serverless | Serverless |
| Parameters | `question` = `上周 GMV 为什么变化了?`<br>`llm_provider` = `none` | `llm_provider` = `zhipu` | `llm_provider` = `zhipu`<br>`max_rounds` = `6`<br>`patience` = `4` |
| Schedule | 周一 09:00 | 每天 02:00 | 周日 03:00 |

⚠ `03` 的 `llm_provider` 默认是 `databricks`,挂 Job 时一定要传 `zhipu`,否则会退回 `ReferenceGenerator`,评估的就不是真模型了。

---

## 六、上线后,循环怎么转

```
线上提问 ──→ assistant_requests
   │                 │
   │   失败 / 👎      ▼
   │        待复核队列(05 第 5 格 / 网页左侧)
   │                 │  人确认正确口径
   │                 ▼
   │        eval/cases.yaml 新增一道题(问题 + 指标/维度/窗口声明,标准答案程序化生成)
   │                 │
   │                 ▼
   │        每日评估:新题进入分数;每周改进:新题进入训练集
   │                 │  闸门采纳的提示
   │                 ▼
   │        PR → review → 合并 prompting.yaml
   │                 │
   └─────────────────┘  线上助手下次启动时读到新提示
```

**这是离线 demo 做不到的部分。** 评估集从线上真实问题里长大:
04 第一次真实运行暴露的「21 道题、闸门分辨率只有 1 道」,会随着线上问题的积累自然缓解。
而且新题来自真实的提问方式,比人凭想象写的题更能代表线上分布。

**哪一步必须有人:** 确认口径(把失败问题变成评估题)和合并提示(PR review)。
这是刻意的 —— 标准答案和线上知识是整个系统的「尺子」,尺子的变更要有人签字。

---

## 七、接真实数据要改什么

| 改什么 | 在哪 | 说明 |
|---|---|---|
| 表结构 | `knowledge/tables.yaml` | 指向真实的订单表、会话表 |
| 指标口径 | `knowledge/metrics/*.yaml` | 按公司定义;GMV = UV × CVR × AOV 恒等式会在加载时做符号验证 |
| 口语说法 | `knowledge/assistant/synonyms.yaml` | 「美国站」「手机端」这类说法。维度取值本身启动时从仓库实时查出,不用列 |
| 「今天」是哪天 | `RCA_AS_OF=today`(App)/ `date.today()`(`05`) | 样例数据是历史快照,所以默认用它的报告日 |
| 评估集 | `eval/cases.yaml` | 从待复核队列里逐步补 |

---

## 八、还没有做的(如实说明)

| 缺口 | 影响 | 下一步 |
|---|---|---|
| 分流是规则 | 没有「为什么」字样的归因问法(如「GMV 跌了是哪个渠道拖的」)会走查数路径 | 判错的问法进待复核队列后补进 `synonyms.yaml`;规则覆盖不了时再加 LLM 分类,并用同样的方式评估它 |
| 比率指标不能归因 | 「转化率为什么跌」目前只给出说明 | 实现 mix/rate 拆分(结构变化 vs 各分组自身变化) |
| 对比窗口只有「上周 / 昨天」 | 「上个月」「同比」不识别,按上周处理,并在回答的「假设」里写明 | 扩充 `comparison_periods` |
| 没有异常检测 | 每周归因是定时的,不是「GMV 异常才触发」 | 加一个每日 Job:GMV 偏离近 4 周同期均值超过阈值时才触发归因并推送 |
| 单实例串行 | 网页端同一时刻只处理一个问题 | 并发量上来后改成连接池 |
