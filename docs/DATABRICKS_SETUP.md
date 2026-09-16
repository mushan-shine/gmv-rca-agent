# 在 Databricks 上运行:逐步操作手册

面向 **Databricks Free Edition**(2025 年起取代 Community Edition)。
本文假设你**不在本地跑任何东西**,全部操作发生在 Databricks workspace 里。

> 本地那套(`pytest -q`、`--target duckdb`)只是改代码时的快速回归,
> 不是交付路径。想完全跳过本地,直接从第 1 步开始。

目录

1. [前置检查:确认你的 workspace 长什么样](#1-前置检查)
2. [把代码放进 workspace](#2-把代码放进-workspace)
3. [确认 catalog 与 schema](#3-确认-catalog-与-schema)
4. [建表灌数(notebook `00_setup`)](#4-建表灌数)
5. [跑分解(notebook `01_decompose`)](#5-跑分解)
6. [出具验收证据(notebook `02_run_tests`)](#6-出具验收证据)
7. [跑循环主线(notebook `03_nl2sql_loop`)](#7-跑循环主线)
8. [纯 SQL 路线(不用 notebook)](#8-纯-sql-路线)
9. [建 Job:把循环挂上定时](#9-建-job)
10. [排查表](#10-排查表)

---

## 1. 前置检查

登录 <https://www.databricks.com/learn/free-edition> 注册并进入 workspace。
左侧栏应当能看到这几项(名称随版本略有出入):

| 你需要用到 | 左侧栏位置 | 本项目用它做什么 |
|---|---|---|
| **Workspace** | 顶部 | 放 notebook / Git folder |
| **Catalog**(Catalog Explorer) | 中部 | 看建出来的 5 张表 |
| **SQL Editor** | SQL 区 | 纯 SQL 路线;临时查数 |
| **Jobs & Pipelines / Workflows** | 底部 | 第 9 步定时任务 |
| **Compute → SQL warehouses** | 底部 | 确认有一个 serverless warehouse |

还要确认一项:左侧 **Serving**(或 **Machine Learning → Serving**)里有没有
`databricks-meta-llama-*` 之类的 Foundation Model 端点 —— 循环的生成环节需要它。
**没有也不影响把项目跑完**,`notebooks/03_nl2sql_loop.py` 会自动退回到
不需要 LLM 的基线生成器,其余环节一模一样。

Free Edition 的限制:**只有 serverless**,一个 2X-Small SQL warehouse,
有每日额度,**出站网络受限**(所以生成环节优先用 workspace 内的 FM API,
请求不出 Databricks)。本项目的样例数据默认约 33.6 万行会话,远在额度之内。

> ⚠ Free Edition 的界面与权限模型仍在演进。下文提到的菜单位置若对不上,
> 用左上角搜索框搜关键词(如 "Git folder"、"SQL warehouse")一般能直接跳过去。

---

## 2. 把代码放进 workspace

### 路线 A:GitHub + Git folder(**推荐**)

简历项目需要一个公开仓库,而且 L3 改进循环的设计就是「知识改动走 Git PR」——
提示与 few-shot 的每一次变更都要能被 review、被回滚。所以先上 GitHub。

**A1. 本地推到 GitHub**(这是唯一需要在本地做的事)

```bash
cd gmv-rca-agent
git add -A
git commit -m "text-to-SQL loop on Databricks: self-repair, eval, improvement gate"
git branch -M main
git remote add origin https://github.com/<你的账号>/gmv-rca-agent.git
git push -u origin main
```

**A2. 在 Databricks 里配 Git 凭据**

1. 右上角头像 → **Settings**
2. 左侧 **Linked accounts**(旧版叫 Git integration)
3. Git provider 选 **GitHub**,填你的 GitHub 用户名
4. Token 处填一个 GitHub **Personal access token**(`repo` 权限)
   —— 在 GitHub → Settings → Developer settings → Personal access tokens 生成
5. Save

**A3. 克隆进 workspace**

1. 左侧 **Workspace** → 进入你的用户目录
2. 右上角 **Create** → **Git folder**
3. Git repository URL 填 `https://github.com/<你的账号>/gmv-rca-agent`
4. Git provider 选 GitHub,Git folder name 保持 `gmv-rca-agent`
5. **Create Git folder**

之后本地 `git push` → 在 Git folder 里点 **Pull** 就能同步。
也可以直接在 Databricks 里改代码,再从 Git folder 界面 commit & push。

### 路线 B:直接上传(不推荐)

Workspace → 用户目录 → **Create** → **File upload**,把整个目录传上去。
能跑,但没有版本控制,与 L3「知识改动走 PR」的设计对不上,简历上也少一个 GitHub 链接。

---

## 3. 确认 catalog 与 schema

**这是最容易卡住的一步。** 不同 workspace 的默认 catalog 名不一样 ——
有的是 `main`,Free Edition 常见的是 `workspace`。先查清楚再往下走。

打开 **SQL Editor**(左侧栏),粘贴执行:

```sql
SHOW CATALOGS;
```

记下你有写权限的那个名字(下文统称 `<CATALOG>`)。再确认能建 schema:

```sql
CREATE SCHEMA IF NOT EXISTS <CATALOG>.gmv_rca;
SHOW SCHEMAS IN <CATALOG>;
```

能建出来就说明权限够了。项目里所有地方的 catalog 都是参数,不需要改代码:

| 在哪 | 怎么传 |
|---|---|
| notebook | 顶部的 `catalog` / `schema` 组件(widget) |
| CLI | `--catalog <CATALOG> --schema gmv_rca` |
| 环境变量 | `DATABRICKS_CATALOG` / `DATABRICKS_SCHEMA` |

---

## 4. 建表灌数

打开 `gmv-rca-agent/notebooks/00_setup.py`。

1. 右上角 **Connect** → 选 **Serverless**(Free Edition 只有这个)
2. notebook 顶部会出现三个输入框(第一次运行后才出现,可先 **Run all** 再回来改):
   - `catalog` → 填第 3 步确认的 `<CATALOG>`
   - `schema` → `gmv_rca`
   - `daily_sessions` → `6000`(默认;想快点可以填 `1200`)
3. 点 **Run all**

**期望看到的输出:**

```
指标: ['aov', 'cvr', 'gmv', 'orders', 'uv']
维度(按钻取优先级): ['channel', 'device', 'market', ...]
GMV 的乘法因子: ['uv', 'cvr', 'aov']

[ 1] create:dim_product            x.xxs
...
[10] insert:fact_ad_spend          x.xxs

sessions             358259
orders_all_status    15670
completed_orders     14416
sessions_converted   14416          ← 必须与上一行相等
...
✓ sessions_converted == completed_orders == 14416
```

最后那条断言是整个数据层的关键:它证明「一个转化会话恰好一个订单」成立,
也就是 `UV × CVR × AOV = GMV` 在数据层没有歧义。**不相等就停下来查,不要往下走。**

**预计耗时:** 首次要等 serverless 冷启动,整体 1~3 分钟。之后重跑约 30~60 秒。

**幂等:** 全部语句是 `CREATE OR REPLACE` + 确定性 `INSERT`(不用 `rand()`),
重跑得到逐行完全相同的数据。放心重跑。

**验证:** 左侧 **Catalog** → `<CATALOG>` → `gmv_rca`,应当能看到 5 张表:
`fact_orders` / `fact_sessions` / `fact_ad_spend` / `dim_product` / `dim_user`。

---

## 5. 跑分解

打开 `notebooks/01_decompose.py`,填同样的 `catalog` / `schema`,**Run all**。

它会:

1. 做一次 `GMV = UV × CVR × AOV` 的对数分解,打印每个因子的份额与分配百分点,
   并**断言残差 < 0.1%**(简报 V0 验收标准 4);
2. 按 `dimensions.yaml` 的优先级逐个维度做加法分解,同样断言闭合;
3. 打印每条 SQL 的**指纹**(可复现性,简报约束 3);
4. 演示两条边界:比率指标的加法分解会被拒绝、跨表口径不对齐的过滤会被拒绝。

**期望看到的关键行:**

```
残差        -1.074e-12 %
恒等式偏差   1.241e-14 %
闭合(<0.1%)True
```

残差应当在 `1e-12` 量级 —— 这是浮点误差,不是近似。
若它变成 `1e-2` 或更大,说明口径漂了,去看 `README.md` 的「口径决策」。

---

## 6. 出具验收证据

打开 `notebooks/02_run_tests.py`,**Run all**。

这一步把**整套验收测试跑在 Databricks 上**,而不是本地:

- 先跑不需要数据库的部分(知识库校验、手算对照、SQL 方言校验、凭据处理);
- 再用 `--rca-target=spark` 把数据相关的测试跑在当前 SparkSession 上,
  写入独立的 `gmv_rca_test` schema(不碰第 4 步建的正式 schema);
- 最后单独打印闭合性的**实测数量级** —— 简历和汇报里要引用的是这个数字,
  不是一句「测试通过」。

**期望看到:**

```
243 passed
因子分解残差      -1.074e-12 %   (要求 < 0.1%)
维度分解最大残差   1.440e-13 %   (要求 < 0.1%)
```

重复跑时可以在第二格的 `pytest.main([...])` 里加 `"--rca-skip-setup"`,跳过重新建库。

跑完想清理:最后一格有一条注释掉的 `DROP SCHEMA ... CASCADE`,取消注释执行即可。

---

## 7. 跑循环主线

打开 `notebooks/03_nl2sql_loop.py`,填同样的 `catalog` / `schema`,**Run all**。

这是项目的核心。它依次做四件事:

**7.1 探测 LLM 是否可用。** 第 2 格会列出你 workspace 里所有的 serving 端点。
看到 `databricks-meta-llama-*` / `databricks-gpt-*` 之类的名字,就把它填进顶部的
`endpoint` 组件,重跑该格。

有端点时,notebook 直接用 workspace 内的 serving 端点 ——
凭据由 Runtime 注入,**不需要手工配 token**,请求也不出 Databricks。

**没有 LLM 也能跑完整条流水线** —— 生成器会自动换成 `ReferenceGenerator`
(永远给出正确 SQL 的「作弊」生成器),守卫、执行、比对、状态表、改进闭环完全一样。
这时的满分结果不是成绩,而是**对评估器本身的校验**:
一个永远正确的生成器如果拿不到满分,说明比对逻辑或标准答案有问题。

**要接外部 API**(OpenAI / 自建代理):把装配格里的 client 换成

```python
OpenAIChatClient(
    base_url="https://api.openai.com/v1",
    api_key=dbutils.secrets.get("my-scope", "openai"),   # 不要把 key 写进 notebook
    model="gpt-4o-mini",
)
```

两点要先想清楚:Free Edition 的**出站网络可能直接不通**;
以及业务 schema 会随 prompt **离开你的 workspace**。

**预算与缓存。** 装配格里设了 `UsageMeter(max_calls=200, max_tokens=500_000)` ——
在发起调用**之前**检查,超了熔断。客户端外面包了一层 prompt 缓存:
评估会把逐字相同的 prompt 打很多遍,第二遍不再付费,同一版本的结果也因此可复现。

**7.2 看一次自修复循环。** 打印每一轮尝试:守卫报了什么、引擎报了什么、第几轮修好的。

**7.3 跑整个评估集。** 26 条 case,输出:

```
pass@1              ...%     第一轮就跑通且答对
pass (final)        ...%     执行式比对通过率
holdout pass        ...%     留出集 —— 回归闸门看的是它
hallucination rate  ...%     引用了不存在的表/列
correct refusal     ...%     不可答问题被正确拒答
false refusal       ...%     可答问题被误拒(堵死「一律拒答」这条捷径)
```

**7.4 改进循环 + 回归闸门。** 按失败类型归因,给失败 case 补 few-shot 示例
(每条都带 `source_run` 与 `fixes_case`),版本号 +1,然后过回归闸门:
**留出集指标不得退步**,否则不许合并。

通过之后,最后一格(默认注释掉)把新知识写回
`knowledge/nl2sql/prompting.yaml`,接着在 Git folder 界面 commit & push 开 PR。

跑完之后去 Catalog 里看这两张表 —— 它们是「这是个循环」的物证:

```sql
SELECT * FROM <CATALOG>.gmv_rca.nl2sql_eval_runs ORDER BY created_at DESC;
SELECT * FROM <CATALOG>.gmv_rca.nl2sql_attempts  ORDER BY created_at DESC LIMIT 50;
```

---

## 8. 纯 SQL 路线

不想用 notebook、只想看数据长什么样时用这条。

1. 打开 **SQL Editor**
2. 打开仓库里的 `sql/setup/databricks_setup.sql`,**整段复制**进去
3. 首行的 `CREATE SCHEMA IF NOT EXISTS main.gmv_rca` 里的 `main` 换成你的 `<CATALOG>`
   —— 或者在本地重新渲染一份:
   ```bash
   python sql/setup/render.py --catalog <CATALOG> --schema gmv_rca
   ```
4. **Run all**(SQL Editor 会按分号顺序执行)

末尾的自检查询会返回 10 行 `check_name / value`,同样要确认
`sessions_converted` 等于 `completed_orders`。

> `sql/setup/*.sql` 是 `src/rca/sampledata.py` 渲染出来的**产物**,不要手改。
> 改了 `knowledge/tables.yaml` 或生成器之后重新跑 `render.py`;
> `tests/test_sql_rendering.py` 会检查产物有没有漂移。

---

## 9. 建 Job

把评估挂成定时任务,就有了一条**跨天累积的指标时间序列** ——
简历上「持续评估」这句话需要有它做底。五分钟。

1. 左侧 **Jobs & Pipelines**(或 Workflows)→ **Create** → **Job**
2. Task name:`setup`
3. Type:**Notebook**
4. Source:**Git provider**(如果用了 Git folder 就选 Workspace 也行)
5. Path:`gmv-rca-agent/notebooks/00_setup.py`
6. Compute:**Serverless**
7. **Parameters** 里加两项(对应 notebook 的 widget):
   - `catalog` = `<CATALOG>`
   - `schema` = `gmv_rca`
8. 右侧 **Schedule** → 需要定时就设,比如每天 07:00
9. **Create** → **Run now** 验证一次

再加第二个 task,依赖第一个:

* Task name:`evaluate`
* Path:`gmv-rca-agent/notebooks/03_nl2sql_loop.py`
* Depends on:`setup`

于是每天:重建样例数据 → 跑评估 → 往 `nl2sql_eval_runs` 追加一行。
**Runs** 页面会留下每次执行的日志与耗时,Delta 表里会积累一条指标时间序列 ——
「这个知识版本比上个版本好在哪」从此有据可查,而不是凭印象。

> 注意判据:定时跑评估本身**还不是**循环。
> 它成为循环,是因为 `03` 里的改进环节会读上一轮写下的失败记录、
> 产出新的 few-shot,而下一轮生成时 prompt 里就有了它。

---

## 10. 排查表

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `[SCHEMA_NOT_FOUND]` / `Catalog 'main' not found` | 你的 workspace 默认 catalog 不叫 `main` | 回第 3 步跑 `SHOW CATALOGS`,把 `catalog` 参数换成实际的名字 |
| `PERMISSION_DENIED` 建 schema 失败 | 当前用户对该 catalog 没有 `CREATE SCHEMA` 权限 | 换一个有权限的 catalog;Free Edition 一般对默认 catalog 是有权限的 |
| `ModuleNotFoundError: No module named 'rca'` | notebook 没找到 `src/` | 确认 notebook 在 `notebooks/` 目录下(它靠相对路径定位仓库根);若是直接上传的散文件,手动 `sys.path.insert(0, "<仓库根>/src")` |
| `ModuleNotFoundError: pydantic` / `yaml` | Runtime 里没有这两个包 | 在 notebook 第一格加 `%pip install pydantic PyYAML`,然后 `dbutils.library.restartPython()` |
| notebook 顶部没有 widget 输入框 | widget 要先执行那一格才会出现 | 先 **Run all** 一次,再回到顶部改值重跑 |
| 第一条语句卡住很久 | serverless 冷启动 | 等 1~2 分钟;Compute → SQL warehouses 里能看到启动状态 |
| `sessions_converted ≠ completed_orders` | 生成器的两处转化判定漂移了 | **停下**。这会让 `UV×CVR×AOV=GMV` 在数据层失效。看 `src/rca/sampledata.py` 里 `_cvr_expr` 与 `_fact_orders_sql` 的 `WHERE` 是否仍然一致 |
| 残差不是 `1e-12` 量级而是百分之几 | 指标口径不一致 | 看 `README.md`「口径决策」;`knowledge.py` 的恒等式符号校验应当能提前拦住大部分情况 |
| `pytest` 在 notebook 里报 rootdir 不对 | 没 `chdir` 到仓库根 | `02_run_tests.py` 已经做了 `os.chdir(REPO_ROOT)`;自己写的话要补上 |
| `03` 里探测不到任何 serving 端点 | 这个 workspace 没开 Foundation Model APIs | 不影响:会自动退回 `ReferenceGenerator`,整条流水线照跑。要接外部 LLM 就用 `CallableGenerator(lambda prompt: ...)` |
| `ModuleNotFoundError: sqlglot` | 守卫的运行时依赖 | notebook 第一格已有 `%pip install sqlglot`;Job 里跑要确认这格执行到了 |
| 模型调用报 `LlmBudgetExceeded` | 触发了预算熔断 | 这是**设计行为**。报告里会显示成 `generator_error`,与「模型变差」区分得开。调高 `UsageMeter` 的上限,或缩小评估集 |
| 模型调用报 `HTTP 403 / 404` | 端点名填错,或当前用户对该端点没有查询权限 | 回第 7.1 步看端点列表里的确切名字 |
| 大量 `generator` 类失败,`generation_error` 写着「被截断」 | `max_tokens` 太小 | 调大装配格里的 `max_tokens`;截断的回复是半条 SQL,修不出来 |
| 接外部 API 时请求超时/连不上 | Free Edition 出站受限 | 改用 workspace 内的 serving 端点(默认路径) |
| `nl2sql_eval_runs` 里只有一行 | 状态表是 append-only,但你只跑过一次 | 再跑一次 `03`;两行之后回归闸门才有基线可比 |
| Git folder 里 Pull 报认证失败 | Linked accounts 里的 GitHub token 过期或权限不足 | 重新生成一个带 `repo` 权限的 GitHub PAT,回 Settings → Linked accounts 更新 |

---

## 附:从本机连 workspace(可选)

如果某天想在本地 IDE 里调试同一套代码,不必改任何东西 ——
配好三个环境变量(或仓库根的 `.env`,已 gitignore)即可:

```bash
DATABRICKS_SERVER_HOSTNAME=<workspace>.cloud.databricks.com   # 不带 https://
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
DATABRICKS_TOKEN=<personal access token>
```

前两项在 **Compute → SQL warehouses → 点开你的 warehouse → Connection details** 页面直接复制。
Token 在 **Settings → Developer → Access tokens** 生成。

```bash
python -m rca.cli setup  --target databricks --catalog <CATALOG>
python -m rca.cli demo   --target databricks --catalog <CATALOG>
pytest -q --rca-target=databricks
```

跑的是同一份 SQL,只是换了条通道。
