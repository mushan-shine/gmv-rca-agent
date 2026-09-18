# 执行记录:从本地代码到 Databricks 上的真实评估

> **日期:** 2026-09-17
> **环境:** Databricks Free Edition(serverless)· 模型 智谱 `glm-4-flash`(免费)· 本机 Windows
> **范围:** 代码推上 GitHub → 在 Databricks 上跑通 00~03 → 接入真实 LLM 跑出两轮评估 → 构建自主改进循环 04 → 04 第一次真实运行与改进
>
> 这份记录保留了**真实遇到的每一个问题**。其中不少问题本地测试发现不了,只有在目标平台上才会出现;
> 还有几个问题出在评估系统自己身上,而不是模型。这些正是项目想展示的东西。

---

## 目录

- [一、结果总览](#一结果总览)
- [二、业务背景:我们在做什么](#二业务背景我们在做什么)
- [三、执行步骤](#三执行步骤)
  - [阶段 A · 代码推上 GitHub](#阶段-a--代码推上-github)
  - [阶段 B · 准备 Databricks 工作区](#阶段-b--准备-databricks-工作区)
  - [阶段 C · 跑通 00~02](#阶段-c--跑通-0002)
  - [阶段 D · 接入智谱 LLM](#阶段-d--接入智谱-llm)
  - [阶段 E · 第 1 轮真实评估(0%)](#阶段-e--第-1-轮真实评估0)
  - [阶段 F · 第 2 轮真实评估(63.6%)](#阶段-f--第-2-轮真实评估636)
  - [阶段 G · 构建自主改进循环](#阶段-g--构建自主改进循环)
  - [阶段 H · 自主改进循环第一次真实运行(0 条采纳)](#阶段-h--自主改进循环第一次真实运行0-条采纳)
  - [阶段 I · 从「一步步执行」到可用的问答助手](#阶段-i--从一步步执行到可用的问答助手)
- [四、问题清单(按类型汇总)](#四问题清单按类型汇总)
- [五、经验总结](#五经验总结)
- [六、下一步](#六下一步)
- [附录:提交记录与诊断代码](#附录提交记录与诊断代码)

---

## 一、结果总览

| 轮次 | 改动 | 最终准确率 | 首轮准确率 | 留出集 | 平均轮数 | 模型调用 | token |
|---|---|---|---|---|---|---|---|
| 第 1 轮 | 基线 | **0%** | 0% | 0% | 1.27 | 32 | 20,385 |
| 第 2 轮 | + 报告日历(**只改上下文,模型不变**) | **63.6%** | 50.0% | 50.0% | 1.35 | 34 | 26,618 |

**自主改进循环(`04`,+ 维度取值检查)** —— 这里的准确率把「可答题答对」和「不可答题正确拒答」合在一起算,与上表口径不同:

| | 训练集(21 道) | 留出集(5 道) | 结果 |
|---|---|---|---|
| 基线 | **66.7%**(14 道) | **80.0%**(4 道) | |
| 第 1 轮候选 | 57.1%(12 道) | 80.0% | ✗ 拒绝:训练集 14 → 12 |
| 第 2 轮候选 | 57.1%(12 道) | 80.0% | ✗ 拒绝:训练集 14 → 12 |
| 最终 | 66.7% | 80.0% | 连续 2 轮未采纳,按 patience 停止。114 次调用 · 9.2 万 token · 14 分钟 |

- **自修复的贡献:** 第 2 轮首轮 50.0% → 最终 63.6%,**+13.6 个百分点**来自 L1 自修复
- **闸门挡下了两次退步:** 提议者给出的两条提示都会让训练集少对 2 道,都没有进入知识库
- **在 Databricks 上验证通过:** 建表灌数自检、分解闭合性(残差 ~1e-12%)、整套验收测试
- **记录并处理的问题:** 23 个,其中 5 个出在评估系统自身,而不是模型(见 [第四节](#四问题清单按类型汇总))

---

## 二、业务背景:我们在做什么

**业务问题。** 电商公司的运营、投放、品类经理每天在群里问大量指标问题
(「上周美国站 GMV 多少」「哪个渠道转化率最高」),数据团队的时间被一次性取数占满。

**方案。** 做一个会写 SQL 的问答助手。但真正难的不是「写 SQL」,而是:

1. **怎么知道它答对了?** —— 答错的数字会直接影响投放预算
2. **怎么让它持续变好,并且证明是真的变好?**

**项目的三层循环:**

| 循环 | 频率 | 做什么 | 业务含义 |
|---|---|---|---|
| L1 自修复 | 一次提问内 | SQL 出错 → 把具体错误喂回去 → 重写,最多 3 轮 | 分析师写错了自己改 |
| L2 评估 | 每次改动后 | 26 道题,标准答案程序化生成,比对执行结果 | 定期考核,分数可追溯 |
| L3 改进 | 跨版本 | 失败 → 提出改进 → 重考 → 不退步才采纳 → 记住试过什么 | 带教新人:总结错题,形成团队规范 |

**评估集的设计。** 26 道题里:

- **21 道训练题**:改进过程可以看它们的失败
- **5 道留出题**:改进过程看不到,专门检查「是真学会了还是背答案」
- 其中 **4 道不可回答**(问客户邮箱、广告计划 GMV、毛利率、退货原因 —— 数据里根本没有):
  正确行为是拒答。一个准确率很高、但会编造答案的助手,在业务上是危险的

---

## 三、执行步骤

每一步按 **操作 → 作用 → 业务含义 → 结果 / 问题** 记录。

### 阶段 A · 代码推上 GitHub

#### A1. 提交代码并推送

**操作**

```bash
git commit -m "..."
git branch -M main
git remote set-url origin https://github.com/<账号>/gmv-rca-agent.git
git push -u origin main
```

**作用:** Databricks 通过 Git folder 从 GitHub 拉代码,代码必须先在 GitHub 上。

**业务含义:** 知识库(指标口径、提示规则)的每一次变更都要能 review、能回滚,Git 是唯一真源。

> **🔴 问题 1:`error: remote origin already exists`**
>
> - **现象:** 执行 `git remote add origin ...` 报错
> - **原因:** 之前已经执行过一次 `remote add`,但地址里还是模板占位符 `<你的账号>`,没有替换成真实用户名。再次 `add` 就报已存在
> - **解法:** 不删除重加,用 `git remote set-url origin <正确地址>` 直接改地址,再 `git push`

#### A2. 生成 GitHub 访问令牌

**操作:** GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens**

| 字段 | 取值 |
|---|---|
| Repository access | Only select repositories → `gmv-rca-agent` |
| Repository permissions | **Contents: Read and write**(其余 No access) |
| Expiration | 90 天 |

**作用:** Databricks 克隆仓库、在 Git folder 里 commit/push 需要凭据。

**为什么选 Fine-grained 而不是 classic:** classic 的 `repo` 权限会放行账号下**所有**仓库。
Fine-grained 只授权这一个仓库,泄露时影响面最小。

---

### 阶段 B · 准备 Databricks 工作区

#### B1. 克隆仓库到 Git folder

**操作:** 头像 → Settings → Linked accounts 填 GitHub 用户名和令牌;
Workspace → Create → Git folder → 填仓库地址。

**作用:** 之后代码更新只需在 Git folder 里点 **Pull**。

#### B2. 确认 catalog 名

**操作:** SQL Editor 执行 `SHOW CATALOGS;`,结果为 `samples` / `system` / `workspace`。

**作用:** 所有表都要建在一个有写权限的 catalog 下。`samples`、`system` 是内置只读目录,只能用 `workspace`。

**业务含义:** Unity Catalog 的 catalog 相当于数据仓库里的「库」,决定数据归属与权限。

> **🔴 问题 2:代码默认 catalog 是 `main`,Free Edition 里不存在**
>
> - **原因:** 开发时按常见的 Unity Catalog 默认值写成了 `main`
> - **影响:** 4 个 notebook 第一次运行都会报 `Catalog 'main' not found`
> - **解法:** 把 Databricks 的默认 catalog 统一改成 `workspace`(`config.py`、`dialects.py`、`cli.py`、
>   `render.py`、4 个 notebook、`.env.example`),重新渲染建表 SQL

#### B3. 运行前的平台走查(未等报错就修掉的 4 个问题)

在让 notebook 真正运行之前,先按 Databricks serverless 的运行环境逐个检查,提前修掉了 4 个
「本地测不出来、上平台必出错」的问题:

| # | 问题 | 原因 | 解法 |
|---|---|---|---|
| 3 | `00` / `01` 没有安装依赖 | serverless 环境不保证有 pydantic v2、PyYAML | 两个 notebook 开头加 `%pip install` + `restartPython()` |
| 4 | `02` 的离线测试会因缺 duckdb 失败 | 有一条测试要创建本地 DuckDB,Databricks 上不装 duckdb | 该测试加 `pytest.importorskip("duckdb")` |
| 5 | `02` **第二次运行**会失败 | 状态表跨运行保留,测试用固定 `run_id`,重跑时出现重复行,聚合断言翻倍 | 远程目标下,测试会话开始时清空测试 schema 里的状态表 |
| 6 | `03` 调 Databricks 模型端点可能 401 | notebook 里是 Runtime 认证,`config.token` 可能为空 | 改用 `config.authenticate()` 取认证头 |

---

### 阶段 C · 跑通 00~02

#### C1. `00_setup` —— 建表灌数

**操作:** 打开 notebook → Connect Serverless → Run all(组件默认值)。

**作用:** 在 `workspace.gmv_rca` 下建 5 张表,灌入 8 周(2025-05-05 ~ 2025-06-29)日粒度数据,约 33.6 万行会话。
数据由确定性生成器产生(不用 `rand()`),重跑逐行一致。

**业务含义:** 模拟一家美/英/德三市场电商的数据仓库:

| 表 | 含义 |
|---|---|
| `fact_sessions` | 访问会话(渠道、设备、市场、是否转化) |
| `fact_orders` | 订单(金额、状态:完成/取消/退款) |
| `fact_ad_spend` | 广告投放花费 |
| `dim_product` | 商品(品类、价格带) |
| `dim_user` | 用户(分层:新客/回头客/VIP) |

**关键自检:** `sessions_converted == completed_orders`。它保证「一个转化会话恰好对应一笔完成订单」,
于是 **GMV = UV × CVR × AOV** 在数据层严格成立 —— 后面所有的分解和标准答案都依赖这个恒等式。

**结果:** ✅ 通过。截图中日 GMV 约 1.7 万、日订单两百多单,符合 6000 日均会话的规模。

#### C2. `01_decompose` —— 指标分解

**操作:** Run all。

**作用:** 把「本周 GMV 比上周变化多少」精确拆成流量、转化率、客单价各自的贡献,
以及各渠道、设备、品类等维度的贡献,并断言**各部分之和等于总变化**。

**业务含义:** 回答「GMV 为什么跌了」—— 是流量少了、转化率低了、还是客单价降了,主要跌在哪个渠道。
在项目里它还兼任**出题老师**:评估集 26 道题的标准答案由它这套口径算出。

**结果:** ✅ 通过,残差约 1e-12%(浮点误差量级)。

#### C3. `02_run_tests` —— 在 Databricks 上跑验收测试

**操作:** Run all。分两段:离线测试(不需要数据库)+ 在 Spark 上跑的数据测试。

**作用:** 证明这套代码**在目标引擎上**确实跑得通、算得对,而不只是在本地 DuckDB 上。

**业务含义:** 出题老师和考试系统本身必须先被验证,否则后面给模型打的分没有意义。

> **🔴 问题 7:离线测试一条都没跑就崩了**
>
> ```
> 退出码: 4
> ImportError while loading conftest '/Workspace/.../tests/conftest.py'.
> OSError: [Errno 95] Operation not supported: '/Workspace/.../tests/__pycache__'
> ```
>
> - **原因:** pytest 加载测试文件时会改写断言并把字节码缓存写进 `tests/__pycache__`。
>   Databricks 的 `/Workspace` 文件系统**不支持创建这个目录**。普通 `import` 遇到这个错误会静默忽略,
>   但 pytest 只容忍「没权限」「只读」这类错误,Errno 95 不在其中,于是直接崩溃
> - **定位方法:** 在本地模拟「创建 `__pycache__` 就报 Errno 95」的环境,完整复现了同样的报错(退出码 4)
> - **解法:** `02` 开头设置 `sys.dont_write_bytecode = True`,并把 `sys.pycache_prefix` 指到 `/tmp`;
>   pytest 加 `-p no:cacheprovider`。本地模拟环境下验证修复后测试全部通过

**结果:** ✅ 修复后两段测试全部通过。这是项目第一份「在真实 Databricks 引擎上验证过」的证据。

---

### 阶段 D · 接入智谱 LLM

#### D1. 选择模型供应商

**决策:** workspace 里没有可用的 Foundation Model 端点,改用智谱开放平台的免费模型 `glm-4-flash`。

**作用:** L1 生成 SQL、L3 提议改进,都需要一个真实模型。

**实现:** 项目的 LLM 客户端只认 OpenAI chat-completions 格式,智谱兼容这个格式,所以只加了一个预设:

| 设置 | 取值 | 原因 |
|---|---|---|
| 地址 | `https://open.bigmodel.cn/api/paas/v4` | 智谱 OpenAI 兼容接口 |
| 模型 | `glm-4-flash` | 免费 |
| `do_sample` | `false` | 关闭采样,**同一个问题两次得到同一个答案**,否则指标涨跌无法归因 |
| 内容拦截 | 识别 `finish_reason=sensitive` | 否则被审核拦下的空回复会被误判成「模型没写 SQL」 |

notebook 03 增加了供应商下拉框(`databricks` / `zhipu` / `none`),以及「连通性 → 读密钥 → 冒烟测试」三步检查。

#### D2. 把 API key 存进 Databricks Secret

**作用:** key 不能写进 notebook(notebook 会进 Git),只能存在 Databricks 的 secret 里,运行时读取。

**业务含义:** 凭据管理。一旦密钥进入代码仓库,就等同于泄露。

secret 只能用 Databricks CLI 创建,这一步在本机遇到了连续 3 个问题:

> **🔴 问题 8:`databricks` 命令无法识别**
>
> - **原因:** ① `winget install` 装好了 CLI,但**当前终端的 PATH 没有刷新**;
>   ② `--host` 参数粘贴了浏览器地址栏的完整 URL(带 `/browse/folders/...?o=...`)
> - **解法:** 重开终端(或用 `Set-Alias` 指向 CLI 的完整路径);`--host` 只保留域名部分

> **🔴 问题 9:`store token: The parameter is incorrect` / `OS keyring unreachable`**
>
> - **原因:** CLI 默认把 OAuth 令牌存进 Windows 凭据管理器,令牌写入失败
> - **解法:** 改用文件存储:`set DATABRICKS_AUTH_STORAGE=plaintext`(PowerShell 为 `$env:DATABRICKS_AUTH_STORAGE="plaintext"`),
>   然后重新 `databricks auth login`。令牌存到用户目录下的 `.databricks/token-cache.json`,只有当前 Windows 用户可读

> **🔴 问题 10:登录显示成功,但仍提示 `no cached credentials`**
>
> - **原因:** 检查 `~/.databrickscfg` 发现两个**名字带前导空格**的脏条目:`[ DEFAULT]` 和
>   `default_profile =  DEFAULT`(第一次登录输入 profile 名时带进了空格)。CLI 去找 `" DEFAULT"` 的令牌,
>   而令牌实际存在 `"DEFAULT"` 下
> - **解法:** 备份为 `.databrickscfg.bak` 后清理配置文件;用 `databricks current-user me` 验证认证恢复
> - **提示:** profile 名输入 `DEFAULT`,之后的命令就不需要每条都加 `--profile`

创建 scope 并存入密钥:

```bash
databricks secrets create-scope llm
databricks secrets put-secret llm zhipu_api_key
```

#### D3. notebook 03 冒烟测试

> **🔴 问题 11:网络通、密钥也读到了,但请求返回 HTML 400**
>
> ```
> open.bigmodel.cn:连通(HTTP 401,网络可达)
> API key:已从 secret llm/zhipu_api_key 读取
> ✗ 冒烟测试失败:HTTP 400: <html>...400 Bad Request...alibaba-ga</html>
> ```
>
> **排查过程(逐步排除):**
>
> | 步骤 | 做法 | 结论 |
> |---|---|---|
> | 1 | 注意到返回的是**网关 HTML**,不是智谱 API 的 JSON 报错 | 请求在网关就被挡了,没到鉴权 |
> | 2 | 本机用假 key 发送多种内容(含 `SELECT`、`DROP TABLE`、真实评估 prompt) | 全部到达 API → **不是请求内容被拦截** |
> | 3 | 在 Databricks 上用诊断代码分别发「假 key」和「真 key」 | 假 key → 401 JSON(到达 API);真 key → 400 HTML → **Databricks 网络路径没问题,问题在 key 本身** |
> | 4 | 检查 key 的格式(只看长度与字符类别,不打印值) | 长度 50,**含控制字符**;智谱 key 正常是 49 位 |
> | 5 | 去掉控制字符后再发 | **HTTP 200** → 确认 |
>
> - **原因:** 在 cmd 里运行 `databricks secrets put-secret` 时,在它的隐藏输入提示里按了 **Ctrl+V**。
>   这个提示不识别粘贴快捷键,把 Ctrl+V 当成控制字符 `\x16` 一起存进了 key。
>   带控制字符的 `Authorization` 请求头不合法,阿里云网关直接回 400
> - **解法(操作):** 改在 PowerShell 里读入变量 → 清除控制字符 → **确认长度为 49** → 用 `--string-value` 写入:
>   ```powershell
>   $k = Read-Host "粘贴 API key(鼠标右键粘贴)"
>   $k = ($k -replace '[\x00-\x1F\x7F]', '').Trim(); "长度: $($k.Length)"
>   databricks secrets put-secret llm zhipu_api_key --string-value $k
>   Remove-Variable k
>   ```
> - **解法(代码):** LLM 客户端在**发请求之前**检查 key 是否含控制字符,有则直接报「key 里有 N 个不可见的控制字符」,
>   报错里只给码点,不带 key 本身。以后不会再出现看不出原因的 HTML 400
> - **附带问题:** 第一次重存后诊断仍显示长度 50。用 CLI 查看 secret 元数据,确认它确实已被更新,
>   之后逐步确认长度为 49 再写入,恢复正常
> - **附带发现:** 诊断代码没有带 `do_sample=false`,同样发 `"hi"` 两次得到了不同的回复
>   (`"Hi 👋! I'm"` 和 `"你好👋！很高兴见到"`)—— 印证了预设里关闭采样是必要的

> **🔴 问题 12:删除诊断格时误删了「1. 选择并验证 LLM」的代码格**
>
> - **原因:** 诊断格插在该格正下方,两格相邻,点错了
> - **解法:** Edit → Undo delete cell;撤销不管用时,Git 窗口里对该 notebook 执行 Discard changes 从 Git 恢复

**结果:** ✅ 冒烟测试通过,`LLM 可用:True`。

---

### 阶段 E · 第 1 轮真实评估(0%)

**操作:** notebook 03 → Run all。

**作用:** 让模型回答 26 道题,与程序化生成的标准答案比对,记录每一轮尝试。

**业务含义:** 新助手第一次考试。

**结果:**

```
pass@1 0.0%  ·  pass (final) 0.0%  ·  holdout 0.0%  ·  avg attempts 1.27
hallucination 0.0%  ·  correct refusal 25.0%  ·  false refusal 0.0%
调用 32 次 · token 20,385
```

> **🔴 问题 13:22 道题 SQL 全部跑通,但结果都是 `None` / `0` / 空集**
>
> 模型写的 SQL:
> ```sql
> WHERE ... dt BETWEEN DATEADD(day, -7, CURRENT_DATE()) AND CURRENT_DATE()
> ```
>
> - **原因:** prompt 里**没有告诉模型「今天」是哪天**。模型把「上周」理解成真实世界(2026 年)的上周,
>   而数据是 2025 年 5~6 月的历史快照,查到的时间段里没有数据
> - **性质:** 这是**评估设计的遗漏**,不是模型能力问题。换一个人类分析师拿到同样的信息,也会这样写
> - **为什么 L1 没发现:** SQL 语法正确、执行成功,只是返回空 —— 自修复循环只看「跑不跑得通」,
>   发现不了「答得对不对」。这恰好证明了 L2 评估的必要性
> - **解法:**
>   1. prompt 增加**报告日历**:今天是 2025-06-30;「上周」= 06-23 至 06-29;「两周前」= 06-16 至 06-22;不要用 `CURRENT_DATE()`。
>      已验证与评估集标准答案的窗口完全一致
>   2. L1 增加一条确定性检查:设了报告日期时,SQL 中出现 `CURRENT_DATE` / `NOW()` 就作为可修复的错误反馈
> - **业务含义:** 真实系统里「上周」「上个月」由语义层或报表日历统一定义。这不是泄题,是业务口径

> **🔴 问题 14:回归闸门报「全量正确率退步 100% → 0%」**
>
> - **原因:** 闸门的基线取的是「时间上最近的上一次评估」,那一次是之前没接 LLM 时用 `ReferenceGenerator`
>   (永远给出正确答案、用于校验评估器本身)跑出的 100%。拿真实模型和它比,结论毫无意义
> - **解法:** 基线只取**同一个生成器(同一个模型)**的上一次评估

> **🔴 问题 15:本地测试中,守卫把 `DATEADD(day, …)` 的 `day` 判成「不存在的列」**
>
> - **原因:** 本地测试用 DuckDB 方言解析,DuckDB 解析器把时间单位 `day` 当成了列名
>   (Databricks 方言下不会误判)。误判会给模型错误的反馈「列 day 不存在」,自修复就会朝错误方向改
> - **解法:** 守卫忽略不是真实列名的时间单位关键字(`day`、`month`、`week` 等),两个方言下结论一致

**本轮产生的改进候选 v2 未合并:** 它加的示例是为了修日期问题产生的,而日期问题的正确解法是补上下文,不是塞示例。

---

### 阶段 F · 第 2 轮真实评估(63.6%)

**操作:** 推送修复 → Git folder Pull → notebook 03 → Run all。**模型不变,只加了报告日历。**

**结果:**

```
pass@1 50.0%  ·  pass (final) 63.6%  ·  holdout 50.0%  ·  avg attempts 1.35
hallucination 0.0%  ·  correct refusal 0.0%  ·  false refusal 0.0%
调用 34 次 · token 26,618
回归闸门:基线 = 同一模型的第 1 轮 → 通过 ✓
```

模型写的 SQL 变成了 `dt BETWEEN DATE '2025-06-23' AND DATE '2025-06-29'`,得到正确结果 142,905.20。

**失败分析方法:** 没有凭失败明细猜测,而是从 `nl2sql_attempts` 表里**只读查询**每道错题的实际 SQL
(通过本机 CLI 调用 SQL Statement API,使用 Serverless Starter Warehouse)。

> **🔴 问题 16:模型正确拒答了,却被判成「该拒答没拒答」**
>
> ```
> unanswerable_customer_email  #1  [guard]
>   CANNOT_ANSWER: The schema provided does not include an email address column.
> ```
>
> - **原因:** 模型把拒答语句包进了 ` ```sql ` 代码块。解析器看到代码块就当成 SQL 交给守卫,
>   守卫报「无法解析」,连错 3 轮后判为失败。**评分器的错,不是模型的错**
> - **同类问题:** 模型返回空内容时,解析器会把它当成「拒答」
> - **为什么必须先修:** 评分器错了,改进循环就会朝错误的方向优化
> - **解法:** 解析器识别代码块内的拒答标记;空回复不再算作拒答。两处都加了回归测试

**剩余 11 道错题的真实原因:**

| 类别 | 题数 | 模型实际写的 | 本质 |
|---|---|---|---|
| **维度取值写错** | 5 | `region='US'`、`market='Germany'`、`device='DESKTOP'`、`category='Paid Search'` | 知道列名,不知道列里有哪些值 |
| **转化率口径错** | 2 | 订单表 JOIN 会话表后相除再 ×100 → 恒为 100 | JOIN 后分母只剩有订单的会话 |
| **结果形状** | 1 | 按设备拆分写成一行两列 | 标准答案是每个设备一行,问题本身有歧义 |
| **编造答案** | 3 | 按日期硬 JOIN 广告表;把价格带当成本;虚构 `RETURNED` 状态 | 数据不支持时没有拒答 |

> **🔴 问题 17:幻觉率显示 0%,但模型实际编造了 3 次**
>
> - **原因:** 守卫只能识别「引用了不存在的列」,识别不了「用存在的列编出不存在的含义」
> - **处理:** 在文档中如实标注该指标会低估语义层面的幻觉;维度取值检查能拦下其中一部分(如 `RETURNED`)

> **🔴 问题 18:两轮评估都记成 `knowledge v1`,只看状态表无法解释 0% → 63.6%**
>
> - **原因:** 知识版本号只跟踪 `prompting.yaml`,而报告日历是代码里加的上下文
> - **解法:** 每次评估记录 **prompt 指纹**(prompt 模板 + 循环配置的哈希)。指纹不同 = 模型看到的上下文不同

---

### 阶段 G · 构建自主改进循环

#### G1. 识别差距

到第 2 轮为止,L1 和 L2 是自动的,但 **L3 是人在闭环**:失败原因是人看出来的,修改是人写的,合不合并是人定的。
按 loop engineering 的标准,这一层还不算闭环。

#### G2. 维度取值检查(L1 新增反馈)

**实现:** `src/rca/nl2sql/domains.py`。启动时**从仓库实时查询**每个维度列的实际取值(只查 ≤ 50 个取值的低基数列)。
SQL 执行前检查 `列 = '值'` / `列 IN (...)`,值不存在就作为可修复错误反馈:

```
region = 'US'        → 列 region 没有取值 'US'。全部取值:EMEA, NA。'US' 实际出现在列:market。
market = 'Germany'   → 列 market 没有取值 'Germany'。全部取值:DE, UK, US。
device = 'DESKTOP'   → 注意大小写,实际是 'desktop'。
```

**设计取舍:** 取值**不写进 prompt**。直接告诉模型所有取值当然能少错几道,但那是人替系统做了改进。
放在反馈里,模型只在写错时看到;能否沉淀成一条通用提示,交给改进循环完成。

**业务含义:** 相当于分析师写错筛选条件时,系统提示「这个字段里没有这个值,可选的是……」。

#### G3. 自主改进循环(L3)

**实现:** `src/rca/nl2sql/improve.py`,入口 `notebooks/04_improvement_loop.py`。

```
读训练集失败 + 台账里该模型以往被拒绝的提示
  → LLM 提议一条通用提示
  → 带着这条提示重跑整个评估集
  → 确定性闸门裁决
  → 采纳:知识版本 +1,成为新基线;拒绝:连同原因记入台账
  → 停止条件:达到目标 / 训练集无错题 / 连续 N 轮未采纳 / 预算用完 / 达到轮数上限
```

| 设计规则 | 原因 |
|---|---|
| 提议者**看不到留出集** | 否则留出集的题也能被「修」掉,闸门形同虚设 |
| 提议者**看不到参考 SQL** | 否则最省事的做法是把答案抄进提示 |
| 被拒绝的提示**跨循环记住**,并出现在下次提议的 prompt 中 | 否则确定性解码的模型会反复提同一个无效修改 |
| 训练集必须**严格多答对一道** | 持平不采纳,prompt 不会在没有证据时变长 |
| 留出集不退步、幻觉率与误拒率不升高 | 防止背答案、防止靠「一律拒答」刷分 |
| 每轮只加一条提示,其余全部不变 | 台账里每一行的指标变化都能归因到那一条提示 |
| 与已有或已拒绝提示重复的候选**不试跑** | 不为已知无效的修改花费调用预算 |

**业务含义:** 相当于带教新人的完整流程:复盘错题 → 提出规范 → 试行 → 看考核结果决定是否纳入团队规范 →
记录「试过但没用的做法」,避免重复踩坑。

#### G4. 状态层

| 变更 | 作用 |
|---|---|
| 新表 `nl2sql_improvement_ledger` | 每个候选(采纳 / 拒绝 / 重复)一行,含前后指标与裁决原因 |
| `nl2sql_eval_runs` 新增 `prompt_fingerprint`、`train_accuracy`、`holdout_accuracy` | 归因与迭代曲线 |
| **表结构自动迁移** | workspace 里已存在的旧表缺少新列,`CREATE TABLE IF NOT EXISTS` 不会更新旧表,插入会失败。启动时比对列并用 `ALTER TABLE ... ADD COLUMNS` 补齐 |

#### G5. 构建过程中发现并修掉的问题

> **🔴 问题 19:中文提示会被全部判为重复**
>
> - **原因:** 去重用的归一化只保留英文字母和数字,中文提示会被抹成空字符串,彼此相等
> - **解法:** 按 Unicode 词字符归一化;归一化结果为空时不参与去重

> **🔴 问题 20:`02` 第二次运行时,自主循环的测试结果会变**
>
> - **原因:** 测试会话开始时清空了测试 schema 里的状态表,但新加的台账表不在清空列表中。
>   上一次运行留下的「已拒绝提示」会让本次候选被判为重复
> - **解法:** 台账表加入清空列表

#### G6. 验证

- 测试:**298 passed**(新增 30 条,覆盖采纳 / 拒绝 / 重复、留出集不可见、不含参考 SQL、跨循环记忆、耐心停止、标准答案只算一次)
- 本地端到端模拟(与 notebook 04 相同的装配方式,假模型扮演生成器与提议者):

```
round 1: accepted   训练集 71.4% → 85.7%,留出集 80% → 100%
round 2: rejected   训练集没有多答对
round 3: duplicate  重复提议被拒绝过的提示 → 未试跑,直接拦下
停止原因:patience(连续 2 轮未采纳)
```

**状态:** 代码已推送(`dd3249c`),第一次真实运行见阶段 H。

---

### 阶段 H · 自主改进循环第一次真实运行(0 条采纳)

#### H1. 运行 `04_improvement_loop`

**操作:** Git folder 里 Pull → 打开 `04` → Connect Serverless → Run all(组件默认值:最多 4 轮,连续 2 轮不采纳即停)。

**作用:** 第一次让系统**自己**完成「找错 → 提改进 → 试行 → 裁决 → 记录」,不经过人。

**结果:**

```
基线          训练集 66.7%(14/21)· 留出集 80.0%(4/5)
第 1 轮 ✗ 拒绝  "Always verify column names and values against the actual data dictionary."
               训练集 14 → 12
第 2 轮 ✗ 拒绝  "Always verify the correct usage of dimensions and metrics in queries."
               训练集 14 → 12
停止原因 patience · 采纳 0 条 · 知识版本 v1 → v1
总用量 114 次调用 · 92,354 token · 836.7 秒
```

剩下的错题:

| 范围 | 题 | 表现 |
|---|---|---|
| 训练 | `cvr_w8`、`cvr_paid_search_w8` | 转化率返回 **100.0**,期望约 0.04 |
| 训练 | `gmv_by_device_w8` | 应返回 2 行(两种设备),只返回了 1 行 |
| 训练 | `gmv_paid_search_mobile_w8`、`uv_wow_change_us` | 3 轮都没写出能跑的 SQL,**原因一栏是空的**(问题 23) |
| 训练 | `unanswerable_campaign_gmv`、`unanswerable_profit_margin` | 本应拒答 |
| 留出 | `holdout_unanswerable_return_reason` | 本应拒答 |

#### H2. 解读:哪部分起了作用,哪部分没有

**起作用的是「裁决」这一半。** 两条候选都会让训练集少对 2 道。如果没有闸门,「让 LLM 改 prompt」会把两条有害的提示直接写进知识库。
这正是闸门存在的理由:**让不可靠的提议者也能安全地参与循环**。

**没起作用的是「提议」这一半。** 两条提示都是空话:没有点名任何表、列、取值或指标,
对着任何 text-to-SQL 系统都说得通,因此什么也修不好。
剩下的错题其实都有很具体的修法(例如 CVR 返回 100.0 说明公式或量纲错了),但提议者没有给出。

**这次运行本身暴露了三个问题**(问题 21~23),都出在循环的设计和观测上,而不是模型。

> **🔴 问题 21:提议者只给出空话**
>
> - **现象:** 两轮提示分别是「核对列名和取值」「核对维度和指标的用法」
> - **原因:** ① 一次把 7 道**各不相同**的错题(数值错、形状错、该拒答没拒答、跑不通)全交给提议者,
>   唯一能同时「覆盖」它们的只有一句空话;② prompt 只要求提示「通用」,没有要求「具体」;
>   ③ 提议者用的是和生成器相同的免费小模型
> - **解法:**
>   - **按失败形态分组,每轮只打一类。** 错题先被确定性地归到 `dimension_value` / `missed_refusal` /
>     `wrong_value` / `wrong_shape` / `no_runnable_sql` / `false_refusal` 之一;
>     每轮只把数量最多、且本次循环还没试过的那一类给提议者,其他类只告诉它「还有几道」
>   - **试跑前的具体性检查。** 提示里必须点名至少一个具体对象(表名、列名、指标、维度,或维度的实际取值,
>     或拒答标记 `CANNOT_ANSWER`),否则记为 `too_generic`,**不试跑**。
>     一次试跑约 37 次调用、4~5 分钟;这次两条空话合计花掉约 9 分钟和 7.5 万 token,检查本身几乎零成本。
>     被判为空话的提示同样进台账,下次提议时出现在「以往被拒绝」列表里
>   - **提议者可以单独换模型**(`proposer_model` 组件)。提议者每轮只调用 1 次,生成器每轮约 37 次;
>     在提议者上用更强的模型,成本增加很少
> - **验证:** 用这次真实被拒的两条原话做测试用例,两条都被判为 `too_generic`,且没有触发试跑

> **🔴 问题 22:21 道题上,±2 道的波动可能只是噪声**
>
> - **现象:** 两条内容不同、但同样空洞的提示,都恰好让训练集 14 → 12
> - **原因:** 模型在确定性解码下,prompt 里多一句话也会改变它在其他题上的输出。
>   也就是说,**在这个评估集规模上,「随便加一句话」本身就会带来约 2 道题的波动**。
>   原闸门只要求总分 +1,一次恰好 +1 的噪声就会被当成改进采纳
> - **解法:**
>   - 闸门新增**归因检查**:这一轮针对的那类错题里,至少要修好一道。总分涨了但涨在别处,按噪声处理,不采纳
>   - 每轮记录**修好了哪些题、改坏了哪些题**(`fixed` / `broken`,写进台账)。
>     总分只是两者之差;这次两轮都是 −2,但当时没有记录具体是哪几道
> - **仍然存在的局限:** 评估集只有 21 道训练题,闸门的分辨率就是 1 道题。更可靠的做法是扩充评估集,
>   或对同一候选用不同的题目顺序多跑几次取稳定结果 —— 都会成倍增加成本,目前没有做

> **🔴 问题 23:「3 轮都没写出能跑的 SQL:」后面是空的**
>
> - **现象:** `gmv_paid_search_mobile_w8`、`uv_wow_change_us` 的失败原因一栏为空
> - **原因:** 评分器写失败原因时,只取「引擎报错」和「守卫信息」两个字段。
>   而维度取值检查、报告日历检查这类**上下文检查**的反馈存在另一个字段里,没有被取到。
>   两个检查都是后来加的,加的时候没有同步更新评分器
> - **影响:** 人看不到原因;失败分组也可能把这类题归错类。
>   (提议者本身看得到这些反馈,它读的是另一个完整字段。)
> - **解法:** 失败原因改为取最后一轮的**全部**反馈。新增测试:被取值检查连续拦下 3 轮的题,
>   原因里必须含具体的错误取值
> - **这两道题的真实原因:** 状态表 `nl2sql_attempts` 的 `guard_errors` 列里有完整记录,可以用下面的 SQL 查

```sql
-- 两道「跑不通」的题,每一轮具体被什么拦下
SELECT case_id, attempt_no, failure_kind, guard_errors, execution_error, generation_error
FROM workspace.gmv_rca.nl2sql_attempts
WHERE run_id = 'loop_07704059_r0'
  AND case_id IN ('gmv_paid_search_mobile_w8', 'uv_wow_change_us')
ORDER BY case_id, attempt_no;

-- 两条被拒的提示分别让哪几道题从对变错
SELECT case_id,
       MAX(CASE WHEN run_id = 'loop_07704059_r0' THEN verdict END) AS baseline,
       MAX(CASE WHEN run_id = 'loop_07704059_r1' THEN verdict END) AS round1,
       MAX(CASE WHEN run_id = 'loop_07704059_r2' THEN verdict END) AS round2
FROM workspace.gmv_rca.nl2sql_attempts
WHERE run_id LIKE 'loop_07704059_r%'
GROUP BY case_id
HAVING COUNT(DISTINCT verdict) > 1
ORDER BY case_id;
```

#### H3. 改进后的循环

```
读训练集失败 → 按失败形态分组 → 选一类(数量最多、本次还没试过)
  → LLM 针对这一类提一条提示
  → 具体性检查(没点名具体对象 → too_generic,不试跑)          ← 新增,几乎零成本
  → 重复检查(与已有 / 已拒绝提示重复 → duplicate,不试跑)
  → 带着这条提示重跑整个评估集                                 ← 昂贵,约 4~5 分钟
  → 闸门:训练集 +1 且 针对的题至少修好一道 且 留出集不退步 且 幻觉/误拒不升高
  → 记录修好 / 改坏了哪些题;没采纳就换下一类错题
```

**业务含义:** 对应带教新人时的两条常识:复盘时**一次只讲一类错误**,而不是泛泛地说「以后细心点」;
新人成绩变好了,要先确认**是不是刚讲的那类题变好了**,而不是碰巧别的题蒙对了。

#### H4. 验证

- 本地测试 **312 passed**(新增 14 条,包括用两条真实被拒的原话验证具体性检查)
- `02` 第 2 段的期望数字更新为 **`133 passed, 1 skipped`**
- **状态:** 尚未在 Databricks 上重新运行 `04`

---

### 阶段 I · 从「一步步执行」到可用的问答助手

#### I1. 识别差距

`00`~`04` 是研发和评估系统:证明 SQL 写得对、能持续变准。但业务同事没法用 ——
要打开 notebook、改代码、逐格运行。要上线,缺三样东西:

| 缺什么 | 之前 | 现在 |
|---|---|---|
| 输入口 | 改 notebook 代码 | notebook `05` 改一个输入框;网页聊天(Databricks Apps);定时 Job |
| 问题分流 | 只会查数 | 「为什么……」走归因,其余走查数 |
| 线上日志与反馈 | 无 | `assistant_requests` / `assistant_feedback`,失败和 👎 进待复核队列 |

#### I2. 关键设计

**归因不交给 LLM。** 归因要的是一组互相勾稽的数字(各因子贡献之和 = 总变化)。
让 LLM 写多条 SQL 再拼,口径稍有不齐就对不上账。所以「为什么」类问题直接调用分解内核,
叙述用模板生成 —— **回答里没有一个数字是模型写的**。这条路径完全不需要 LLM。

**分流是确定性的。** 关键词加业务词表(`knowledge/assistant/synonyms.yaml`,业务方维护、走 PR)。
维度取值不用列:启动时从仓库实时查出,问题里直接出现 `US`、`mobile` 就能识别。

**假设必须说出来。** 每个回答都附带它采用的假设:用的哪两周、按什么过滤、没点名指标时用了默认指标、
没识别出时间范围时按上周处理。答案可以不完美,但不能悄悄替提问人做决定。

**只允许不破坏恒等式的过滤。** 品类要 join 商品表,只在订单侧可用;按它过滤 GMV,
UV 和 GMV 就落在不同口径上,UV × CVR × AOV 不再成立。所以只有在全部相关事实表上都是原生列的维度
(渠道、设备、市场、大区)才会被识别为过滤条件。

**线上身份与权限。** 网页入口用 Databricks Apps 自己的 service principal(OAuth)连 SQL Warehouse,
不经手任何个人 token;Unity Catalog 里只授予这个 schema 的读权限和写日志的权限。

**线上日志是下一轮评估的原料。** 没答出来的、被点 👎 的问题进入待复核队列,
人确认口径后加进 `eval/cases.yaml`。这样评估集会从线上真实问题里长大 ——
这正是阶段 H 暴露的「21 道题、闸门分辨率只有 1 道」的长期解法。

#### I3. 构建过程中发现并修掉的问题

| 问题 | 原因 | 解法 |
|---|---|---|
| 维度贡献显示成实际的 100 倍 | 因子分解的贡献是小数,维度分解的贡献已经是百分点,两者单位不同 | 叙述层分别处理,测试断言「各因子贡献之和 = 总变化」 |
| 查数结果显示成 `29312.60`,没有千分位 | 引擎返回 `Decimal`,格式化只认 `int` / `float` | 数值判断加上 `Decimal` |
| 网页上日期被划掉 | Markdown 把两个 `~` 之间的内容渲染成删除线(`06-23~06-29 vs 06-16~06-22`) | 展示前转义 `~` |
| 常驻进程的 SQL 审计记录无限增长 | 执行器把每条 SQL 记进列表,批处理跑完就退出无所谓,常驻服务会一直涨 | 改为只保留最近 500 条 |

#### I4. 验证

- 本地测试 **351 passed**(新增 39 条:分流 28 条(不需要数据库)、OAuth 连接 2 条、端到端 9 条,包括归因数字与直接写 SQL 算出的一致、
  查数经过自修复、失败和 👎 进入待复核队列、线上提示来自合并后的 `prompting.yaml`)
- 网页入口在本机(DuckDB + 样例数据)用浏览器实际点过:归因、过滤、明细表、👎 反馈、待复核队列
- `02` 的期望数字更新为 **`198 passed, 1 skipped`** 和 **`142 passed, 1 skipped`**
- **状态:** 尚未在 Databricks 上运行 `05`;workspace 是否支持 Apps 待确认

部署步骤见 [DEPLOYMENT.md](DEPLOYMENT.md)。

---

## 四、问题清单(按类型汇总)

| # | 问题 | 类型 | 根因一句话 | 解法一句话 |
|---|---|---|---|---|
| 1 | `remote origin already exists` | Git 操作 | 地址里残留模板占位符 | `git remote set-url` |
| 2 | catalog `main` 不存在 | 平台差异 | Free Edition 默认 catalog 是 `workspace` | 统一改默认值 |
| 3 | 00/01 缺依赖 | 平台差异 | serverless 不保证预装 pydantic v2 | 加 `%pip install` |
| 4 | 离线测试依赖 duckdb | 平台差异 | Databricks 不装 duckdb | `importorskip` |
| 5 | 02 重跑失败 | 状态管理 | 状态表跨运行保留 | 测试前清空测试 schema |
| 6 | notebook 认证可能 401 | 平台差异 | Runtime 认证下 `config.token` 可能为空 | 用 `config.authenticate()` |
| 7 | pytest `Errno 95` | **平台差异** | `/Workspace` 不支持 `__pycache__` | 关字节码写入,缓存引到 `/tmp` |
| 8 | CLI 命令找不到 | 本机环境 | PATH 未刷新;host 带了路径 | 重开终端;只填域名 |
| 9 | 令牌存不进凭据管理器 | 本机环境 | Windows keyring 写入失败 | `DATABRICKS_AUTH_STORAGE=plaintext` |
| 10 | 登录成功仍无凭据 | 本机环境 | 配置文件 profile 名带空格 | 清理 `.databrickscfg` |
| 11 | 智谱 HTML 400 | **凭据** | Ctrl+V 把 `\x16` 存进了 key | PowerShell 清洗后重存;客户端预检 |
| 12 | 误删代码格 | 操作 | 相邻格点错 | Undo / Git discard |
| 13 | 22 道题返回空 | **评估设计** | prompt 缺报告日期 | 报告日历 + L1 检查 |
| 14 | 闸门基线错误 | **评估系统** | 与参考生成器比较 | 只比同一模型 |
| 15 | `day` 被误判为幻觉列 | **评估系统** | DuckDB 解析器差异 | 忽略时间单位关键字 |
| 16 | 正确拒答被判失败 | **评估系统** | 拒答包在代码块里 | 解析器识别 |
| 17 | 幻觉率低估 | **评估系统** | 守卫识别不了语义编造 | 如实标注 + 取值检查 |
| 18 | 版本号无法解释分数变化 | 可观测性 | 上下文变更未记录 | prompt 指纹 |
| 19 | 中文提示全判重复 | 代码缺陷 | 归一化抹掉中文 | Unicode 归一化 |
| 20 | 02 重跑结果变化 | 状态管理 | 台账表未清空 | 加入清空列表 |
| 21 | 提议者只给出空话 | **循环设计** | 7 类错题一起给、只要求「通用」 | 按失败形态分组 + 试跑前具体性检查 |
| 22 | ±2 道的波动可能只是噪声 | **评估系统** | 21 道题上,多一句话就能改变别的题 | 归因检查 + 记录修好/改坏的题 |
| 23 | 跑不通的题原因为空 | 可观测性 | 评分器漏取上下文检查的反馈 | 取最后一轮全部反馈 |

**按类型:** 平台差异 5 · 本机环境 3 · **评估系统自身 5** · 评估设计 1 · 循环设计 1 · 状态管理 2 · 凭据 1 · 其他 5(Git 操作、误删、可观测性 ×2、代码缺陷)

---

## 五、经验总结

**1. 本地测试通过 ≠ 目标平台能跑。**
Errno 95、catalog 名、Runtime 认证方式,这些问题本地一个都测不出来。
项目因此设计了三层验证:本地 DuckDB(算术)→ sqlglot(方言语法)→ `02` 在 Spark 上跑(真机)。

**2. 先怀疑评估系统,再怀疑模型。**
第 1 轮 0% 的主因是 prompt 缺少业务上下文;第 2 轮里有一道「失败」其实是评分器解析错了。
如果直接开始调 prompt,会在一把坏掉的尺子上优化。`ReferenceGenerator`(永远正确)拿满分,是评估器的基本自检。

**3. 看证据,不猜。**
HTML 400 用「假 key vs 真 key」对照实验逐步排除;剩余错题从状态表里查出实际 SQL 再归类。
两次都比根据报错文字推测更快得到正确结论。

**4. 一次只改一个变量。**
第 1 轮到第 2 轮只加了报告日历、模型不变,所以 0% → 63.6% 可以完整归因到这一个改动。
自主改进循环沿用同一原则:每轮只加一条提示。

**5. 能跑通但必然错的 SQL,是最危险的失败。**
用 `CURRENT_DATE()` 查历史数据、`market='Germany'`,语法都对、执行都成功、结果都是空。
把这类错误从「L2 考试时才发现」前移到「L1 执行前就拦下并反馈」,是本次最有价值的两项改进。

**6. 凭据问题要能自我诊断。**
一个不可见字符造成的 HTML 400 花了最多排查时间。现在客户端在发请求前就会指出「key 里有控制字符」。

**7. 自主循环里,裁决比提议更重要。**
第一次真实运行 0 条采纳,但闸门挡下了两条会让准确率下降的提示。提议者弱,循环依然安全;
反过来,如果闸门不可靠,提议者越强,越可能把有害的修改写进知识库。
提议的质量可以靠分组、具体性检查、换更强的模型逐步提升,裁决的可靠性必须一开始就有。

**8. 先确认分数变化是信号还是噪声。**
两条内容不同的空话都让分数变了 2 道题,这个数字本身就是评估集的噪声量级。
只看总分的闸门,会把运气当成改进。

---

## 六、下一步

1. ~~在 Databricks 上运行 `04_improvement_loop`~~ ✅ 已完成(阶段 H:0 条采纳,闸门挡下 2 条有害提示)
2. Pull 最新代码,**重新运行 `02`**(期望 `198 passed, 1 skipped` 和 `142 passed, 1 skipped`),再**重新运行 `04`**:
   - 验证跨循环记忆:上次被拒的两条提示出现在提议者 prompt 的「Previously rejected」里
   - 验证新机制:每轮打印「针对 …」「修好 / 改坏」,空话显示为「⊘ 太笼统(未试跑)」
   - 可选:`max_rounds` / `patience` 调到 6 / 4,让它把各类错题都试一遍
3. 可选:`proposer_model` 填一个更强的模型,对比提议质量(只换提议者,生成器不变,仍然只改一个变量)
4. 有提示被采纳后,写回 `prompting.yaml` 并提交,形成知识版本的 Git 记录
5. 运行 `05_assistant` 提几个问题;确认 workspace 有没有 Apps,有就按 DEPLOYMENT.md 部署网页入口
6. 用 `databricks.yml` 部署三个 Job(每日评估、每周改进、每周归因),先手动跑一次再打开调度

---

## 附录:提交记录与诊断代码

### 提交记录

| 提交 | 时间 | 内容 |
|---|---|---|
| `0d5cfb5` | 09-17 00:14 | text-to-SQL 循环:自修复、评估、改进闸门 |
| `bfb5224` | 09-17 16:27 | 智谱接入、Databricks 运行时修复、默认 catalog 改为 workspace |
| `a8a6945` | 09-17 17:34 | 修复 Databricks 上 pytest 的 Errno 95 |
| `376a351` | 09-17 18:55 | 报告日历、同模型回归基线、时间单位守卫修复 |
| `7921c12` | 09-17 19:32 | 修复评分器:代码块内拒答与空回复的误判 |
| `dd3249c` | 09-17 19:48 | 自主改进循环、维度取值检查、状态表迁移 |

### 诊断代码:排查 LLM 请求被拒(不打印密钥)

在 notebook 03 中临时插入运行,排查完**立即删除**。

```python
import json, urllib.request, urllib.error, unicodedata
from rca.nl2sql.llm import ZHIPU_BASE_URL

key = dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)
print("key 长度:", len(key))
print("含控制字符:", [(i, hex(ord(c))) for i, c in enumerate(key) if unicodedata.category(c).startswith("C")])

def probe(label, token):
    body = json.dumps({"model": "glm-4-flash", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode()
    req = urllib.request.Request(ZHIPU_BASE_URL + "/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            code, text = r.status, r.read()[:120]
    except urllib.error.HTTPError as e:
        code, text = e.code, e.read()[:120]
    kind = "HTML(被网关挡下)" if text.lstrip().startswith(b"<") else "JSON(到达 API)"
    print(f"{label}: HTTP {code}  {kind}")

probe("假 key", "bogus.key")        # 到达 API → 网络路径正常
probe("真 key", key)                # 若为 HTML 400 → 问题在 key 本身
```

判读:**假 key 到达 API、真 key 被网关挡下** → 检查 key 格式;**两者都被挡下** → 网络路径问题。
