"""样例数据生成(建表 DDL + 集合式 INSERT)。

设计要点
--------

**一份生成器,两个方言。** 所有 SQL 由本模块构造,经 :mod:`rca.dialects`
渲染成 Databricks 或 DuckDB 两种写法。``sql/setup/*.sql`` 是本模块渲染出的
**产物**(由 ``sql/setup/render.py`` 重新生成),因此建表脚本与
``knowledge/tables.yaml`` 不可能漂移。

**集合式,不是逐行插入。** 每张表一条 ``INSERT ... SELECT``,全部在仓库侧计算。
Free Edition 只有一个 2X-Small warehouse,逐行插入 33 万行会直接烧掉配额。

**确定性,不用 ``rand()``。** 所有随机量都是 ``hash(种子字符串)`` 的派生值,
种子来自行本身的业务键。这样:

* 同一方言下重跑,数据逐行一致(简报约束 3);
* ``fact_orders`` 可以从 ``fact_sessions`` **重新推导**出来 —— 订单级随机量
  全部由 ``session_id`` 派生,不需要中间表或临时视图。

**会话与订单的对齐(最关键的口径决定)。**

    UV   = 会话数                    (fact_sessions)
    CVR  = 已完成订单数 / 会话数
    AOV  = GMV / 已完成订单数
    => UV × CVR × AOV = GMV          精确成立

为让这个恒等式在**数据层**也毫无歧义,生成器保证:
**一个转化会话恰好产生一个订单**。于是 ``fact_sessions.converted``
(定义为「该会话产生了一笔 COMPLETED 订单」)的计数与 ``orders`` 指标完全相等,
CVR 可以从两侧交叉验证 —— 这正是 ``tests/test_sample_data.py`` 做的事。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from .dialects import SqlDialect
from .knowledge import Knowledge, TableDef

# ---------------------------------------------------------------------------
# 业务画像(刻意写死:样例数据必须可复现,不接受运行期随机配置)
# ---------------------------------------------------------------------------
# key -> (流量份额, CVR 乘数, AOV 乘数, 代码)
CHANNELS: dict[str, tuple[float, float, float, str]] = {
    "Paid Search": (0.30, 0.90, 1.00, "PS"),
    "Organic": (0.28, 1.15, 1.05, "OR"),
    "Email": (0.10, 1.40, 1.10, "EM"),
    "Social": (0.17, 0.60, 0.90, "SO"),
    "Direct": (0.15, 1.25, 1.15, "DI"),
}
DEVICES: dict[str, tuple[float, float, float, str]] = {
    "mobile": (0.62, 0.80, 0.88, "M"),
    "desktop": (0.38, 1.30, 1.20, "D"),
}
# key -> (流量份额, CVR 乘数, AOV 乘数, 大区)
MARKETS: dict[str, tuple[float, float, float, str]] = {
    "US": (0.55, 1.00, 1.00, "NA"),
    "UK": (0.25, 0.95, 0.95, "EMEA"),
    "DE": (0.20, 0.90, 0.92, "EMEA"),
}

CATEGORIES: list[tuple[str, float]] = [
    ("Apparel", 0.24),
    ("Electronics", 0.18),
    ("Home", 0.16),
    ("Beauty", 0.14),
    ("Sports", 0.13),
    ("Toys", 0.09),
    ("Grocery", 0.06),
]
# 价格带 -> (占比, 基准客单价)
PRICE_TIERS: list[tuple[str, float, float]] = [
    ("BUDGET", 0.40, 28.0),
    ("MID", 0.42, 78.0),
    ("PREMIUM", 0.18, 190.0),
]
USER_TIERS: list[tuple[str, float]] = [
    ("NEW", 0.45),
    ("RETURNING", 0.42),
    ("VIP", 0.13),
]

BASE_CVR = 0.045
"""基准转化率(会话 -> 已完成订单)。"""

P_COMPLETED = 0.92
P_CANCELLED = 0.97  # 累积概率:[0.92, 0.97) 为 CANCELLED,其余为 REFUNDED
"""订单状态分布。GMV/orders 只统计 COMPLETED,这里留出 8% 的非完成订单,
让「数据问题 vs 业务问题」的区分在 V1 注入异常时有可操作的抓手。"""

# 周内节律,索引 0 = 周一。刻意让周末高于工作日,使日粒度数据有真实的波动结构。
WEEKDAY_FACTORS = [1.00, 0.98, 0.97, 0.99, 1.04, 1.12, 1.08]

CTR = 0.030
CPC = 0.85


@dataclass(frozen=True)
class SampleDataConfig:
    """样例数据规模。默认值面向 Databricks Free Edition 的配额。"""

    start_date: date = date(2025, 5, 5)
    """必须是周一 —— 周内节律的索引由「距 start_date 的天数 % 7」得到。"""

    weeks: int = 8
    daily_sessions: int = 6000
    """全站日均会话数。默认 6000 × 56 天 ≈ 33.6 万行,远低于百万行上限。"""

    n_users: int = 50_000
    n_products: int = 240

    def __post_init__(self) -> None:
        if self.start_date.weekday() != 0:
            raise ValueError("start_date 必须是周一,否则 WEEKDAY_FACTORS 的索引会错位")
        if self.weeks < 2:
            raise ValueError("至少需要 2 周数据才能做同比/环比")
        for value, label in ((self.daily_sessions, "daily_sessions"), (self.n_users, "n_users"), (self.n_products, "n_products")):
            if value <= 0:
                raise ValueError(f"{label} 必须为正")

    @property
    def n_days(self) -> int:
        return self.weeks * 7

    @property
    def end_date(self) -> date:
        return date.fromordinal(self.start_date.toordinal() + self.n_days - 1)

    @property
    def user_bounds(self) -> dict[str, tuple[int, int]]:
        """按市场把 user_id 切成互不重叠的区间,保证会话的 market 与用户的 market 一致。"""
        bounds: dict[str, tuple[int, int]] = {}
        lo = 1
        remaining = list(MARKETS.items())
        for idx, (market, (share, *_rest)) in enumerate(remaining):
            if idx == len(remaining) - 1:
                hi = self.n_users
            else:
                hi = lo + max(1, round(self.n_users * share)) - 1
            bounds[market] = (lo, min(hi, self.n_users))
            lo = bounds[market][1] + 1
        return bounds


# ---------------------------------------------------------------------------
# SQL 片段构造工具
# ---------------------------------------------------------------------------
def _case_numeric(column: str, mapping: dict[str, float]) -> str:
    """``CASE <column> WHEN 'a' THEN 1.0 ... END``(数值查表)。"""
    arms = " ".join(
        f"WHEN {SqlDialect.string_literal(k)} THEN {v!r}" for k, v in mapping.items()
    )
    return f"(CASE {column} {arms} END)"


def _case_string(column: str, mapping: dict[str, str]) -> str:
    arms = " ".join(
        f"WHEN {SqlDialect.string_literal(k)} THEN {SqlDialect.string_literal(v)}"
        for k, v in mapping.items()
    )
    return f"(CASE {column} {arms} END)"


def _case_index(index_expr: str, values: list[float]) -> str:
    """按整数索引查表。"""
    arms = " ".join(f"WHEN {i} THEN {v!r}" for i, v in enumerate(values))
    return f"(CASE {index_expr} {arms} END)"


def _weighted_pick(rand_expr: str, items: list[tuple[str, float]]) -> str:
    """按权重把 [0,1) 随机数映射成类别值。权重会自动归一化。"""
    total = sum(w for _, w in items)
    arms: list[str] = []
    cumulative = 0.0
    for label, weight in items[:-1]:
        cumulative += weight / total
        arms.append(
            f"WHEN {rand_expr} < {cumulative:.6f} THEN {SqlDialect.string_literal(label)}"
        )
    arms.append(f"ELSE {SqlDialect.string_literal(items[-1][0])}")
    return "(CASE " + " ".join(arms) + " END)"


def _values_cte(alias: str, column: str, values: list[str]) -> str:
    unions = "\n    UNION ALL ".join(
        f"SELECT {SqlDialect.string_literal(v)} AS {column}" for v in values
    )
    return f"{alias} AS (\n    {unions}\n  )"


def _int_series(d: SqlDialect, bound: int, alias: str) -> str:
    """产生 ``1..bound`` 的 BIGINT 序列子查询。

    两个方言的生成器函数(``explode`` / ``unnest``)都只能出现在 SELECT 列表的
    最外层,不能被 ``CAST(...)`` 包住 —— 所以这里必须多套一层子查询。
    """
    return (
        f"(SELECT CAST(n AS BIGINT) AS {alias} "
        f"FROM (SELECT {d.explode_int_range(str(bound))} AS n) _s)"
    )


def _floor_int(expr: str, sql_type: str = "BIGINT") -> str:
    """向下取整后转整数。

    **必须显式 FLOOR。** DuckDB 的 ``CAST(DOUBLE AS BIGINT)`` 是四舍五入,
    Spark/Databricks 是截断 —— 直接 CAST 会让两个方言产生不同的取值范围,
    ``1 + CAST(r * 240 AS BIGINT)`` 在 DuckDB 上会溢出到 241,
    进而在 join dim_product 时变成孤儿行。
    """
    return f"CAST(FLOOR({expr}) AS {sql_type})"


def _seed(literal: str, *parts: str) -> str:
    """拼一个用于 ``rand01`` 的确定性种子表达式。"""
    return " || ".join([SqlDialect.string_literal(literal), *parts])


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
def render_create_table(table: TableDef, dialect: SqlDialect) -> str:
    """由 ``tables.yaml`` 的 :class:`TableDef` 渲染建表语句。

    用 ``CREATE OR REPLACE TABLE`` 而非 ``CREATE TABLE IF NOT EXISTS``:
    整套 setup 脚本必须是幂等的,重跑一次得到一模一样的数据,
    而不是把确定性的行再插一遍。
    """
    cols = []
    for name, col in table.columns.items():
        null_sql = "" if col.nullable else " NOT NULL"
        cols.append(f"  {name} {dialect.type_sql(col.type)}{null_sql}")
    suffix = dialect.create_table_suffix()
    tail = f"\n{suffix}" if suffix else ""
    return (
        f"CREATE OR REPLACE TABLE {dialect.qualify(table.name)} (\n"
        + ",\n".join(cols)
        + f"\n){tail}"
    )


def render_schema_ddl(dialect: SqlDialect) -> list[str]:
    stmts: list[str] = []
    if dialect.catalog:
        stmts.append(f"CREATE SCHEMA IF NOT EXISTS {dialect.catalog}.{dialect.schema}")
    elif dialect.schema != "main":
        stmts.append(f"CREATE SCHEMA IF NOT EXISTS {dialect.schema}")
    return stmts


# ---------------------------------------------------------------------------
# 各表的数据生成
# ---------------------------------------------------------------------------
def _dim_product_sql(d: SqlDialect, cfg: SampleDataConfig) -> str:
    pid = d.cast_string("product_id")
    category = _weighted_pick(d.rand01(_seed("cat|", pid)), CATEGORIES)
    sub_idx = f"1 + {_floor_int(d.rand01(_seed('sub|', pid)) + ' * 4', 'INT')}"
    tier = _weighted_pick(
        d.rand01(_seed("tier|", pid)), [(t, w) for t, w, _ in PRICE_TIERS]
    )
    return f"""INSERT INTO {d.qualify('dim_product')}
SELECT
  product_id,
  category,
  category || '-' || {d.cast_string('sub_idx')} AS sub_category,
  price_tier
FROM (
  SELECT
    product_id,
    {category} AS category,
    {sub_idx} AS sub_idx,
    {tier} AS price_tier
  FROM {_int_series(d, cfg.n_products, 'product_id')} p
) q"""


def _dim_user_sql(d: SqlDialect, cfg: SampleDataConfig) -> str:
    bounds = cfg.user_bounds
    uid = d.cast_string("user_id")
    market_case_arms = " ".join(
        f"WHEN user_id <= {hi} THEN {SqlDialect.string_literal(m)}"
        for m, (_lo, hi) in list(bounds.items())[:-1]
    )
    last_market = list(bounds)[-1]
    market_case = (
        f"(CASE {market_case_arms} ELSE {SqlDialect.string_literal(last_market)} END)"
    )
    region_case = _case_string("market", {m: v[3] for m, v in MARKETS.items()})
    signup_days = "-" + _floor_int(d.rand01(_seed("signup|", uid)) + " * 730", "INT")
    signup = d.date_shift(d.date_literal(cfg.start_date), signup_days)
    user_tier = _weighted_pick(d.rand01(_seed("utier|", uid)), USER_TIERS)
    return f"""INSERT INTO {d.qualify('dim_user')}
SELECT
  user_id,
  {region_case} AS region,
  market,
  {signup} AS signup_dt,
  {user_tier} AS user_tier
FROM (
  SELECT user_id, {market_case} AS market
  FROM {_int_series(d, cfg.n_users, 'user_id')} u
) m"""


def _cvr_expr(d: SqlDialect, dt: str, channel: str, device: str, market: str) -> str:
    """群组转化率。**只依赖 (dt, channel, device, market)**。

    这条约束是刻意的:``fact_orders`` 需要从 ``fact_sessions`` 重新推导出
    「这个会话有没有下单」,而 ``fact_sessions`` 上恰好只有这四列可用。
    """
    noise_seed = f"concat_ws('|', {d.cast_string(dt)}, {channel}, {device}, {market}, 'cvr')"
    return (
        f"({BASE_CVR!r}"
        f" * {_case_numeric(channel, {k: v[1] for k, v in CHANNELS.items()})}"
        f" * {_case_numeric(device, {k: v[1] for k, v in DEVICES.items()})}"
        f" * {_case_numeric(market, {k: v[1] for k, v in MARKETS.items()})}"
        f" * (0.94 + 0.12 * {d.rand01(noise_seed)}))"
    )


def _fact_sessions_sql(d: SqlDialect, cfg: SampleDataConfig) -> str:
    di = d.day_index("d.dt", cfg.start_date)
    sessions_noise_seed = (
        "concat_ws('|', " + d.cast_string("d.dt") + ", c.channel, v.device, m.market, 'sessions')"
    )
    n_sessions = (
        f"CAST(GREATEST(1, ROUND("
        f"{cfg.daily_sessions}"
        f" * {_case_numeric('c.channel', {k: v[0] for k, v in CHANNELS.items()})}"
        f" * {_case_numeric('v.device', {k: v[0] for k, v in DEVICES.items()})}"
        f" * {_case_numeric('m.market', {k: v[0] for k, v in MARKETS.items()})}"
        f" * {_case_index(f'({di} % 7)', WEEKDAY_FACTORS)}"
        f" * (1 + 0.0015 * {di})"
        f" * (0.92 + 0.16 * {d.rand01(sessions_noise_seed)})"
        f")) AS INT)"
    )
    session_id = (
        "concat_ws('_', 's', replace("
        + d.cast_string("dt")
        + ", '-', ''), "
        + _case_string("channel", {k: v[3] for k, v in CHANNELS.items()})
        + " || "
        + _case_string("device", {k: v[3] for k, v in DEVICES.items()})
        + ", market, lpad("
        + d.cast_string("i")
        + ", 6, '0'))"
    )
    bounds = cfg.user_bounds
    r_user = d.rand01(_seed("user|", "session_id"))
    user_arms = " ".join(
        f"WHEN {SqlDialect.string_literal(m)} THEN "
        f"{lo} + {_floor_int(f'{r_user} * {hi - lo + 1}')}"
        for m, (lo, hi) in bounds.items()
    )
    user_id = f"(CASE market {user_arms} END)"
    region_case = _case_string("market", {m: v[3] for m, v in MARKETS.items()})
    r_conv = d.rand01("session_id || '|conv'")
    r_status = d.rand01("session_id || '|status'")
    cvr = _cvr_expr(d, "dt", "channel", "device", "market")
    return f"""INSERT INTO {d.qualify('fact_sessions')}
WITH
  dates AS ({d.date_series(cfg.start_date, cfg.end_date)}),
  {_values_cte('channels', 'channel', list(CHANNELS))},
  {_values_cte('devices', 'device', list(DEVICES))},
  {_values_cte('markets', 'market', list(MARKETS))},
  grid AS (
    SELECT d.dt, c.channel, v.device, m.market,
           {n_sessions} AS n_sessions
    FROM dates d
    CROSS JOIN channels c
    CROSS JOIN devices v
    CROSS JOIN markets m
  ),
  expanded AS (
    SELECT g.dt, g.channel, g.device, g.market,
           {d.explode_int_range('g.n_sessions')} AS i
    FROM grid g
  ),
  sess AS (
    SELECT dt, channel, device, market, {session_id} AS session_id
    FROM expanded
  )
SELECT
  session_id,
  {user_id} AS user_id,
  channel,
  device,
  {region_case} AS region,
  market,
  -- converted 的定义:该会话产生了一笔 COMPLETED 订单。
  -- 与 fact_orders 的生成条件逐字相同,因此 COUNT(converted) 必然等于 orders 指标。
  (({r_conv}) < {cvr} AND ({r_status}) < {P_COMPLETED!r}) AS converted,
  dt
FROM sess"""


def _fact_orders_sql(d: SqlDialect, cfg: SampleDataConfig) -> str:
    cvr = _cvr_expr(d, "dt", "channel", "device", "market")
    r_conv = d.rand01("session_id || '|conv'")
    r_status = d.rand01("session_id || '|status'")
    r_prod = d.rand01("session_id || '|prod'")
    r_amt = d.rand01("session_id || '|amt'")
    price_base = _case_numeric("p.price_tier", {t: base for t, _w, base in PRICE_TIERS})
    amount = (
        f"CAST(ROUND({price_base}"
        f" * {_case_numeric('o.channel', {k: v[2] for k, v in CHANNELS.items()})}"
        f" * {_case_numeric('o.device', {k: v[2] for k, v in DEVICES.items()})}"
        f" * {_case_numeric('o.market', {k: v[2] for k, v in MARKETS.items()})}"
        f" * (0.70 + 0.60 * o.r_amt), 2) AS {d.type_sql('decimal(12,2)')})"
    )
    status = (
        f"(CASE WHEN o.r_status < {P_COMPLETED!r} THEN 'COMPLETED'"
        f" WHEN o.r_status < {P_CANCELLED!r} THEN 'CANCELLED'"
        f" ELSE 'REFUNDED' END)"
    )
    return f"""INSERT INTO {d.qualify('fact_orders')}
-- 订单完全由 fact_sessions 推导:转化判定与订单级随机量都从 session_id 派生,
-- 因此这条语句可以独立重跑,结果不变(简报约束 3)。
WITH ordered AS (
  SELECT
    s.session_id, s.user_id, s.channel, s.device, s.region, s.market, s.dt,
    1 + {_floor_int(f'{r_prod} * {cfg.n_products}')} AS product_id,
    {r_status} AS r_status,
    {r_amt} AS r_amt
  FROM {d.qualify('fact_sessions')} s
  WHERE ({r_conv}) < {cvr}
)
SELECT
  'o' || substr(o.session_id, 2) AS order_id,
  o.session_id,
  o.user_id,
  o.product_id,
  o.channel,
  o.device,
  o.region,
  o.market,
  {amount} AS order_amount,
  {status} AS order_status,
  o.dt
FROM ordered o
JOIN {d.qualify('dim_product')} p ON o.product_id = p.product_id"""


def _fact_ad_spend_sql(d: SqlDialect, cfg: SampleDataConfig) -> str:
    paid_channels = ["Paid Search", "Social"]
    di = d.day_index("d.dt", cfg.start_date)
    noise_seed = (
        "concat_ws('|', " + d.cast_string("d.dt") + ", c.channel, m.market, "
        + d.cast_string("k.camp_no") + ", 'spend')"
    )
    clicks = (
        f"CAST(GREATEST(1, ROUND("
        f"{cfg.daily_sessions}"
        f" * {_case_numeric('c.channel', {k: CHANNELS[k][0] for k in paid_channels})}"
        f" * {_case_numeric('m.market', {k: v[0] for k, v in MARKETS.items()})}"
        f" * {_case_index(f'({di} % 7)', WEEKDAY_FACTORS)}"
        f" / 3.0"
        f" * (0.85 + 0.30 * {d.rand01(noise_seed)})"
        f")) AS BIGINT)"
    )
    return f"""INSERT INTO {d.qualify('fact_ad_spend')}
WITH
  dates AS ({d.date_series(cfg.start_date, cfg.end_date)}),
  {_values_cte('channels', 'channel', paid_channels)},
  {_values_cte('markets', 'market', list(MARKETS))},
  campaigns AS (SELECT {d.explode_int_range('3')} AS camp_no)
SELECT
  concat_ws('-', 'cmp', {_case_string('channel', {k: CHANNELS[k][3] for k in paid_channels})},
            market, lpad({d.cast_string('camp_no')}, 2, '0')) AS campaign_id,
  channel,
  market,
  CAST(ROUND(clicks * {CPC!r}, 2) AS {d.type_sql('decimal(12,2)')}) AS spend,
  CAST(ROUND(clicks / {CTR!r}) AS BIGINT) AS impressions,
  clicks,
  dt
FROM (
  SELECT d.dt, c.channel, m.market, k.camp_no, {clicks} AS clicks
  FROM dates d
  CROSS JOIN channels c
  CROSS JOIN markets m
  CROSS JOIN campaigns k
) x"""


# 生成顺序有依赖:dim_product 必须先于 fact_orders(join),fact_sessions 先于 fact_orders。
_GENERATORS = [
    ("dim_product", _dim_product_sql),
    ("dim_user", _dim_user_sql),
    ("fact_sessions", _fact_sessions_sql),
    ("fact_orders", _fact_orders_sql),
    ("fact_ad_spend", _fact_ad_spend_sql),
]


@dataclass(frozen=True)
class SetupStatement:
    label: str
    sql: str


def build_setup_statements(
    knowledge: Knowledge,
    dialect: SqlDialect,
    config: SampleDataConfig | None = None,
) -> list[SetupStatement]:
    """返回完整的建库脚本:建 schema -> 建表 -> 灌数。

    Args:
        knowledge: 已校验的知识库,提供 ``tables.yaml`` 中的 schema。
        dialect: 目标方言。
        config: 数据规模;缺省为面向 Free Edition 配额的默认值。

    Returns:
        按**执行顺序**排列的语句列表。每条都可单独重跑,整体幂等。
    """
    cfg = config or SampleDataConfig()
    stmts = [SetupStatement(f"schema:{i}", s) for i, s in enumerate(render_schema_ddl(dialect))]

    for table_name, _ in _GENERATORS:
        table = knowledge.table(table_name)
        stmts.append(
            SetupStatement(f"create:{table_name}", render_create_table(table, dialect))
        )
    for table_name, builder in _GENERATORS:
        stmts.append(SetupStatement(f"insert:{table_name}", builder(dialect, cfg)))
    return stmts


def build_verification_sql(dialect: SqlDialect) -> str:
    """自检查询:行数、日期覆盖,以及 CVR 的两侧交叉核对。

    返回 ``(check_name, value)`` 两列。**``sessions_converted`` 必须等于
    ``completed_orders``** —— 不相等说明生成器的两处转化判定已经漂移,
    ``UV × CVR × AOV = GMV`` 在数据层就不再成立,后续所有分解都不可信。

    刻意写成 ``UNION ALL`` 而不是「无 FROM 的标量子查询并排」:
    后者在各引擎上的支持度参差,而这段 SQL 要能直接贴进 Databricks SQL Editor。
    """
    d = dialect
    string_type = d.type_sql("string")

    def count_row(name: str, table: str, predicate: str = "") -> str:
        where = f" WHERE {predicate}" if predicate else ""
        return (
            f"SELECT {SqlDialect.string_literal(name):<22} AS check_name, "
            f"CAST(COUNT(*) AS {string_type}) AS value FROM {d.qualify(table)}{where}"
        )

    def scalar_row(name: str, expression: str, table: str) -> str:
        return (
            f"SELECT {SqlDialect.string_literal(name):<22} AS check_name, "
            f"CAST({expression} AS {string_type}) AS value FROM {d.qualify(table)}"
        )

    parts = [
        count_row("sessions", "fact_sessions"),
        count_row("orders_all_status", "fact_orders"),
        count_row("completed_orders", "fact_orders", "order_status = 'COMPLETED'"),
        count_row("sessions_converted", "fact_sessions", "converted"),
        count_row("products", "dim_product"),
        count_row("users", "dim_user"),
        count_row("ad_spend_rows", "fact_ad_spend"),
        scalar_row("first_dt", "MIN(dt)", "fact_sessions"),
        scalar_row("last_dt", "MAX(dt)", "fact_sessions"),
        scalar_row("distinct_days", "COUNT(DISTINCT dt)", "fact_sessions"),
    ]
    return "\nUNION ALL ".join(parts)


def run_verification(executor: Any, dialect: SqlDialect) -> dict[str, str]:
    """跑一次自检并返回 ``{check_name: value}``。"""
    rows = executor.run(build_verification_sql(dialect))
    return {row["check_name"]: row["value"] for row in rows}
