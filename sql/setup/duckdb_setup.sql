-- ===========================================================================
-- 本文件由 sql/setup/render.py 自动生成,请勿手工编辑。
-- 生成器:src/rca/sampledata.py    Schema 真源:knowledge/tables.yaml
-- 方言:duckdb    目标:main.<table>
--
-- 使用方式(Databricks):在 SQL Editor 中按顺序整段执行。
-- 全部语句均为幂等:CREATE OR REPLACE + 确定性 INSERT,重跑得到完全相同的数据。
-- 规模:8 周日粒度,日均 6000 会话,约 336,000 行会话数据。
-- ===========================================================================

-- [create:dim_product]
CREATE OR REPLACE TABLE main.dim_product (
  product_id BIGINT NOT NULL,
  category VARCHAR NOT NULL,
  sub_category VARCHAR NOT NULL,
  price_tier VARCHAR NOT NULL
);

-- [create:dim_user]
CREATE OR REPLACE TABLE main.dim_user (
  user_id BIGINT NOT NULL,
  region VARCHAR NOT NULL,
  market VARCHAR NOT NULL,
  signup_dt DATE NOT NULL,
  user_tier VARCHAR NOT NULL
);

-- [create:fact_sessions]
CREATE OR REPLACE TABLE main.fact_sessions (
  session_id VARCHAR NOT NULL,
  user_id BIGINT NOT NULL,
  channel VARCHAR NOT NULL,
  device VARCHAR NOT NULL,
  region VARCHAR NOT NULL,
  market VARCHAR NOT NULL,
  converted BOOLEAN NOT NULL,
  dt DATE NOT NULL
);

-- [create:fact_orders]
CREATE OR REPLACE TABLE main.fact_orders (
  order_id VARCHAR NOT NULL,
  session_id VARCHAR NOT NULL,
  user_id BIGINT NOT NULL,
  product_id BIGINT NOT NULL,
  channel VARCHAR NOT NULL,
  device VARCHAR NOT NULL,
  region VARCHAR NOT NULL,
  market VARCHAR NOT NULL,
  order_amount DECIMAL(12,2) NOT NULL,
  order_status VARCHAR NOT NULL,
  dt DATE NOT NULL
);

-- [create:fact_ad_spend]
CREATE OR REPLACE TABLE main.fact_ad_spend (
  campaign_id VARCHAR NOT NULL,
  channel VARCHAR NOT NULL,
  market VARCHAR NOT NULL,
  spend DECIMAL(12,2) NOT NULL,
  impressions BIGINT NOT NULL,
  clicks BIGINT NOT NULL,
  dt DATE NOT NULL
);

-- [insert:dim_product]
INSERT INTO main.dim_product
SELECT
  product_id,
  category,
  category || '-' || CAST(sub_idx AS VARCHAR) AS sub_category,
  price_tier
FROM (
  SELECT
    product_id,
    (CASE WHEN ((hash('cat|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.240000 THEN 'Apparel' WHEN ((hash('cat|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.420000 THEN 'Electronics' WHEN ((hash('cat|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.580000 THEN 'Home' WHEN ((hash('cat|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.720000 THEN 'Beauty' WHEN ((hash('cat|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.850000 THEN 'Sports' WHEN ((hash('cat|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.940000 THEN 'Toys' ELSE 'Grocery' END) AS category,
    1 + CAST(FLOOR(((hash('sub|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) * 4) AS INT) AS sub_idx,
    (CASE WHEN ((hash('tier|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.400000 THEN 'BUDGET' WHEN ((hash('tier|' || CAST(product_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.820000 THEN 'MID' ELSE 'PREMIUM' END) AS price_tier
  FROM (SELECT CAST(n AS BIGINT) AS product_id FROM (SELECT unnest(generate_series(1, 240)) AS n) _s) p
) q;

-- [insert:dim_user]
INSERT INTO main.dim_user
SELECT
  user_id,
  (CASE market WHEN 'US' THEN 'NA' WHEN 'UK' THEN 'EMEA' WHEN 'DE' THEN 'EMEA' END) AS region,
  market,
  (DATE '2025-05-05' + CAST(-CAST(FLOOR(((hash('signup|' || CAST(user_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) * 730) AS INT) AS INTEGER)) AS signup_dt,
  (CASE WHEN ((hash('utier|' || CAST(user_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.450000 THEN 'NEW' WHEN ((hash('utier|' || CAST(user_id AS VARCHAR)) % 1000000)::DOUBLE / 1000000.0) < 0.870000 THEN 'RETURNING' ELSE 'VIP' END) AS user_tier
FROM (
  SELECT user_id, (CASE WHEN user_id <= 27500 THEN 'US' WHEN user_id <= 40000 THEN 'UK' ELSE 'DE' END) AS market
  FROM (SELECT CAST(n AS BIGINT) AS user_id FROM (SELECT unnest(generate_series(1, 50000)) AS n) _s) u
) m;

-- [insert:fact_sessions]
INSERT INTO main.fact_sessions
WITH
  dates AS (SELECT CAST(unnest(generate_series(DATE '2025-05-05', DATE '2025-06-29', INTERVAL 1 DAY)) AS DATE) AS dt),
  channels AS (
    SELECT 'Paid Search' AS channel
    UNION ALL SELECT 'Organic' AS channel
    UNION ALL SELECT 'Email' AS channel
    UNION ALL SELECT 'Social' AS channel
    UNION ALL SELECT 'Direct' AS channel
  ),
  devices AS (
    SELECT 'mobile' AS device
    UNION ALL SELECT 'desktop' AS device
  ),
  markets AS (
    SELECT 'US' AS market
    UNION ALL SELECT 'UK' AS market
    UNION ALL SELECT 'DE' AS market
  ),
  grid AS (
    SELECT d.dt, c.channel, v.device, m.market,
           CAST(GREATEST(1, ROUND(6000 * (CASE c.channel WHEN 'Paid Search' THEN 0.3 WHEN 'Organic' THEN 0.28 WHEN 'Email' THEN 0.1 WHEN 'Social' THEN 0.17 WHEN 'Direct' THEN 0.15 END) * (CASE v.device WHEN 'mobile' THEN 0.62 WHEN 'desktop' THEN 0.38 END) * (CASE m.market WHEN 'US' THEN 0.55 WHEN 'UK' THEN 0.25 WHEN 'DE' THEN 0.2 END) * (CASE (datediff('day', DATE '2025-05-05', d.dt) % 7) WHEN 0 THEN 1.0 WHEN 1 THEN 0.98 WHEN 2 THEN 0.97 WHEN 3 THEN 0.99 WHEN 4 THEN 1.04 WHEN 5 THEN 1.12 WHEN 6 THEN 1.08 END) * (1 + 0.0015 * datediff('day', DATE '2025-05-05', d.dt)) * (0.92 + 0.16 * ((hash(concat_ws('|', CAST(d.dt AS VARCHAR), c.channel, v.device, m.market, 'sessions')) % 1000000)::DOUBLE / 1000000.0)))) AS INT) AS n_sessions
    FROM dates d
    CROSS JOIN channels c
    CROSS JOIN devices v
    CROSS JOIN markets m
  ),
  expanded AS (
    SELECT g.dt, g.channel, g.device, g.market,
           unnest(generate_series(1, g.n_sessions)) AS i
    FROM grid g
  ),
  sess AS (
    SELECT dt, channel, device, market, concat_ws('_', 's', replace(CAST(dt AS VARCHAR), '-', ''), (CASE channel WHEN 'Paid Search' THEN 'PS' WHEN 'Organic' THEN 'OR' WHEN 'Email' THEN 'EM' WHEN 'Social' THEN 'SO' WHEN 'Direct' THEN 'DI' END) || (CASE device WHEN 'mobile' THEN 'M' WHEN 'desktop' THEN 'D' END), market, lpad(CAST(i AS VARCHAR), 6, '0')) AS session_id
    FROM expanded
  )
SELECT
  session_id,
  (CASE market WHEN 'US' THEN 1 + CAST(FLOOR(((hash('user|' || session_id) % 1000000)::DOUBLE / 1000000.0) * 27500) AS BIGINT) WHEN 'UK' THEN 27501 + CAST(FLOOR(((hash('user|' || session_id) % 1000000)::DOUBLE / 1000000.0) * 12500) AS BIGINT) WHEN 'DE' THEN 40001 + CAST(FLOOR(((hash('user|' || session_id) % 1000000)::DOUBLE / 1000000.0) * 10000) AS BIGINT) END) AS user_id,
  channel,
  device,
  (CASE market WHEN 'US' THEN 'NA' WHEN 'UK' THEN 'EMEA' WHEN 'DE' THEN 'EMEA' END) AS region,
  market,
  -- converted 的定义:该会话产生了一笔 COMPLETED 订单。
  -- 与 fact_orders 的生成条件逐字相同,因此 COUNT(converted) 必然等于 orders 指标。
  ((((hash(session_id || '|conv') % 1000000)::DOUBLE / 1000000.0)) < (0.045 * (CASE channel WHEN 'Paid Search' THEN 0.9 WHEN 'Organic' THEN 1.15 WHEN 'Email' THEN 1.4 WHEN 'Social' THEN 0.6 WHEN 'Direct' THEN 1.25 END) * (CASE device WHEN 'mobile' THEN 0.8 WHEN 'desktop' THEN 1.3 END) * (CASE market WHEN 'US' THEN 1.0 WHEN 'UK' THEN 0.95 WHEN 'DE' THEN 0.9 END) * (0.94 + 0.12 * ((hash(concat_ws('|', CAST(dt AS VARCHAR), channel, device, market, 'cvr')) % 1000000)::DOUBLE / 1000000.0))) AND (((hash(session_id || '|status') % 1000000)::DOUBLE / 1000000.0)) < 0.92) AS converted,
  dt
FROM sess;

-- [insert:fact_orders]
INSERT INTO main.fact_orders
-- 订单完全由 fact_sessions 推导:转化判定与订单级随机量都从 session_id 派生,
-- 因此这条语句可以独立重跑,结果不变(简报约束 3)。
WITH ordered AS (
  SELECT
    s.session_id, s.user_id, s.channel, s.device, s.region, s.market, s.dt,
    1 + CAST(FLOOR(((hash(session_id || '|prod') % 1000000)::DOUBLE / 1000000.0) * 240) AS BIGINT) AS product_id,
    ((hash(session_id || '|status') % 1000000)::DOUBLE / 1000000.0) AS r_status,
    ((hash(session_id || '|amt') % 1000000)::DOUBLE / 1000000.0) AS r_amt
  FROM main.fact_sessions s
  WHERE (((hash(session_id || '|conv') % 1000000)::DOUBLE / 1000000.0)) < (0.045 * (CASE channel WHEN 'Paid Search' THEN 0.9 WHEN 'Organic' THEN 1.15 WHEN 'Email' THEN 1.4 WHEN 'Social' THEN 0.6 WHEN 'Direct' THEN 1.25 END) * (CASE device WHEN 'mobile' THEN 0.8 WHEN 'desktop' THEN 1.3 END) * (CASE market WHEN 'US' THEN 1.0 WHEN 'UK' THEN 0.95 WHEN 'DE' THEN 0.9 END) * (0.94 + 0.12 * ((hash(concat_ws('|', CAST(dt AS VARCHAR), channel, device, market, 'cvr')) % 1000000)::DOUBLE / 1000000.0)))
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
  CAST(ROUND((CASE p.price_tier WHEN 'BUDGET' THEN 28.0 WHEN 'MID' THEN 78.0 WHEN 'PREMIUM' THEN 190.0 END) * (CASE o.channel WHEN 'Paid Search' THEN 1.0 WHEN 'Organic' THEN 1.05 WHEN 'Email' THEN 1.1 WHEN 'Social' THEN 0.9 WHEN 'Direct' THEN 1.15 END) * (CASE o.device WHEN 'mobile' THEN 0.88 WHEN 'desktop' THEN 1.2 END) * (CASE o.market WHEN 'US' THEN 1.0 WHEN 'UK' THEN 0.95 WHEN 'DE' THEN 0.92 END) * (0.70 + 0.60 * o.r_amt), 2) AS DECIMAL(12,2)) AS order_amount,
  (CASE WHEN o.r_status < 0.92 THEN 'COMPLETED' WHEN o.r_status < 0.97 THEN 'CANCELLED' ELSE 'REFUNDED' END) AS order_status,
  o.dt
FROM ordered o
JOIN main.dim_product p ON o.product_id = p.product_id;

-- [insert:fact_ad_spend]
INSERT INTO main.fact_ad_spend
WITH
  dates AS (SELECT CAST(unnest(generate_series(DATE '2025-05-05', DATE '2025-06-29', INTERVAL 1 DAY)) AS DATE) AS dt),
  channels AS (
    SELECT 'Paid Search' AS channel
    UNION ALL SELECT 'Social' AS channel
  ),
  markets AS (
    SELECT 'US' AS market
    UNION ALL SELECT 'UK' AS market
    UNION ALL SELECT 'DE' AS market
  ),
  campaigns AS (SELECT unnest(generate_series(1, 3)) AS camp_no)
SELECT
  concat_ws('-', 'cmp', (CASE channel WHEN 'Paid Search' THEN 'PS' WHEN 'Social' THEN 'SO' END),
            market, lpad(CAST(camp_no AS VARCHAR), 2, '0')) AS campaign_id,
  channel,
  market,
  CAST(ROUND(clicks * 0.85, 2) AS DECIMAL(12,2)) AS spend,
  CAST(ROUND(clicks / 0.03) AS BIGINT) AS impressions,
  clicks,
  dt
FROM (
  SELECT d.dt, c.channel, m.market, k.camp_no, CAST(GREATEST(1, ROUND(6000 * (CASE c.channel WHEN 'Paid Search' THEN 0.3 WHEN 'Social' THEN 0.17 END) * (CASE m.market WHEN 'US' THEN 0.55 WHEN 'UK' THEN 0.25 WHEN 'DE' THEN 0.2 END) * (CASE (datediff('day', DATE '2025-05-05', d.dt) % 7) WHEN 0 THEN 1.0 WHEN 1 THEN 0.98 WHEN 2 THEN 0.97 WHEN 3 THEN 0.99 WHEN 4 THEN 1.04 WHEN 5 THEN 1.12 WHEN 6 THEN 1.08 END) / 3.0 * (0.85 + 0.30 * ((hash(concat_ws('|', CAST(d.dt AS VARCHAR), c.channel, m.market, CAST(k.camp_no AS VARCHAR), 'spend')) % 1000000)::DOUBLE / 1000000.0)))) AS BIGINT) AS clicks
  FROM dates d
  CROSS JOIN channels c
  CROSS JOIN markets m
  CROSS JOIN campaigns k
) x;

-- [verify] 自检:sessions_converted 必须等于 completed_orders
SELECT 'sessions'             AS check_name, CAST(COUNT(*) AS VARCHAR) AS value FROM main.fact_sessions
UNION ALL SELECT 'orders_all_status'    AS check_name, CAST(COUNT(*) AS VARCHAR) AS value FROM main.fact_orders
UNION ALL SELECT 'completed_orders'     AS check_name, CAST(COUNT(*) AS VARCHAR) AS value FROM main.fact_orders WHERE order_status = 'COMPLETED'
UNION ALL SELECT 'sessions_converted'   AS check_name, CAST(COUNT(*) AS VARCHAR) AS value FROM main.fact_sessions WHERE converted
UNION ALL SELECT 'products'             AS check_name, CAST(COUNT(*) AS VARCHAR) AS value FROM main.dim_product
UNION ALL SELECT 'users'                AS check_name, CAST(COUNT(*) AS VARCHAR) AS value FROM main.dim_user
UNION ALL SELECT 'ad_spend_rows'        AS check_name, CAST(COUNT(*) AS VARCHAR) AS value FROM main.fact_ad_spend
UNION ALL SELECT 'first_dt'             AS check_name, CAST(MIN(dt) AS VARCHAR) AS value FROM main.fact_sessions
UNION ALL SELECT 'last_dt'              AS check_name, CAST(MAX(dt) AS VARCHAR) AS value FROM main.fact_sessions
UNION ALL SELECT 'distinct_days'        AS check_name, CAST(COUNT(DISTINCT dt) AS VARCHAR) AS value FROM main.fact_sessions;
