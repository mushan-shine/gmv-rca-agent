"""运行目标(target)的装配。

本项目的**目标平台是 Databricks**。有三种运行形态,SQL 完全相同:

======================  ====================================================
target                  什么时候用
======================  ====================================================
``databricks``          从本机/CI 连 SQL Warehouse(``databricks-sql-connector``)
``spark``               代码跑在 Databricks Notebook 或 Job 里,直接用 SparkSession
``duckdb``              本地快速自测。**不是部署目标**,只是同一份 SQL 生成器的
                        第二个方言,让算术闭合性可以在毫秒内验证。
======================  ====================================================

凭据只从**环境变量**读取,绝不进代码、不进 YAML、不进日志。
:class:`DatabricksSettings` 的 ``repr`` 会把 token 打码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .dialects import DatabricksDialect, DuckDBDialect, SqlDialect
from .errors import RcaError
from .warehouse import DatabricksExecutor, DuckDBExecutor, SparkExecutor, SqlExecutor

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CATALOG = "workspace"
DEFAULT_SCHEMA = "gmv_rca"

ENV_HOSTNAME = "DATABRICKS_SERVER_HOSTNAME"
ENV_HTTP_PATH = "DATABRICKS_HTTP_PATH"
ENV_TOKEN = "DATABRICKS_TOKEN"
ENV_CATALOG = "DATABRICKS_CATALOG"
ENV_SCHEMA = "DATABRICKS_SCHEMA"
ENV_WAREHOUSE_ID = "DATABRICKS_WAREHOUSE_ID"


class ConfigError(RcaError):
    """运行目标配置缺失或不合法。"""


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """把仓库根的 ``.env`` 读进 ``os.environ``(已存在的环境变量优先)。

    刻意不引第三方依赖:格式只支持 ``KEY=VALUE``、``#`` 注释、可选的 ``export`` 前缀
    和成对引号。``.env`` 在 ``.gitignore`` 里,**不要提交** —— 它装的是 token。

    Returns:
        本次真正写入 ``os.environ`` 的变量名(便于排查,**不含值**)。
    """
    path = Path(path) if path is not None else REPO_ROOT / ".env"
    if not path.is_file():
        return []

    applied: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:   # 真实环境变量优先于 .env
            os.environ[key] = value
            applied.append(key)
    return applied


@dataclass(frozen=True)
class DatabricksSettings:
    """连接一个 SQL Warehouse 所需的全部信息。"""

    server_hostname: str
    http_path: str
    access_token: str = field(repr=False)
    catalog: str = DEFAULT_CATALOG
    schema: str = DEFAULT_SCHEMA

    def __repr__(self) -> str:  # 防止 token 出现在异常栈/日志里
        return (
            f"DatabricksSettings(server_hostname={self.server_hostname!r}, "
            f"http_path={self.http_path!r}, access_token='***', "
            f"catalog={self.catalog!r}, schema={self.schema!r})"
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "DatabricksSettings":
        """从环境变量读取。缺变量时给出可直接照做的错误信息。"""
        if env is None:
            load_dotenv()
            env = os.environ
        missing = [
            name for name in (ENV_HOSTNAME, ENV_HTTP_PATH, ENV_TOKEN) if not env.get(name)
        ]
        if missing:
            raise ConfigError(
                "缺少 Databricks 连接配置:" + ", ".join(missing) + "\n\n"
                "在 shell 里设置(不要写进代码或提交进 Git):\n"
                f"  {ENV_HOSTNAME}=<workspace>.cloud.databricks.com   # 不带 https://\n"
                f"  {ENV_HTTP_PATH}=/sql/1.0/warehouses/<warehouse-id>\n"
                f"  {ENV_TOKEN}=<personal access token>\n"
                f"  {ENV_CATALOG}={DEFAULT_CATALOG}      # 可选\n"
                f"  {ENV_SCHEMA}={DEFAULT_SCHEMA}   # 可选\n\n"
                "前两项在 Databricks 的 SQL Warehouse → Connection details 页面可以直接复制。\n"
                "也可以把它们写进仓库根的 .env(已在 .gitignore 中,不会被提交),"
                "格式见 .env.example。"
            )
        hostname = env[ENV_HOSTNAME].strip()
        for prefix in ("https://", "http://"):
            if hostname.startswith(prefix):
                hostname = hostname[len(prefix) :]
        return cls(
            server_hostname=hostname.rstrip("/"),
            http_path=env[ENV_HTTP_PATH].strip(),
            access_token=env[ENV_TOKEN].strip(),
            catalog=(env.get(ENV_CATALOG) or DEFAULT_CATALOG).strip(),
            schema=(env.get(ENV_SCHEMA) or DEFAULT_SCHEMA).strip(),
        )

    @classmethod
    def available(cls, env: Mapping[str, str] | None = None) -> bool:
        if env is None:
            load_dotenv()
            env = os.environ
        return all(env.get(name) for name in (ENV_HOSTNAME, ENV_HTTP_PATH, ENV_TOKEN))


@dataclass
class Target:
    """一个可执行的运行目标:方言 + 执行器。"""

    name: str
    dialect: SqlDialect
    executor: SqlExecutor
    description: str = ""

    def close(self) -> None:
        close = getattr(self.executor, "close", None)
        if callable(close):
            close()


TARGET_NAMES = ("databricks", "spark", "duckdb")


def build_target(
    name: str,
    *,
    catalog: str | None = None,
    schema: str | None = None,
    duckdb_path: str = ":memory:",
    spark: Any | None = None,
) -> Target:
    """按名字装配运行目标。

    Args:
        name: ``databricks`` / ``spark`` / ``duckdb``。
        catalog: 覆盖目录名(Databricks 与 spark 目标)。
        schema: 覆盖 schema 名。
        duckdb_path: 本地目标的库文件路径,默认内存库。
        spark: 已有的 SparkSession;``spark`` 目标下省略则取当前活跃会话。

    Raises:
        ConfigError: 目标名未知,或 Databricks 凭据缺失。
    """
    if name == "databricks":
        settings = DatabricksSettings.from_env()
        dialect = DatabricksDialect(
            catalog=catalog or settings.catalog, schema=schema or settings.schema
        )
        executor = DatabricksExecutor(
            server_hostname=settings.server_hostname,
            http_path=settings.http_path,
            access_token=settings.access_token,
        )
        return Target(
            name=name,
            dialect=dialect,
            executor=executor,
            description=f"SQL Warehouse {settings.server_hostname} → {dialect.qualify('*')}",
        )

    if name == "spark":
        load_dotenv()
        session = spark if spark is not None else _active_spark_session()
        dialect = DatabricksDialect(
            catalog=catalog or os.environ.get(ENV_CATALOG) or DEFAULT_CATALOG,
            schema=schema or os.environ.get(ENV_SCHEMA) or DEFAULT_SCHEMA,
        )
        return Target(
            name=name,
            dialect=dialect,
            executor=SparkExecutor(session),
            description=f"SparkSession → {dialect.qualify('*')}",
        )

    if name == "duckdb":
        import duckdb  # 本地开发依赖,延迟导入

        dialect = DuckDBDialect(schema=schema or "main")
        return Target(
            name=name,
            dialect=dialect,
            executor=DuckDBExecutor(duckdb.connect(duckdb_path)),
            description=f"本地 DuckDB ({duckdb_path})",
        )

    raise ConfigError(f"未知运行目标 {name!r};可选:{TARGET_NAMES}")


def build_app_target(
    *,
    catalog: str | None = None,
    schema: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Target:
    """Databricks Apps 里的运行目标:用 App 自己的 service principal 连 SQL Warehouse。

    Apps 会自动注入 ``DATABRICKS_HOST`` / ``DATABRICKS_CLIENT_ID`` / ``DATABRICKS_CLIENT_SECRET``,
    ``databricks-sdk`` 的 ``Config()`` 据此做 OAuth;warehouse 由 ``app.yaml`` 里声明的资源
    注入成 ``DATABRICKS_WAREHOUSE_ID``。全程不经手任何个人 token。

    这个 service principal 在 Unity Catalog 里只需要 schema 上的 ``SELECT``
    (外加写两张问答日志表的权限)—— 守卫之外,权限层再挡一次。
    """
    env = os.environ if env is None else env
    warehouse_id = (env.get(ENV_WAREHOUSE_ID) or "").strip()
    if not warehouse_id:
        raise ConfigError(
            f"缺少 {ENV_WAREHOUSE_ID}。在 App 的 Resources 里添加一个 SQL warehouse,"
            f"资源键填 sql-warehouse(与 app.yaml 一致)。"
        )
    try:
        from databricks.sdk.core import Config
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise ConfigError("缺少 databricks-sdk;Databricks Apps 环境里是预装的") from exc

    sdk_config = Config()
    hostname = (sdk_config.host or "").removeprefix("https://").removeprefix("http://").rstrip("/")
    dialect = DatabricksDialect(
        catalog=catalog or env.get(ENV_CATALOG) or DEFAULT_CATALOG,
        schema=schema or env.get(ENV_SCHEMA) or DEFAULT_SCHEMA,
    )
    executor = DatabricksExecutor(
        server_hostname=hostname,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: sdk_config.authenticate,
    )
    return Target(
        name="databricks-app",
        dialect=dialect,
        executor=executor,
        description=f"SQL Warehouse(App service principal)→ {dialect.qualify('*')}",
    )


def _active_spark_session() -> Any:
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:  # pragma: no cover - 仅在 Databricks 运行时可用
        raise ConfigError(
            "target=spark 需要在 Databricks Notebook / Job 里运行(依赖 pyspark)。"
            "从本机连远端请改用 target=databricks。"
        ) from exc

    session = SparkSession.getActiveSession()
    if session is None:  # pragma: no cover
        session = SparkSession.builder.getOrCreate()
    return session
