"""运行目标的装配与凭据处理。

凭据是这个项目唯一会造成**不可逆后果**的东西(token 泄漏进日志、进 Git)。
所以这里的断言全部围绕一件事:**token 不得出现在任何可打印的地方。**
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rca.config import (
    ENV_HOSTNAME,
    ENV_HTTP_PATH,
    ENV_TOKEN,
    ConfigError,
    DatabricksSettings,
    build_target,
    load_dotenv,
)

FAKE_TOKEN = "dapi-THIS-IS-NOT-A-REAL-TOKEN"

GOOD_ENV = {
    ENV_HOSTNAME: "demo.cloud.databricks.com",
    ENV_HTTP_PATH: "/sql/1.0/warehouses/deadbeef",
    ENV_TOKEN: FAKE_TOKEN,
}


# ---------------------------------------------------------------------------
# 凭据不得泄漏
# ---------------------------------------------------------------------------
def test_token_is_redacted_in_repr():
    settings = DatabricksSettings.from_env(GOOD_ENV)
    assert FAKE_TOKEN not in repr(settings)
    assert "***" in repr(settings)


def test_token_is_redacted_in_executor_repr(monkeypatch):
    for key, value in GOOD_ENV.items():
        monkeypatch.setenv(key, value)
    target = build_target("databricks")
    assert FAKE_TOKEN not in repr(target.executor)
    assert FAKE_TOKEN not in repr(target)


def test_dotenv_return_value_never_contains_values(tmp_path: Path):
    """``load_dotenv`` 返回变量**名**,不返回值 —— 它的结果经常被打印出来排查。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(f"{k}={v}" for k, v in GOOD_ENV.items()), encoding="utf-8"
    )
    applied = load_dotenv(env_file)
    assert set(applied) == set(GOOD_ENV)
    assert FAKE_TOKEN not in " ".join(applied)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    ["demo.cloud.databricks.com", "https://demo.cloud.databricks.com", "https://demo.cloud.databricks.com/"],
)
def test_hostname_is_normalised(raw):
    """Connection details 页面复制出来可能带 https:// 和尾部斜杠,连接器都不接受。"""
    settings = DatabricksSettings.from_env({**GOOD_ENV, ENV_HOSTNAME: raw})
    assert settings.server_hostname == "demo.cloud.databricks.com"


def test_dotenv_parses_export_prefix_and_quotes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(ENV_HTTP_PATH, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# 注释\n\nexport DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/abc"\n', encoding="utf-8"
    )
    load_dotenv(env_file)
    import os

    assert os.environ[ENV_HTTP_PATH] == "/sql/1.0/warehouses/abc"


def test_real_environment_wins_over_dotenv(tmp_path: Path, monkeypatch):
    """CI 里注入的真实环境变量不能被仓库里某个陈旧的 .env 覆盖掉。"""
    monkeypatch.setenv(ENV_HTTP_PATH, "/sql/1.0/warehouses/from-env")
    env_file = tmp_path / ".env"
    env_file.write_text("DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/from-file\n", encoding="utf-8")
    load_dotenv(env_file)
    import os

    assert os.environ[ENV_HTTP_PATH] == "/sql/1.0/warehouses/from-env"


def test_missing_dotenv_is_not_an_error(tmp_path: Path):
    assert load_dotenv(tmp_path / "nope.env") == []


# ---------------------------------------------------------------------------
# 错误信息要能直接照做
# ---------------------------------------------------------------------------
def test_missing_credentials_error_names_every_missing_variable():
    with pytest.raises(ConfigError) as info:
        DatabricksSettings.from_env({})
    message = str(info.value)
    for name in (ENV_HOSTNAME, ENV_HTTP_PATH, ENV_TOKEN):
        assert name in message
    assert "Connection details" in message


def test_unknown_target_is_rejected():
    with pytest.raises(ConfigError, match="未知运行目标"):
        build_target("snowflake")


# ---------------------------------------------------------------------------
# 目标装配
# ---------------------------------------------------------------------------
def test_databricks_target_uses_databricks_dialect(monkeypatch):
    for key, value in GOOD_ENV.items():
        monkeypatch.setenv(key, value)
    target = build_target("databricks", catalog="my_cat", schema="my_schema")
    assert target.dialect.name == "databricks"
    assert target.dialect.qualify("fact_orders") == "my_cat.my_schema.fact_orders"
    assert target.executor.dialect_name == "databricks"


def test_duckdb_target_is_clearly_local():
    target = build_target("duckdb")
    try:
        assert target.dialect.name == "duckdb"
        assert target.dialect.qualify("fact_orders") == "main.fact_orders"
    finally:
        target.close()
