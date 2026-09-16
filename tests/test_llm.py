"""真实 LLM 接入层:重试、缓存、预算、凭据。

这一层不需要数据库,也不需要网络 —— 用假的 ``urlopen`` 把 HTTP 行为钉死。

最重要的一条断言在 :func:`test_transport_retry_is_not_a_self_repair_attempt`:
**传输层重发同一个 prompt,不得被记成一次自修复尝试。**
混同会让一次网络抖动被读成「模型改了两轮才对」,直接污染 ``avg_attempts``。
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from rca.nl2sql import llm as llm_module
from rca.nl2sql.generate import LlmSqlGenerator, PromptContext
from rca.nl2sql.llm import (
    CachingChatClient,
    LlmBudgetExceeded,
    LlmConfigError,
    LlmResponse,
    LlmTransportError,
    OpenAIChatClient,
    StaticChatClient,
    UsageMeter,
    build_chat_client,
    prompt_fingerprint,
)

FAKE_KEY = "sk-THIS-IS-NOT-A-REAL-KEY"


def _chat_body(text: str, *, finish_reason: str = "stop", model: str = "m") -> bytes:
    return json.dumps(
        {
            "model": model,
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
    ).encode("utf-8")


class _FakeHttp:
    """记录请求、按剧本返回响应的假 urlopen。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[Any] = []

    def __call__(self, request, timeout=None):  # noqa: ANN001
        self.requests.append(request)
        item = self.script.pop(0) if self.script else b""
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _http_error(code: int, body: str = "boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=code, msg="err", hdrs=None, fp=io.BytesIO(body.encode())
    )


@pytest.fixture()
def no_sleep(monkeypatch):
    """退避是真的会 sleep 的。测试里跳过,但仍断言它被调用了几次。"""
    calls: list[int] = []
    monkeypatch.setattr(llm_module, "_sleep_backoff", lambda attempt, **kw: calls.append(attempt))
    return calls


def _client(**kwargs: Any) -> OpenAIChatClient:
    defaults = dict(
        base_url="https://example.invalid/v1", api_key=FAKE_KEY, model="test-model"
    )
    defaults.update(kwargs)
    return OpenAIChatClient(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 请求与响应
# ---------------------------------------------------------------------------
def test_request_shape_is_openai_chat_completions(monkeypatch):
    http = _FakeHttp([_chat_body("```sql\nSELECT 1\n```")])
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", http)

    response = _client(temperature=0.0, max_tokens=256).complete("hello")

    assert response.text == "```sql\nSELECT 1\n```"
    assert response.prompt_tokens == 120 and response.completion_tokens == 30
    assert response.total_tokens == 150

    request = http.requests[0]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    payload = json.loads(request.data)
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0.0
    assert payload["messages"][-1] == {"role": "user", "content": "hello"}


def test_trailing_slash_in_base_url_is_tolerated(monkeypatch):
    http = _FakeHttp([_chat_body("ok")])
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", http)
    _client(base_url="https://example.invalid/v1/").complete("hi")
    assert http.requests[0].full_url == "https://example.invalid/v1/chat/completions"


def test_temperature_defaults_to_zero():
    """评估要求可复现。采样噪声会让指标的涨跌无法归因到任何一次改动。"""
    assert _client().temperature == 0.0


def test_truncated_response_is_flagged(monkeypatch):
    monkeypatch.setattr(
        llm_module.urllib.request,
        "urlopen",
        _FakeHttp([_chat_body("SELECT SUM(order_amo", finish_reason="length")]),
    )
    response = _client().complete("q")
    assert response.truncated


def test_response_without_choices_is_a_transport_error(monkeypatch):
    monkeypatch.setattr(
        llm_module.urllib.request, "urlopen", _FakeHttp([b'{"choices": []}'])
    )
    with pytest.raises(LlmTransportError):
        _client().complete("q")


# ---------------------------------------------------------------------------
# 重试
# ---------------------------------------------------------------------------
def test_rate_limit_is_retried_with_the_same_prompt(monkeypatch, no_sleep):
    http = _FakeHttp([_http_error(429), _http_error(503), _chat_body("ok")])
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", http)

    response = _client().complete("same prompt")

    assert response.text == "ok"
    assert response.attempts == 3
    assert len(no_sleep) == 2, "两次失败之间都要退避"
    sent = [json.loads(r.data)["messages"][-1]["content"] for r in http.requests]
    assert sent == ["same prompt"] * 3, "重试发的必须是同一个 prompt —— 它没有新信息"


def test_client_error_is_not_retried(monkeypatch, no_sleep):
    """400 / 401 重发多少次都一样,立刻失败并把原因说清楚。"""
    http = _FakeHttp([_http_error(401, "invalid token"), _chat_body("ok")])
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", http)

    with pytest.raises(LlmTransportError, match="401"):
        _client().complete("q")
    assert len(http.requests) == 1
    assert no_sleep == []


def test_retries_are_bounded(monkeypatch, no_sleep):
    http = _FakeHttp([_http_error(503)] * 10)
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", http)

    with pytest.raises(LlmTransportError):
        _client(max_retries=3).complete("q")
    assert len(http.requests) == 3


def test_timeout_is_retried(monkeypatch, no_sleep):
    http = _FakeHttp([TimeoutError("slow"), _chat_body("ok")])
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", http)
    assert _client().complete("q").text == "ok"


# ---------------------------------------------------------------------------
# 凭据
# ---------------------------------------------------------------------------
def test_api_key_is_never_in_repr():
    assert FAKE_KEY not in repr(_client())
    assert "***" in repr(_client())


def test_api_key_is_never_in_the_error_message(monkeypatch, no_sleep):
    monkeypatch.setattr(
        llm_module.urllib.request, "urlopen", _FakeHttp([_http_error(500, "server exploded")])
    )
    with pytest.raises(LlmTransportError) as info:
        _client(max_retries=1).complete("q")
    assert FAKE_KEY not in str(info.value)


def test_api_key_goes_in_the_header_not_the_url(monkeypatch):
    http = _FakeHttp([_chat_body("ok")])
    monkeypatch.setattr(llm_module.urllib.request, "urlopen", http)
    _client().complete("q")
    request = http.requests[0]
    assert FAKE_KEY not in request.full_url
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def test_identical_prompt_is_served_from_cache():
    """评估会把逐字相同的 prompt 打很多遍。第二遍不该再付费。"""
    inner = StaticChatClient({"": "reply"}, model="m")
    client = CachingChatClient(inner)

    first = client.complete("same")
    second = client.complete("same")

    assert not first.cached and second.cached
    assert second.text == first.text
    assert len(inner.calls) == 1
    assert (client.hits, client.misses) == (1, 1)


def test_different_prompts_are_not_confused():
    inner = StaticChatClient({"alpha": "A", "beta": "B"}, model="m")
    client = CachingChatClient(inner)
    assert client.complete("alpha").text == "A"
    assert client.complete("beta").text == "B"
    assert len(inner.calls) == 2


def test_cache_key_covers_model_and_params():
    """换模型或换 temperature 必须失效,否则会拿旧模型的答案冒充新模型的成绩。"""
    base = prompt_fingerprint("m1", {"temperature": 0.0}, "p")
    assert base != prompt_fingerprint("m2", {"temperature": 0.0}, "p")
    assert base != prompt_fingerprint("m1", {"temperature": 0.7}, "p")
    assert base == prompt_fingerprint("m1", {"temperature": 0.0}, "p")


def test_cache_can_be_backed_by_an_external_store():
    class Store:
        def __init__(self) -> None:
            self.data: dict[str, LlmResponse] = {}

        def get(self, key: str) -> LlmResponse | None:
            return self.data.get(key)

        def put(self, key: str, value: LlmResponse) -> None:
            self.data[key] = value

    store = Store()
    inner = StaticChatClient({"": "reply"}, model="m")
    CachingChatClient(inner, store=store).complete("q")
    assert len(store.data) == 1

    fresh = CachingChatClient(StaticChatClient({}, model="m", default="MISS"), store=store)
    assert fresh.complete("q").text == "reply", "新进程应当能命中持久化缓存"


# ---------------------------------------------------------------------------
# 预算
# ---------------------------------------------------------------------------
def test_budget_is_checked_before_spending():
    meter = UsageMeter(max_calls=2)
    meter.calls = 2
    with pytest.raises(LlmBudgetExceeded, match="调用上限"):
        meter.check_budget()


def test_token_budget_trips():
    meter = UsageMeter(max_tokens=100)
    meter.record(LlmResponse(text="x", model="m", prompt_tokens=80, completion_tokens=30))
    with pytest.raises(LlmBudgetExceeded, match="token"):
        meter.check_budget()


def test_cached_calls_do_not_consume_budget():
    """命中缓存没有真实计费,不该占用预算 —— 否则缓存就白做了。"""
    meter = UsageMeter(max_calls=1)
    meter.record(LlmResponse(text="x", model="m", prompt_tokens=10, cached=True))
    meter.check_budget()
    assert meter.calls == 0 and meter.cached_calls == 1


def test_meter_summary_reports_cache_hits():
    meter = UsageMeter()
    meter.record(LlmResponse(text="x", model="m", prompt_tokens=10, completion_tokens=5))
    meter.record(LlmResponse(text="x", model="m", cached=True))
    assert "命中缓存 1 次" in meter.summary()


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------
def test_databricks_base_url_is_derived_from_the_hostname():
    client = build_chat_client(
        {
            "RCA_LLM_PROVIDER": "databricks",
            "DATABRICKS_SERVER_HOSTNAME": "https://demo.cloud.databricks.com/",
            "DATABRICKS_TOKEN": FAKE_KEY,
        },
        cache=False,
    )
    assert client.base_url == "https://demo.cloud.databricks.com/serving-endpoints"
    assert client.model == llm_module.DEFAULT_DATABRICKS_MODEL


def test_databricks_requests_stay_inside_the_workspace():
    """出站网络受限时,模型请求必须打到 workspace 自己的域名上。"""
    client = build_chat_client(
        {
            "RCA_LLM_PROVIDER": "databricks",
            "DATABRICKS_SERVER_HOSTNAME": "demo.cloud.databricks.com",
            "DATABRICKS_TOKEN": FAKE_KEY,
        },
        cache=False,
    )
    assert "databricks.com" in client.base_url
    assert "openai.com" not in client.base_url


def test_openai_provider_uses_the_public_endpoint():
    client = build_chat_client(
        {"RCA_LLM_PROVIDER": "openai", "OPENAI_API_KEY": FAKE_KEY}, cache=False
    )
    assert client.base_url == "https://api.openai.com/v1"


def test_custom_provider_requires_base_url_and_model():
    with pytest.raises(LlmConfigError, match="RCA_LLM_BASE_URL"):
        build_chat_client({"RCA_LLM_PROVIDER": "custom"}, cache=False)


def test_missing_databricks_hostname_names_the_variable():
    with pytest.raises(LlmConfigError, match="DATABRICKS_SERVER_HOSTNAME"):
        build_chat_client({"RCA_LLM_PROVIDER": "databricks", "DATABRICKS_TOKEN": FAKE_KEY}, cache=False)


def test_unknown_provider_is_rejected():
    with pytest.raises(LlmConfigError, match="未知供应商"):
        build_chat_client({"RCA_LLM_PROVIDER": "bedrock"}, cache=False)


def test_build_wraps_in_cache_by_default():
    client = build_chat_client(
        {
            "RCA_LLM_PROVIDER": "databricks",
            "DATABRICKS_SERVER_HOSTNAME": "d.cloud.databricks.com",
            "DATABRICKS_TOKEN": FAKE_KEY,
        }
    )
    assert isinstance(client, CachingChatClient)


# ---------------------------------------------------------------------------
# 生成器
# ---------------------------------------------------------------------------
def _context(question: str = "What was GMV last week?") -> PromptContext:
    return PromptContext(question=question, schema_text="main.fact_orders(order_amount DOUBLE)")


def test_generator_name_carries_the_model():
    """指标变了,要能分清是换了知识版本还是换了模型。"""
    generator = LlmSqlGenerator(StaticChatClient({}, model="llama-3.3-70b"))
    assert generator.name == "llm:llama-3.3-70b"


def test_generator_passes_usage_through():
    generator = LlmSqlGenerator(StaticChatClient({"": "```sql\nSELECT 1\n```"}, model="m"))
    generation = generator.generate(_context())
    assert generation.sql == "SELECT 1"
    assert generation.prompt_tokens > 0


def test_generator_checks_budget_before_calling():
    client = StaticChatClient({"": "```sql\nSELECT 1\n```"}, model="m")
    meter = UsageMeter(max_calls=0)
    with pytest.raises(LlmBudgetExceeded):
        LlmSqlGenerator(client, meter=meter).generate(_context())
    assert client.calls == [], "熔断必须发生在花钱之前"


def test_truncated_reply_becomes_actionable_feedback():
    """截断的回复里是半条 SQL。直接交给守卫会得到一个误导性的语法错,
    模型下一轮会去「修」一条本来没写完的 SQL。"""

    class Truncating:
        model = "m"
        params: dict[str, Any] = {}

        def complete(self, prompt: str) -> LlmResponse:
            return LlmResponse(text="SELECT SUM(order_amo", model="m", finish_reason="length")

    generation = LlmSqlGenerator(Truncating()).generate(_context())
    assert generation.sql == ""
    assert "截断" in generation.error
    assert not generation.refused, "截断不是拒答"


def test_reply_without_sql_becomes_actionable_feedback():
    generator = LlmSqlGenerator(StaticChatClient({"": "Ask the data team."}, model="m"))
    generation = generator.generate(_context())
    assert generation.error and "SQL" in generation.error
    assert not generation.refused


def test_refusal_passes_through():
    generator = LlmSqlGenerator(
        StaticChatClient({"": "CANNOT_ANSWER: no campaign_id in the schema"}, model="m")
    )
    generation = generator.generate(_context())
    assert generation.refused
    assert "campaign_id" in (generation.refusal_reason or "")


def test_prompt_is_recorded_for_debugging():
    generator = LlmSqlGenerator(StaticChatClient({"": "```sql\nSELECT 1\n```"}, model="m"))
    generator.generate(_context("How many sessions?"))
    assert "How many sessions?" in generator.last_prompt
    assert "main.fact_orders" in generator.last_prompt


def test_usage_delta_isolates_one_run_from_the_next():
    """计量器跨跑批累计(预算要管整个进程),但 eval_runs 记的是**这一次**的用量。

    不做差的话,第二次跑批会把第一次的 token 再报一遍,
    「这一版比上一版贵了多少」就永远算不对。
    """
    meter = UsageMeter()
    meter.record(LlmResponse(text="a", model="m", prompt_tokens=100, completion_tokens=20))

    checkpoint = meter.snapshot()
    meter.record(LlmResponse(text="b", model="m", prompt_tokens=30, completion_tokens=5))
    meter.record(LlmResponse(text="c", model="m", cached=True))

    run = meter.delta_from(checkpoint)
    assert run.calls == 1 and run.cached_calls == 1
    assert run.total_tokens == 35
    assert meter.total_tokens == 155, "累计值本身不受影响"
