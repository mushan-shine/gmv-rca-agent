"""LLM 客户端:一种线格式,多个供应商。

只实现 **OpenAI chat-completions 线格式**,因为下面这些都讲它:

* Databricks Foundation Model APIs / Model Serving
  (``https://<workspace>/serving-endpoints``)
* OpenAI (``https://api.openai.com/v1``)
* 智谱开放平台 (``https://open.bigmodel.cn/api/paas/v4``,GLM-4-Flash 免费)
* 各类 OpenAI 兼容端点(自建代理、vLLM、Anthropic 的兼容层)

于是换供应商只是换 ``base_url`` + ``model``,不引第三方 SDK ——
用标准库 ``urllib`` 发请求,在 Databricks notebook、CI、本机里行为一致。

三件真实接入 LLM 才会遇到、而 stub 遇不到的事
--------------------------------------------

**1. 传输层重试 ≠ 自修复循环。**
``429`` / ``503`` / 超时要重发**同一个 prompt**,退避等待 —— 这里没有任何新信息,
它不该被记成一次「尝试」。而 L1 自修复发的是**带错误反馈的新 prompt**,那才是一次尝试。
两者混同会直接污染 ``avg_attempts`` 这个指标:一次网络抖动会被读成「模型改了两轮才对」。
所以退避重试封在这一层,循环层根本看不见。

**2. 评估会把同一个 prompt 打很多遍。**
26 条 case × 每次改动重跑 × 多个知识版本 —— 大量请求的 prompt 逐字相同。
:class:`CachingChatClient` 按 (模型 + 参数 + prompt) 的指纹缓存,
让迭代不必反复付费,也让同一版本的评估结果可复现。

**3. 预算必须能熔断。**
:class:`UsageMeter` 累计调用数与 token,超限直接抛错而不是继续烧。
这是原简报 §10「预算熔断」在模型侧的对应物。

凭据只从环境变量读,不进日志、不进 ``repr``、不拼进 URL。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ..errors import RcaError

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LlmError(RcaError):
    """调用模型失败。"""


class LlmTransportError(LlmError):
    """网络/服务端错误,重试之后仍未成功。"""


class LlmBudgetExceeded(LlmError):
    """超出本次跑批的调用或 token 预算 —— 熔断,不再继续。"""


class LlmConfigError(LlmError):
    """缺少必要的连接配置。"""


# ---------------------------------------------------------------------------
# 结果与用量
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LlmResponse:
    """一次模型调用的结果。"""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    """**传输层**重发次数,不是自修复轮数。"""

    cached: bool = False
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def filtered(self) -> bool:
        """回复被供应商的内容安全策略拦截(智谱返回 ``finish_reason="sensitive"``)。

        被拦截的回复通常是空的。不单独识别的话,它会被当成「模型没写出 SQL」,
        自修复循环会徒劳地让模型「把 SQL 放进代码块」。
        """
        return self.finish_reason in {"sensitive", "content_filter"}

    @property
    def truncated(self) -> bool:
        """回复被 ``max_tokens`` 截断。

        截断的回复里常常是半条 SQL —— 让它进守卫会得到一个误导性的语法错,
        模型下一轮会去改一条本来没错的 SQL。调用方应当直接加大 max_tokens。
        """
        return self.finish_reason == "length"


@dataclass
class UsageMeter:
    """累计用量,并在超限时熔断。"""

    max_calls: int | None = None
    max_tokens: int | None = None
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def check_budget(self) -> None:
        """在**发起调用之前**检查。超了就抛,不要先花钱再报告。"""
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise LlmBudgetExceeded(
                f"已达调用上限({self.calls}/{self.max_calls})。"
                f"评估跑批到此熔断 —— 调高 max_calls 或缩小评估集。"
            )
        if self.max_tokens is not None and self.total_tokens >= self.max_tokens:
            raise LlmBudgetExceeded(
                f"已达 token 上限({self.total_tokens}/{self.max_tokens})。"
            )

    def record(self, response: LlmResponse) -> None:
        if response.cached:
            self.cached_calls += 1
            return
        self.calls += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.total_latency_ms += response.latency_ms

    def snapshot(self) -> "UsageMeter":
        """当前计数的快照(不带预算)。用于算「某一段区间的用量」。"""
        return UsageMeter(
            calls=self.calls,
            cached_calls=self.cached_calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_latency_ms=self.total_latency_ms,
        )

    def delta_from(self, earlier: "UsageMeter") -> "UsageMeter":
        """自 ``earlier`` 以来的增量。

        计量器是**跨跑批累计**的(预算要管整个进程),
        而 ``nl2sql_eval_runs`` 要记的是**这一次跑批**花了多少 ——
        不做差的话,第二次跑批会把第一次的用量再报一遍。
        """
        return UsageMeter(
            calls=self.calls - earlier.calls,
            cached_calls=self.cached_calls - earlier.cached_calls,
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=self.completion_tokens - earlier.completion_tokens,
            total_latency_ms=self.total_latency_ms - earlier.total_latency_ms,
        )

    def summary(self) -> str:
        return (
            f"调用 {self.calls} 次(命中缓存 {self.cached_calls} 次) · "
            f"token {self.prompt_tokens}+{self.completion_tokens}={self.total_tokens} · "
            f"累计 {self.total_latency_ms / 1000:.1f}s"
        )


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------
@runtime_checkable
class ChatClient(Protocol):
    model: str

    def complete(self, prompt: str) -> LlmResponse:
        ...


def prompt_fingerprint(model: str, params: Mapping[str, Any], prompt: str) -> str:
    """(模型 + 参数 + prompt) 的指纹。缓存键,也是可复现性的锚点。"""
    payload = json.dumps(
        {"model": model, "params": dict(sorted(params.items())), "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class OpenAIChatClient:
    """任何讲 OpenAI chat-completions 的端点。

    Args:
        base_url: 例如 ``https://<workspace>.cloud.databricks.com/serving-endpoints``
            或 ``https://api.openai.com/v1``。末尾斜杠可有可无。
        api_key: 凭据。**只从环境变量传进来**,不会进日志或 ``repr``。
        model: Databricks 下是 serving endpoint 名;OpenAI 下是模型名。
        temperature: 默认 0 —— 评估要求可复现,采样噪声会让指标的涨跌无法归因。
    """

    base_url: str
    api_key: str = field(repr=False)
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    system_prompt: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    """供应商特有的请求字段,原样并入请求体。例如智谱的 ``{"do_sample": False}``。"""

    def __repr__(self) -> str:  # 防止 key 出现在异常栈里
        return (
            f"OpenAIChatClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"api_key='***', temperature={self.temperature})"
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            # 进缓存键:do_sample 开/关的回复不能互相冒充
            "extra_body": json.dumps(dict(self.extra_body), sort_keys=True),
        }

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _payload(self, prompt: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **dict(self.extra_body),
        }

    def complete(self, prompt: str) -> LlmResponse:
        """发一次请求。``429`` / ``5xx`` / 超时会退避重发**同一个 prompt**。

        Raises:
            LlmTransportError: 重试用尽仍失败,或返回体不是预期结构。
        """
        body = json.dumps(self._payload(prompt)).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **dict(self.extra_headers),
        }
        started = time.perf_counter()
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                self._endpoint(), data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return _parse_chat_response(
                    raw,
                    fallback_model=self.model,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempt,
                )
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise LlmTransportError(
                        f"调用 {self.model} 失败({last_error})"
                    ) from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries:
                    raise LlmTransportError(
                        f"调用 {self.model} 失败({last_error})"
                    ) from None
            _sleep_backoff(attempt)

        raise LlmTransportError(f"调用 {self.model} 失败({last_error})")


def _sleep_backoff(attempt: int, base: float = 0.5, cap: float = 8.0) -> None:
    """指数退避 + 抖动。

    抖动不是讲究:评估是**批量**打请求的,同步退避会让所有请求在同一刻再次撞上限流,
    形成自我强化的尖峰。
    """
    delay = min(cap, base * (2 ** (attempt - 1)))
    time.sleep(delay * (0.5 + random.random() * 0.5))


def _read_error_body(exc: urllib.error.HTTPError, limit: int = 300) -> str:
    try:
        text = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 读错误体失败不该盖过原始错误
        return exc.reason or ""
    return text[:limit]


def _parse_chat_response(
    raw: Mapping[str, Any], *, fallback_model: str, latency_ms: int, attempts: int
) -> LlmResponse:
    choices = raw.get("choices") or []
    if not choices:
        raise LlmTransportError(f"返回体里没有 choices:{str(raw)[:300]}")
    choice = choices[0]
    message = choice.get("message") or {}
    usage = raw.get("usage") or {}
    return LlmResponse(
        text=message.get("content") or "",
        model=str(raw.get("model") or fallback_model),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_ms=latency_ms,
        attempts=attempts,
        finish_reason=str(choice.get("finish_reason") or ""),
    )


@dataclass
class CachingChatClient:
    """按 (模型 + 参数 + prompt) 指纹缓存的包装器。

    评估会把逐字相同的 prompt 打很多遍(重跑、对比知识版本、调试单条 case)。
    缓存让这些重复不再付费,也让**同一版本的评估结果完全可复现** ——
    否则即便 ``temperature=0``,供应商侧的变动也会让两次跑批对不上。

    默认只在进程内缓存。传 ``store`` 可以接任意持久化后端
    (``get(key)`` / ``put(key, value)``),例如一张 Delta 表。
    """

    inner: ChatClient
    store: Any = None
    hits: int = 0
    misses: int = 0
    _memory: dict[str, LlmResponse] = field(default_factory=dict, repr=False)

    @property
    def model(self) -> str:
        return self.inner.model

    def key_for(self, prompt: str) -> str:
        params = getattr(self.inner, "params", {})
        return prompt_fingerprint(self.inner.model, params, prompt)

    def complete(self, prompt: str) -> LlmResponse:
        key = self.key_for(prompt)
        cached = self._memory.get(key)
        if cached is None and self.store is not None:
            cached = self.store.get(key)
        if cached is not None:
            self.hits += 1
            return LlmResponse(
                text=cached.text,
                model=cached.model,
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
                latency_ms=0,
                attempts=cached.attempts,
                cached=True,
                finish_reason=cached.finish_reason,
            )

        self.misses += 1
        response = self.inner.complete(prompt)
        self._memory[key] = response
        if self.store is not None:
            self.store.put(key, response)
        return response


@dataclass
class StaticChatClient:
    """返回预设回复,不发任何网络请求。测试与离线演示用。"""

    responses: Mapping[str, str]
    model: str = "static"
    default: str = ""
    calls: list[str] = field(default_factory=list)

    @property
    def params(self) -> dict[str, Any]:
        return {}

    def complete(self, prompt: str) -> LlmResponse:
        self.calls.append(prompt)
        for needle, reply in self.responses.items():
            if needle in prompt:
                return LlmResponse(text=reply, model=self.model, prompt_tokens=len(prompt) // 4)
        return LlmResponse(text=self.default, model=self.model)


# ---------------------------------------------------------------------------
# 从配置装配
# ---------------------------------------------------------------------------
ENV_PROVIDER = "RCA_LLM_PROVIDER"
ENV_MODEL = "RCA_LLM_MODEL"
ENV_BASE_URL = "RCA_LLM_BASE_URL"
ENV_API_KEY = "RCA_LLM_API_KEY"
ENV_MAX_CALLS = "RCA_LLM_MAX_CALLS"
ENV_MAX_TOKENS = "RCA_LLM_MAX_TOKENS"

PROVIDERS = ("databricks", "zhipu", "openai", "custom")

DEFAULT_DATABRICKS_MODEL = "databricks-meta-llama-3-3-70b-instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_ZHIPU_MODEL = "glm-4-flash"
"""智谱开放平台的免费模型。"""

ZHIPU_EXTRA_BODY: dict[str, Any] = {"do_sample": False}
"""智谱的贪婪解码开关。

``do_sample=False`` 时 temperature / top_p 不生效,模型走确定性解码 ——
评估要求同一个 prompt 两次得到同一个答案,否则指标的涨跌无法归因。
"""


def build_chat_client(
    env: Mapping[str, str] | None = None,
    *,
    cache: bool = True,
    **overrides: Any,
) -> ChatClient:
    """按环境变量装配一个客户端。

    ============================  =================================================
    ``RCA_LLM_PROVIDER``          ``databricks``(默认) / ``zhipu`` / ``openai`` / ``custom``
    ``RCA_LLM_MODEL``             endpoint 名或模型名
    ``RCA_LLM_BASE_URL``          ``custom`` 必填;另两者可覆盖默认推导
    ``RCA_LLM_API_KEY``           凭据;``databricks`` 缺省用 ``DATABRICKS_TOKEN``,
                                  ``zhipu`` 缺省用 ``ZHIPUAI_API_KEY``
    ============================  =================================================

    ``databricks`` 的 base_url 由 ``DATABRICKS_SERVER_HOSTNAME`` 推导成
    ``https://<host>/serving-endpoints`` —— 请求不出 Databricks,
    符合 Free Edition 出站受限的约束。

    Raises:
        LlmConfigError: 缺少必要配置,错误信息里会写明要设哪几个变量。
    """
    from ..config import ENV_HOSTNAME, ENV_TOKEN, load_dotenv

    if env is None:
        load_dotenv()
        env = os.environ

    provider = (overrides.pop("provider", None) or env.get(ENV_PROVIDER) or "databricks").strip()
    if provider not in PROVIDERS:
        raise LlmConfigError(f"未知供应商 {provider!r};可选:{PROVIDERS}")

    base_url = overrides.pop("base_url", None) or env.get(ENV_BASE_URL) or ""
    api_key = overrides.pop("api_key", None) or env.get(ENV_API_KEY) or ""
    model = overrides.pop("model", None) or env.get(ENV_MODEL) or ""

    if provider == "databricks":
        model = model or DEFAULT_DATABRICKS_MODEL
        api_key = api_key or env.get(ENV_TOKEN, "")
        if not base_url:
            hostname = (env.get(ENV_HOSTNAME) or "").strip().rstrip("/")
            for prefix in ("https://", "http://"):
                if hostname.startswith(prefix):
                    hostname = hostname[len(prefix) :]
            if not hostname:
                raise LlmConfigError(
                    f"provider=databricks 需要 {ENV_HOSTNAME}(或显式传 {ENV_BASE_URL})。\n"
                    f"在 Databricks notebook 里运行时不必手工设 —— "
                    f"见 notebooks/03_nl2sql_loop.py 的装配格。"
                )
            base_url = f"https://{hostname}/serving-endpoints"
        if not api_key:
            raise LlmConfigError(
                f"provider=databricks 需要 {ENV_TOKEN}(或 {ENV_API_KEY})。"
            )
    elif provider == "zhipu":
        model = model or DEFAULT_ZHIPU_MODEL
        base_url = base_url or ZHIPU_BASE_URL
        api_key = api_key or env.get("ZHIPUAI_API_KEY", "")
        overrides.setdefault("extra_body", dict(ZHIPU_EXTRA_BODY))
        if not api_key:
            raise LlmConfigError(
                f"provider=zhipu 需要 {ENV_API_KEY} 或 ZHIPUAI_API_KEY。\n"
                f"在 https://open.bigmodel.cn 控制台的「API Keys」页面创建。\n"
                f"⚠ 请求会出站到 open.bigmodel.cn,Databricks Free Edition 可能连不上 —— "
                f"先用 check_connectivity() 确认。"
            )
    elif provider == "openai":
        model = model or DEFAULT_OPENAI_MODEL
        base_url = base_url or "https://api.openai.com/v1"
        api_key = api_key or env.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LlmConfigError(
                f"provider=openai 需要 {ENV_API_KEY} 或 OPENAI_API_KEY。\n"
                f"⚠ 走外部 API 意味着请求要出站。Databricks Free Edition 的出站网络"
                f"可能被限制,并且业务数据的 schema 会离开你的 workspace —— "
                f"这两点在方案里要写清楚。"
            )
    else:  # custom
        if not base_url or not model:
            raise LlmConfigError(
                f"provider=custom 需要同时给出 {ENV_BASE_URL} 与 {ENV_MODEL}。"
            )

    client: ChatClient = OpenAIChatClient(
        base_url=base_url, api_key=api_key, model=model, **overrides
    )
    return CachingChatClient(client) if cache else client


def build_usage_meter(env: Mapping[str, str] | None = None) -> UsageMeter:
    """从环境变量读预算上限。不设就是不限。"""
    env = os.environ if env is None else env

    def _int(name: str) -> int | None:
        raw = env.get(name)
        return int(raw) if raw and raw.strip().isdigit() else None

    return UsageMeter(max_calls=_int(ENV_MAX_CALLS), max_tokens=_int(ENV_MAX_TOKENS))


def check_connectivity(base_url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """检查当前运行环境能否连到 ``base_url`` 所在的主机。

    **只要拿到任何 HTTP 响应就算连通** —— 包括 401 / 404,那说明网络是通的,
    只是请求本身不对。只有 DNS 失败、连接被拒、超时才算不通。

    为什么需要它:Databricks Free Edition 限制出站网络。连不上外部模型时,
    自修复循环会把每条 case 都记成 ``generator_error`` —— 看起来像模型全挂了,
    其实是网络。先查清楚,免得白跑一批评估。

    Returns:
        ``(是否连通, 说明)``。说明里不包含任何凭据。
    """
    request = urllib.request.Request(base_url.rstrip("/") + "/", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, f"连通(HTTP {response.status})"
    except urllib.error.HTTPError as exc:
        return True, f"连通(HTTP {exc.code},网络可达)"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"连不上:{type(exc).__name__}: {reason}"
