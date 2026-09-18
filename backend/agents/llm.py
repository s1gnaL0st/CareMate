"""Central model configuration for chat and vision calls.

All configured providers use an OpenAI-compatible API. ``ark`` remains the
default so existing deployments continue to work without changing ``.env``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from config import get_settings


ModelPurpose = Literal["chat", "vision"]


class LLMConfigurationError(ValueError):
    """Raised when model provider environment variables are incomplete."""


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    api_key: str = field(repr=False)
    model: str
    base_url: str | None
    extra_body: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _ProviderProfile:
    api_key_envs: tuple[str, ...]
    model_envs: tuple[str, ...]
    base_url_envs: tuple[str, ...] = ()
    vision_model_envs: tuple[str, ...] = ()
    default_base_url: str | None = None
    default_model: str | None = None
    requires_base_url: bool = False
    extra_body: Mapping[str, Any] | None = None


_PROVIDERS: dict[str, _ProviderProfile] = {
    "ark": _ProviderProfile(
        api_key_envs=("ARK_API_KEY",),
        model_envs=("ARK_MODEL_ID",),
        base_url_envs=("ARK_BASE_URL",),
        vision_model_envs=("ARK_VISION_MODEL_ID",),
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-seed-1-6-flash-250828",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "openai": _ProviderProfile(
        api_key_envs=("OPENAI_API_KEY",),
        model_envs=("OPENAI_MODEL",),
        base_url_envs=("OPENAI_BASE_URL",),
        vision_model_envs=("OPENAI_VISION_MODEL",),
    ),
    "deepseek": _ProviderProfile(
        api_key_envs=("DEEPSEEK_API_KEY",),
        model_envs=("DEEPSEEK_MODEL",),
        base_url_envs=("DEEPSEEK_BASE_URL",),
        vision_model_envs=("DEEPSEEK_VISION_MODEL",),
        default_base_url="https://api.deepseek.com",
    ),
    "qwen": _ProviderProfile(
        api_key_envs=("DASHSCOPE_API_KEY",),
        model_envs=("QWEN_MODEL", "DASHSCOPE_MODEL"),
        base_url_envs=("DASHSCOPE_BASE_URL",),
        vision_model_envs=("QWEN_VISION_MODEL",),
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "zhipu": _ProviderProfile(
        api_key_envs=("ZAI_API_KEY", "ZHIPU_API_KEY"),
        model_envs=("ZHIPU_MODEL", "ZAI_MODEL"),
        base_url_envs=("ZHIPU_BASE_URL", "ZAI_BASE_URL"),
        vision_model_envs=("ZHIPU_VISION_MODEL", "ZAI_VISION_MODEL"),
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
    ),
    "custom": _ProviderProfile(
        api_key_envs=(),
        model_envs=(),
        requires_base_url=True,
    ),
}

_PROVIDER_ALIASES = {
    "volcengine": "ark",
    "doubao": "ark",
    "dashscope": "qwen",
    "tongyi": "qwen",
    "glm": "zhipu",
    "zai": "zhipu",
    "compatible": "custom",
    "openai-compatible": "custom",
}

_PRESETS: dict[str, float] = {
    "router": 0.0,
    "precise": 0.2,
    "balanced": 0.4,
    "fast": 0.7,
}


def _first_value(env: Mapping[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = env.get(name, "").strip()
        if value:
            return value
    return ""


def _canonical_provider(raw_provider: str) -> str:
    normalized = raw_provider.strip().lower().replace("_", "-") or "ark"
    provider = _PROVIDER_ALIASES.get(normalized, normalized)
    if provider not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS)
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER '{raw_provider}'. Supported providers: {supported}"
        )
    return provider


def resolve_model_settings(
    env: Mapping[str, str] | None = None,
    *,
    purpose: ModelPurpose = "chat",
) -> ModelSettings:
    """Resolve and validate provider settings without making a network call."""
    if purpose not in ("chat", "vision"):
        raise LLMConfigurationError("Model purpose must be 'chat' or 'vision'")

    values = os.environ if env is None else env
    chat_provider = _canonical_provider(values.get("LLM_PROVIDER", "ark"))
    vision_provider_raw = values.get("VISION_PROVIDER", "")
    provider = (
        _canonical_provider(vision_provider_raw)
        if purpose == "vision" and vision_provider_raw.strip()
        else chat_provider
    )
    profile = _PROVIDERS[provider]
    inherits_chat_settings = purpose == "chat" or provider == chat_provider

    api_key_names = ("LLM_API_KEY", *profile.api_key_envs)
    model_names = ("LLM_MODEL", *profile.model_envs)
    base_url_names = ("LLM_BASE_URL", *profile.base_url_envs)

    if purpose == "vision":
        api_key_names = (
            "VISION_API_KEY",
            *(("LLM_API_KEY",) if inherits_chat_settings else ()),
            *profile.api_key_envs,
        )
        model_names = (
            "VISION_MODEL",
            *profile.vision_model_envs,
            *(("LLM_MODEL",) if inherits_chat_settings else ()),
            *profile.model_envs,
        )
        base_url_names = (
            "VISION_BASE_URL",
            *(("LLM_BASE_URL",) if inherits_chat_settings else ()),
            *profile.base_url_envs,
        )

    api_key = _first_value(values, api_key_names)
    if not api_key:
        raise LLMConfigurationError(
            f"Missing model API key. Configure one of: {', '.join(api_key_names)}"
        )

    model = _first_value(values, model_names) or profile.default_model or ""
    if not model:
        raise LLMConfigurationError(
            f"Missing model name. Configure one of: {', '.join(model_names)}"
        )

    base_url = _first_value(values, base_url_names) or profile.default_base_url
    if profile.requires_base_url and not base_url:
        raise LLMConfigurationError(
            f"Missing compatible API base URL. Configure one of: {', '.join(base_url_names)}"
        )

    return ModelSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        extra_body=profile.extra_body,
    )


def resolve_fallback_model_settings(
    env: Mapping[str, str] | None = None,
) -> ModelSettings | None:
    """Resolve an optional generic fallback without exposing provider secrets.

    Fallback settings intentionally use a provider-neutral prefix so operators
    can change vendors without adding provider-specific code to deployment.
    """
    values = os.environ if env is None else env
    provider = values.get("LLM_FALLBACK_PROVIDER", "").strip()
    if not provider:
        return None
    fallback_values = dict(values)
    fallback_values.update(
        {
            "LLM_PROVIDER": provider,
            "LLM_API_KEY": values.get("LLM_FALLBACK_API_KEY", ""),
            "LLM_MODEL": values.get("LLM_FALLBACK_MODEL", ""),
            "LLM_BASE_URL": values.get("LLM_FALLBACK_BASE_URL", ""),
        }
    )
    return resolve_model_settings(fallback_values)


def _build_chat_openai(
    settings: ModelSettings,
    *,
    streaming: bool,
    temperature: float,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> ChatOpenAI:
    resolved_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    )
    resolved_retries = (
        max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", "2"))
    )
    kwargs: dict[str, Any] = {
        "api_key": settings.api_key,
        "model": settings.model,
        "temperature": temperature,
        "streaming": streaming,
        "timeout": resolved_timeout,
        "max_retries": max(0, resolved_retries),
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.extra_body:
        kwargs["extra_body"] = settings.extra_body
    return ChatOpenAI(**kwargs)


def get_chat_llm(
    preset: str = "balanced",
    *,
    streaming: bool = True,
    temperature: float | None = None,
) -> Any:
    """Return the configured OpenAI-compatible LangChain chat model."""
    settings = resolve_model_settings()
    resolved_temp = temperature if temperature is not None else _PRESETS.get(preset, 0.4)

    primary = _build_chat_openai(
        settings,
        streaming=streaming,
        temperature=resolved_temp,
    )
    fallback_settings = resolve_fallback_model_settings()
    if fallback_settings is None:
        return primary
    fallback = _build_chat_openai(
        fallback_settings,
        streaming=streaming,
        temperature=resolved_temp,
    )
    return primary.with_fallbacks([fallback])


def get_clinic_llm(*, temperature: float = 0.0) -> Any:
    """Return the optional clinic-only model without changing global routing.

    The dedicated endpoint is intentionally not part of ``get_chat_llm``:
    supervisor, planner, responder, RAG and every other agent keep their
    existing provider. If no clinic endpoint is configured, the clinic falls
    back to the current global model for backwards-compatible local runs.
    """
    settings = get_settings()
    base_url = settings.clinic_llm_base_url.strip()
    if not settings.clinic_llm_enabled or not base_url:
        return get_chat_llm("balanced", streaming=False, temperature=temperature)
    dedicated = ModelSettings(
        provider="clinic_local",
        api_key=settings.clinic_llm_api_key or "dummy",
        model=settings.clinic_llm_model or "grpo-200",
        base_url=base_url,
    )
    return _build_chat_openai(
        dedicated,
        streaming=False,
        temperature=temperature,
        timeout_seconds=settings.clinic_llm_timeout_seconds,
        # A local clinic endpoint should fail fast when the SSH tunnel is down;
        # retrying a refused connection only makes the chat appear frozen.
        max_retries=0,
    )
