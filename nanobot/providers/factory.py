"""Build a concrete :class:`~nanobot.providers.base.LLMProvider` from :class:`~nanobot.config.schema.Config`."""

from __future__ import annotations

from loguru import logger

from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMProvider


class LlmBuildError(Exception):
    """Raised when the configured LLM route cannot be constructed."""


def _config_for_hop(full: Config, provider: str, model: str) -> Config:
    """Clone *full* with a single primary route and an empty fallback chain."""
    data = full.model_dump(mode="json", by_alias=True)
    agents = data.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    defaults["provider"] = provider
    defaults["model"] = model
    defaults["llmFallbackChain"] = []
    return Config.model_validate(data)


def _validate_leg(config: Config) -> None:
    """Raise :exc:`LlmBuildError` if this hop cannot call the model API."""
    from nanobot.providers.registry import find_by_name

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            raise LlmBuildError("Azure OpenAI requires api_key and api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise LlmBuildError(
                f"No API key configured for provider '{provider_name}' (model {model!r})."
            )


def build_single_leg_provider(config: Config) -> LLMProvider:
    """Instantiate exactly one backend from *config* (must already be a single hop)."""
    from nanobot.providers.registry import find_by_name

    _validate_leg(config)

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    if backend == "openai_codex":
        from nanobot.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "github_copilot":
        from nanobot.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "azure_openai":
        from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key, api_base=p.api_base, default_model=model
        )
    elif backend == "anthropic":
        from nanobot.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    else:
        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def _route_label(provider: str, model: str) -> str:
    return f"{provider}/{model}"


def build_llm_provider(config: Config) -> LLMProvider:
    """Build primary provider, or a :class:`~nanobot.providers.fallback_chain.FallbackChainProvider` when configured."""
    from nanobot.providers.fallback_chain import FallbackChainProvider

    defaults = config.agents.defaults
    hop_specs: list[tuple[str, str, str]] = [
        (_route_label(defaults.provider, defaults.model), defaults.provider, defaults.model)
    ]
    for entry in defaults.llm_fallback_chain:
        hop_specs.append((_route_label(entry.provider, entry.model), entry.provider, entry.model))

    legs: list[tuple[str, LLMProvider]] = []
    for label, prov, mdl in hop_specs:
        hop_cfg = _config_for_hop(config, prov, mdl)
        try:
            legs.append((label, build_single_leg_provider(hop_cfg)))
        except LlmBuildError as exc:
            logger.info("Skipping LLM hop {}: {}", label, exc)

    if not legs:
        raise LlmBuildError(
            "No usable LLM hop: check API keys for the primary model and any llmFallbackChain entries."
        )
    if len(legs) == 1:
        return legs[0][1]
    return FallbackChainProvider(legs)
