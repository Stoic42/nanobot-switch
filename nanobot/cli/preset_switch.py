"""One-shot presets to switch the default agent between MiniMax (China), OpenAI GPT, and Kimi."""

from __future__ import annotations

from collections.abc import Callable

from nanobot.config.schema import Config, LlmFallbackEntry

# Defaults align with docs/tests (MiniMax M2.7, Moonshot K2.5, OpenAI GPT-4.1 for coding).
_MINIMAX_CN_OPENAI = "https://api.minimaxi.com/v1"
_MINIMAX_CN_ANTHROPIC = "https://api.minimaxi.com/anthropic"
_MOONSHOT_INTL = "https://api.moonshot.ai/v1"
_MOONSHOT_CN = "https://api.moonshot.cn/v1"


def _set_minimax_cn_bases(config: Config) -> None:
    config.providers.minimax.api_base = _MINIMAX_CN_OPENAI
    config.providers.minimax_anthropic.api_base = _MINIMAX_CN_ANTHROPIC


def _set_moonshot_base(config: Config, url: str) -> None:
    config.providers.moonshot.api_base = url


def _clear_fallback_chain(config: Config) -> None:
    config.agents.defaults.llm_fallback_chain = []


def _after_minimax_cn(config: Config) -> None:
    """Primary: MiniMax mainland; fall back to Kimi (CN) then Qwen on transient failures."""
    _set_minimax_cn_bases(config)
    _set_moonshot_base(config, _MOONSHOT_CN)
    config.agents.defaults.llm_fallback_chain = [
        LlmFallbackEntry(provider="moonshot", model="kimi-k2.5"),
        LlmFallbackEntry(provider="dashscope", model="qwen-max"),
    ]


def _after_gpt(config: Config) -> None:
    _clear_fallback_chain(config)


def _after_kimi_intl(config: Config) -> None:
    _set_moonshot_base(config, _MOONSHOT_INTL)
    _clear_fallback_chain(config)


def _after_kimi_cn(config: Config) -> None:
    _set_moonshot_base(config, _MOONSHOT_CN)
    _clear_fallback_chain(config)


PresetHandler = Callable[[Config], None]

PRESET_CHOICES: dict[str, tuple[str, str, PresetHandler | None]] = {
    # name -> (provider, default_model, post_hook)
    "minimax-cn": ("minimax", "MiniMax-M2.7", _after_minimax_cn),
    "gpt": ("openai", "gpt-4.1", _after_gpt),
    "kimi": ("moonshot", "kimi-k2.5", _after_kimi_intl),
    "kimi-cn": ("moonshot", "kimi-k2.5", _after_kimi_cn),
}


def list_presets() -> list[str]:
    return sorted(PRESET_CHOICES.keys())


def describe_presets() -> str:
    lines = [
        "  minimax-cn  — MiniMax 国内 Coding Plan（minimaxi.com），并写入 llmFallbackChain：",
        "                Kimi（moonshot.cn）→ 通义 Qwen（dashscope）在上游连续失败时依次尝试",
        "  gpt         — OpenAI（清空 fallback 链）",
        "  kimi        — Kimi 国际站（api.moonshot.ai），清空 fallback 链",
        "  kimi-cn     — Kimi 中国站（api.moonshot.cn），清空 fallback 链",
    ]
    return "\n".join(lines)


def apply_llm_preset(config: Config, preset: str, model_override: str | None = None) -> tuple[str, str]:
    """Apply preset in place. Returns (provider, model) after apply."""
    key = preset.strip().lower()
    if key not in PRESET_CHOICES:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(list_presets())}")

    provider, default_model, hook = PRESET_CHOICES[key]
    model = (model_override or default_model).strip()
    config.agents.defaults.provider = provider
    config.agents.defaults.model = model
    if hook is not None:
        hook(config)
    return provider, model
