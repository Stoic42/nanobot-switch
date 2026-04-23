"""Tests for nanobot.cli.preset_switch."""

from nanobot.cli.preset_switch import PRESET_CHOICES, apply_llm_preset
from nanobot.config.schema import Config, LlmFallbackEntry


def test_apply_minimax_cn_sets_bases_model_and_fallback_chain():
    config = Config()
    apply_llm_preset(config, "minimax-cn")
    assert config.agents.defaults.provider == "minimax"
    assert config.agents.defaults.model == "MiniMax-M2.7"
    assert config.providers.minimax.api_base == "https://api.minimaxi.com/v1"
    assert config.providers.minimax_anthropic.api_base == "https://api.minimaxi.com/anthropic"
    assert config.providers.moonshot.api_base == "https://api.moonshot.cn/v1"
    assert config.agents.defaults.llm_fallback_chain == [
        LlmFallbackEntry(provider="moonshot", model="kimi-k2.5"),
        LlmFallbackEntry(provider="dashscope", model="qwen-max"),
    ]


def test_apply_gpt():
    config = Config()
    config.agents.defaults.llm_fallback_chain = [
        LlmFallbackEntry(provider="moonshot", model="kimi-k2.5"),
    ]
    apply_llm_preset(config, "gpt")
    assert config.agents.defaults.provider == "openai"
    assert config.agents.defaults.model == "gpt-4.1"
    assert config.agents.defaults.llm_fallback_chain == []


def test_apply_kimi_moonshot_urls():
    config = Config()
    apply_llm_preset(config, "kimi")
    assert config.agents.defaults.provider == "moonshot"
    assert config.providers.moonshot.api_base == "https://api.moonshot.ai/v1"

    apply_llm_preset(config, "kimi-cn")
    assert config.providers.moonshot.api_base == "https://api.moonshot.cn/v1"


def test_model_override():
    config = Config()
    apply_llm_preset(config, "gpt", model_override="gpt-4o")
    assert config.agents.defaults.model == "gpt-4o"


def test_unknown_preset_raises():
    config = Config()
    try:
        apply_llm_preset(config, "nope")
    except ValueError as e:
        assert "Unknown preset" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_all_presets_registered():
    assert set(PRESET_CHOICES.keys()) == {"minimax-cn", "gpt", "kimi", "kimi-cn"}
