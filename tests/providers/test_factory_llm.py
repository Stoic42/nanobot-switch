"""Tests for :mod:`nanobot.providers.factory`."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nanobot.config.schema import Config
from nanobot.providers.factory import LlmBuildError, build_llm_provider
from nanobot.providers.fallback_chain import FallbackChainProvider


def _mock_async_openai():
    return patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=MagicMock())


def test_build_llm_provider_single_openai():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "openai", "model": "gpt-4.1"}},
            "providers": {"openai": {"apiKey": "sk-test"}},
        }
    )
    with _mock_async_openai():
        p = build_llm_provider(config)
    assert p.__class__.__name__ == "OpenAICompatProvider"


def test_build_llm_provider_skips_hops_without_keys():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "minimax",
                    "model": "MiniMax-M2.7",
                    "llmFallbackChain": [{"provider": "openai", "model": "gpt-4.1"}],
                }
            },
            "providers": {"openai": {"apiKey": "sk-fallback"}},
        }
    )
    with _mock_async_openai():
        p = build_llm_provider(config)
    assert p.__class__.__name__ == "OpenAICompatProvider"
    assert p.get_default_model() == "gpt-4.1"


def test_build_llm_provider_fallback_chain_two_openai_compat_legs():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "minimax",
                    "model": "MiniMax-M2.7",
                    "llmFallbackChain": [{"provider": "moonshot", "model": "kimi-k2.5"}],
                }
            },
            "providers": {
                "minimax": {"apiKey": "sk-mini"},
                "moonshot": {"apiKey": "sk-moon"},
            },
        }
    )
    with _mock_async_openai():
        p = build_llm_provider(config)
    assert isinstance(p, FallbackChainProvider)


def test_build_llm_provider_raises_when_nothing_usable():
    config = Config.model_validate(
        {
            "agents": {
                "defaults": {
                    "provider": "openai",
                    "model": "gpt-4.1",
                    "llmFallbackChain": [{"provider": "moonshot", "model": "kimi-k2.5"}],
                }
            },
            "providers": {},
        }
    )
    with pytest.raises(LlmBuildError):
        build_llm_provider(config)
