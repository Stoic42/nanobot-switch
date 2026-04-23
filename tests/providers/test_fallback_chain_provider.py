"""Tests for :class:`~nanobot.providers.fallback_chain.FallbackChainProvider`."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.fallback_chain import FallbackChainProvider


class _StubLeg(LLMProvider):
    def __init__(self, label: str, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.label = label
        self._responses = responses
        self.calls = 0

    def get_default_model(self) -> str:
        return "stub-model"

    async def chat(
        self,
        messages,
        tools=None,
        model=None,
        max_tokens=4096,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    ) -> LLMResponse:
        raise AssertionError("unused in this test")

    async def chat_with_retry(self, **kwargs) -> LLMResponse:
        r = self._responses[self.calls]
        self.calls += 1
        return r

    async def chat_stream_with_retry(self, **kwargs) -> LLMResponse:
        return await self.chat_with_retry(**kwargs)


@pytest.mark.asyncio
async def test_fallback_moves_to_next_leg_on_transient_error():
    err = LLMResponse(content="rate limit exceeded", finish_reason="error")
    ok = LLMResponse(content="hello", finish_reason="stop")
    a = _StubLeg("a", [err])
    b = _StubLeg("b", [ok])
    chain = FallbackChainProvider([("leg-a", a), ("leg-b", b)])

    with patch("nanobot.providers.fallback_chain.write_active_llm_route") as mock_write:
        r = await chain.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert r.content == "hello"
    assert a.calls == 1
    assert b.calls == 1
    mock_write.assert_called_once_with("leg-b")


@pytest.mark.asyncio
async def test_fallback_stops_on_non_transient_error():
    hard = LLMResponse(content="insufficient_quota", finish_reason="error")
    a = _StubLeg("a", [hard])
    b = _StubLeg("b", [LLMResponse(content="no", finish_reason="stop")])
    chain = FallbackChainProvider([("leg-a", a), ("leg-b", b)])

    with patch("nanobot.providers.fallback_chain.write_active_llm_route") as mock_write:
        r = await chain.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert "quota" in (r.content or "").lower()
    assert a.calls == 1
    assert b.calls == 0
    mock_write.assert_not_called()
