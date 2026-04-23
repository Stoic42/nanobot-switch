"""Try multiple LLM backends in order when the previous one fails with a transient error."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.active_llm import write_active_llm_route
from nanobot.providers.base import LLMProvider, LLMResponse


class FallbackChainProvider(LLMProvider):
    """Delegate to the first sub-provider; on transient ``finish_reason=error``, try the next."""

    def __init__(self, legs: list[tuple[str, LLMProvider]]) -> None:
        super().__init__()
        if not legs:
            raise ValueError("FallbackChainProvider requires at least one leg")
        self._legs = legs

    def get_default_model(self) -> str:
        return self._legs[0][1].get_default_model()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        last: LLMResponse | None = None
        for label, sub in self._legs:
            leg_model = sub.get_default_model()
            last = await sub.chat(
                messages=messages,
                tools=tools,
                model=leg_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
            if last.finish_reason != "error":
                write_active_llm_route(label)
                return last
            if not self._is_transient_response(last):
                return last
            logger.warning("LLM fallback (chat): {} returned error, trying next backend", label)
        return last if last is not None else LLMResponse(content="No LLM backend available", finish_reason="error")

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        last: LLMResponse | None = None
        for label, sub in self._legs:
            leg_model = sub.get_default_model()
            last = await sub.chat_with_retry(
                messages=messages,
                tools=tools,
                model=leg_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
                retry_mode=retry_mode,
                on_retry_wait=on_retry_wait,
            )
            if last.finish_reason != "error":
                write_active_llm_route(label)
                return last
            if not self._is_transient_response(last):
                return last
            logger.warning(
                "LLM fallback: {} failed after retries (transient), trying next backend",
                label,
            )
        return last if last is not None else LLMResponse(content="No LLM backend available", finish_reason="error")

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        retry_mode: str = "standard",
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        last: LLMResponse | None = None
        for label, sub in self._legs:
            leg_model = sub.get_default_model()
            last = await sub.chat_stream_with_retry(
                messages=messages,
                tools=tools,
                model=leg_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
                on_content_delta=on_content_delta,
                retry_mode=retry_mode,
                on_retry_wait=on_retry_wait,
            )
            if last.finish_reason != "error":
                write_active_llm_route(label)
                return last
            if not self._is_transient_response(last):
                return last
            logger.warning(
                "LLM fallback (stream): {} failed after retries (transient), trying next backend",
                label,
            )
        return last if last is not None else LLMResponse(content="No LLM backend available", finish_reason="error")
