"""Shared fakes that mimic the shape of the OpenAI SDK's chat completion
response, used by both tests/test_agent_loop.py and tests/test_test_writer.py
so a scripted sequence of LLM turns can be fed to the real loop code without
a running Ollama server.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[List[FakeToolCall]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeCompletion:
    def __init__(self, message: FakeMessage) -> None:
        self.choices = [FakeChoice(message)]


class FakeChatCompletions:
    def __init__(self, responses: List[FakeCompletion]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.received_kwargs: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeCompletion:
        self.received_kwargs.append(kwargs)
        if self.calls >= len(self._responses):
            # Ran out of scripted responses: just keep saying nothing to do.
            return FakeCompletion(FakeMessage(content="done", tool_calls=None))
        response = self._responses[self.calls]
        self.calls += 1
        return response


class FakeChat:
    def __init__(self, responses: List[FakeCompletion]) -> None:
        self.completions = FakeChatCompletions(responses)


class FakeClient:
    def __init__(self, responses: List[FakeCompletion]) -> None:
        self.chat = FakeChat(responses)


def tool_call(call_id: str, name: str, arguments: Dict[str, Any]) -> FakeToolCall:
    return FakeToolCall(call_id, name, json.dumps(arguments))


def stop_turn() -> FakeCompletion:
    """A turn where the model calls no tool - used to end a phase."""
    return FakeCompletion(FakeMessage(content="done", tool_calls=None))
