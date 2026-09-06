import sys
from types import SimpleNamespace

import pytest

from cosq.backends.openai import OpenAIBackend
from cosq.types import GenerationParams


class _FakeCompletions:
    def __init__(self):
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer: ok"))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_openai_backend_builds_payload(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = OpenAIBackend("gpt-4o-mini", revision="hosted-test", system_prompt="be brief")
    fake = _FakeClient()
    backend._client = fake

    out = backend.generate("Question: x", GenerationParams(max_new_tokens=12))

    payload = fake.chat.completions.payload
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"][0] == {"role": "system", "content": "be brief"}
    assert payload["messages"][1] == {"role": "user", "content": "Question: x"}
    assert payload["max_tokens"] == 12
    assert out.text == "Answer: ok"
    assert out.prompt_tokens == 7
    assert out.completion_tokens == 2


def test_openai_backend_requires_a_token(monkeypatch):
    class FakeOpenAIModule:
        class OpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "openai", FakeOpenAIModule)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAIBackend("gpt-4o-mini")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        backend._client_or_create()
