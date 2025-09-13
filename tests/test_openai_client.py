# tests/test_openai_client.py
import pytest
from app import openai_client


class DummyMessage:
    def __init__(self, content):
        self.content = content


class DummyChoice:
    def __init__(self, content):
        self.message = DummyMessage(content)


class DummyUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class DummyResp:
    def __init__(self, text):
        self.choices = [DummyChoice(text)]
        self.usage = DummyUsage()


def test_ask_openai_mock(monkeypatch):
    """
    Testa ask_openai() sem chamar a API real, mockando a resposta.
    """

    # força settings a ter uma API key dummy
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    # patch no _build_client -> retorna objeto com chat.completions.create
    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return DummyResp("Paris")

    monkeypatch.setattr(openai_client, "_build_client", lambda timeout=None: DummyClient())

    result = openai_client.ask_openai("Qual a capital da França?")

    assert result["provider"] == "openai"
    assert result["model"] == openai_client.settings.OPENAI_MODEL
    assert result["answer"] == "Paris"
    assert result["usage"]["total_tokens"] == 15
