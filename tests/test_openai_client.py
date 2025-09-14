# =============================================================================
# File: tests/test_openai_client.py
# Version: 2025-09-14 17:00:00 -03 (America/Sao_Paulo)
# Changes:
# - CORREÇÃO DEFINITIVA: A função mock `mock_create` agora aceita o
#   argumento `self` para simular corretamente um método de instância.
# =============================================================================
import pytest
import asyncio
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

@pytest.mark.asyncio
async def test_ask_openai_mock(monkeypatch):
    monkeypatch.setattr(openai_client.settings, "OPENAI_API_KEY", "dummy-key-for-test")

    # CORREÇÃO: A função de mock precisa aceitar `self` como primeiro argumento
    # para simular corretamente um método de uma instância de classe.
    async def mock_create(self, **kwargs):
        await asyncio.sleep(0) # simula I/O
        return DummyResp("Paris")

    monkeypatch.setattr(
        "openai.resources.chat.completions.AsyncCompletions.create",
        mock_create
    )

    result = await openai_client.ask_openai("Qual a capital da França?")

    assert result["provider"] == "openai"
    assert result["model"] == openai_client.settings.OPENAI_MODEL
    assert result["answer"] == "Paris"
    assert result["usage"]["total_tokens"] == 15

