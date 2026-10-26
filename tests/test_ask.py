"""ask.py の会話履歴のテスト（LLMもAPIキーも使わない）。"""

from types import SimpleNamespace

from google.genai import types

from app import ask


class FakeModels:
    """generate_content に渡された contents を記録し、決まった返事を返す。"""

    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append(list(contents))
        reply = f"返事{len(self.calls)}"
        return SimpleNamespace(
            text=reply,
            candidates=[
                SimpleNamespace(
                    content=types.Content(role="model", parts=[types.Part(text=reply)])
                )
            ],
            usage_metadata=None,
        )


def fake_input(lines):
    it = iter(lines)

    def read(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return read


def test_chat_sends_previous_turns_with_each_question():
    client = SimpleNamespace(models=FakeModels())
    history = ask.chat(client, read=fake_input(["私の名前は佐藤です", "私の名前は？"]))

    first, second = client.models.calls
    assert [c.parts[0].text for c in first] == ["私の名前は佐藤です"]
    # 2回目には、1回目の質問とモデルの返事も一緒に送られている
    assert [c.parts[0].text for c in second] == ["私の名前は佐藤です", "返事1", "私の名前は？"]
    assert [c.role for c in history] == ["user", "model", "user", "model"]


def test_chat_ends_on_exit_and_skips_empty_input():
    client = SimpleNamespace(models=FakeModels())
    history = ask.chat(client, read=fake_input(["", "exit", "届かない質問"]))
    assert client.models.calls == []
    assert history == []


def test_make_config():
    assert ask.make_config(None) is None
    assert ask.make_config(0.5).temperature == 0.5
