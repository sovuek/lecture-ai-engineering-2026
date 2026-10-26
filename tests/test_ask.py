"""ask.py のテスト（LLMもAPIキーも使わない）。"""

from app import ask


def test_make_config():
    assert ask.make_config(None) is None
    assert ask.make_config(0.5).temperature == 0.5
