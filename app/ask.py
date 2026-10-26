"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --temperature 1.5 "LoRAとは何ですか？"
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（宿題・実装）

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

LLMのAPI自体は会話を覚えていない。--chat では、これまでのやりとりを
history（types.Content のリスト）として毎回まとめて送ることで、続けて質問できるようにする。
"""

import argparse
from collections.abc import Callable

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# --chat を終了する入力
EXIT_WORDS = {"exit", "quit", "q", "終了"}


def make_config(temperature: float | None) -> types.GenerateContentConfig | None:
    """temperature を指定したときだけ config を作る（None ならモデルの既定値）。"""
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=temperature)


def print_usage(response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[トークン数] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL, contents=question, config=make_config(temperature)
    )
    print_usage(response)
    return response.text


def chat(
    client: genai.Client,
    temperature: float | None = None,
    read: Callable[[str], str] = input,
) -> list[types.Content]:
    """会話履歴を保ったまま、続けて質問できるループ。最後の履歴を返す。"""
    history: list[types.Content] = []
    config = make_config(temperature)
    while True:
        try:
            question = read("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question in EXIT_WORDS:
            break
        if not question:
            continue

        history.append(types.Content(role="user", parts=[types.Part(text=question)]))
        try:
            response = client.models.generate_content(
                model=MODEL, contents=history, config=config
            )
        except errors.APIError as e:
            # 失敗した質問は履歴から外し、続けて質問できるようにする
            history.pop()
            print(api_error_message(e.code, e.message))
            continue

        # モデルの返答も履歴に積む。次の質問ではこれも一緒に送られる
        history.append(response.candidates[0].content)
        print(f"\nモデル> {response.text}")
        print_usage(response)
        print(f"[履歴] {len(history)} メッセージ")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--temperature", "-t", type=float, default=None,
        help="0.0〜2.0。省略するとモデルの既定値",
    )
    parser.add_argument(
        "--chat", action="store_true", help="会話履歴を保ったまま続けて質問する"
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.temperature is not None:
        print(f"[temperature] {args.temperature}")

    if args.chat:
        print("続けて質問できます（終了: exit / Ctrl+D）")
        chat(genai.Client(api_key=get_api_key()), args.temperature)
        return

    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
