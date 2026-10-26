"""第1回 宿題（基礎）の実験スクリプト。

使い方:
    uv run python -m scripts.lec01_experiment   # モデル・temperature を変えて、同じ質問を3回ずつ

結果は lectures/01/results/basic.json に保存する（APIキーは保存しない）。
"""

import argparse
import json
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.ask import DEFAULT_QUESTION
from app.config import get_api_key

RESULT_PATH = Path(__file__).resolve().parent.parent / "lectures" / "01" / "results" / "basic.json"

LITE = "gemini-3.5-flash-lite"
FLASH = "gemini-3.5-flash"

# (モデル, temperature) の組。None はモデルの既定値
SETTINGS = [(LITE, 0.0), (LITE, 1.0), (LITE, 2.0), (FLASH, None)]


def call(client, model, question, temperature=None, retries=3, wait=30):
    """1回呼んで、回答・応答時間・トークン数を dict で返す。429/503 のときは待って再試行する。"""
    config = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    rate_limited = 0
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=question, config=config)
        except errors.APIError as e:
            elapsed = time.perf_counter() - start
            if e.code in (429, 503) and attempt < retries:
                rate_limited += 1
                print(f"  {e.code}: {wait}秒待って再試行")
                time.sleep(wait)
                continue
            return {"model": model, "temperature": temperature, "question": question,
                    "error": e.code, "error_message": (e.message or "")[:200],
                    "seconds": round(elapsed, 2), "retries": rate_limited}
        elapsed = time.perf_counter() - start
        usage = response.usage_metadata
        return {
            "model": model,
            "temperature": temperature,
            "question": question,
            "answer": response.text,
            "seconds": round(elapsed, 2),
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
            "thoughts_tokens": usage.thoughts_token_count,
            "total_tokens": usage.total_token_count,
            "retries": rate_limited,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--interval", type=float, default=5.0, help="呼び出しの間隔（秒）")
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    jobs = [(m, t) for m, t in SETTINGS for _ in range(args.repeat)]
    results = []
    for i, (model, temperature) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {model} t={temperature}")
        r = call(client, model, DEFAULT_QUESTION, temperature)
        results.append(r)
        print(f"  -> {r.get('error') or 'ok'} {r['seconds']}s out={r.get('output_tokens')}")
        # 保存は毎回行い、途中で止まっても結果が残るようにする
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.interval and i < len(jobs):
            time.sleep(args.interval)
    print(f"保存しました: {RESULT_PATH}")


if __name__ == "__main__":
    main()
