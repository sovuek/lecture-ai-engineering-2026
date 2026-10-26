"""第1回 宿題（基礎・発展）の実験スクリプト。

使い方:
    uv run python -m scripts.lec01_experiment basic   # 基礎: モデル・temperature を変えて3回ずつ
    uv run python -m scripts.lec01_experiment bench   # 発展: 応答時間とトークン数を比較
    uv run python -m scripts.lec01_experiment burst   # 発展: 待たずに連続で呼び、レート制限を観察

結果は lectures/01/results/<実験名>.json に保存する（APIキーは保存しない）。
"""

import argparse
import json
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.ask import DEFAULT_QUESTION
from app.config import get_api_key

RESULTS_DIR = Path(__file__).resolve().parent.parent / "lectures" / "01" / "results"

LITE = "gemini-3.5-flash-lite"
FLASH = "gemini-3.5-flash"
OLD_LITE = "gemini-3.1-flash-lite"

QUESTIONS = {
    "short": "1+1は？数字だけで答えてください。",
    "lora": "LoRAとは何ですか？",
    "course": DEFAULT_QUESTION,
}

# (モデル, temperature) の組。None はモデルの既定値
BASIC_SETTINGS = [(LITE, 0.0), (LITE, 1.0), (LITE, 2.0), (FLASH, None)]
BENCH_MODELS = [LITE, FLASH, OLD_LITE]


def call(client, model, question, temperature=None, retries=3, wait=30):
    """1回呼んで、回答・応答時間・トークン数を dict で返す。429 のときは待って再試行する。"""
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


def run(name, jobs, interval):
    client = genai.Client(api_key=get_api_key())
    results = []
    for i, (model, question, temperature) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {model} t={temperature} {question[:20]}")
        r = call(client, model, question, temperature)
        results.append(r)
        print(f"  -> {r.get('error') or 'ok'} {r['seconds']}s out={r.get('output_tokens')}")
        # 保存は毎回行い、途中で止まっても結果が残るようにする
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{name}.json"
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if interval and i < len(jobs):
            time.sleep(interval)
    print(f"保存しました: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["basic", "bench", "burst"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--interval", type=float, default=5.0, help="呼び出しの間隔（秒）")
    args = parser.parse_args()

    if args.experiment == "basic":
        jobs = [(m, DEFAULT_QUESTION, t) for m, t in BASIC_SETTINGS for _ in range(args.repeat)]
        run("basic", jobs, args.interval)
    elif args.experiment == "bench":
        jobs = [(m, q, None) for m in BENCH_MODELS for q in QUESTIONS.values() for _ in range(2)]
        run("bench", jobs, args.interval)
    else:
        # 短い質問を待たずに連続で投げ、何回目で 429 になるかを見る（再試行はしない）
        client = genai.Client(api_key=get_api_key())
        results = []
        start = time.perf_counter()
        for i in range(20):
            r = call(client, LITE, QUESTIONS["short"], retries=0)
            r["t_since_start"] = round(time.perf_counter() - start, 2)
            results.append(r)
            print(f"[{i + 1}] {r.get('error') or 'ok'} t={r['t_since_start']}s")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "burst.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
