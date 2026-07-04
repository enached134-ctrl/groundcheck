"""Cost-parity report — does the fine-tuned local judge match a frontier judge, for free?

Runs the tuned local judge AND a frontier teacher (Gemini) on the same held-out cases,
scores both against the by-construction ground truth, and reports agreement + cost. The
point of distillation isn't "cheaper tokens" — it's a judge that runs locally in CI on
every commit with no API dependency, no rate limits, and nothing leaving the box.

    GEMINI_API_KEY=... python src/costparity.py --limit 60
"""

from __future__ import annotations

import argparse
import json
import os
import time

import requests
import urllib3

from judge import Judge
from prompt import SYSTEM, user_turn

urllib3.disable_warnings()

# gemini-2.5-flash list price (per 1M tokens), 2026 — used only for the illustrative
# cost column; the real point is $0 marginal + no dependency for the local judge.
TEACHER_IN_PER_M = 0.075
TEACHER_OUT_PER_M = 0.30
_URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_judge(question: str, context: str, answer: str) -> tuple[bool, int, int]:
    key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": user_turn(question, context, answer)}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    last_exc: Exception | None = None
    for attempt in range(5):
        try:
            r = requests.post(_URL.format(m=model), params={"key": key}, json=body,
                              timeout=60, verify=False)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            time.sleep(3 * (attempt + 1))
    else:
        raise last_exc or RuntimeError("teacher call failed after retries")
    data = r.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    usage = data.get("usageMetadata", {})
    try:
        grounded = bool(json.loads(text).get("grounded", False))
    except json.JSONDecodeError:
        grounded = '"grounded": true' in text.lower()
    return grounded, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="out/adapter")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    rows = load(args.data)[: args.limit]
    print(f"Cost-parity on {len(rows)} held-out cases (local judge vs Gemini teacher)...\n")

    local = Judge(args.base, adapter=args.adapter)
    local_correct = teacher_correct = agree = 0
    in_tok = out_tok = 0
    for r in rows:
        gold = bool(r["grounded"])
        lp = bool(local.judge(r["question"], r["context"], r["answer"]).get("grounded", False))
        tp, ti, to = teacher_judge(r["question"], r["context"], r["answer"])
        in_tok += ti
        out_tok += to
        local_correct += lp == gold
        teacher_correct += tp == gold
        agree += lp == tp

    n = len(rows)
    teacher_cost = (in_tok * TEACHER_IN_PER_M + out_tok * TEACHER_OUT_PER_M) / 1e6
    per_1k = teacher_cost / n * 1000 if n else 0.0
    result = {
        "cases": n,
        "local_accuracy": round(local_correct / n, 3),
        "teacher_accuracy": round(teacher_correct / n, 3),
        "agreement": round(agree / n, 3),
        "teacher_cost_per_1k_calls_usd": round(per_1k, 4),
        "local_cost_per_1k_calls_usd": 0.0,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/cost-parity.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"  local judge (fine-tuned) accuracy : {result['local_accuracy']:.3f}")
    print(f"  teacher (Gemini) accuracy         : {result['teacher_accuracy']:.3f}")
    print(f"  local–teacher agreement           : {result['agreement']:.3f}")
    print(f"  teacher cost / 1k calls           : ${per_1k:.4f}")
    print("  local cost / 1k calls             : $0.0000  (own GPU, in CI, no API)")
    print("\n  wrote results/cost-parity.json")


if __name__ == "__main__":
    main()
