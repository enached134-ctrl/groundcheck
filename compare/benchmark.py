"""Head-to-head: groundcheck's local judge vs Ragas + DeepEval faithfulness.

Runs each judge over the SAME held-out set (data/test.jsonl, labels correct by
construction) and reports, per judge:
  • agreement-F1 / accuracy vs the gold labels
  • refusal-correctness  (the case faithfulness metrics classically miss)
  • latency  (wall-clock, measured)
  • cost  (local = $0; API judges = estimated from token count × price)

Nothing here is pre-computed. Run it to get numbers:

    pip install -r compare/requirements.txt
    export OPENAI_API_KEY=...          # only the Ragas/DeepEval judges need it
    python compare/benchmark.py --limit 60

groundcheck runs locally (uses the repo's own model + adapter, $0/call). Ragas
and DeepEval each call a frontier model, so they need a key and cost money — that
asymmetry is part of what the benchmark measures. Results -> compare/results.json.

Honest scope: this measures agreement with groundcheck's synthetic, correct-by-
construction labels, not a universal groundedness benchmark. Faithfulness metrics
also grade a slightly different thing (claim-support ratio) than the binary
grounded/not label, so we threshold their score at --faith-threshold. Read the
per-judge notes in the output before quoting any single number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

PRICE_PER_1K_INPUT_USD = 0.00015  # gpt-4o-mini input; override with --price


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_preds(preds: list[bool], rows: list[dict]) -> dict:
    """Same metric definition as src/evaluate.py, over precomputed predictions."""
    tp = fp = tn = fn = 0
    refusal_total = refusal_ok = 0
    for pred, r in zip(preds, rows):
        gold = bool(r["grounded"])
        if r.get("kind") == "refusal":
            refusal_total += 1
            refusal_ok += int(pred == gold)
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and not gold:
            tn += 1
        else:
            fn += 1
    n = len(rows)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": prec, "recall": rec, "f1": f1,
        "refusal_correct": refusal_ok / refusal_total if refusal_total else 0.0,
        "n": n,
    }


def _tokens(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)  # rough fallback


# --- judges ---------------------------------------------------------------

def run_groundcheck(rows, base, adapter, **_):
    from judge import Judge
    j = Judge(base, adapter=adapter)
    preds, t0 = [], time.perf_counter()
    for r in rows:
        preds.append(bool(j.judge(r["question"], r["context"], r["answer"]).get("grounded", False)))
    return preds, time.perf_counter() - t0, 0.0, "local model, $0/call"


def run_deepeval(rows, model, faith_threshold, price, **_):
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    metric = FaithfulnessMetric(threshold=faith_threshold, model=model, include_reason=False, async_mode=False)
    preds, cost, t0 = [], 0.0, time.perf_counter()
    for r in rows:
        tc = LLMTestCase(input=r["question"], actual_output=r["answer"], retrieval_context=[r["context"]])
        metric.measure(tc)
        preds.append(bool((metric.score or 0.0) >= faith_threshold))
        cost += _tokens(r["question"] + r["context"] + r["answer"]) / 1000 * price
    return preds, time.perf_counter() - t0, cost, f"{model}, est. cost"


def run_ragas(rows, model, faith_threshold, price, **_):
    import asyncio
    from langchain_openai import ChatOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    scorer = Faithfulness(llm=LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0)))
    preds, cost, t0 = [], 0.0, time.perf_counter()
    for r in rows:
        s = SingleTurnSample(user_input=r["question"], response=r["answer"], retrieved_contexts=[r["context"]])
        val = asyncio.run(scorer.single_turn_ascore(s))
        preds.append(bool((val or 0.0) >= faith_threshold))
        cost += _tokens(r["question"] + r["context"] + r["answer"]) / 1000 * price
    return preds, time.perf_counter() - t0, cost, f"{model}, est. cost"


JUDGES = {"groundcheck": run_groundcheck, "deepeval": run_deepeval, "ragas": run_ragas}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="first N rows (0 = all)")
    ap.add_argument("--judges", default="groundcheck,deepeval,ragas")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=os.path.join(os.path.dirname(__file__), "..", "out", "adapter"))
    ap.add_argument("--model", default="gpt-4o-mini", help="frontier model for Ragas/DeepEval")
    ap.add_argument("--faith-threshold", type=float, default=0.5)
    ap.add_argument("--price", type=float, default=PRICE_PER_1K_INPUT_USD, help="USD per 1k input tokens (cost estimate)")
    args = ap.parse_args()

    rows = load(args.data)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Head-to-head on {len(rows)} held-out cases · faithfulness threshold {args.faith_threshold}\n")

    out = {}
    for name in [j.strip() for j in args.judges.split(",") if j.strip()]:
        fn = JUDGES.get(name)
        if not fn:
            print(f"  ? unknown judge '{name}' — skipping")
            continue
        try:
            preds, secs, cost, note = fn(
                rows, base=args.base, adapter=args.adapter, model=args.model,
                faith_threshold=args.faith_threshold, price=args.price,
            )
        except Exception as e:  # missing dep / no key / API drift — keep going
            print(f"  ! {name}: skipped ({type(e).__name__}: {e})")
            continue
        m = score_preds(preds, rows)
        out[name] = {**m, "seconds_total": round(secs, 2),
                     "ms_per_case": round(1000 * secs / max(1, len(rows)), 1),
                     "cost_usd_total": round(cost, 4),
                     "cost_usd_per_1k_calls": round(cost / max(1, len(rows)) * 1000, 3), "note": note}
        print(f"  ✓ {name}: F1 {m['f1']:.3f} · acc {m['accuracy']:.3f} · refusal {m['refusal_correct']:.3f} · "
              f"{out[name]['ms_per_case']}ms/case · ${out[name]['cost_usd_per_1k_calls']}/1k  ({note})")

    if out:
        print("\n" + "judge".ljust(14) + "F1     acc    refusal  ms/case   $/1k")
        print("-" * 60)
        for name, m in out.items():
            print(f"{name:14s}{m['f1']:.3f}  {m['accuracy']:.3f}  {m['refusal_correct']:.3f}    "
                  f"{m['ms_per_case']:>6}    {m['cost_usd_per_1k_calls']:>6}")
    dst = os.path.join(os.path.dirname(__file__), "results.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"cases": len(rows), "faith_threshold": args.faith_threshold, "judges": out}, f, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
