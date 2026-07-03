"""Judge the judge: base vs fine-tuned on the held-out test split.

Prints the before/after table that IS the release gate. Exits non-zero if the
fine-tuned model does not beat the base model on F1 — no ship.

    python src/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --adapter out/adapter
"""

from __future__ import annotations

import argparse
import json

from judge import Judge


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(judge: Judge, rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    refusal_total = refusal_ok = 0
    for r in rows:
        pred = bool(judge.judge(r["question"], r["context"], r["answer"]).get("grounded", False))
        gold = bool(r["grounded"])
        if r["kind"] == "refusal":
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
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "refusal_correct": (refusal_ok / refusal_total if refusal_total else 0.0),
        "n": n,
    }


def row(label: str, m: dict) -> str:
    return (f"{label:16s} {m['accuracy']:.3f}    {m['precision']:.3f}     "
            f"{m['recall']:.3f}   {m['f1']:.3f}   {m['refusal_correct']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="out/adapter")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows")
    args = ap.parse_args()

    rows = load(args.data)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Evaluating on {len(rows)} held-out cases...\n")
    base_m = score(Judge(args.base), rows)
    tuned_m = score(Judge(args.base, adapter=args.adapter), rows)

    print("model            acc      prec      rec     f1      refusal")
    print("-" * 62)
    print(row("base (0-shot)", base_m))
    print(row("groundcheck", tuned_m))
    print("-" * 62)
    delta = tuned_m["f1"] - base_m["f1"]
    print(f"\nF1 delta: {delta:+.3f}")

    results = {"base": base_m, "tuned": tuned_m, "f1_delta": delta}
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/latest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if delta <= 0:
        print("\nGATE: FAIL — fine-tune does not beat base. Not shipping.")
        raise SystemExit(1)
    print("\nGATE: PASS — fine-tune beats base on F1. Ship it.")


if __name__ == "__main__":
    main()
