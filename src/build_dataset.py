"""Build a groundedness-judge dataset with labels that are correct BY CONSTRUCTION.

Every record is a (question, context, answer) triple plus a boolean `grounded` label.
Because we generate the context first and then either preserve or deliberately corrupt
the answer, the label needs no human annotation and no LLM labelling — it is a property
of how the record was built.

Four balanced case types:
  1. SUPPORTED   — the answer states a fact present in the context            -> grounded=True
  2. CONTRADICT  — the answer states a value contradicting the context        -> grounded=False
  3. FABRICATED  — the answer adds a specific fact absent from the context     -> grounded=False
  4. REFUSAL     — asked something absent, the answer correctly declines       -> grounded=True

Usage:
    python src/build_dataset.py --n 1400 --seed 7 --out data
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

# --- content pools (fictional, so nothing depends on world knowledge) -------------

PRODUCTS = [
    "the Halcyon X2 router", "the Meridian S5 sensor", "the Cobalt Q9 drive",
    "the Vantage T3 gateway", "the Lumen P7 panel", "the Orbit M4 controller",
    "the Zephyr K2 module", "the Beacon R8 relay", "the Cascade V6 pump",
    "the Nimbus L1 array", "the Talon F5 actuator", "the Verity C4 hub",
]
ATTRS = {
    "release year": [str(y) for y in range(2017, 2027)],
    "peak throughput": [f"{n} Gbps" for n in (4, 8, 12, 20, 40, 100)],
    "operating latency": [f"{n} ms" for n in (2, 5, 9, 14, 22, 60)],
    "rated capacity": [f"{n} TB" for n in (1, 2, 4, 8, 16, 32)],
    "power draw": [f"{n} W" for n in (12, 18, 24, 45, 90, 130)],
    "housing material": ["anodized aluminium", "reinforced polymer", "cast magnesium",
                         "brushed steel", "carbon composite", "ceramic-coated alloy"],
    "warranty period": [f"{n} years" for n in (1, 2, 3, 5)],
    "operating range": [f"{a} to {b} C" for a, b in ((-10, 55), (0, 40), (-20, 70), (5, 45))],
}
ATTR_NAMES = list(ATTRS.keys())

REFUSAL_ATTRS = [
    "firmware licence", "retail price", "manufacturing plant",
    "certification body", "shipping weight", "default password",
]


def _sentence(attr: str, value: str, product: str) -> str:
    return {
        "release year": f"{product.capitalize()} was released in {value}.",
        "peak throughput": f"{product.capitalize()} delivers a peak throughput of {value}.",
        "operating latency": f"{product.capitalize()} operates at a latency of {value}.",
        "rated capacity": f"{product.capitalize()} has a rated capacity of {value}.",
        "power draw": f"{product.capitalize()} draws {value} under load.",
        "housing material": f"{product.capitalize()} uses a {value} housing.",
        "warranty period": f"{product.capitalize()} ships with a {value} warranty.",
        "operating range": f"{product.capitalize()} is rated for {value}.",
    }[attr]


def _make_context(rng: random.Random, product: str) -> tuple[str, dict[str, str]]:
    chosen = rng.sample(ATTR_NAMES, k=4)
    facts = {a: rng.choice(ATTRS[a]) for a in chosen}
    sentences = [_sentence(a, v, product) for a, v in facts.items()]
    rng.shuffle(sentences)
    return " ".join(sentences), facts


def _record(question: str, context: str, answer: str, grounded: bool, kind: str) -> dict:
    return {
        "question": question,
        "context": context,
        "answer": answer,
        "grounded": grounded,
        "kind": kind,
    }


def build(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    while len(out) < n:
        product = rng.choice(PRODUCTS)
        context, facts = _make_context(rng, product)
        attr = rng.choice(list(facts))
        value = facts[attr]
        q = f"What is the {attr} of {product}?"
        kind = rng.choice(["supported", "contradict", "fabricated", "refusal"])

        if kind == "supported":
            out.append(_record(q, context, f"The {attr} of {product} is {value} [1].", True, kind))
        elif kind == "contradict":
            wrong = rng.choice([v for v in ATTRS[attr] if v != value])
            out.append(_record(q, context, f"The {attr} of {product} is {wrong} [1].", False, kind))
        elif kind == "fabricated":
            extra_attr = rng.choice([a for a in ATTR_NAMES if a not in facts])
            extra_val = rng.choice(ATTRS[extra_attr])
            ans = (f"The {attr} of {product} is {value}, and its {extra_attr} "
                   f"is {extra_val} [1].")
            out.append(_record(q, context, ans, False, kind))
        else:  # refusal
            missing = rng.choice(REFUSAL_ATTRS)
            rq = f"What is the {missing} of {product}?"
            ans = (f"The provided context does not state the {missing} of {product}, "
                   f"so I cannot answer from these sources.")
            out.append(_record(rq, context, ans, True, kind))
    rng.shuffle(out)
    return out


def split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed + 1)
    rng.shuffle(records)
    n = len(records)
    tr, va = int(n * 0.8), int(n * 0.9)
    return {"train": records[:tr], "val": records[tr:va], "test": records[va:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = split(build(args.n, args.seed), args.seed)
    for name, rows in parts.items():
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pos = sum(1 for r in rows if r["grounded"])
        print(f"{name:5s}: {len(rows):4d} rows  ({pos} grounded / {len(rows) - pos} not)")


if __name__ == "__main__":
    main()
