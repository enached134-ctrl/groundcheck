# groundcheck

**A small open judge model for RAG groundedness — fine-tuned with QLoRA, and shipped only
because its eval suite said yes.**

Every serious RAG system needs a groundedness judge: something that reads *(question,
retrieved context, answer)* and decides whether every claim in the answer is actually
supported by the context. Today that judge is usually a frontier-model API call — which makes
every CI eval run cost money, leak data off-box, and rate-limit your test suite.

`groundcheck` fine-tunes a **small open model (1–3B) into a dedicated groundedness judge**
that runs locally on consumer hardware, so an eval gate can grade hundreds of cases per CI
run at zero marginal cost.

## The rule this repo lives by

> **The fine-tune ships only if the evals say it beats the base model.**
> Training is cheap. Judgment is the product. The eval suite — not vibes — decides ship/no-ship.

## Architecture

```
1. DATASET  (src/build_dataset.py)
   Labels correct BY CONSTRUCTION — no human annotation, no LLM labelling.
   Generate the context first, then preserve or deliberately corrupt the answer.
   Four balanced case types:
     - supported  : answer states a fact present in the context      -> grounded=True
     - contradict : answer swaps in a value that conflicts            -> grounded=False
     - fabricated : answer adds a specific fact absent from context   -> grounded=False
     - refusal    : asked something absent, answer correctly declines -> grounded=True

2. TRAIN    (src/train.py)
   QLoRA (4-bit NF4) on an 8 GB consumer GPU (RTX 5070 laptop):
   base = small instruct model (Qwen2.5-1.5B-Instruct class)
   task = (question, context, answer) → {"grounded": bool, "feedback": "..."}
   completions-only loss — the model learns to PRODUCE the verdict, not echo the prompt.

3. JUDGE THE JUDGE  (src/evaluate.py)
   Held-out agreement vs constructed labels: accuracy / precision / recall / F1 /
   refusal-case correctness — base zero-shot vs fine-tuned. The before/after table
   IS the release gate: evaluate.py exits non-zero if the fine-tune doesn't win.

4. SERVE    (src/judge.py + src/promptfoo_provider.py)
   Drop-in grader: a promptfoo Python provider so any eval suite (including
   agentic-rag-mcp's CI gate) can swap the frontier-API judge for this local one.
```

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # torch: cu128 wheels
python src/build_dataset.py --n 1400 --seed 7 --out data               # deterministic corpus
python src/train.py --base Qwen/Qwen2.5-1.5B-Instruct --epochs 2       # QLoRA, ~8 GB VRAM
python src/evaluate.py --adapter out/adapter                          # base vs tuned + gate
```

## Skills this exercises, deliberately

Fine-tuning (LoRA/QLoRA, PEFT, TRL) · PyTorch · quantization (bitsandbytes 4-bit) · dataset
engineering · LLM-as-judge methodology · evaluation harnesses · Hugging Face transformers ·
GPU training on consumer hardware — the exact stack that July-2026 AI Engineer postings ask
for most, verified against a live analysis of 96 remote job descriptions.

## Status

- [x] Design (this document)
- [ ] Dataset builder
- [ ] QLoRA training run
- [ ] Eval: base vs tuned (the ship/no-ship table)
- [ ] promptfoo provider + integration example with agentic-rag-mcp

## License

MIT
