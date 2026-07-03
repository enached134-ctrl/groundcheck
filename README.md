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
   Labeled-by-construction pairs from real document corpora:
   - grounded answers  = faithful, extractive/paraphrased summaries of the context
   - ungrounded answers = controlled corruptions: facts injected from OTHER documents,
     negated claims, fabricated specifics (numbers, names), unsupported causal leaps
   → every label is correct by construction, no human annotation needed

2. TRAIN    (src/train.py)
   QLoRA (4-bit NF4) on an 8 GB consumer GPU (RTX 5070 laptop):
   base = small instruct model (Qwen2.5-1.5B-Instruct class)
   task = (question, context, answer) → {"grounded": bool, "feedback": "..."}

3. JUDGE THE JUDGE  (src/evaluate.py)
   Held-out agreement vs constructed labels: accuracy / F1 / refusal-case correctness,
   base model zero-shot vs fine-tuned — the before/after table IS the release gate.

4. SERVE    (src/judge.py + promptfoo provider)
   Drop-in grader: a promptfoo Python provider so any eval suite (including
   agentic-rag-mcp's CI gate) can swap the API judge for this local one.
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
