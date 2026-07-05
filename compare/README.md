# Head-to-head: groundcheck vs Ragas vs DeepEval

`groundcheck`'s claim is that a small, local, fine-tuned judge can match a frontier
judge on RAG groundedness — at zero marginal cost and with no data leaving the box.
This harness puts that claim next to the two eval frameworks teams actually reach
for, on the **same** held-out set, and reports the numbers side by side.

It is deliberately **run-it-yourself**: no results are committed here. You run it,
you get `results.json`. That's the honest version of a benchmark.

## What it measures

For each judge, over `../data/test.jsonl` (labels correct by construction):

| Metric | Meaning |
|---|---|
| **F1 / accuracy** | agreement with the gold grounded/not label |
| **refusal-correct** | share of *correct-refusal* cases graded right — the case faithfulness metrics classically fumble |
| **ms/case** | measured wall-clock latency |
| **$/1k calls** | `$0` for the local judge; an **estimate** (tokens × price) for the API judges |

## Run it

```bash
pip install -r compare/requirements.txt        # ragas, deepeval, langchain-openai, tiktoken
export OPENAI_API_KEY=sk-...                    # only the Ragas/DeepEval judges call an API
python compare/benchmark.py --limit 60          # groundcheck runs locally, off your model + adapter
```

Pick a subset of judges with `--judges groundcheck,deepeval`. If a framework isn't
installed or the key is missing, that judge is skipped and the others still run.

## Read the numbers honestly

- **Scope.** This is agreement with groundcheck's *synthetic, correct-by-construction*
  labels — it proves the judge reproduces a known ground truth, not that it's the best
  judge on your production traffic. Point it at your own labeled traces for that.
- **Different targets.** Ragas/DeepEval "faithfulness" grade a claim-support *ratio*;
  groundcheck emits a binary grounded/not. We threshold their score at
  `--faith-threshold` (default 0.5) to compare like with like — move it and the
  numbers move.
- **Cost is an estimate.** API cost is `input_tokens × --price` (default gpt-4o-mini
  input pricing), not a billed figure. The local judge is genuinely `$0`.

The interesting column is usually **refusal-correct**: a correct "I don't know" is
where content-faithfulness metrics tend to break, and it's the exact failure
groundcheck was built to catch.
