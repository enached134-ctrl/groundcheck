"""promptfoo custom provider — plug the fine-tuned judge into an eval suite.

This is the bridge that lets groundcheck act as the grounded/ungrounded gate for
another project's RAG pipeline (e.g. agentic-rag-mcp). promptfoo passes a JSON string
with question/context/answer; the provider returns the verdict.

promptfooconfig.yaml:
    providers:
      - id: file://src/promptfoo_provider.py
        config: { adapter: out/adapter }
"""

from __future__ import annotations

import json
from typing import Any

_JUDGE = None


def _get_judge(config: dict[str, Any]):
    global _JUDGE
    if _JUDGE is None:
        from judge import Judge

        _JUDGE = Judge(
            config.get("base", "Qwen/Qwen2.5-1.5B-Instruct"),
            adapter=config.get("adapter", "out/adapter"),
        )
    return _JUDGE


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(prompt)
    judge = _get_judge(options.get("config", {}))
    verdict = judge.judge(payload["question"], payload["context"], payload["answer"])
    return {"output": json.dumps(verdict)}
