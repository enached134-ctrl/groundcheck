"""Shared chat formatting for the groundedness judge — used by train, eval, and serve.

Keeping the prompt in one place guarantees training and inference agree token-for-token.
"""

from __future__ import annotations

import json

SYSTEM = (
    "You are a strict groundedness judge. Given a QUESTION, the retrieved CONTEXT, and an "
    "ANSWER, decide whether every claim in the ANSWER is supported by the CONTEXT. "
    "An answer that correctly declines because the context lacks the information is grounded. "
    "An answer that states, contradicts, or invents any fact not supported by the context is "
    'not grounded. Reply with JSON only: {"grounded": true|false, "feedback": "<one sentence>"}.'
)


def user_turn(question: str, context: str, answer: str) -> str:
    return f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"


def target_turn(grounded: bool, kind: str) -> str:
    feedback = {
        "supported": "The answer's claim is stated in the context.",
        "contradict": "The answer contradicts the value given in the context.",
        "fabricated": "The answer adds a fact that is absent from the context.",
        "refusal": "The answer correctly declines because the context lacks the information.",
    }.get(kind, "Checked against the context.")
    return json.dumps({"grounded": grounded, "feedback": feedback})


def messages(record: dict, include_target: bool) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_turn(record["question"], record["context"], record["answer"])},
    ]
    if include_target:
        msgs.append({"role": "assistant", "content": target_turn(record["grounded"], record["kind"])})
    return msgs
