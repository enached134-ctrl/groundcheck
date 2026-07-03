"""Inference wrapper: load base (+ optional LoRA adapter) and judge one triple.

    from judge import Judge
    j = Judge("Qwen/Qwen2.5-1.5B-Instruct", adapter="out/adapter")
    j.judge(question, context, answer)  # -> {"grounded": bool, "feedback": str}
"""

from __future__ import annotations

import json
import re

from prompt import SYSTEM, user_turn

_VERDICT = re.compile(r'"grounded"\s*:\s*(true|false)', re.I)


class Judge:
    def __init__(self, base: str, adapter: str | None = None, load_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(base)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def _generate(self, question: str, context: str, answer: str) -> str:
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_turn(question, context, answer)},
        ]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def judge(self, question: str, context: str, answer: str) -> dict:
        raw = self._generate(question, context, answer)
        try:
            start = raw.index("{")
            end = raw.index("}", start) + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            m = _VERDICT.search(raw)
            grounded = m.group(1).lower() == "true" if m else False
            return {"grounded": grounded, "feedback": raw.strip()[:200]}
