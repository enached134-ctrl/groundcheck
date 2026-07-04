"""QLoRA fine-tune of a small instruct model into a groundedness judge.

Runs on an 8 GB consumer GPU (tested target: RTX 5070 laptop). 4-bit NF4 base +
LoRA adapters. Trained on prompt/completion pairs so TRL masks the prompt and the
model learns to PRODUCE the verdict, not echo the question.

    python src/train.py --base Qwen/Qwen2.5-1.5B-Instruct --epochs 2 --out out/adapter
"""

from __future__ import annotations

import argparse
import json
import pathlib

from prompt import SYSTEM, target_turn, user_turn


def load_split(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out/adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_pc(rows: list[dict]) -> Dataset:
        # prompt = system+user rendered with the assistant generation prefix;
        # completion = the JSON verdict. TRL masks the prompt (completion-only loss).
        prompts, completions = [], []
        for r in rows:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_turn(r["question"], r["context"], r["answer"])},
            ]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            completions.append(target_turn(r["grounded"], r["kind"]))
        return Dataset.from_dict({"prompt": prompts, "completion": completions})

    train_ds = to_pc(load_split(f"{args.data}/train.jsonl"))
    val_ds = to_pc(load_split(f"{args.data}/val.jsonl"))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_len,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )
    trainer.train()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"adapter saved -> {out}")


if __name__ == "__main__":
    main()
