"""QLoRA fine-tune of a small instruct model into a groundedness judge.

Runs on an 8 GB consumer GPU (tested target: RTX 5070 laptop). 4-bit NF4 base +
LoRA adapters, trained on completions only so the model learns to PRODUCE the verdict,
not to echo the prompt.

    python src/train.py --base Qwen/Qwen2.5-1.5B-Instruct --epochs 2 --out out/adapter
"""

from __future__ import annotations

import argparse
import pathlib

from prompt import messages


def load_split(path: str) -> list[dict]:
    import json

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
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(rows: list[dict]) -> Dataset:
        texts = [tok.apply_chat_template(messages(r, include_target=True), tokenize=False)
                 for r in rows]
        return Dataset.from_dict({"text": texts})

    train_ds = to_text(load_split(f"{args.data}/train.jsonl"))
    val_ds = to_text(load_split(f"{args.data}/val.jsonl"))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Train on completions only: mask everything up to the assistant turn.
    resp_template = "<|im_start|>assistant\n"
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_len,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        completion_only_loss=True,
        assistant_only_loss=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )
    # Fallback for TRL versions that need an explicit collator for completion masking.
    if not getattr(cfg, "completion_only_loss", False):
        from trl import DataCollatorForCompletionOnlyLM

        trainer.data_collator = DataCollatorForCompletionOnlyLM(resp_template, tokenizer=tok)

    trainer.train()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"adapter saved -> {out}")


if __name__ == "__main__":
    main()
