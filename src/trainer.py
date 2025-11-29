import os
from typing import Dict, Any

import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)

from .config import train_config, paths
from .data import load_vihealthqa


# ================ 1. Format câu hỏi - trả lời ==================


def format_example(example: Dict[str, Any]) -> str:
    """
    Format 1 sample ViHealthQA thành prompt text cho decoder-only LM.
    Bạn có thể chỉnh template này cho đẹp hơn nếu muốn.
    """
    q = (example["question"] or "").strip()
    a = (example["answer"] or "").strip()

    text = (
        "### Câu hỏi:\n"
        f"{q}\n\n"
        "### Trả lời:\n"
        f"{a}"
    )
    return text


# ================ 2. Tokenization / Preprocessing ==============


def preprocess_dataset(raw_ds: DatasetDict, tokenizer) -> DatasetDict:
    """
    Tokenize toàn bộ ViHealthQA cho bài toán causal LM.
    - padding = max_length để collator không bị lỗi shape
    - labels = input_ids (decoder-only)
    """

    def _process_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = [
            format_example({"question": q, "answer": a})
            for q, a in zip(batch["question"], batch["answer"])
        ]

        tokenized = tokenizer(
            texts,
            max_length=train_config.max_seq_length,
            truncation=True,
            padding="max_length",  # QUAN TRỌNG: để default_collator không lỗi
        )

        # decoder-only: labels chính là input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("🔹 Preprocessing dataset (tokenization)…")
    tokenized_ds = raw_ds.map(
        _process_batch,
        batched=True,
        remove_columns=["id", "question", "answer", "link"],
        desc="Tokenizing ViHealthQA",
    )
    return tokenized_ds


# ================ 3. Load tokenizer + model ====================


def get_tokenizer_and_model():
    """
    Load tokenizer + Qwen2.5-1.5B với dtype = bfloat16 cho RTX 5080.
    """
    print(f"🔹 Loading tokenizer & model: {train_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

    # Đảm bảo có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,  # BF16 cho 5080
        device_map=None,             # Trainer sẽ tự move sang GPU
    )

    # Resize embedding nếu tokenizer có thêm token
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


# ================ 4. Train loop với Trainer ====================


def train() -> None:
    # Seed cố định
    set_seed(train_config.seed)
    paths.make_dirs()

    # 1) Load dataset từ CSV local (đã export sẵn)
    raw_ds = load_vihealthqa()

    # 2) Load tokenizer & model
    tokenizer, model = get_tokenizer_and_model()

    # Bật gradient checkpointing để tiết kiệm VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # bắt buộc khi dùng gradient checkpointing

    # (Tuỳ chọn) thêm dropout nhẹ để giảm overfit
    if hasattr(model.config, "dropout") and model.config.dropout == 0.0:
        model.config.dropout = 0.1
    if hasattr(model.config, "hidden_dropout") and model.config.hidden_dropout == 0.0:
        model.config.hidden_dropout = 0.1
    if hasattr(model.config, "attention_dropout") and model.config.attention_dropout == 0.0:
        model.config.attention_dropout = 0.1

    # 3) Tokenize dataset
    tokenized_ds = preprocess_dataset(raw_ds, tokenizer)

    # 4) TrainingArguments (API mới của HF, dùng eval_strategy)
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,

        # ----- TRAINING -----
        num_train_epochs=train_config.num_train_epochs,
        learning_rate=train_config.learning_rate,
        warmup_ratio=train_config.warmup_ratio,
        weight_decay=train_config.weight_decay,

        # ----- BATCH & GRADIENT -----
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        max_grad_norm=1.0,  # clip grad để tránh nổ

        # ----- LOGGING / EVAL / SAVE -----
        logging_steps=train_config.logging_steps,
        eval_strategy="steps",                 # transformers mới
        eval_steps=train_config.eval_steps,
        save_strategy="steps",
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ----- PRECISION -----
        fp16=False,
        bf16=True,

        # ----- KHÁC -----
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        # data_collator = None => dùng default (ổn vì đã padding sẵn)
    )

    print("🚀 Start training decoder-only model on ViHealthQA…")
    trainer.train()
    print("✅ Training finished!")
    print(f"Best model saved to: {training_args.output_dir}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
