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


# ===================== 1. Format 1 sample =====================


def format_example(example: Dict[str, Any]) -> str:
    """
    Format 1 sample ViHealthQA thành text cho decoder-only LM.
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


# ================== 2. Tokenization / Preprocess ==============


def preprocess_dataset(raw_ds: DatasetDict, tokenizer) -> DatasetDict:
    """
    Tokenize toàn bộ ViHealthQA cho bài toán causal LM.
    - padding = max_length để default collator không lỗi shape
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
            padding="max_length",
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


# ================== 3. Load tokenizer + model =================


def get_tokenizer_and_model():
    """
    Load tokenizer + Qwen2.5-1.5B với dtype = bfloat16.
    """
    print(f"🔹 Loading tokenizer & model: {train_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

    # Đảm bảo có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,  # BF16 cho GPU đời mới (5080)
        device_map=None,             # Trainer sẽ tự move khi train
    )

    # Resize embedding nếu tokenizer có thêm token
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


# ======================= 4. Train loop ========================


def train() -> None:
    # Seed cố định cho reproducibility
    set_seed(train_config.seed)
    paths.make_dirs()

    # 1) Load dataset từ CSV local (đã export sẵn)
    raw_ds = load_vihealthqa()

    print()
    print(raw_ds)
    print("\n📌 Sample train row:")
    print(raw_ds["train"][0])

    # 2) Load tokenizer & model
    tokenizer, model = get_tokenizer_and_model()

    # ==== PRINT MODEL INFORMATION ====
    print("\n================ MODEL INFORMATION ================\n")
    print(f"📌 Model name: {train_config.model_name}")

    # Tổng số parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧠 Total parameters: {total_params:,}")
    print(f"🧩 Trainable parameters: {trainable_params:,}")
    print(f"🔧 Frozen parameters: {total_params - trainable_params:,}")

    # In ra các config quan trọng
    cfg = model.config
    print(f"\nHidden size: {cfg.hidden_size}")
    print(f"Intermediate size: {getattr(cfg, 'intermediate_size', 'N/A')}")
    print(f"Num attention heads: {cfg.num_attention_heads}")
    print(f"Num layers (blocks): {cfg.num_hidden_layers}")
    print(f"Vocab size: {cfg.vocab_size}")
    print(f"Max sequence length: {cfg.max_position_embeddings}")

    # Attention (RoPE, sliding window,…)
    if hasattr(cfg, "rope_theta"):
        print(f"RoPE theta: {cfg.rope_theta}")
    if hasattr(cfg, "rope_scaling"):
        print(f"RoPE scaling: {cfg.rope_scaling}")

    # Dropout
    if hasattr(cfg, "attention_dropout"):
        print(f"Attention dropout: {cfg.attention_dropout}")
    if hasattr(cfg, "hidden_dropout"):
        print(f"Hidden dropout: {cfg.hidden_dropout}")
    if hasattr(cfg, "dropout"):
        print(f"General dropout: {cfg.dropout}")

    # Ki architecture (Qwen2.5 support)
    if hasattr(cfg, "use_sliding_window"):
        print(f"Sliding window: {cfg.use_sliding_window}")
    if hasattr(cfg, "sliding_window"):
        print(f"Sliding window size: {cfg.sliding_window}")
    if hasattr(cfg, "attention_bias"):
        print(f"Attention bias: {cfg.attention_bias}")

    print("\n====================================================\n")

    # 3) Tokenize dataset
    tokenized_ds = preprocess_dataset(raw_ds, tokenizer)

    # 4) TrainingArguments
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
        eval_strategy="steps",                     # transformers mới
        eval_steps=train_config.eval_steps,
        save_strategy="epoch",                     # lưu theo epoch
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
        report_to="none",                          # không log ra wandb, tb
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,   # Warning deprecate nhưng vẫn chạy được
    )

    print("🚀 Start training decoder-only model on ViHealthQA…")
    trainer.train()
    print("🚀 Training finished!")
    print(f"Best model (by eval_loss) saved under: {training_args.output_dir}")

    # 6) Đánh giá trên test set (giống pipeline encode-decode trước)
    print("\n[INFO] Evaluating on test set...")
    test_metrics = trainer.evaluate(
        tokenized_ds["test"],
        metric_key_prefix="test",
    )

    print("[RESULT] Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
