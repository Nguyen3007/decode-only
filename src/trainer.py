# src/trainer.py

from typing import Dict, List

import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

from .config import train_config, paths
from .data import load_vihealthqa


# ====== 1. LOAD TOKENIZER & MODEL ======

def get_tokenizer_and_model():
    """
    Load tokenizer + model decoder-only (Qwen…) cho bài ViHealthQA.
    """
    print(f"🔹 Loading tokenizer & model: {train_config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)

    # Đảm bảo có pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype="auto",  # để transformers tự chọn (float16 / bfloat16 nếu có)
    )

    # Nếu cần, resize embedding nếu vocab thay đổi
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer, model


# ====== 2. BUILD CHAT TEXT TỪ QUESTION/ANSWER ======

def build_chat_text(example: Dict, tokenizer) -> str:
    """
    Từ một mẫu {question, answer} tạo thành 1 đoạn hội thoại theo chat template.
    """
    question = example["question"]
    answer = example["answer"]

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Bạn là một trợ lý y tế tiếng Việt hữu ích, trả lời chính xác, "
                "ngắn gọn, dễ hiểu, dựa trên kiến thức y khoa đáng tin cậy."
            ),
        },
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # Nếu tokenizer có chat_template thì dùng luôn (thường Qwen có)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # vì ta đang train, không cần prompt để generate
        )
    else:
        # Fallback nếu model không có chat_template
        text = (
            "<|system|> Bạn là trợ lý y tế tiếng Việt hữu ích.\n"
            f"<|user|> {question}\n"
            f"<|assistant|> {answer}"
        )

    return text


# ====== 3. PREPROCESS DATASET ======

def preprocess_dataset(raw_ds: DatasetDict, tokenizer):
    """
    Từ raw DatasetDict (id, question, answer, link) → dataset tokenized
    cho bài causal LM (input_ids, attention_mask, labels).
    """

    def _preprocess(batch):
        texts = []

        questions = batch["question"]
        answers = batch["answer"]

        for q, a in zip(questions, answers):
            ex = {"question": q, "answer": a}
            text = build_chat_text(ex, tokenizer)
            texts.append(text)

        tokenized = tokenizer(
            texts,
            max_length=train_config.max_seq_length,
            truncation=True,
            padding="max_length",  # PAD HẾT về max_seq_length
        )

        # Với causal LM: labels = input_ids (dự đoán token kế tiếp)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    print("🔹 Preprocessing dataset (tokenization)…")

    tokenized_ds = raw_ds.map(
        _preprocess,
        batched=True,
        remove_columns=["id", "question", "answer", "link"],
    )

    print(tokenized_ds)
    return tokenized_ds


# ====== 4. TRAINING LOOP ======

def train():
    # Đảm bảo thư mục tồn tại
    paths.make_dirs()

    # Set seed cho reproducibility
    set_seed(train_config.seed)

    # 1) Load dataset raw
    raw_ds = load_vihealthqa()

    # 2) Load tokenizer + model
    tokenizer, model = get_tokenizer_and_model()

    # 3) Tokenize dataset theo chat template
    tokenized_ds = preprocess_dataset(raw_ds, tokenizer)

    # 4) TrainingArguments
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        logging_steps=train_config.logging_steps,
        eval_strategy="steps",
        eval_steps=train_config.eval_steps,
        save_steps=train_config.save_steps,
        save_total_limit=train_config.save_total_limit,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        report_to="none",  # tắt wandb/mlflow
    )

    # 5) Data collator
    collator = default_data_collator

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=collator,
    )

    # 7) Train
    print("🚀 Start training decoder-only model on ViHealthQA…")
    trainer.train()

    # 8) Save model + tokenizer
    save_dir = train_config.output_dir
    print(f"💾 Saving final model to: {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


def main():
    train()


if __name__ == "__main__":
    main()
