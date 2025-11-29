# src/eval_generation.py

import torch
from pathlib import Path

from datasets import DatasetDict
import evaluate
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import train_config
from .data import load_vihealthqa, format_example


def main():
    # ----------------------------------------------------
    # 1) Thiết bị
    # ----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ----------------------------------------------------
    # 2) Tìm checkpoint fine-tuned mới nhất
    # ----------------------------------------------------
    ckpt_root = Path(train_config.output_dir)
    assert ckpt_root.exists(), f"Output dir not found: {ckpt_root}"

    ckpt_dirs = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    )

    if not ckpt_dirs:
        raise RuntimeError(
            f"Không tìm thấy checkpoint nào trong {ckpt_root}. "
            f"Hãy kiểm tra lại bạn đã train xong và có save checkpoint chưa."
        )

    best_ckpt_dir = ckpt_dirs[-1]
    print(f"🔹 Loading fine-tuned model from {best_ckpt_dir}")

    # ----------------------------------------------------
    # 3) Load tokenizer (từ model gốc) + model (từ checkpoint)
    # ----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(best_ckpt_dir),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # ----------------------------------------------------
    # 4) Load dataset test (dùng loader local của bạn)
    # ----------------------------------------------------
    ds: DatasetDict = load_vihealthqa()
    test_ds = ds["test"]

    print("🔹 Loaded ViHealthQA (local) for evaluation.")
    print(test_ds)
    print("\n📌 Sample test row:")
    print(test_ds[0])

    # ----------------------------------------------------
    # 5) Chuẩn bị metric BLEU + ROUGE
    # ----------------------------------------------------
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    preds = []
    refs = []

    # Có thể giới hạn số mẫu để test nhanh (None = full test)
    # Ví dụ: N_SAMPLES = 200 để chạy nhanh hơn
    N_SAMPLES = None  # đổi thành số cụ thể nếu muốn
    if N_SAMPLES is not None:
        eval_ds = test_ds.select(range(min(N_SAMPLES, len(test_ds))))
    else:
        eval_ds = test_ds

    print(f"\n🔹 Evaluating on {len(eval_ds)} test examples...\n")

    # ----------------------------------------------------
    # 6) Loop generate từng mẫu với progress bar
    # ----------------------------------------------------
    max_input_len = min(train_config.max_seq_length, 1024)
    max_new_tokens = 256

    for ex in tqdm(eval_ds, desc="Generating", ncols=100):
        question = ex["question"]
        answer_ref = ex["answer"].strip()

        # Prompt giống lúc train nhưng KHÔNG kèm answer
        prompt = format_example(question, answer=None)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy để đánh giá ổn định
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Cắt phần prompt, chỉ lấy tokens mới sinh
        generated = output_ids[0][inputs["input_ids"].shape[1]:]

        pred_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        preds.append(pred_text)
        refs.append(answer_ref)

    # ----------------------------------------------------
    # 7) Tính BLEU + ROUGE trên toàn bộ test
    # ----------------------------------------------------
    print("\n🔹 Computing BLEU & ROUGE on generated answers...")

    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_score = rouge.compute(predictions=preds, references=refs)

    print("\n===== GENERATION METRICS ON TEST =====")
    print(f"BLEU:     {bleu_score['bleu']:.4f}")
    print(f"ROUGE-1:  {rouge_score['rouge1']:.4f}")
    print(f"ROUGE-2:  {rouge_score['rouge2']:.4f}")
    print(f"ROUGE-L:  {rouge_score['rougeL']:.4f}")
    print("=======================================")


if __name__ == "__main__":
    main()
