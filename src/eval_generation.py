# src/eval_generation.py
import torch
from pathlib import Path
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import train_config
from .data import load_vihealthqa, format_example  # nếu format_example nằm trong trainer thì copy sang data

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 1) Xác định thư mục checkpoint mới nhất
    ckpt_root = Path(train_config.output_dir)
    assert ckpt_root.exists(), f"Output dir not found: {ckpt_root}"

    # Tìm các folder kiểu checkpoint-*
    ckpt_dirs = sorted(
        [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    )

    if not ckpt_dirs:
        raise RuntimeError(
            f"Không tìm thấy checkpoint nào trong {ckpt_root}. "
            f"Hãy kiểm tra lại bạn đã train xong chưa."
        )

    best_ckpt_dir = ckpt_dirs[-1]  # lấy checkpoint mới nhất
    print("🔹 Loading fine-tuned model from", best_ckpt_dir)

    # 2) Load tokenizer từ model gốc
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Load model từ checkpoint fine-tuned
    model = AutoModelForCausalLM.from_pretrained(
        str(best_ckpt_dir),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    # 2) Load raw test set (chưa tokenize)
    ds = load_vihealthqa()
    test_ds = ds["test"]

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    preds = []
    refs = []

    # 3) Loop qua test set và generate
    for ex in test_ds:
        question = ex["question"]
        answer_ref = ex["answer"].strip()

        # Prompt chỉ chứa câu hỏi
        prompt = (
            "### Câu hỏi:\n"
            f"{question}\n\n"
            "### Trả lời:\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=train_config.max_seq_length,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,          # greedy cho ổn định
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Cắt phần prompt, chỉ lấy new tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]

        pred_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        preds.append(pred_text)
        refs.append(answer_ref)

    # 4) Tính BLEU + ROUGE
    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_score = rouge.compute(predictions=preds, references=refs)

    print("\n===== GENERATION METRICS ON TEST =====")
    print(f"BLEU: {bleu_score['bleu']:.4f}")
    print(f"ROUGE-1: {rouge_score['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_score['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")

if __name__ == "__main__":
    main()
