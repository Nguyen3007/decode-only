Dự án này thực hiện Supervised Fine-Tuning (SFT) mô hình Qwen/Qwen2.5-1.5B-Instruct theo dạng decoder-only causal language modeling trên tập dữ liệu ViHealthQA (Vietnamese Health Q&A).

Project được thiết kế tối giản, dễ huấn luyện trên GPU (Vast.ai, Colab Pro, Kaggle), với kiến trúc rõ ràng và dễ mở rộng.

📁 Project Structure
decode-only/
│
├── src/
│   ├── config.py        # Cấu hình training + đường dẫn
│   ├── data.py          # Load và kiểm tra dataset ViHealthQA
│   ├── trainer.py       # Fine-tune Qwen2.5-1.5B theo kiểu causal LM
│   └── __init__.py
│
├── requirements.txt     # Các thư viện cần thiết
└── README.md

📦 Dataset: ViHealthQA

Dataset sử dụng:
tarudesu/ViHealthQA
(Nội dung: câu hỏi – trả lời y tế tiếng Việt, 3 split: train / validation / test)

Ví dụ 1 mẫu:

{
  "id": 1,
  "question": "Đang chích ngừa viêm gan B có chích ngừa Covid-19 được không?",
  "answer": "Nếu anh/chị đang tiêm ngừa vaccine phòng bệnh viêm gan B... ",
  "link": "https://vnexpress.net/..."
}

🚀 1. Cài đặt môi trường

Trong môi trường Python 3.10+:

pip install -r requirements.txt

🔍 2. Kiểm tra dataset

Bạn có thể chạy thử việc load dataset:

python -m src.data


Output sẽ hiển thị tổng số mẫu và 1 sample để kiểm tra.

🏋️ 3. Fine-tune mô hình (SFT Decoder-Only)

Huấn luyện mô hình Qwen2.5-1.5B trên ViHealthQA:

python -m src.trainer


trainer.py sẽ tự động:

Load dataset

Load tokenizer + model

Xây dựng chat template cho dạng Q&A

Tokenize & sinh labels (causal LM)

Huấn luyện với Trainer()

Lưu checkpoint vào:

checkpoints/qwen2_5_1_5b_vihealthqa/

🧩 4. Cấu hình training

Mọi hyperparameter nằm trong src/config.py.

Ví dụ:

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
max_seq_length = 1024
batch_size = 2
gradient_accumulation_steps = 8
num_train_epochs = 3
learning_rate = 1e-5


Bạn có thể chỉnh ở đây thay vì sửa nhiều file.

🧪 5. Evaluate (sẽ thêm sau)

Dự án sẽ sớm bổ sung file evaluate.py để:

Generate câu trả lời từ checkpoint đã fine-tune

So sánh với ground truth

Tính ROUGE / BLEU

🔧 6. Huấn luyện trên GPU (Vast.ai)

Khi clone repo trên Vast.ai:

git clone https://github.com/Nguyen3007/decode-only.git
cd decode-only
pip install -r requirements.txt
python -m src.trainer


GPU 12GB trở lên được khuyến nghị cho Qwen 1.5B.

📌 Ghi chú

Local CPU có thể chạy tokenization, nhưng không phù hợp để train Qwen2.5-1.5B.

Khi train trên GPU 12GB, nên giảm:

max_seq_length = 512

per_device_train_batch_size = 1

gradient_accumulation_steps = 16

✨ Mục tiêu dự án

Xây dựng pipeline SFT decoder-only rõ ràng và dễ tái sử dụng cho mô hình LLM.

Fine-tune chuyên sâu mô hình Qwen trên nhiệm vụ Q&A y tế tiếng Việt.

Chuẩn bị để mở rộng sang:

LoRA / QLoRA

RAG

Evaluate nâng cao

Deployment (FastAPI, HF Spaces)

👤 Tác giả

Nguyen3007
Sinh viên ngành Khoa học máy tính — yêu thích NLP, LLM, Recommender Systems.
