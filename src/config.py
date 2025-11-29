# src/config.py
from dataclasses import dataclass
from pathlib import Path


# ====== 1. ĐƯỜNG DẪN CƠ BẢN ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathConfig:
    """
    Quản lý các đường dẫn chính trong project.
    """

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"

    def make_dirs(self) -> None:
        """
        Tạo sẵn các thư mục nếu chưa tồn tại.
        Gọi hàm này 1 lần ở đầu chương trình.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


paths = PathConfig()


# ====== 2. CẤU HÌNH TRAINING DECODER-ONLY ======

@dataclass
class TrainingConfig:
    """
    Cấu hình chính cho bài toán decoder-only SFT trên ViHealthQA.
    """

    # ----- Model & Dataset -----
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "tarudesu/ViHealthQA"

    # ----- Tokenization -----
    max_seq_length: int = 1024  # có thể giảm nếu thiếu VRAM

    # ----- Batch & Gradient -----
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # 2 * 8 = 16 effective batch size

    # ----- Training -----
    num_train_epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # ----- Logging & Save -----
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 2

    # ----- Precision -----
    fp16: bool = True   # nếu GPU hỗ trợ
    bf16: bool = False  # bật nếu GPU hỗ trợ bf16 tốt hơn

    # ----- Misc -----
    seed: int = 42

    @property
    def output_dir(self) -> str:
        """
        Nơi lưu checkpoint của mô hình decoder-only ViHealthQA.
        """
        return str(paths.checkpoints_dir / "qwen2_5_1_5b_vihealthqa")


train_config = TrainingConfig()
