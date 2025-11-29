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
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "tarudesu/ViHealthQA"

    # 16GB VRAM → không nên 1024
    max_seq_length: int = 768

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1

    # tăng accumulation để bù batch size
    gradient_accumulation_steps: int = 16

    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3

    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2

    fp16: bool = True   # phù hợp nhất cho 5080
    bf16: bool = False

    seed: int = 42

    @property
    def output_dir(self):
        return str(paths.checkpoints_dir / "qwen2_5_1_5b_vihealthqa")


train_config = TrainingConfig()
