from dataclasses import dataclass
from pathlib import Path


# Gốc project: decode-only/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    """
    Quản lý đường dẫn chính trong project.
    """
    root_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"

    def make_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


paths = Paths()


@dataclass
class TrainingConfig:
    """
    Cấu hình training cho decoder-only Qwen2.5-1.5B trên ViHealthQA.
    Tối ưu cho GPU 16GB (RTX 5080) với bfloat16 + gradient checkpointing.
    """

    # ----- Model -----
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # ----- Tokenization -----
    # 16GB VRAM: 768 là hợp lý cho QA, không quá tốn bộ nhớ.
    max_seq_length: int = 768

    # ----- Batch & Gradient -----
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # => effective batch ~16

    # ----- Training -----
    num_train_epochs: int = 3
    learning_rate: float = 5e-6   # giảm nhẹ để tránh nổ gradient
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # ----- Logging / Eval / Save -----
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2

    # ----- Precision -----
    fp16: bool = False           # KHÔNG dùng fp16
    bf16: bool = True            # Dùng bfloat16 (RTX 5080 hỗ trợ tốt)

    seed: int = 42

    @property
    def output_dir(self) -> str:
        return str(paths.checkpoints_dir / "qwen2_5_1_5b_vihealthqa")


train_config = TrainingConfig()
