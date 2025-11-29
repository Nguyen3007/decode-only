# src/data.py
from typing import Optional

from datasets import DatasetDict, load_dataset

from .config import train_config, paths


# src/data.py
from typing import Optional

from datasets import DatasetDict, load_dataset

from .config import train_config, paths


def load_vihealthqa(
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load full ViHealthQA từ HuggingFace Datasets.

    Dataset có 3 split: train / validation / test
    với các cột chính: id, question, answer, link.
    """
    paths.make_dirs()

    print(f"🔹 Loading dataset: {train_config.dataset_name}")
    ds = load_dataset(
        train_config.dataset_name,
        cache_dir=cache_dir,
        # KHÔNG còn dùng trust_remote_code ở đây
    )

    print(ds)
    print("\n📌 Sample train row:")
    print(ds["train"][0])

    return ds



def main() -> None:
    """
    Cho phép chạy file này trực tiếp:
    python -m src.data
    để test việc load dataset.
    """
    ds = load_vihealthqa()
    print("\n✅ Loaded ViHealthQA successfully!")
    print("Splits:", ds.keys())
    for split in ds.keys():
        print(f"{split}: {len(ds[split])} examples")


if __name__ == "__main__":
    main()
