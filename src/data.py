# src/data.py
from typing import Optional

from datasets import DatasetDict, load_dataset

from .config import train_config, paths


# src/data.py
from typing import Optional

from datasets import DatasetDict, load_dataset

from .config import train_config, paths


def load_vihealthqa(cache_dir=None):
    paths.make_dirs()
    data_dir = paths.data_dir / "raw"

    train_path = str(data_dir / "train.csv")
    val_path   = str(data_dir / "val.csv")
    test_path  = str(data_dir / "test.csv")

    print("🔹 Using LOCAL CSV instead of HuggingFace downloads.")

    ds = DatasetDict({
        "train": load_dataset("csv", data_files=train_path)["train"],
        "validation": load_dataset("csv", data_files=val_path)["validation"],
        "test": load_dataset("csv", data_files=test_path)["test"],
    })

    print(ds)
    print("\n📌 Sample train row:", ds["train"][0])
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
