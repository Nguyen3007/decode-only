# src/data.py
from typing import Optional

from datasets import DatasetDict, load_dataset

from .config import train_config, paths


# src/data.py
from typing import Optional

from datasets import DatasetDict, load_dataset

from .config import train_config, paths


def load_vihealthqa(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load ViHealthQA từ 3 file CSV local (train/val/test) đã upload lên Vast.
    """

    paths.make_dirs()
    data_dir = paths.data_dir / "raw"

    train_path = str(data_dir / "train.csv")
    val_path   = str(data_dir / "val.csv")
    test_path  = str(data_dir / "test.csv")

    print("🔹 Using LOCAL CSV instead of HuggingFace downloads.")
    print("   Train:", train_path)
    print("   Val  :", val_path)
    print("   Test :", test_path)

    # ⭐ QUAN TRỌNG: dùng dict {split_name: path}
    train_ds = load_dataset("csv", data_files={"train": train_path})["train"]
    val_ds   = load_dataset("csv", data_files={"validation": val_path})["validation"]
    test_ds  = load_dataset("csv", data_files={"test": test_path})["test"]

    ds = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    print(ds)
    print("\n📌 Sample train row:")
    print(ds["train"][0])

    return ds


def main() -> None:
    ds = load_vihealthqa()
    print("\n✅ Loaded ViHealthQA from local CSV successfully!")
    print("Splits:", ds.keys())
    for split in ds.keys():
        print(f"{split}: {len(ds[split])} examples")


if __name__ == "__main__":
    main()
