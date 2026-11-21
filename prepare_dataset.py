"""Prepare Real-World Fruit Quality Dataset.

This script aggregates and organizes real-world photographic images into:
  data/train/<class>/
  data/val/<class>/
  data/test/<class>/

Classes:
  - fresh  (Grade A: Fresh, crisp, unblemished fruit - export quality)
  - medium (Grade B: Table ripe, moderate shelf-life, minor cosmetic spots)
  - rotten (Grade C: Spoiled, moldy, oxidized, or defective fruit - discard)
"""
import os
import shutil
import random
from pathlib import Path

# Random seed for reproducible dataset splits
SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Known local cache and archive paths
KAGGLE_DATASET = Path(r"C:\Users\91914\.cache\kagglehub\datasets\sriramr\fruits-fresh-and-rotten-for-classification\versions\1\dataset")
ROBOFLOW_DATASET = Path(r"C:\Users\91914\Downloads\archive\Fruit Quality Classification")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images_from_dir(directory):
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.suffix.lower() in EXTS]


def build_dataset(target_total=4000):
    print("=" * 60)
    print("Building Real-World Fruit Quality Grading Dataset")
    print(f"Target dataset size: ~{target_total} images")
    print("=" * 60)

    # 1. Collect from Kaggle dataset
    fresh_kaggle = []
    medium_kaggle = []
    rotten_kaggle = []

    if KAGGLE_DATASET.exists():
        for split in ["train", "test"]:
            sdir = KAGGLE_DATASET / split
            # Fresh apples and oranges -> fresh (Grade A)
            fresh_kaggle.extend(collect_images_from_dir(sdir / "freshapples"))
            fresh_kaggle.extend(collect_images_from_dir(sdir / "freshoranges"))
            # Ripe bananas -> medium (Grade B)
            medium_kaggle.extend(collect_images_from_dir(sdir / "freshbanana"))
            # Rotten fruits -> rotten (Grade C)
            rotten_kaggle.extend(collect_images_from_dir(sdir / "rottenapples"))
            rotten_kaggle.extend(collect_images_from_dir(sdir / "rottenbanana"))
            rotten_kaggle.extend(collect_images_from_dir(sdir / "rottenoranges"))

    # 2. Collect from Roboflow dataset
    fresh_rf = []
    medium_rf = []
    rotten_rf = []

    if ROBOFLOW_DATASET.exists():
        for split in ["train", "valid"]:
            img_dir = ROBOFLOW_DATASET / split / "images"
            if not img_dir.exists():
                continue
            for img_p in collect_images_from_dir(img_dir):
                name = img_p.name.lower()
                # Classify based on filename or roboflow naming
                if "bad" in name or "0-1" in name or "0day" in name or "0-day" in name:
                    rotten_rf.append(img_p)
                elif "banana" in name or "guava" in name:
                    medium_rf.append(img_p)
                else:
                    fresh_rf.append(img_p)

    all_fresh = fresh_kaggle + fresh_rf
    all_medium = medium_kaggle + medium_rf
    all_rotten = rotten_kaggle + rotten_rf

    print(f"Discovered real images:")
    print(f"  Fresh images available:  {len(all_fresh)}")
    print(f"  Medium images available: {len(all_medium)}")
    print(f"  Rotten images available: {len(all_rotten)}")

    # Target per class: ~1400 fresh, ~1200 medium, ~1400 rotten
    random.shuffle(all_fresh)
    random.shuffle(all_medium)
    random.shuffle(all_rotten)

    selected_fresh = all_fresh[:1400]
    selected_medium = all_medium[:1200]
    selected_rotten = all_rotten[:1400]

    classes_data = {
        "fresh": selected_fresh,
        "medium": selected_medium,
        "rotten": selected_rotten,
    }

    # Clean existing data splits if needed
    for split in ["train", "val", "test"]:
        s_path = DATA_DIR / split
        if s_path.exists():
            shutil.rmtree(s_path)

    summary = {"train": 0, "val": 0, "test": 0}

    for cls_name, file_list in classes_data.items():
        n = len(file_list)
        n_tr = int(0.70 * n)
        n_va = int(0.15 * n)

        splits = {
            "train": file_list[:n_tr],
            "val": file_list[n_tr:n_tr + n_va],
            "test": file_list[n_tr + n_va:],
        }

        for split_name, files in splits.items():
            dest_dir = DATA_DIR / split_name / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for idx, src in enumerate(files):
                ext = src.suffix.lower()
                dst = dest_dir / f"{cls_name}_{idx:04d}{ext}"
                shutil.copy2(src, dst)
            summary[split_name] += len(files)
            print(f"  [{split_name:5s}] {cls_name:8s}: {len(files)} images copied")

    total_copied = sum(summary.values())
    print("-" * 60)
    print(f"Successfully organized {total_copied} real-world images!")
    print(f"  Train: {summary['train']} (70%)")
    print(f"  Val:   {summary['val']} (15%)")
    print(f"  Test:  {summary['test']} (15%)")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset()
