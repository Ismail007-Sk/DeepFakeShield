from pathlib import Path
import shutil
import random

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent

PRE_REAL = BASE_DIR / "Dataset" / "Real"
PRE_FAKE = BASE_DIR / "Dataset" / "Fake"

OUTPUT_DIR = BASE_DIR / "SplitData"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def split_and_copy(src_folder, class_name):
    images = list(src_folder.glob("*.*"))
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    train_dir = OUTPUT_DIR / "train" / class_name
    val_dir = OUTPUT_DIR / "val" / class_name
    test_dir = OUTPUT_DIR / "test" / class_name

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for file in train_files:
        shutil.copy(file, train_dir / file.name)

    for file in val_files:
        shutil.copy(file, val_dir / file.name)

    for file in test_files:
        shutil.copy(file, test_dir / file.name)

    print(f"{class_name}: {len(train_files)} train | {len(val_files)} val | {len(test_files)} test")


if __name__ == "__main__":
    print("Splitting dataset for DeepFakeShield...")

    split_and_copy(PRE_REAL, "Real")
    split_and_copy(PRE_FAKE, "Fake")

    print("Dataset split completed ✅")
