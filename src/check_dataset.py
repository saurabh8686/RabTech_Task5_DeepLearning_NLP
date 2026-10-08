
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "aclImdb"

print("IMDb DATASET VERIFICATION")
print("=" * 35)

for split in ["train", "test"]:
    for label in ["pos", "neg"]:
        folder = DATA_DIR / split / label

        if not folder.exists():
            print(f"ERROR: Missing folder: {folder}")
            continue

        files = list(folder.glob("*.txt"))
        print(f"{split}/{label}: {len(files)} reviews")

print("\nVerification completed.")
