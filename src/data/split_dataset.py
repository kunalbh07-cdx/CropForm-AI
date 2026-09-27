from pathlib import Path
import random

SEED = 42
TRAIN_RATIO = 0.70 
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# PATHS
BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = BASE_DIR / "dataset" / "raw" / "images"
MASK_DIR = BASE_DIR / "dataset" / "raw" / "annotations"
SPLIT_DIR = BASE_DIR / "dataset" / "splits"

# CREATE SPLIT DIRECTORY
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# GET IMAGES
valid_extensions = [".jpg", ".jpeg", ".png"]
images = sorted(
    [
        file for file in IMAGE_DIR.iterdir()
        if file.suffix.lower() in valid_extensions
    ]
)

# CHECK MATCHING MASKS
valid_pairs = []
for image_path in images:
    mask_path = MASK_DIR / image_path.name
    if mask_path.exists():
        valid_pairs.append(image_path.stem)
    else:
        print(f"WARNING: Missing mask for {image_path.name}")
print("=" * 60)
print("          DATASET SPLITTING")
print("=" * 60)
print(f"Total valid images: {len(valid_pairs)}")

#SHUFFLE
random.seed(SEED)
random.shuffle(valid_pairs)

# CALCULATE SPLIT
total = len(valid_pairs)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)
train_files = valid_pairs[:train_end]
val_files = valid_pairs[train_end:val_end]
test_files = valid_pairs[val_end:]

# SAVE SPLITS
def save_split(filename, files):
    path = SPLIT_DIR / filename
    with open(path, "w") as f:
        for file_name in files:
            f.write(file_name + "\n")
save_split("train.txt", train_files)
save_split("val.txt", val_files)
save_split("test.txt", test_files)

# RESULTS
print("\nSPLIT RESULTS")
print("-" * 60)
print(f"Training   : {len(train_files)}")
print(f"Validation : {len(val_files)}")
print(f"Testing    : {len(test_files)}")
print("\nFiles created:")
print(SPLIT_DIR / "train.txt")
print(SPLIT_DIR / "val.txt")
print(SPLIT_DIR / "test.txt")
print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)