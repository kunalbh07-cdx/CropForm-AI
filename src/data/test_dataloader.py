from pathlib import Path

from torch.utils.data import DataLoader

from dataset import CropSegmentationDataset


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "dataset" / "raw" / "images"
MASK_DIR = BASE_DIR / "dataset" / "raw" / "annotations"
SPLIT_DIR = BASE_DIR / "dataset" / "splits"


# ==========================================
# CREATE DATASET
# ==========================================

train_dataset = CropSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    split_file=SPLIT_DIR / "train.txt",
    image_size=256
)


# ==========================================
# CREATE DATALOADER
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)


# ==========================================
# GET ONE BATCH
# ==========================================

images, masks = next(iter(train_loader))


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("              DATALOADER TEST")
print("=" * 60)

print(f"Number of training samples : {len(train_dataset)}")

print(f"\nImage batch shape          : {images.shape}")
print(f"Mask batch shape           : {masks.shape}")

print(f"\nImage dtype                : {images.dtype}")
print(f"Mask dtype                 : {masks.dtype}")

print(f"\nImage minimum              : {images.min().item()}")
print(f"Image maximum              : {images.max().item()}")

print(f"Mask minimum               : {masks.min().item()}")
print(f"Mask maximum               : {masks.max().item()}")

print("\n" + "=" * 60)
print("              TEST COMPLETE")
print("=" * 60)