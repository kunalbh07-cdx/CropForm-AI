import torch
from torch.utils.data import DataLoader

from data.dataset import CropSegmentationDataset
from models.unet import UNet


# ==========================================
# Configuration
# ==========================================

IMAGE_DIR = "dataset/raw/images"
MASK_DIR = "dataset/raw/annotations"

TEST_SPLIT = "dataset/splits/test.txt"

MODEL_PATH = "models/unet_best.pth"

IMAGE_SIZE = 256

BATCH_SIZE = 2

NUM_WORKERS = 0

THRESHOLD = 0.5


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==========================================
# Test Dataset
# ==========================================

test_dataset = CropSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    split_file=TEST_SPLIT,
    image_size=IMAGE_SIZE
)

print("Test samples:", len(test_dataset))


# ==========================================
# Test DataLoader
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ==========================================
# Load U-Net
# ==========================================

model = UNet(
    in_channels=3,
    out_channels=1
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Loaded trained model:", MODEL_PATH)


# ==========================================
# Metric counters
# ==========================================

total_intersection = 0.0
total_union = 0.0

total_predicted = 0.0
total_target = 0.0

total_true_positive = 0.0


# ==========================================
# Evaluation
# ==========================================

with torch.no_grad():

    for images, masks in test_loader:

        # Move data to GPU/CPU
        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        # Model prediction
        logits = model(images)

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Convert probabilities to binary mask
        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # Calculate intersection
        intersection = (
            predictions * masks
        ).sum().item()

        # Calculate union
        union = (
            predictions + masks
            - predictions * masks
        ).sum().item()

        # Number of predicted positive pixels
        predicted_pixels = (
            predictions.sum().item()
        )

        # Number of ground-truth positive pixels
        target_pixels = (
            masks.sum().item()
        )

        # True positive pixels
        true_positive = intersection

        # Add to totals
        total_intersection += intersection
        total_union += union

        total_predicted += predicted_pixels
        total_target += target_pixels

        total_true_positive += true_positive


# ==========================================
# Calculate metrics
# ==========================================

epsilon = 1e-8


# IoU
iou = (
    total_intersection
    /
    (total_union + epsilon)
)


# Dice Score
dice = (
    2 * total_intersection
    /
    (
        total_predicted
        + total_target
        + epsilon
    )
)


# Precision
precision = (
    total_true_positive
    /
    (total_predicted + epsilon)
)


# Recall
recall = (
    total_true_positive
    /
    (total_target + epsilon)
)


# ==========================================
# Print Results
# ==========================================

print("\n========== Evaluation Results ==========")

print(
    f"IoU       : {iou:.4f}"
)

print(
    f"Dice Score: {dice:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print("========================================")