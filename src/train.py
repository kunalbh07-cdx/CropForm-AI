import os

import torch
from torch.utils.data import DataLoader

from data.dataset import CropSegmentationDataset
from models.unet import UNet
from models.losses import BCEDiceLoss


# =========================
# Configuration
# =========================

IMAGE_DIR = "dataset/raw/images"
MASK_DIR = "dataset/raw/annotations"

TRAIN_SPLIT = "dataset/splits/train.txt"
VAL_SPLIT = "dataset/splits/val.txt"

MODEL_DIR = "models"

IMAGE_SIZE = 256

BATCH_SIZE = 2

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

NUM_WORKERS = 0


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================
# Create datasets
# =========================

train_dataset = CropSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    split_file=TRAIN_SPLIT,
    image_size=IMAGE_SIZE
)

val_dataset = CropSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    split_file=VAL_SPLIT,
    image_size=IMAGE_SIZE
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# =========================
# Create DataLoaders
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# =========================
# Create model
# =========================

model = UNet(
    in_channels=3,
    out_channels=1
)

model = model.to(device)


# =========================
# Loss function
# =========================

criterion = BCEDiceLoss()


# =========================
# Optimizer
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Mixed Precision
# =========================

use_amp = torch.cuda.is_available()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# =========================
# Create model directory
# =========================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


best_val_loss = float("inf")


# =========================
# Training loop
# =========================

for epoch in range(NUM_EPOCHS):

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward pass
        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

        # Backpropagation
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # =========================
    # Validation
    # =========================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    masks
                )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # =========================
    # Print progress
    # =========================

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # =========================
    # Save best model
    # =========================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        save_path = os.path.join(
            MODEL_DIR,
            "unet_best.pth"
        )

        torch.save(
            model.state_dict(),
            save_path
        )

        print(
            "Best model saved:",
            save_path
        )


print("Training completed.")