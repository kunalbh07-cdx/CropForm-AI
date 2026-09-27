import os

import torch
import matplotlib.pyplot as plt
from PIL import Image

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

THRESHOLD = 0.5

OUTPUT_DIR = "results/predictions"

NUM_IMAGES = 5


# ==========================================
# Create output folder
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# Load test dataset
# ==========================================

test_dataset = CropSegmentationDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    split_file=TEST_SPLIT,
    image_size=IMAGE_SIZE
)

print(
    "Test samples:",
    len(test_dataset)
)


# ==========================================
# Load trained model
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

print(
    "Loaded model:",
    MODEL_PATH
)


# ==========================================
# Generate predictions
# ==========================================

for index in range(
    min(NUM_IMAGES, len(test_dataset))
):

    # Get image and ground-truth mask
    image, mask = test_dataset[index]

    # Add batch dimension
    input_tensor = image.unsqueeze(0)

    # Move to GPU/CPU
    input_tensor = input_tensor.to(device)


    # Model prediction
    with torch.no_grad():

        output = model(
            input_tensor
        )

        probability = torch.sigmoid(
            output
        )

        prediction = (
            probability >= THRESHOLD
        ).float()


    # Remove batch/channel dimensions
    prediction = prediction.squeeze().cpu().numpy()

    mask = mask.squeeze().numpy()

    image_display = image.permute(
        1, 2, 0
    ).numpy()


    # ======================================
    # Create visualization
    # ======================================

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )


    # Original image
    axes[0].imshow(
        image_display
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")


    # Ground truth
    axes[1].imshow(
        mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")


    # Prediction
    axes[2].imshow(
        prediction,
        cmap="gray"
    )

    axes[2].set_title(
        "U-Net Prediction"
    )

    axes[2].axis("off")


    # Overall title
    figure.suptitle(
        f"Sample {index + 1}"
    )


    # Save result
    output_path = os.path.join(
        OUTPUT_DIR,
        f"prediction_{index + 1}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Saved:",
        output_path
    )


print(
    "\nVisualization completed."
)