import os

import cv2
import numpy as np
import pandas as pd
import torch

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

OUTPUT_FILE = "results/features.csv"


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

print("Test samples:", len(test_dataset))


# ==========================================
# Load trained U-Net
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
# Store extracted features
# ==========================================

all_features = []


# ==========================================
# Process every test image
# ==========================================

for index in range(len(test_dataset)):

    # --------------------------------------
    # Get image
    # --------------------------------------

    image_tensor, _ = test_dataset[index]

    # Add batch dimension
    input_tensor = image_tensor.unsqueeze(0)

    # Move image to GPU
    input_tensor = input_tensor.to(device)


    # --------------------------------------
    # Generate segmentation prediction
    # --------------------------------------

    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(output)

        prediction = (
            probability >= THRESHOLD
        ).float()


    # --------------------------------------
    # Convert prediction to NumPy
    # --------------------------------------

    mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # Convert 0/1 mask to 0/255
    mask = mask * 255


    # --------------------------------------
    # Convert image to NumPy
    # --------------------------------------

    image_rgb = (
        image_tensor
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    # Convert 0-1 to 0-255
    image_rgb = (
        image_rgb * 255
    ).astype(np.uint8)


    # ======================================
    # SHAPE FEATURES
    # ======================================

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) > 0:

        # Total segmented area
        total_area = sum(
            cv2.contourArea(contour)
            for contour in contours
        )

        # Total perimeter
        total_perimeter = sum(
            cv2.arcLength(
                contour,
                True
            )
            for contour in contours
        )

        # Largest contour
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        largest_area = cv2.contourArea(
            largest_contour
        )

        # Bounding box
        x, y, width, height = cv2.boundingRect(
            largest_contour
        )

        # Aspect ratio
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 0

        # Circularity
        if total_perimeter > 0:
            circularity = (
                4 * np.pi * total_area
                /
                (total_perimeter ** 2)
            )
        else:
            circularity = 0

    else:

        total_area = 0
        total_perimeter = 0
        largest_area = 0
        width = 0
        height = 0
        aspect_ratio = 0
        circularity = 0


    # ======================================
    # SIZE FEATURES
    # ======================================

    segmented_pixel_count = np.sum(
        mask > 0
    )

    image_area = (
        mask.shape[0] *
        mask.shape[1]
    )

    if image_area > 0:

        vegetation_ratio = (
            segmented_pixel_count
            /
            image_area
        )

    else:

        vegetation_ratio = 0


    # ======================================
    # COLOUR FEATURES
    # ======================================

    # Select only segmented pixels
    segmented_pixels = image_rgb[
        mask > 0
    ]


    if len(segmented_pixels) > 0:

        # Mean RGB
        mean_red = np.mean(
            segmented_pixels[:, 0]
        )

        mean_green = np.mean(
            segmented_pixels[:, 1]
        )

        mean_blue = np.mean(
            segmented_pixels[:, 2]
        )


        # Convert RGB image to HSV
        image_hsv = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2HSV
        )

        segmented_hsv = image_hsv[
            mask > 0
        ]


        # Mean HSV
        mean_hue = np.mean(
            segmented_hsv[:, 0]
        )

        mean_saturation = np.mean(
            segmented_hsv[:, 1]
        )

        mean_value = np.mean(
            segmented_hsv[:, 2]
        )

    else:

        mean_red = 0
        mean_green = 0
        mean_blue = 0

        mean_hue = 0
        mean_saturation = 0
        mean_value = 0


    # ======================================
    # Store features
    # ======================================

    features = {

        "image_index": index + 1,

        # Shape
        "total_area": total_area,
        "total_perimeter": total_perimeter,
        "largest_contour_area": largest_area,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,

        # Size
        "segmented_pixel_count":
            segmented_pixel_count,

        "bounding_box_width":
            width,

        "bounding_box_height":
            height,

        "vegetation_ratio":
            vegetation_ratio,

        # RGB
        "mean_red": mean_red,
        "mean_green": mean_green,
        "mean_blue": mean_blue,

        # HSV
        "mean_hue": mean_hue,
        "mean_saturation":
            mean_saturation,
        "mean_value": mean_value
    }


    all_features.append(features)


    # Progress
    if (index + 1) % 50 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(test_dataset)} images"
        )


# ==========================================
# Create DataFrame
# ==========================================

df = pd.DataFrame(
    all_features
)


# ==========================================
# Create results folder
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)


# ==========================================
# Save features
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFeature extraction completed.")

print(
    "Saved features to:",
    OUTPUT_FILE
)

print(
    "\nFeature shape:",
    df.shape
)

print(
    "\nFirst 5 rows:"
)

print(
    df.head()
)