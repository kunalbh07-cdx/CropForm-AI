import os

import numpy as np
import pandas as pd


# ==========================================
# Configuration
# ==========================================

INPUT_FILE = "results/features.csv"

OUTPUT_FILE = "results/fused_features.csv"

NORMALIZED_OUTPUT_FILE = (
    "results/fused_features_normalized.csv"
)


# ==========================================
# Load extracted features
# ==========================================

df = pd.read_csv(INPUT_FILE)

print("Loaded feature data.")
print("Original shape:", df.shape)


# ==========================================
# Select feature groups
# ==========================================

shape_features = [
    "total_area",
    "total_perimeter",
    "largest_contour_area",
    "aspect_ratio",
    "circularity"
]


size_features = [
    "segmented_pixel_count",
    "bounding_box_width",
    "bounding_box_height",
    "vegetation_ratio"
]


colour_features = [
    "mean_red",
    "mean_green",
    "mean_blue",
    "mean_hue",
    "mean_saturation",
    "mean_value"
]


# ==========================================
# Check required columns
# ==========================================

required_features = (
    shape_features
    + size_features
    + colour_features
)

missing_features = [
    feature
    for feature in required_features
    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "Missing features: "
        + str(missing_features)
    )


# ==========================================
# Create fused feature table
# ==========================================

fused_features = df[
    required_features
].copy()


# ==========================================
# Create fused feature vector
# ==========================================

feature_matrix = fused_features.values.astype(
    np.float32
)


print(
    "Fused feature matrix shape:",
    feature_matrix.shape
)


# ==========================================
# Save raw fused features
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

fused_features.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Saved raw fused features to:",
    OUTPUT_FILE
)


# ==========================================
# Feature normalization
# ==========================================

mean = feature_matrix.mean(
    axis=0
)

std = feature_matrix.std(
    axis=0
)

# Prevent division by zero
std[std == 0] = 1.0


normalized_matrix = (
    feature_matrix - mean
) / std


# ==========================================
# Convert normalized features to DataFrame
# ==========================================

normalized_features = pd.DataFrame(
    normalized_matrix,
    columns=required_features
)


# ==========================================
# Save normalized features
# ==========================================

normalized_features.to_csv(
    NORMALIZED_OUTPUT_FILE,
    index=False
)


print(
    "Saved normalized fused features to:",
    NORMALIZED_OUTPUT_FILE
)


# ==========================================
# Display results
# ==========================================

print(
    "\nNumber of fused features:",
    len(required_features)
)

print(
    "Number of samples:",
    len(normalized_features)
)

print(
    "\nFirst 5 fused feature rows:"
)

print(
    normalized_features.head()
)

print(
    "\nFeature fusion completed successfully."
)