import sys
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image


# ==========================================
# Allow importing from src
# ==========================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from models.unet import UNet


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/unet_best.pth"

IMAGE_SIZE = 256

THRESHOLD = 0.5


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Crop Segmentation AI",
    page_icon="🌾",
    layout="wide"
)


# ==========================================
# Title
# ==========================================

st.title(
    "🌾 Crop Segmentation AI"
)

st.write(
    "U-Net based segmentation with "
    "Shape, Size and Colour feature extraction."
)


# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# Load model
# ==========================================

@st.cache_resource
def load_model():

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

    return model


model = load_model()


# ==========================================
# Upload image
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a crop/vegetation image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    # ======================================
    # Read image
    # ======================================

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ======================================
    # Display original
    # ======================================

    st.subheader(
        "Original Image"
    )

    st.image(
        original_image,
        use_container_width=True
    )


    # ======================================
    # Preprocess
    # ======================================

    resized_image = original_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.array(
        resized_image
    ).astype(
        np.float32
    ) / 255.0


    # HWC → CHW
    image_tensor = torch.from_numpy(
        image_array
    ).permute(
        2, 0, 1
    )


    # Add batch dimension
    input_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    # ======================================
    # U-Net prediction
    # ======================================

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


    # ======================================
    # Convert mask
    # ======================================

    mask = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # ======================================
    # Display segmentation
    # ======================================

    st.subheader(
        "Segmentation Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            resized_image,
            caption="Input Image",
            use_container_width=True
        )

    with col2:

        st.image(
            mask * 255,
            caption="Predicted Segmentation Mask",
            use_container_width=True
        )


    # ======================================
    # Feature extraction
    # ======================================

    mask_255 = mask * 255

    image_rgb = np.array(
        resized_image
    )


    # ======================================
    # Shape features
    # ======================================

    contours, _ = cv2.findContours(
        mask_255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) > 0:

        total_area = sum(
            cv2.contourArea(c)
            for c in contours
        )

        total_perimeter = sum(
            cv2.arcLength(
                c,
                True
            )
            for c in contours
        )

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        largest_area = cv2.contourArea(
            largest_contour
        )

        x, y, width, height = cv2.boundingRect(
            largest_contour
        )

        if height > 0:

            aspect_ratio = (
                width / height
            )

        else:

            aspect_ratio = 0


        if total_perimeter > 0:

            circularity = (
                4 * np.pi * total_area
                /
                (
                    total_perimeter ** 2
                )
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
    # Size features
    # ======================================

    segmented_pixels = np.sum(
        mask > 0
    )

    total_pixels = (
        mask.shape[0]
        *
        mask.shape[1]
    )

    vegetation_ratio = (
        segmented_pixels
        /
        total_pixels
    )


    # ======================================
    # Colour features
    # ======================================

    selected_pixels = image_rgb[
        mask > 0
    ]


    if len(selected_pixels) > 0:

        mean_red = np.mean(
            selected_pixels[:, 0]
        )

        mean_green = np.mean(
            selected_pixels[:, 1]
        )

        mean_blue = np.mean(
            selected_pixels[:, 2]
        )


        image_hsv = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2HSV
        )

        hsv_pixels = image_hsv[
            mask > 0
        ]


        mean_hue = np.mean(
            hsv_pixels[:, 0]
        )

        mean_saturation = np.mean(
            hsv_pixels[:, 1]
        )

        mean_value = np.mean(
            hsv_pixels[:, 2]
        )

    else:

        mean_red = 0
        mean_green = 0
        mean_blue = 0

        mean_hue = 0
        mean_saturation = 0
        mean_value = 0


    # ======================================
    # Display features
    # ======================================

    st.subheader(
        "🌿 Extracted Features"
    )


    col1, col2, col3 = st.columns(3)


    # Shape
    with col1:

        st.markdown(
            "### Shape"
        )

        st.write(
            "Area:",
            round(
                total_area,
                2
            )
        )

        st.write(
            "Perimeter:",
            round(
                total_perimeter,
                2
            )
        )

        st.write(
            "Circularity:",
            round(
                circularity,
                4
            )
        )

        st.write(
            "Aspect Ratio:",
            round(
                aspect_ratio,
                4
            )
        )


    # Size
    with col2:

        st.markdown(
            "### Size"
        )

        st.write(
            "Pixel Count:",
            int(
                segmented_pixels
            )
        )

        st.write(
            "Bounding Box Width:",
            width
        )

        st.write(
            "Bounding Box Height:",
            height
        )

        st.write(
            "Vegetation Ratio:",
            round(
                vegetation_ratio,
                4
            )
        )


    # Colour
    with col3:

        st.markdown(
            "### Colour"
        )

        st.write(
            "Mean Red:",
            round(
                mean_red,
                2
            )
        )

        st.write(
            "Mean Green:",
            round(
                mean_green,
                2
            )
        )

        st.write(
            "Mean Blue:",
            round(
                mean_blue,
                2
            )
        )

        st.write(
            "Mean Hue:",
            round(
                mean_hue,
                2
            )
        )

        st.write(
            "Mean Saturation:",
            round(
                mean_saturation,
                2
            )
        )

        st.write(
            "Mean Value:",
            round(
                mean_value,
                2
            )
        )


    # ======================================
    # Final information
    # ======================================

    st.success(
        "Segmentation and feature extraction completed!"
    )