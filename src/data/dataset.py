from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CropSegmentationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        split_file,
        image_size=256
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        # Read image names from split file
        with open(split_file, "r") as file:
            self.image_names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print(
            f"Loaded {len(self.image_names)} samples "
            f"from {Path(split_file).name}"
        )

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]

        # ==========================================
        # FIND IMAGE
        # ==========================================

        image_path = None

        for extension in [".png", ".jpg", ".jpeg"]:

            candidate = self.image_dir / (
                image_name + extension
            )

            if candidate.exists():
                image_path = candidate
                break

        # ==========================================
        # FIND MASK
        # ==========================================

        mask_path = None

        for extension in [".png", ".jpg", ".jpeg"]:

            candidate = self.mask_dir / (
                image_name + extension
            )

            if candidate.exists():
                mask_path = candidate
                break

        # ==========================================
        # CHECK FILES
        # ==========================================

        if image_path is None:
            raise FileNotFoundError(
                f"Image not found: {image_name}"
            )

        if mask_path is None:
            raise FileNotFoundError(
                f"Mask not found: {image_name}"
            )

        # ==========================================
        # LOAD IMAGE
        # ==========================================

        image = Image.open(image_path).convert("RGB")

        # Load segmentation mask
        mask = Image.open(mask_path).convert("L")

        # ==========================================
        # RESIZE
        # ==========================================

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        # IMPORTANT:
        # NEAREST is used for masks so that
        # class labels are not interpolated.

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        # ==========================================
        # CONVERT TO NUMPY
        # ==========================================

        image = np.array(
            image,
            dtype=np.float32
        )

        mask = np.array(
            mask,
            dtype=np.float32
        )

        # ==========================================
        # NORMALIZE IMAGE
        # ==========================================

        image = image / 255.0

        # ==========================================
        # CONVERT MASK TO BINARY
        # ==========================================

        mask = mask / 255.0

        mask = (
            mask > 0.5
        ).astype(np.float32)

        # ==========================================
        # IMAGE:
        # H x W x C
        # →
        # C x H x W
        # ==========================================

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ==========================================
        # NUMPY → PYTORCH
        # ==========================================

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        # ==========================================
        # MASK:
        # H x W
        # →
        # 1 x H x W
        # ==========================================

        mask = mask.unsqueeze(0)

        return image, mask 