from pathlib import Path
from PIL import Image


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "dataset" / "raw" / "images"
MASK_DIR = BASE_DIR / "dataset" / "raw" / "annotations"


# ==========================================
# CHECK FOLDERS
# ==========================================

print("=" * 60)
print("       CROP SEGMENTATION DATASET CHECK")
print("=" * 60)

if not IMAGE_DIR.exists():
    print("ERROR: Images folder not found!")
    print(IMAGE_DIR)
    exit()

if not MASK_DIR.exists():
    print("ERROR: Annotations folder not found!")
    print(MASK_DIR)
    exit()


# ==========================================
# FIND FILES
# ==========================================

valid_extensions = [".jpg", ".jpeg", ".png"]

image_files = sorted(
    [
        file for file in IMAGE_DIR.iterdir()
        if file.suffix.lower() in valid_extensions
    ]
)

mask_files = sorted(
    [
        file for file in MASK_DIR.iterdir()
        if file.suffix.lower() in valid_extensions
    ]
)


print("\nDATASET FILES")
print("-" * 60)

print(f"Images found      : {len(image_files)}")
print(f"Annotations found : {len(mask_files)}")


# ==========================================
# CHECK MATCHING FILES
# ==========================================

image_names = {file.stem for file in image_files}
mask_names = {file.stem for file in mask_files}

missing_masks = image_names - mask_names
missing_images = mask_names - image_names


print("\nMATCHING CHECK")
print("-" * 60)

print(f"Images without annotation : {len(missing_masks)}")
print(f"Annotations without image : {len(missing_images)}")


# ==========================================
# CHECK FIRST IMAGE
# ==========================================

if image_files:

    image_path = image_files[0]

    print("\nSAMPLE IMAGE")
    print("-" * 60)

    try:

        with Image.open(image_path) as image:

            print(f"Filename : {image_path.name}")
            print(f"Size     : {image.size}")
            print(f"Mode     : {image.mode}")

    except Exception as error:

        print("Could not open image.")
        print(error)


# ==========================================
# CHECK FIRST ANNOTATION
# ==========================================

if mask_files:

    mask_path = mask_files[0]

    print("\nSAMPLE ANNOTATION")
    print("-" * 60)

    try:

        with Image.open(mask_path) as mask:

            print(f"Filename : {mask_path.name}")
            print(f"Size     : {mask.size}")
            print(f"Mode     : {mask.mode}")

    except Exception as error:

        print("Could not open annotation.")
        print(error)


# ==========================================
# CHECK IMAGE / MASK SIZE
# ==========================================

if image_files and mask_files:

    common_names = image_names.intersection(mask_names)

    if common_names:

        sample_name = sorted(common_names)[0]

        image_path = next(
            (
                IMAGE_DIR / f"{sample_name}{extension}"
                for extension in valid_extensions
                if (IMAGE_DIR / f"{sample_name}{extension}").exists()
            ),
            None
        )

        mask_path = next(
            (
                MASK_DIR / f"{sample_name}{extension}"
                for extension in valid_extensions
                if (MASK_DIR / f"{sample_name}{extension}").exists()
            ),
            None
        )

        if image_path and mask_path:

            try:

                with Image.open(image_path) as image:
                    image_size = image.size

                with Image.open(mask_path) as mask:
                    mask_size = mask.size

                print("\nIMAGE / MASK SIZE CHECK")
                print("-" * 60)

                print(f"Image size : {image_size}")
                print(f"Mask size  : {mask_size}")

                if image_size == mask_size:
                    print("STATUS     : PASS")
                else:
                    print("STATUS     : WARNING")

            except Exception as error:

                print("Could not compare sizes.")
                print(error)


# ==========================================
# FINAL STATUS
# ==========================================

print("\n" + "=" * 60)

if (
    len(image_files) > 0
    and len(mask_files) > 0
    and len(missing_masks) == 0
    and len(missing_images) == 0
):

    print("DATASET STATUS: READY FOR NEXT STEP")

else:

    print("DATASET STATUS: NEEDS ATTENTION")

print("=" * 60)