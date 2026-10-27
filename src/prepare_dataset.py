from pathlib import Path
import random
import shutil

import cv2
import numpy as np


# -----------------------------------
# 1. Configure directories
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data/raw/PennFudanPed"

IMAGES_DIR = RAW_DIR / "PNGImages"
MASKS_DIR = RAW_DIR / "PedMasks"

OUTPUT_DIR = ROOT / "data/processed"


# -----------------------------------
# 2. Create reproducible train/val/test split
# -----------------------------------

images = sorted(IMAGES_DIR.glob("*.png"))

if len(images) != 170:
    raise RuntimeError(
        f"Expected 170 source images, found {len(images)}."
    )

random.seed(42)
random.shuffle(images)

train_images = images[:119]
val_images = images[119:144]
test_images = images[144:]


splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images
}


# -----------------------------------
# 3. Convert masks to YOLO boxes
# -----------------------------------

def mask_to_yolo(mask):

    height, width = mask.shape[:2]

    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids != 0]

    labels = []

    for object_id in object_ids:

        ys, xs = np.where(mask == object_id)

        if len(xs) == 0:
            continue

        # Pixel coordinates
        x_min = xs.min()
        x_max = xs.max() + 1

        y_min = ys.min()
        y_max = ys.max() + 1

        # Normalize coordinates
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height

        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        labels.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return labels


# -----------------------------------
# 4. Process each dataset split
# -----------------------------------

def process_split(split_name, image_paths):

    output_images = OUTPUT_DIR / "images" / split_name
    output_labels = OUTPUT_DIR / "labels" / split_name

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    total_objects = 0

    for image_path in image_paths:

        mask_path = (
            MASKS_DIR / f"{image_path.stem}_mask.png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED
        )

        if mask is None or mask.ndim != 2:
            raise ValueError(f"Invalid mask: {mask_path}")

        labels = mask_to_yolo(mask)

        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        label_path = (
            output_labels / f"{image_path.stem}.txt"
        )

        label_path.write_text(
            "\n".join(labels) + ("\n" if labels else "")
        )

        total_objects += len(labels)

    print(
        f"{split_name}: "
        f"{len(image_paths)} images, "
        f"{total_objects} pedestrians"
    )


# -----------------------------------
# 5. Run preparation
# -----------------------------------

if __name__ == "__main__":

    # Clear old generated data to avoid stale files
    for folder_name in ("images", "labels"):
        folder = OUTPUT_DIR / folder_name

        if folder.exists():
            shutil.rmtree(folder)

    for split_name, image_paths in splits.items():
        process_split(split_name, image_paths)

    print("\nDataset preparation completed!")