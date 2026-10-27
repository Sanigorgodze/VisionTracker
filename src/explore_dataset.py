from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Project directories
ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "data/raw/PennFudanPed/PNGImages"
MASKS_DIR = ROOT / "data/raw/PennFudanPed/PedMasks"


# Find all images
images = sorted(IMAGES_DIR.glob("*.png"))

print("Total images:", len(images))


# Select the first image
image_path = images[0]

# Find its segmentation mask
mask_path = MASKS_DIR / f"{image_path.stem}_mask.png"


# Load image and mask
image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

if image is None or mask is None:
    raise FileNotFoundError("Image or mask could not be loaded.")


# Convert BGR to RGB for Matplotlib
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Find unique object identifiers
object_ids = np.unique(mask)

print("Image:", image_path.name)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Object IDs:", object_ids)


# Display image and mask
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask, cmap="nipy_spectral")
axes[1].set_title("Segmentation Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()