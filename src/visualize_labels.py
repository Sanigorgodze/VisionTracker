from pathlib import Path

import cv2
import random


# -----------------------------------
# 1. Define project directories
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "data/processed"

IMAGES_DIR = DATASET / "images"
LABELS_DIR = DATASET / "labels"

OUTPUT_DIR = ROOT / "outputs/plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# 2. Convert YOLO boxes to pixels
# -----------------------------------

def yolo_to_pixels(x_center, y_center, box_width, box_height, img_w, img_h):

    x_min = int((x_center - box_width / 2) * img_w)
    y_min = int((y_center - box_height / 2) * img_h)

    x_max = int((x_center + box_width / 2) * img_w)
    y_max = int((y_center + box_height / 2) * img_h)

    return x_min, y_min, x_max, y_max


# -----------------------------------
# 3. Draw bounding boxes
# -----------------------------------

def visualize_image(image_path, split_name):

    # Find matching label file
    label_path = (
        LABELS_DIR
        / split_name
        / f"{image_path.stem}.txt"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    img_h, img_w = image.shape[:2]

    # Read YOLO annotations
    lines = label_path.read_text().splitlines()

    for line in lines:

        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(f"Invalid annotation: {line}")

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])

        # Convert normalized coordinates to pixels
        x_min, y_min, x_max, y_max = yolo_to_pixels(
            x_center,
            y_center,
            box_width,
            box_height,
            img_w,
            img_h
        )

        # Draw green bounding box
        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # Add class label
        cv2.putText(
            image,
            "person",
            (x_min, max(20, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return image


# -----------------------------------
# 4. Main function
# -----------------------------------

def main():

    split_name = "train"

    image_paths = sorted(
        (IMAGES_DIR / split_name).glob("*.png")
    )

    if not image_paths:
        raise RuntimeError("No images found. Run prepare_dataset.py first.")

    random.seed(42)

    # Select 5 random images
    selected_images = random.sample(
        image_paths,
        min(5, len(image_paths))
    )

    for index, image_path in enumerate(selected_images, start=1):

        image = visualize_image(image_path, split_name)

        output_path = OUTPUT_DIR / f"annotation_{index}.jpg"

        cv2.imwrite(str(output_path), image)

        print(f"Saved: {output_path}")

        cv2.imshow("YOLO Ground Truth", image)

        # Wait until any key is pressed
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()