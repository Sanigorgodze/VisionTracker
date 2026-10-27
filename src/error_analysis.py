from pathlib import Path
import random

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "data/processed/images/val"
LABELS_DIR = ROOT / "data/processed/labels/val"

OUTPUT_DIR = ROOT / "outputs/error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load both models
pretrained = YOLO("yolo11n.pt")

finetuned = YOLO(
    str(ROOT / "experiments/finetune_5_epochs/weights/best.pt")
)


def draw_ground_truth(image, label_path):

    height, width = image.shape[:2]

    for line in label_path.read_text().splitlines():

        if not line.strip():
            continue

        class_id, xc, yc, w, h = map(float, line.split())

        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)

        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return image


def main():

    images = sorted(IMAGES_DIR.glob("*.png"))

    random.seed(42)
    selected = random.sample(images, min(5, len(images)))

    for index, image_path in enumerate(selected, start=1):

        print("Analyzing:", image_path.name)

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Cannot load {image_path}")

        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        # 1. Ground truth
        ground_truth = draw_ground_truth(
            image.copy(),
            label_path
        )

        # 2. Pretrained predictions
        original_results = pretrained.predict(
            source=str(image_path),
            imgsz=320,
            conf=0.25,
            classes=[0],
            device="cpu",
            verbose=False
        )

        original_image = original_results[0].plot()

        # 3. Fine-tuned predictions
        finetuned_results = finetuned.predict(
            source=str(image_path),
            imgsz=320,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        finetuned_image = finetuned_results[0].plot()

        # Save three images
        cv2.imwrite(
            str(OUTPUT_DIR / f"{index}_ground_truth.jpg"),
            ground_truth
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{index}_pretrained.jpg"),
            original_image
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{index}_finetuned.jpg"),
            finetuned_image
        )

        print(
            "Original detections:",
            len(original_results[0].boxes)
        )

        print(
            "Fine-tuned detections:",
            len(finetuned_results[0].boxes)
        )

        print("------------------------")


if __name__ == "__main__":
    main()