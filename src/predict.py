from pathlib import Path

from ultralytics import YOLO
import cv2


# -----------------------------------
# 1. Project directories
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

IMAGES_DIR = ROOT / "data/processed/images/val"

OUTPUT_DIR = ROOT / "outputs/predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# 2. Load pretrained YOLO model
# -----------------------------------

model = YOLO("yolo11n.pt")


# -----------------------------------
# 3. Select validation image
# -----------------------------------

image_paths = sorted(IMAGES_DIR.glob("*.png"))

if not image_paths:
    raise RuntimeError("No validation images found.")

image_path = image_paths[0]

print("Testing image:", image_path.name)


# -----------------------------------
# 4. Run object detection
# -----------------------------------

results = model.predict(
    source=str(image_path),
    imgsz=320,
    conf=0.25,
    classes=[0],
    device="cpu",
    verbose=False
)


# -----------------------------------
# 5. Inspect predictions
# -----------------------------------

result = results[0]

for box in result.boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    coordinates = box.xyxy[0].tolist()

    print("Class:", model.names[class_id])
    print("Confidence:", round(confidence, 3))
    print("Bounding box:", coordinates)
    print("-----------------------")


# -----------------------------------
# 6. Save predictions
# -----------------------------------

annotated_image = result.plot()

output_path = OUTPUT_DIR / "pretrained_prediction.jpg"

cv2.imwrite(str(output_path), annotated_image)

print("Prediction saved to:", output_path)