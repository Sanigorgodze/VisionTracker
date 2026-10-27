from pathlib import Path
from time import perf_counter
import base64

import cv2
import numpy as np

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO


# ----------------------------------
# Configuration
# ----------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "yolo11n.pt"
HTML_PATH = ROOT / "static/index.html"

IMAGE_SIZE = 320
CONFIDENCE = 0.31

MAX_FILE_SIZE = 10 * 1024 * 1024


# ----------------------------------
# Initialize application
# ----------------------------------

app = FastAPI(
    title="VisionTrack",
    description="AI Pedestrian Detection",
    version="1.0"
)

model = YOLO(str(MODEL_PATH))


# ----------------------------------
# Frontend
# ----------------------------------

@app.get("/")
def home():
    return FileResponse(str(HTML_PATH))


# ----------------------------------
# Health check
# ----------------------------------

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "YOLO11n",
        "device": "cpu"
    }


# ----------------------------------
# Pedestrian detection
# ----------------------------------

@app.post("/predict")
def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    image_bytes = file.file.read(MAX_FILE_SIZE + 1)

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is too large (maximum 10 MB)"
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    start = perf_counter()

    # YOLO prediction
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        classes=[0],
        device="cpu",
        verbose=False
    )

    result = results[0]

    # Draw boxes on image
    annotated = result.plot()

    # Convert image to JPEG bytes
    success, encoded = cv2.imencode(
        ".jpg",
        annotated
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode result image"
        )

    # Convert image to Base64 string
    image_base64 = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    # Extract detections
    detections = []

    for box in result.boxes:

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": "person",
            "confidence": round(
                float(box.conf[0]), 4
            ),
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ]
        })

    processing_ms = (
        perf_counter() - start
    ) * 1000

    return {
        "person_count": len(detections),
        "processing_ms": round(processing_ms, 2),
        "model": "YOLO11n",
        "image_size": IMAGE_SIZE,
        "detections": detections,
        "annotated_image": image_base64
    }