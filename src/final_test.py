from pathlib import Path
from datetime import datetime, timezone

import json

from ultralytics import YOLO


# -----------------------------------
# 1. Project configuration
# -----------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset.yaml"

MODEL_PATH = ROOT / "yolo11n.pt"

EXPERIMENTS = ROOT / "experiments"

EXPERIMENTS.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# 2. Final evaluation
# -----------------------------------

def main():

    print("\nStarting final model evaluation...")

    # Load our selected model
    model = YOLO(str(MODEL_PATH))

    # Evaluate using TEST images
    metrics = model.val(
        data=str(DATASET),

        split="test",

        imgsz=320,
        batch=1,

        device="cpu",
        workers=0,

        classes=[0],

        plots=True,

        project=str(EXPERIMENTS),
        name="final_test",

        exist_ok=True
    )

    # -----------------------------------
    # 3. Collect results
    # -----------------------------------

    results = {
        "model": "YOLO11n pretrained",
        "dataset": "Penn-Fudan Pedestrian",
        "split": "test",
        "image_size": 320,
        "device": "cpu",

        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),

        "inference_ms": float(
            metrics.speed["inference"]
        ),

        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    # -----------------------------------
    # 4. Save metrics to JSON
    # -----------------------------------

    output_path = (
        EXPERIMENTS / "final_test_metrics.json"
    )

    with open(output_path, "w") as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # -----------------------------------
    # 5. Display results
    # -----------------------------------

    print("\n========== FINAL TEST RESULTS ==========")

    print(f"Model:       {results['model']}")
    print(f"Precision:   {results['precision']:.4f}")
    print(f"Recall:      {results['recall']:.4f}")
    print(f"mAP50:       {results['mAP50']:.4f}")
    print(f"mAP50-95:    {results['mAP50_95']:.4f}")

    print(
        f"Inference:   {results['inference_ms']:.2f} ms"
    )

    print("\nSaved to:", output_path)

    print("========================================")


if __name__ == "__main__":
    main()