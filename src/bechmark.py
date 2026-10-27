from pathlib import Path
from ultralytics import YOLO
import csv

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset.yaml"
EXPERIMENTS = ROOT / "experiments"

# Our selected model
MODEL_PATH = ROOT / "yolo11n.pt"

# Resolutions to benchmark
IMAGE_SIZES = [320, 416, 640]


def main():

    model = YOLO(str(MODEL_PATH))

    results = []

    for size in IMAGE_SIZES:

        print(f"\nBenchmarking resolution: {size}")

        metrics = model.val(
            data=str(DATASET),
            split="val",
            imgsz=size,
            batch=1,
            device="cpu",
            workers=0,
            classes=[0],
            plots=False,
            project=str(EXPERIMENTS),
            name=f"benchmark_{size}",
            exist_ok=True
        )

        # Ultralytics reports milliseconds per image
        inference_ms = metrics.speed["inference"]

        result = {
            "imgsz": size,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "map50": metrics.box.map50,
            "map50_95": metrics.box.map,
            "inference_ms": inference_ms,
            "fps": 1000 / inference_ms
            if inference_ms > 0 else 0
        }

        results.append(result)

        print("mAP50:", round(result["map50"], 4))
        print("mAP50-95:", round(result["map50_95"], 4))
        print("Inference:", round(inference_ms, 2), "ms")
        print("FPS:", round(result["fps"], 2))

    # Save benchmark results
    csv_path = EXPERIMENTS / "benchmark_results.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "imgsz",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "inference_ms",
                "fps"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nBenchmark completed!")
    print("Saved results to:", csv_path)


if __name__ == "__main__":
    main()