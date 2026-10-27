from pathlib import Path
from ultralytics import YOLO
import csv


ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset.yaml"
EXPERIMENTS = ROOT / "experiments"


MODELS = {
    "Pretrained": ROOT / "yolo11n.pt",

    "Fine-tuned LR 0.001": (
        EXPERIMENTS
        / "finetune_5_epochs/weights/best.pt"
    ),

    "Fine-tuned LR 0.0001": (
        EXPERIMENTS
        / "finetune_15_epochs_low_lr/weights/best.pt"
    )
}


def evaluate_model(name, model_path):

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\nEvaluating: {name}")

    model = YOLO(str(model_path))

    assert model.names[0] == "person"

    metrics = model.val(
        data=str(DATASET),
        split="val",

        imgsz=320,
        batch=1,

        device="cpu",
        workers=0,

        classes=[0],

        project=str(EXPERIMENTS),
        name="comparison_" + name.lower().replace(" ", "_"),
        exist_ok=True
    )

    return {
        "model": name,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map
    }


def main():

    results = []

    for name, path in MODELS.items():
        result = evaluate_model(name, path)
        results.append(result)

    print("\n========== RESULTS ==========")

    for result in results:

        print(f"\n{result['model']}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall:    {result['recall']:.4f}")
        print(f"mAP50:     {result['map50']:.4f}")
        print(f"mAP50-95:  {result['map50_95']:.4f}")

    # Save results as CSV
    csv_path = EXPERIMENTS / "model_comparison.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "precision",
                "recall",
                "map50",
                "map50_95"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:", csv_path)


if __name__ == "__main__":
    main()