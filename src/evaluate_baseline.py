from pathlib import Path

from ultralytics import YOLO


def main():

    # -----------------------------------
    # 1. Project paths
    # -----------------------------------

    ROOT = Path(__file__).resolve().parent.parent

    DATASET = ROOT / "dataset.yaml"

    EXPERIMENTS = ROOT / "experiments"

    # -----------------------------------
    # 2. Load pretrained model
    # -----------------------------------

    model = YOLO("yolo11n.pt")

    print("Model classes:", len(model.names))
    print("Class 0:", model.names[0])

    # -----------------------------------
    # 3. Validate model
    # -----------------------------------

    metrics = model.val(
        data=str(DATASET),
        split="val",
        imgsz=320,
        batch=1,
        device="cpu",
        workers=0,
        classes=[0],
        plots=True,
        project=str(EXPERIMENTS),
        name="baseline_pretrained",
        exist_ok=True
    )

    # -----------------------------------
    # 4. Print evaluation metrics
    # -----------------------------------

    print("\n===== BASELINE RESULTS =====")

    print("Precision:", round(metrics.box.mp, 4))
    print("Recall:", round(metrics.box.mr, 4))

    print("mAP50:", round(metrics.box.map50, 4))
    print("mAP50-95:", round(metrics.box.map, 4))

    print("============================")


if __name__ == "__main__":
    main()