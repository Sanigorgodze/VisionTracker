from pathlib import Path
from ultralytics import YOLO
import time 


#----------------------
# Project configuration
#-----------------------
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset.yaml"
EXPERIMENTS = ROOT / "experiments"


def main():
    print("Starting Vision trackng....")

    model = YOLO("yolo11n.pt")

    start_time = time.perf_counter()

    #Train model 
    model.train(
        data = str(DATASET),
        epochs = 15,
        imgsz = 320,
        batch = 2,

        device = "cpu",
        workers = 0,
        seed=42,
        deterministic=True,
        optimizer="AdamW",
        lr0 = 0.0001,
        project = str(EXPERIMENTS),
        name = "finetune_15_epochs_low_lr",
        plots = True
    )

    elapsed = time.perf_counter() - start_time

    print("\n==================")
    print("TRAINING COMPLLETED")
    print("====================")

    print(f"Training duration: {elapsed:.2f} seconds")

    print("Model saved to: ", EXPERIMENTS / "finetune_5_epochs/weights/best.pt")

if __name__ == "__main__":
    main()