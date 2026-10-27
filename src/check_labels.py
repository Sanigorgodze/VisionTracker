from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data/processed/labels"

total_boxes = 0
errors = []

for label_path in sorted(LABELS_DIR.rglob("*.txt")):

    lines = label_path.read_text().splitlines()

    for line_number, line in enumerate(lines, start=1):

        if not line.strip():
            continue

        try:
            values = line.split()

            if len(values) != 5:
                raise ValueError("Expected five values")

            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])

            if class_id != 0:
                raise ValueError("Unexpected class ID")

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
                and 0 <= x - w / 2
                and x + w / 2 <= 1
                and 0 <= y - h / 2
                and y + h / 2 <= 1
            ):
                raise ValueError("Invalid bounding-box coordinates")

            total_boxes += 1

        except (ValueError, TypeError) as error:
            errors.append(
                f"{label_path.name}:{line_number} - {error}"
            )


print("Total valid bounding boxes:", total_boxes)
print("Errors:", len(errors))

for error in errors:
    print(error)