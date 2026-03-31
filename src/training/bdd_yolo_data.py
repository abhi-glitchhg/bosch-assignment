import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------

BASE_PATH = Path("../../data/assignment_data_bdd")

TRAIN_JSON = BASE_PATH / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
VAL_JSON = BASE_PATH / "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

OUTPUT_LABELS = BASE_PATH / "labels"


CLASS_NAMES = [
    "person", "rider", "car", "truck", "bus",
    "train", "bike", "motor",
    "traffic light", "traffic sign"
]

CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}



def load_bdd(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for img in tqdm(data, desc=f"Parsing {json_path.name}"):
        img_name = img["name"]

        width = img.get("attributes", {}).get("resolution", {}).get("width", 1280)
        height = img.get("attributes", {}).get("resolution", {}).get("height", 720)

        if "labels" not in img:
            continue

        for obj in img["labels"]:
            if "box2d" not in obj:
                continue

            cls = obj.get("category")
            if cls not in CLASS_MAP:
                continue

            box = obj["box2d"]

            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]

            w = x2 - x1
            h = y2 - y1

            rows.append({
                "image": img_name,
                "class": cls,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "area": w * h
            })

    return pd.DataFrame(rows)


def remove_small_bboxes(df, ratio=0.0002):
    """
    Remove very small boxes (adaptive threshold).
    Default ≈ 0.02% of image area
    """
    img_area = df["width"] * df["height"]
    return df[df["area"] > img_area * ratio]

# ----------------------------
# CONVERT TO YOLO
# ----------------------------

def convert_to_yolo(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby("image")

    for img_name, group in tqdm(grouped, desc=f"Writing {output_dir}"):
        label_path = Path(output_dir) / img_name.replace(".jpg", ".txt")

        with open(label_path, "w") as f:
            for _, row in group.iterrows():
                cls_id = CLASS_MAP[row["class"]]

                x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                W, H = row["width"], row["height"]

                # YOLO format (normalized)
                x = ((x1 + x2) / 2) / W
                y = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H

                f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# ----------------------------
# MAIN PIPELINE
# ----------------------------

def process_split(json_path, split_name):
    df = load_bdd(json_path)

    print(f"\n{split_name} before filtering: {len(df)}")
    df = remove_small_bboxes(df)
    print(f"{split_name} after filtering: {len(df)}")

    convert_to_yolo(
        df,
        OUTPUT_LABELS / split_name
    )


def main():
    process_split(TRAIN_JSON, "train")
    process_split(VAL_JSON, "val")

    print(" DONE: YOLO labels ready")


if __name__ == "__main__":
    main()